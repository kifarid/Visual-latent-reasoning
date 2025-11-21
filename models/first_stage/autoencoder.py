import math
import random

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.optim.lr_scheduler import LambdaLR

from util import instantiate_from_config


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, detach_mean=False):
        self.detach_mean = detach_mean
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        noise = self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        if self.detach_mean:
            x = self.mean.detach() + noise
        else:
            x = self.mean + noise
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    

class AutoencoderKL(pl.LightningModule):
    def __init__(self, 
                encoder_config,
                decoder_config,
                loss_config,
                grad_acc_steps=1,
                ckpt_path=None,
                ignore_keys=[],
                monitor=None,
                distill_model_type="VIT_DINOv2",
                min_lr_multiplier=0.1,
                only_decoder=False,
                scale_equivariance=None,
                ):
        super().__init__()
        
        self.automatic_optimization = False

        if not hasattr(decoder_config, 'params'):
            decoder_config.params = encoder_config.params
        #print (encoder_config)
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.grad_acc_steps = grad_acc_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.only_decoder = only_decoder
        self.scale_equivariance = scale_equivariance
        self.distill_model_type = distill_model_type
        
        assert (not scale_equivariance) or len(scale_equivariance) == 2, "if defined, scale_equivariance should be a list of two lists"
        self.scale_equivariance = scale_equivariance
        
        self.image_size = encoder_config.params['resolution']
        self.patch_size = encoder_config.params["patch_size"]
        

        self.quant_conv = torch.nn.Conv2d(encoder_config.params["z_channels"], 2*encoder_config.params['e_dim'], 1)
        self.post_quant_conv = torch.nn.Conv2d(decoder_config.params['e_dim'], decoder_config.params["z_channels"], 1)
        self.if_distill_loss = False if loss_config.params.get('distill_loss_weight', 0.0) == 0.0 else True
        
        self.grad_acc_steps = grad_acc_steps

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
            
        self._init_distill_model(distill_model_type)
            
            
    def _init_distill_model(self, distill_type):
        q_e_dim = self.encoder_config.params["e_dim"]
        z_channels = self.decoder_config.params["z_channels"]

        def conv1x1(in_c, out_c): return nn.Conv2d(in_c, out_c, 1)

        if distill_type == "VIT_DINO":
            self.distill = timm.create_model("timm/vit_base_patch16_224.dino", img_size=self.image_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2":
            img_size = self._compute_scaled_size(self.image_size, self.patch_size)
            self.distill = timm.create_model("timm/vit_base_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2g":
            img_size = int(self.image_size * 14 / self.patch_size)
            self.distill = timm.create_model("timm/vit_giant_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, 1536)
        elif distill_type == "VIT_DINOv2_large":
            img_size = int(self.image_size * 14 / self.patch_size)
            self.distill = timm.create_model("timm/vit_large_patch14_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "VIT_DINOv2_large_reg4":
            img_size = int(self.image_size * 14 / self.patch_size)
            self.distill = timm.create_model("timm/vit_large_patch14_reg4_dinov2.lvd142m", img_size=img_size, pretrained=True).eval()
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
        elif distill_type == "SAM_VIT":
            self.distill = timm.create_model("samvit_large_patch16.sa1b", pretrained=True)
            self.post_quant_conv_distill = nn.Identity()
        elif distill_type == "SAM_VIT_w_conv":
            self.distill = timm.create_model("samvit_large_patch16.sa1b", pretrained=True)
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)

        elif distill_type == "depth_anything_VIT_L14":
            self.distill = timm.create_model("vit_large_patch14_dinov2.lvd142m", img_size=224, pretrained=False)
            state_dict = torch.load("./pretrained_models/depth_anything_vitl14.pth")
            state_dict = {k.replace("pretrained.", "", 1): v for k, v in state_dict.items()}
            state_dict["pos_embed"] = resample_abs_pos_embed(state_dict["pos_embed"], new_size=(16, 16))
            self.distill.load_state_dict(state_dict, strict=False)
            self.post_quant_conv_distill = conv1x1(q_e_dim, z_channels)
            
    @staticmethod
    def _compute_scaled_size(image_size, patch_size):
        if isinstance(image_size, int):
            return [image_size * 14 // patch_size] * 2
        return [image_size[0] * 14 // patch_size, image_size[1] * 14 // patch_size]
            
    def get_warmup_scheduler(self, optimizer, warmup_steps, min_lr_multiplier):
        min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                #print(f"Step: {step}, Warmup: {warmup_steps}, Warmup Start LR: {warmup_start_lr}, Max LR: {max_lr}")
                return step/warmup_steps
            # After warmup_steps, we just return 1. This could be modified to implement your own schedule
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr) * cosine_decay + min_lr
                return decayed
        return LambdaLR(optimizer, lr_lambda)


    def get_input(self, batch):
        x = batch['images']
        return x.float()
    
    def encode(self, x, detach_mean=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, detach_mean=detach_mean)
        ret = {
            'posterior': posterior,
            'continuous': posterior.sample()
        }
        return ret

    def decode(self, z, return_distill_conv_out=False):
        z_post_quant = self.post_quant_conv(z)
        dec = self.decoder(z_post_quant)
        if return_distill_conv_out:
            distill_conv_out = self.post_quant_conv_distill(z)
            return dec, distill_conv_out
        return dec
    

    def forward(self, input, detach_mean=False):
        encoded = self.encode(input, detach_mean=detach_mean)
        dec, distill_conv_out = self.decode(encoded['continuous'], return_distill_conv_out=True)
        return dec, encoded['posterior'], distill_conv_out

    
    def forward_se(self, input):        
        encoded = self.encode(input)
        dec, distill_conv_out = self.decode(encoded['continuous'], return_distill_conv_out=True)
        
        random_scale = [random.choice(self.scale_equivariance[0]), random.choice(self.scale_equivariance[1])]
        downscale_factor = [1/random_scale[0], 1/random_scale[1]]
        z_se = F.interpolate(encoded['continuous'], scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        dec_se = self.decode(z_se)
        
        input_se = F.interpolate(input, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        decs = [dec, dec_se]
        inputs = [input, input_se]
        return inputs, decs, encoded['posterior'], distill_conv_out
    
    
    def distill_loss(self, distill_output, decoder_distill_output):
        #print(f'DINO loss calculation')
        if 'VIT' in self.distill_model_type:
            if 'reg4' in self.distill_model_type:
                distill_output = distill_output[:, 5:, :] # [CLS, Register*4, Embeddings]
            elif 'reg4' not in self.distill_model_type and 'DINO' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :] # uncomment for DINOv1 
            elif 'depth_anything' in self.distill_model_type:
                distill_output = distill_output[:, 1:, :]
            elif self.distill_model_type == 'SAM_VIT':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                distill_output = F.normalize(distill_output, p=2, dim=2) # without post_conv layer
            elif self.distill_model_type == 'SAM_VIT_w_conv':
                distill_output = distill_output.permute(0, 2, 3, 1).contiguous().view(distill_output.shape[0], -1, distill_output.shape[1])
                # without L2 normalization
            distill_output = distill_output.permute(0, 2, 1).contiguous()
        
        elif self.distill_model_type == 'CNN':
            distill_output = distill_output.view(distill_output.shape[0], distill_output.shape[1], -1)
        decoder_distill_output = decoder_distill_output.view(decoder_distill_output.shape[0], decoder_distill_output.shape[1], -1)
        #print (f'distill_output.shape: {distill_output.shape} decoder_distill_output.shape: {decoder_distill_output.shape}')
        cos_similarity = F.cosine_similarity(decoder_distill_output, distill_output, dim=1)
        cosine_loss = 1 - cos_similarity
        distill_loss = cosine_loss.mean()
        return distill_loss


    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()
        
        if self.only_decoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv_distill.parameters():
                param.requires_grad = False

        x = self.get_input(batch)

        if self.scale_equivariance:
            xs, xrec, posterior, decoder_distill_output = self.forward_se(x)
        else:
            xrec, posterior, decoder_distill_output = self(x)
            xs = x
            
        distill_loss = self.distill_loss(self.get_distill_gt(x), decoder_distill_output) if self.if_distill_loss else torch.tensor(0.0, device=x.device)


        optimizer_idx = 0

        aeloss, log_dict_ae = self.loss(posterior, distill_loss, xs, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

        
        optimizer_idx = 1
        # train the discriminator
        discloss, log_dict_disc  = self.loss(posterior, distill_loss, xs, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()

    def validation_step(self, batch, batch_idx):
        
        x = self.get_input(batch)
        
        xrec, posterior, decoder_distill_output = self(x)
        
        distill_loss = self.distill_loss(self.get_distill_gt(x), decoder_distill_output) if self.if_distill_loss else torch.tensor(0.0, device=x.device)

        optimizer_idx = 1

        discloss, log_dict_disc  = self.loss(posterior, distill_loss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="val")

        self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        optimizer_idx = 0
        aeloss, log_dict_ae = self.loss(posterior, distill_loss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))

        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.min_lr_multiplier)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]
    
    def get_distill_gt(self, x):
        with torch.no_grad():
            if 'VIT' in self.distill_model_type:
                # resize image x to 224x224
                if 'VIT_DINOv2' in self.distill_model_type or 'depth_anything' in self.distill_model_type:
                    image_size = (self.image_size*14//self.patch_size, self.image_size*14//self.patch_size) if isinstance(self.image_size, int) else (self.image_size[0]*14//self.patch_size, self.image_size[1]*14//self.patch_size)
                    x_224 = F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)
                    distill_output = self.distill.forward_features(x_224)
                else: # for VIT-DINOv1, VIT-SAM models
                    distill_output = self.distill.forward_features(x)

            elif self.distill_model_type == 'CNN':
                distill_output = self.distill(x)
        return distill_output
    
    
    
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch)
        x = x.to(self.device)
        xrec = self(x)[0]
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


    def get_last_layer(self):
        return self.decoder.conv_out.weight


class AutoencoderKLIF(AutoencoderKL):
    def __init__(self, 
                encoder_config,
                decoder_config,
                loss_config,
                grad_acc_steps=1,
                ckpt_path=None,
                ignore_keys=[],
                monitor=None,
                distill_model_type="VIT_DINOv2",
                min_lr_multiplier=0.1,
                only_decoder=False,
                scale_equivariance=None,
                ):
        super().__init__(                
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                loss_config=loss_config,
                grad_acc_steps=grad_acc_steps,
                ckpt_path=ckpt_path,
                ignore_keys=ignore_keys,
                monitor=monitor,
                distill_model_type=distill_model_type,
                min_lr_multiplier=min_lr_multiplier,
                only_decoder=only_decoder,
                scale_equivariance=scale_equivariance)
        
        self.encoder2 = instantiate_from_config(encoder_config)
        self.quant_conv2 = torch.nn.Conv2d(encoder_config.params["z_channels"], encoder_config.params['e_dim'], 1)
        
    
    def encode(self, x):
        h_ = self.encoder(x)
        moments = self.quant_conv(h_)
        posterior = DiagonalGaussianDistribution(moments)
        h = posterior.sample()

        h2 = self.encoder2(x)
        h2 = self.quant_conv2(h2)
        
        ret = {
            'posterior': posterior,
            'continuous': (h, h2)
        }
        return ret
    
    
    def decode(self, z, return_distill_conv_out=False):
        z_recon = torch.concat(z, dim=1)
        z_post_recon = self.post_quant_conv(z_recon)
        dec = self.decoder(z_post_recon)
        if return_distill_conv_out:
            z_semantic = z[1]
            distill_conv_out = self.post_quant_conv_distill(z_semantic)
            return dec, distill_conv_out
        return dec
    
    
    def forward_se(self, input):        
        encoded = self.encode(input)
        dec, distill_conv_out = self.decode(encoded['continuous'], return_distill_conv_out=True)
        
        random_scale = [random.choice(self.scale_equivariance[0]), random.choice(self.scale_equivariance[1])]
        downscale_factor = [1/random_scale[0], 1/random_scale[1]]
        z = torch.concat(encoded['continuous'], dim=1)
        z_down = F.interpolate(z, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        z_down_tuple = torch.chunk(z_down, 2, dim=1)
        dec_se = self.decode(z_down_tuple)
        
        input_se = F.interpolate(input, scale_factor=downscale_factor, mode='bilinear', align_corners=False)
        decs = [dec, dec_se]
        inputs = [input, input_se]
        return inputs, decs, encoded['posterior'], distill_conv_out
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.encoder2.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.quant_conv2.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.post_quant_conv_distill.parameters()),
                                  lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(self.loss.beta_1, self.loss.beta_2))

        scheduler_ae_warmup = self.get_warmup_scheduler(opt_ae, self.loss.warmup_steps, self.min_lr_multiplier)
        scheduler_disc_warmup = self.get_warmup_scheduler(opt_disc, self.loss.warmup_steps, self.min_lr_multiplier)
        

        return [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup]

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        [scheduler_ae_warmup, scheduler_disc_warmup] = self.lr_schedulers()
        
        if self.only_decoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv_distill.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            for param in self.quant_conv2.parameters():
                param.requires_grad = False
                
                
        x = self.get_input(batch)

        if self.scale_equivariance:
            xs, xrec, posterior, decoder_distill_output = self.forward_se(x)
        else:
            xrec, posterior, decoder_distill_output = self(x)
            xs = x
            
        distill_loss = self.distill_loss(self.get_distill_gt(x), decoder_distill_output) if self.if_distill_loss else torch.tensor(0.0, device=x.device)


        optimizer_idx = 0

        aeloss, log_dict_ae = self.loss(posterior, distill_loss, xs, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.grad_acc_steps
        self.manual_backward(aeloss) 
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()
            scheduler_ae_warmup.step()

        
        optimizer_idx = 1
        # train the discriminator
        discloss, log_dict_disc  = self.loss(posterior, distill_loss, xs, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss = discloss / self.grad_acc_steps
        self.manual_backward(discloss)
        if (batch_idx+1) % self.grad_acc_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            scheduler_disc_warmup.step()