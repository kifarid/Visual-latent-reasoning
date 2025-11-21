import os
import math
from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from omegaconf import OmegaConf, ListConfig

import time

start_time = time.time() 

from util import instantiate_from_config

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """ª
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
        
def init_ema_model(model):
    ema_model = deepcopy(model)
    requires_grad(ema_model, False)
    update_ema(ema_model, model, decay=0)
    ema_model.eval()
    return ema_model


def build_mlp(hidden_size, projector_dim, z_dim):
    mlp = nn.Linear(hidden_size, projector_dim)
    torch.nn.init.xavier_uniform_(mlp.weight)
    if mlp.bias is not None:
        nn.init.constant_(mlp.bias, 0)
    return mlp

class ModelSR(pl.LightningModule):
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, sigma_min=1e-5, timescale=1.0, enc_scale=4, warmup_steps=5000, min_lr_multiplier=0.1,num_pred_frames=1, sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                sr_loss_weight=1.0,
                ctx_noise_prob=0.8,
                tube_ctx_mask = True,
                loss_type = "cos_sim",
                ctx_noise_exp = 0.5
    ):
        super().__init__()

        self.num_pred_frames = num_pred_frames
        self.enc_scale = enc_scale

        # Training parameters
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.loss_type = loss_type

        
        # denoising backbone
        self.vit = self.build_generator(generator_config)
        
        # Tokenizer
        self.ae = self.build_tokenizer(tokenizer_config)

        # EMA
        self.ema_vit = init_ema_model(self.vit)
        self.sigma_min = sigma_min
        self.timescale = timescale

        self.sr_latents = sr_latents
        self.add_projector = add_projector
        self.add_norm = add_norm
        self.layer_norm = layer_norm
        self.recon_loss_weight = recon_loss_weight
        self.sr_loss_weight = sr_loss_weight
        self.ctx_noise_exp =ctx_noise_exp

        # Build the predictor for the SL loss if needed
        self.build_projector(generator_config.params.hidden_size)
        self.ctx_noise_prob = ctx_noise_prob
        self.tube_ctx_mask = tube_ctx_mask


    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)
    
    def get_betas(self, n_timesteps):
        return torch.zeros(n_timesteps) # Not VP and not supported

    def build_tokenizer(self, tokenizer_config):
        """
        Instantiate the tokenizer model from the config.
        """
        tokenizer_folder = os.path.expandvars(tokenizer_config.folder)
        ckpt_path = tokenizer_config.ckpt_path if tokenizer_config.ckpt_path else "checkpoints/last.ckpt"
        # Load config and create
        tokenizer_config = OmegaConf.load(os.path.join(tokenizer_folder, "config.yaml"))
        model = instantiate_from_config(tokenizer_config.model)
        # Load checkpoint
        checkpoint = torch.load(os.path.join(tokenizer_folder, ckpt_path), map_location="cpu", weights_only=True)["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        return model
    
    def build_generator(self, generator_config):
        """
        Instantiate the denoising backbone model from the config.
        """
        model = instantiate_from_config(generator_config)
        return model
    
    def build_projector(self, generator_hidden_dim):
        if self.add_projector:
            if self.add_norm:
                self.predictor_norm = [instantiate_from_config(self.layer_norm) for _ in range(len(self.sr_latents)) ]
            else:
                #identity layers
                self.predictor_norm = [nn.Identity() for _ in range(len(self.sr_latents)) ]
            self.predictor_proj = [build_mlp(generator_hidden_dim, generator_hidden_dim, generator_hidden_dim) for _ in range(len(self.sr_latents)) ]
            #move to GPU
            self.predictor_proj = nn.ModuleList(self.predictor_proj).to(device=self.device)
            self.predictor_norm = nn.ModuleList(self.predictor_norm).to(device=self.device)
        else:
            self.predictor_norm = None
            self.predictor_proj = None
        
    def get_warmup_scheduler(self, optimizer, warmup_steps=1, min_lr_multiplier=1.0):
        #min_lr = self.learning_rate * min_lr_multiplier
        total_steps = self.trainer.max_epochs * self.num_iters_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step/warmup_steps
            else:
                progress = (min(step, total_steps) - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = (1 - min_lr_multiplier) * cosine_decay + min_lr_multiplier
                return decayed      
        
        return LambdaLR(optimizer, lr_lambda)
    
    def configure_optimizers(self):
        '''
        Prepare optimizer and schedule (linear warmup and cosine decay)
        add the predictor parameters if needed'''
        params = list(self.vit.parameters())
        if self.predictor_proj is not None:
            # add each projector exactly once
            for proj in self.predictor_proj:
                params += list(proj.parameters())
            # likewise for any norms
            if self.predictor_norm is not None:
                for norm in self.predictor_norm:
                    params += list(norm.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01, eps=1e-8)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        
        return  [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def get_input(self, batch, k):
        if type(batch) == dict:
            x = batch[k]
            frame_rate = batch.get('frame_rate', None)
        else:
            x = batch
            frame_rate = None
        assert len(x.shape) == 5, 'input must be 5D tensor'
        return x, frame_rate
    
    def patch_second_stage(self, x):
        """
        Turn a 5D latent grid into per-patch tensors. If `tube_ctx_mask` is True, we
        keep temporal tubes per spatial patch (shape: (b*h_p*w_p, f, e, p, p)).
        Otherwise, we create independent patches per frame (shape: (b*f*h_p*w_p, e, p, p)).
        Returns the patched tensor *and* a meta tuple used by `unpatch_second_stage`.
        """
        patch_size = self.vit.patch_size
        b, f, e, h, w = x.shape
        assert h % patch_size == 0 and w % patch_size == 0, 'Height and width must be divisible by patch size'

        h_p = h // patch_size
        w_p = w // patch_size

        if self.tube_ctx_mask:
            # Keep time dimension: one tube per spatial patch across all frames
            x = rearrange(
                x,
                'b f e (hp p1) (wp p2) -> (b hp wp) f e p1 p2',
                hp=h_p, wp=w_p, p1=patch_size, p2=patch_size
            )
        else:
            # Independent patches per frame
            x = rearrange(
                x,
                'b f e (hp p1) (wp p2) -> (b f hp wp) e p1 p2',
                hp=h_p, wp=w_p, p1=patch_size, p2=patch_size
            )
        return x, (b, f, e, h_p, w_p, patch_size)

    def unpatch_second_stage(self, x, meta):
        """
        Invert `patch_second_stage`. `meta` is the tuple returned by `patch_second_stage`:
        (b, f, e, h_p, w_p, patch_size)
        """
        b, f, e, h_p, w_p, patch_size = meta
        if self.tube_ctx_mask:
            # x: (b*h_p*w_p, f, e, p, p) -> (b, f, e, h, w)
            x = rearrange(
                x,
                '(b hp wp) f e p1 p2 -> b f e (hp p1) (wp p2)',
                b=b, hp=h_p, wp=w_p, p1=patch_size, p2=patch_size
            )
        else:
            # x: (b*f*h_p*w_p, e, p, p) -> (b, f, e, h, w)
            x = rearrange(
                x,
                '(b f hp wp) e p1 p2 -> b f e (hp p1) (wp p2)',
                b=b, f=f, hp=h_p, wp=w_p, p1=patch_size, p2=patch_size
            )
        return x

    def add_noise_ctx(self, x, noise=None):
        """
        Add diffusion-style noise to context latents per patch. Supports two modes:
        - tube_ctx_mask=True: one time/noise per spatial patch across all frames (tubes)
        - tube_ctx_mask=False: independent time/noise per (frame, patch)
        Returns the possibly noised context and the corresponding noise mapped back to
        the original grid.
        """
        # Patch
        x_patched, meta = self.patch_second_stage(x)

        # Sample one t per patched item (matches x_patched.shape[0])
        t = torch.rand((x_patched.shape[0],), device=x.device)
        t = t.pow(self.ctx_noise_exp)

        # Sample one mask per patched item
        mask = torch.rand((x_patched.shape[0],), device=x.device) < self.ctx_noise_prob

        # Prepare noise and apply forward noising
        noise = torch.randn_like(x_patched) if noise is None else noise
        s = [x_patched.shape[0]] + [1] * (x_patched.dim() - 1)
        x_t_noised = self.alpha(t).view(*s) * x_patched + self.sigma(t).view(*s) * noise

        # Mix noised vs. clean per-patch with ctx_noise_prob
        #print(mask.shape, x_t_noised.shape, x_patched.shape, x.shape)
        x_t_noised = torch.where(mask.view(-1, 1, 1, 1, 1), x_t_noised, x_patched)

        # Unpatch back to grid
        x_t_noised = self.unpatch_second_stage(x_t_noised, meta)
        noise = self.unpatch_second_stage(noise, meta)


        # Mix noised vs. clean per-sample with ctx_noise_prob
        b = x.shape[0]
        batch_mask = (torch.rand(b, device=x.device) < 0.95).view(b, 1, 1, 1, 1)
        x = torch.where(batch_mask, x_t_noised, x)
        return x, noise

    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    @torch.no_grad()
    def encode_frames(self, images):
        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1
        x = self.ae.encode(images)['continuous']
        x = x * (self.enc_scale)
        x = rearrange(x, '(b f) e h w -> b f e h w',b=b, f=f)
        return x
    
    @torch.no_grad()
    def decode_frames(self, x):
        frame = x[:,0] 
        frame = frame / (self.enc_scale)

        frame = self.ae.post_quant_conv(frame)
        samples = (self.ae.decoder(frame)).unsqueeze(1)
        for idx in range(1, x.shape[1]):
            frame = x[:,idx] 
            frame = frame / (self.enc_scale)
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            samples = torch.cat([samples, (frame).unsqueeze(1)], dim=1)
        return samples
  
    def v_loss(self, target, pred, noise, t):
        # -dxt/dt
        v = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - v.float()) ** 2)
        return loss

    def sr_loss(self, pred_latents, post_latents):
        sr_loss = 0
        sr_loss_layer = {}
        for i in range(len(pred_latents)):
            if self.predictor_proj is not None and self.predictor_norm is not None:
                pred_latents[i] = self.predictor_norm[i](self.predictor_proj[i]((pred_latents[i])))
                post_latents[i] = self.predictor_norm[i](post_latents[i])
            sr_loss_layer_unred = 0
            # Compute MSE loss if requested
            if self.loss_type in ["mse", "both"]:
                sr_loss_layer_unred = sr_loss_layer_unred + F.mse_loss(pred_latents[i], post_latents[i].detach(), reduction='none')
            # Compute cosine similarity loss if requested
            if self.loss_type in ["cos_sim", "both"]:
                sr_loss_layer_unred = sr_loss_layer_unred + (1 - F.cosine_similarity(pred_latents[i], post_latents[i].detach(), dim=-1, eps=1e-8))
            sr_loss_layer_red = sr_loss_layer_unred.mean()
            sr_loss_layer[f"layer_{self.sr_latents[i]}"] = sr_loss_layer_red
            sr_loss += sr_loss_layer_red

        sr_loss = sr_loss / len(self.sr_latents)
        return sr_loss, sr_loss_layer

        
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')

        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        post_pred, post_latents = self.ema_vit(target, context, t, return_latents=self.sr_latents)

        v_loss = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss.mean()
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, "mean") else sr_loss_val

        # Total loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Log losses as scalars
        self.log("train/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sr_loss", sr_loss_val_mean.item() if hasattr(sr_loss_val_mean, "item") else sr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # Log per-layer sr losses
        for layer_name, layer_val in sr_loss_layers.items():
            scalar_val = layer_val.item() if hasattr(layer_val, "item") else layer_val
            self.log(f"train/sr/{layer_name}", scalar_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA at the end of each epoch as fallback"""
        update_ema(self.ema_vit, self.vit)



    def validation_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        post_pred, post_latents = self.ema_vit(target, context, t, return_latents=self.sr_latents)

        v_loss = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss.mean()
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, "mean") else sr_loss_val

        # Total loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Log losses as scalars
        self.log("val/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/sr_loss", sr_loss_val_mean.item() if hasattr(sr_loss_val_mean, "item") else sr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # Log per-layer sr losses
        for layer_name, layer_val in sr_loss_layers.items():
            scalar_val = layer_val.item() if hasattr(layer_val, "item") else layer_val
            self.log(f"val/sr/{layer_name}", scalar_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    def roll_out(self, x_0, num_gen_frames=25, latent_input=True, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8):
        b, f = x_0.size(0), x_0.size(1)
        if latent_input:
            x_c = x_0.clone()
        else:
            x_c = self.encode_frames(x_0)
        
        x_all = x_c.clone()
        start_time = time.time()

        for idx in range(num_gen_frames):
            x_last_t = self.sample(images=x_c, latent=True, eta=eta, NFE=NFE, sample_with_ema=sample_with_ema, num_samples=num_samples)

            x_all = torch.cat([x_all, x_last_t[:, -self.num_pred_frames:]], dim=1)
            x_c =  x_last_t[:, -self.num_pred_frames:] if self.num_pred_frames > x_c.size(1) else torch.cat([x_c[:, self.num_pred_frames:], x_last_t[:, -self.num_pred_frames:]], dim=1)

        end_time = time.time() 
        # Calculate FPS
        total_time = end_time - start_time
        fps = num_gen_frames / total_time

        print(f"Generated {num_gen_frames} frames in {total_time:.2f} seconds ({fps:.2f} FPS)")
        samples = self.decode_frames(x_all)

        return x_all, samples
    
    @torch.no_grad()
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None, return_sample=False):
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device
        
        if images is not None:
            b, f, e, h, w = images.size()
            if not latent:
                context = self.encode_frames(images)
                b, f, e, h, w = context.size()
            else:
                context = images.clone()
        else:
            context = None

        if frame_rate is None:
            frame_rate = torch.full_like( torch.ones((num_samples,)), 5, device=device)
            
        input_h, input_w = self.vit.input_size[0], self.vit.input_size[1] if isinstance(self.vit.input_size, (list, tuple, ListConfig)) else self.vit.input_size
        target_t = torch.randn(num_samples, self.num_pred_frames, self.vit.in_channels, input_h, input_w, device=device)
        
        t_steps = torch.linspace(1, 0, NFE + 1, device=device)

        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i].repeat(target_t.shape[0])
                neg_v = net(target_t, context, t=t * self.timescale)
                dt = t_steps[i] - t_steps[i+1] 
                dw = torch.randn(target_t.size()).to(target_t.device) * torch.sqrt(dt)
                diffusion = dt
                target_t  = target_t + neg_v * dt + eta *  torch.sqrt(2 * diffusion) * dw
        if return_sample:
            images = self.decode_frames(target_t.clone())
            return target_t, images
        else:
            return target_t

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        images, frame_rate = self.get_input(batch, 'images')
        N = min(5, images.size(0))
        images = images[:N]
        frame_rate = frame_rate[:N] if frame_rate is not None else None
        b, f, e, h, w = images.size()

        l_visual_recon = [images[:,f] for f in range(images.size(1))]
        l_visual_recon_ema = [images[:,f] for f in range(images.size(1))]
        
        
        images = images[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None

        # sample
        # with self.vit.summon_full_params(self.vit, recurse=True):
        samples = self.sample(images, eta=0.0, NFE=30, sample_with_ema=False, num_samples=N, frame_rate=frame_rate, return_sample=True)[1]

        # Only keep the first generated frame
        samples = samples[:N]

        for i in range(samples.size(1)):
            l_visual_recon.append(samples[:,i])

        l_visual_recon = torch.cat(l_visual_recon, dim=0)
        chunks = torch.chunk(l_visual_recon, 2 + 2, dim=0)
        sampled = torch.cat(chunks, 0)
        sampled = vutils.make_grid(sampled, nrow=N, padding=2, normalize=False,)
        
        # sample
        # with self.ema_vit.summon_full_params(self.ema_vit, recurse=True):
        samples_ema = self.sample(images, eta=0.0, NFE=30, sample_with_ema=True, num_samples=N, return_sample=True)[1]
        # Only keep the first generated frame
        samples_ema = samples_ema[:N]

        for i in range(samples_ema.size(1)):
            l_visual_recon_ema.append(samples_ema[:,i])

        l_visual_recon_ema = torch.cat(l_visual_recon_ema, dim=0)
        chunks_ema = torch.chunk(l_visual_recon_ema, 2 + 2, dim=0)
        sampled_ema = torch.cat(chunks_ema, 0)
        sampled_ema = vutils.make_grid(sampled_ema, nrow=N, padding=2, normalize=False)

        log["ema_sampled"] = sampled_ema
        log["sampled"] = sampled
        self.vit.train()
        return log
    
    
    

class ModelSRIF(ModelSR):
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, 
                 sigma_min=1e-5, timescale=1.0, enc_scale=1.89066, enc_scale_dino=3.45062, warmup_steps=5000, min_lr_multiplier=0.1, num_pred_frames=1,
                 sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                sr_loss_weight=1.0,
                ctx_noise_prob=0.8,
                tube_ctx_mask = True,
                loss_type = "cos_sim",
                ctx_noise_exp = 0.5
    ):

        super().__init__(
            tokenizer_config=tokenizer_config,
            generator_config=generator_config,
            adjust_lr_to_batch_size=adjust_lr_to_batch_size,
            sigma_min=sigma_min,
            timescale=timescale,
            enc_scale=enc_scale,
            warmup_steps=warmup_steps,
            min_lr_multiplier=min_lr_multiplier,
            num_pred_frames=num_pred_frames,
            sr_latents=sr_latents,
            add_projector=add_projector,
            add_norm=add_norm,
            layer_norm=layer_norm,
            recon_loss_weight=recon_loss_weight,
            sr_loss_weight=sr_loss_weight,
            ctx_noise_prob=ctx_noise_prob,
            tube_ctx_mask=tube_ctx_mask,
            loss_type=loss_type,
            ctx_noise_exp=ctx_noise_exp
        )
        self.enc_scale_dino = enc_scale_dino
        
    
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        # Optional context noising (inherits from ModelSR)
        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        # Predict with return_latents
        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        post_pred, post_latents = self.ema_vit(target, context, t, return_latents=self.sr_latents)

        # v-loss over all channels; also keep split logs (recon/sem)
        v_loss_full = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss_full.mean()
        loss_recon, loss_sem = map(lambda x: x.mean(), torch.chunk(v_loss_full, 2, dim=2))

        # SR loss from latents
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, 'mean') else sr_loss_val

        # Total weighted loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Logging
        self.log("train/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_recon", loss_recon.item() if hasattr(loss_recon, "item") else loss_recon, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_sem", loss_sem.item() if hasattr(loss_sem, "item") else loss_sem, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sr_loss", sr_loss_val_mean.item() if hasattr(sr_loss_val_mean, "item") else sr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        for layer_name, layer_val in sr_loss_layers.items():
            scalar_val = layer_val.item() if hasattr(layer_val, "item") else layer_val
            self.log(f"train/sr/{layer_name}", scalar_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss



    def validation_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        post_pred, post_latents = self.ema_vit(target, context, t, return_latents=self.sr_latents)

        v_loss_full = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss_full.mean()
        loss_recon, loss_sem = map(lambda x: x.mean(), torch.chunk(v_loss_full, 2, dim=2))

        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, 'mean') else sr_loss_val

        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        self.log("val/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_recon", loss_recon.item() if hasattr(loss_recon, "item") else loss_recon, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_sem", loss_sem.item() if hasattr(loss_sem, "item") else loss_sem, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/sr_loss", sr_loss_val_mean.item() if hasattr(sr_loss_val_mean, "item") else sr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        for layer_name, layer_val in sr_loss_layers.items():
            scalar_val = layer_val.item() if hasattr(layer_val, "item") else layer_val
            self.log(f"val/sr/{layer_name}", scalar_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    @torch.no_grad()
    def encode_frames(self, images):
        if images.ndim == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1

        x = self.ae.encode(images)['continuous']
        x0, x1 = x[0] * self.enc_scale, x[1] * self.enc_scale_dino
        x = torch.cat([x0, x1], dim=1)
        x = rearrange(x, '(b f) e h w -> b f e h w', b=b, f=f)
        return x

    @torch.no_grad()
    def decode_frames(self, x):
        frame = x[:,0] 
        half = frame.size(1) // 2
        frame = torch.cat([
            frame[:, :half] / self.enc_scale,
            frame[:, half:] / self.enc_scale_dino
        ], dim=1)
        frame = self.ae.post_quant_conv(frame)
        samples = (self.ae.decoder(frame)).unsqueeze(1)
        for idx in range(1, x.shape[1]):
            frame = x[:,idx] 
            half = frame.size(1) // 2
            frame = torch.cat([
                frame[:, :half] / self.enc_scale,
                frame[:, half:] / self.enc_scale_dino
            ], dim=1)
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            samples = torch.cat([samples, (frame).unsqueeze(1)], dim=1)
        return samples