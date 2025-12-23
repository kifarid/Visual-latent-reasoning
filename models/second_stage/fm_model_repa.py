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
from omegaconf import OmegaConf, ListConfig, DictConfig

import time
import random

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

class ModelREPA(pl.LightningModule):
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, sigma_min=1e-5, timescale=1.0, enc_scale=4, warmup_steps=5000, min_lr_multiplier=0.1,num_pred_frames=1, sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                sr_loss_weight=1.0,
                ctx_noise_prob=0.8,
                tube_ctx_mask = True,
                loss_type = "cos_sim",
                latent_dim= 768,
                ctx_noise_exp=0.5,
                max_patch_size_second_stage=(6,16),
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
        self.ctx_noise_exp = ctx_noise_exp

        # Build the predictor for the SL loss if needed
        self.build_projector(generator_config.params.hidden_size, hidden_dim=latent_dim)
        self.ctx_noise_prob = ctx_noise_prob
        self.tube_ctx_mask = tube_ctx_mask
        self.max_patch_size_second_stage = max_patch_size_second_stage


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
    
    def build_projector(self, generator_hidden_dim, hidden_dim):
        if self.add_projector:
            if self.add_norm:
                self.predictor_norm = [instantiate_from_config(self.layer_norm) for _ in range(len(self.sr_latents)) ]
            else:
                #identity layers
                self.predictor_norm = [nn.Identity() for _ in range(len(self.sr_latents)) ]
            self.predictor_proj = [build_mlp(generator_hidden_dim, hidden_dim, hidden_dim) for _ in range(len(self.sr_latents)) ]
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
            frame_rate = None
        else:
            x = batch
            frame_rate = None
        assert len(x.shape) == 5, 'input must be 5D tensor'
        return x, frame_rate
    
    def patch_second_stage(self, x, patch_size=None):
        """
        Turn a 5D latent grid into per-patch tensors. If `tube_ctx_mask` is True, we
        keep temporal tubes per spatial patch (shape: (b*h_p*w_p, f, e, p, p)).
        Otherwise, we create independent patches per frame (shape: (b*f*h_p*w_p, e, p, p)).
        Returns the patched tensor *and* a meta tuple used by `unpatch_second_stage`.
        """
        patch_size_h, patch_size_w = (self.vit.patch_size, self.vit.patch_size) if patch_size is None else patch_size
        b, f, e, h, w = x.shape
        assert h % patch_size_h == 0 and w % patch_size_w == 0, 'Height and width must be divisible by patch size'

        h_p = h // patch_size_h
        w_p = w // patch_size_w

        if self.tube_ctx_mask:
            # Keep time dimension: one tube per spatial patch across all frames
            x = rearrange(
                x,
                'b f e (hp p1) (wp p2) -> (b hp wp) f e p1 p2',
                hp=h_p, wp=w_p, p1=patch_size_h, p2=patch_size_w
            )
        else:
            # Independent patches per frame
            x = rearrange(
                x,
                'b f e (hp p1) (wp p2) -> (b f hp wp) e p1 p2',
                hp=h_p, wp=w_p, p1=patch_size_h, p2=patch_size_w
            )
        return x, (b, f, e, h_p, w_p, patch_size_h, patch_size_w)

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

    def add_noise_ctx(self, x, noise=None, patch_size=None, mask=None, batch_mask=None, t=None):
        """
        Add diffusion-style noise to context latents per patch. Supports two modes:
        - tube_ctx_mask=True: one time/noise per spatial patch across all frames (tubes)
        - tube_ctx_mask=False: independent time/noise per (frame, patch)
        Returns the possibly noised context and the corresponding noise mapped back to
        the original grid.
        """
        # Patch
        # random patch size between ViT patch and configured max, respecting divisibility
        _, _, _, h, w = x.shape
        vit_patch = self.vit.patch_size
        if isinstance(vit_patch, (tuple, list)):
            vit_patch_h, vit_patch_w = vit_patch[:2]
        else:
            vit_patch_h = vit_patch_w = vit_patch

        if patch_size is not None:
            if isinstance(patch_size, int):
                patch_size_h = patch_size_w = patch_size
            else:
                patch_size_h, patch_size_w = patch_size[:2]
        else:
            max_patch = self.max_patch_size_second_stage
            if isinstance(max_patch, (tuple, list, ListConfig)):
                if len(max_patch) == 1:
                    max_patch_h = max_patch_w = max_patch[0]
                else:
                    max_patch_h, max_patch_w = max_patch[:2]
            else:
                max_patch_h = max_patch_w = max_patch
            max_patch_h = max(max_patch_h, vit_patch_h)
            max_patch_w = max(max_patch_w, vit_patch_w)
            candidates_h = [p for p in range(vit_patch_h, max_patch_h + 1) if h % p == 0] or [vit_patch_h]
            candidates_w = [p for p in range(vit_patch_w, max_patch_w + 1) if w % p == 0] or [vit_patch_w]
            patch_size_h = random.choice(candidates_h)
            patch_size_w = random.choice(candidates_w)

        x_patched, meta = self.patch_second_stage(x, patch_size=(patch_size_h, patch_size_w))

        # Sample one t per patched item (matches x_patched.shape[0])
        t = torch.rand((x_patched.shape[0],), device=x.device).pow(self.ctx_noise_exp) if t is None else t

        # Sample one mask per patched item
        mask = torch.rand((x_patched.shape[0],), device=x.device) < self.ctx_noise_prob if mask is None else mask

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
        batch_mask = (torch.rand(b, device=x.device) < 0.95).view(b, 1, 1, 1, 1) if batch_mask is None else batch_mask
        x = torch.where(batch_mask, x_t_noised, x)
        return x, (noise, mask, batch_mask, t, (patch_size_h, patch_size_w))

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
        # Debug prints
        #print(f"post_latents_shapes: {tuple(post_latents.shape)}")
        # post is (B, F, C, H, W); each pred[i] is  (B, F, N, D)
        Bp, Fp, C, H, W = post_latents.shape

        sr_loss = 0.0
        sr_loss_layer = {}
        post_base = post_latents

        for i in range(len(pred_latents)):
            pred = pred_latents[i]                          # (B, F, C, H, W)
            B, F_, N, D = pred.shape
            assert B == Bp and F_ == Fp, "Batch/frames mismatch"

            # --- downsample (H,W) -> (~h_n, ~w_n) so that h_n*w_n ≈ N
            aspect_ratio = W / H
            w_n = max(1, int(math.floor(math.sqrt(N * aspect_ratio))))
            h_n = max(1, int(math.ceil(N / w_n)))


            post_latents_p = F.adaptive_avg_pool2d(post_base.view(B*F_, C, H, W), (h_n, w_n))
            post_seq = post_latents_p.view(B, F_, C, h_n * w_n).transpose(-1, -2)  # (B, F, N', C)

            # trim/pad sequence to exactly N
            Np = post_seq.shape[-2]
            if Np < N:
                post_seq = F.pad(post_seq, (0, 0, 0, N - Np))  # pad along N
            elif Np > N:
                post_seq = post_seq[..., :N, :]

            # project channels to D if needed if C!=D
            if post_seq.shape[-1] != D:
                if self.predictor_proj is None or self.predictor_proj[i] is None:
                    raise ValueError(f"Need predictor_proj[{i}] to map {D} -> {post_seq.shape[-1]}")
                pred = self.predictor_proj[i](pred)    # (..., D)->(..., C)

            # optional norm
            if self.predictor_norm is not None and self.predictor_norm[i] is not None:
                pred = self.predictor_norm[i](pred)
                post = self.predictor_norm[i](post_seq)
            else:
                post = post_seq

            # --- loss
            loss_unred = 0.0
            if self.loss_type in ["mse", "both"]:
                loss_unred = loss_unred + F.mse_loss(pred, post.detach(), reduction='none')
            if self.loss_type in ["cos_sim", "both"]:
                cos = 1 - F.cosine_similarity(pred, post.detach(), dim=-1, eps=1e-8)  # (B,F,N)
                if self.loss_type == "cos_sim":
                    sr_layer = cos.mean()
                    sr_loss_layer[f"layer_{self.sr_latents[i]}"] = sr_layer
                    sr_loss += sr_layer
                    continue
                else:
                    loss_unred = loss_unred + cos.unsqueeze(-1)  # (B,F,N,1) broadcast-safe

            sr_layer = loss_unred.mean()
            sr_loss_layer[f"layer_{self.sr_latents[i]}"] = sr_layer
            sr_loss += sr_layer

        sr_loss = sr_loss / len(self.sr_latents)
        return sr_loss, sr_loss_layer

        
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        post_latents, frame_rate = self.get_input(batch, 'latents')

        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents) 

        v_loss = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss.mean()
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, "mean") else sr_loss_val

        # Total loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Log losses as scalar
        self.log("train/loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
        post_latents, frame_rate = self.get_input(batch, 'latents')

        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        
        v_loss = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss.mean()
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, "mean") else sr_loss_val

        # Total loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Log losses as scalars
        self.log("val/loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
    
    
    

class ModelREPAIF(ModelREPA):
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
                latent_dim = 1024,
                base_feature_channels=None,
                ctx_noise_exp=0.5,
                dino_predictor_only=True,
                max_patch_size_second_stage=(6,16),
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
            latent_dim=latent_dim,
            ctx_noise_exp=ctx_noise_exp,
            max_patch_size_second_stage=max_patch_size_second_stage,
        )
        self.enc_scale_dino = enc_scale_dino
        self.base_feature_channels = base_feature_channels

    def _resolve_base_feature_channels(self, total_channels: int) -> int:
        base_ch = self.base_feature_channels
        if base_ch is None:
            base_ch = (total_channels + 1) // 2  # mirror torch.chunk behavior
        if base_ch <= 0 or base_ch >= total_channels:
            raise ValueError(
                f"base_feature_channels must be in [1, {total_channels - 1}], got {base_ch}"
            )
        return base_ch

    def _split_recon_semantic(self, tensor: torch.Tensor, dim: int):
        total = tensor.size(dim)
        base = self._resolve_base_feature_channels(total)
        return torch.split(tensor, [base, total - base], dim=dim)
        
    
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        post_latents, frame_rate = self.get_input(batch, 'latents')
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
        
        # v-loss over all channels; also keep split logs (recon/sem)
        v_loss_full = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss_full.mean()
        loss_recon_map, loss_sem_map = self._split_recon_semantic(v_loss_full, dim=2)
        loss_recon = loss_recon_map.mean()
        loss_sem = loss_sem_map.mean()

        # SR loss from latents
        sr_loss_val, sr_loss_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_val_mean = sr_loss_val.mean() if hasattr(sr_loss_val, 'mean') else sr_loss_val

        # Total weighted loss
        loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_val_mean

        # Logging
        self.log("train/loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
        post_latents, frame_rate = self.get_input(batch, 'latents')
        x = self.encode_frames(images)

        b, f, e, h, w = x.size()

        context = x[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = x[:,-self.num_pred_frames:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        pred, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)

        v_loss_full = self.v_loss(target, pred, noise, t)
        v_loss_mean = v_loss_full.mean()
        loss_recon_map, loss_sem_map = self._split_recon_semantic(v_loss_full, dim=2)
        loss_recon = loss_recon_map.mean()
        loss_sem = loss_sem_map.mean()

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
        frame_recon, frame_sem = self._split_recon_semantic(frame, dim=1)
        frame = torch.cat([
            frame_recon / self.enc_scale,
            frame_sem / self.enc_scale_dino
        ], dim=1)
        frame = self.ae.post_quant_conv(frame)
        samples = (self.ae.decoder(frame)).unsqueeze(1)
        for idx in range(1, x.shape[1]):
            frame = x[:,idx] 
            frame_recon, frame_sem = self._split_recon_semantic(frame, dim=1)
            frame = torch.cat([
                frame_recon / self.enc_scale,
                frame_sem / self.enc_scale_dino
            ], dim=1)
            frame = self.ae.post_quant_conv(frame)
            frame = self.ae.decoder(frame)
            samples = torch.cat([samples, (frame).unsqueeze(1)], dim=1)
        return samples


class ModelREPATeacher(ModelREPA):
    """Distillation variant that sources SR targets from a frozen teacher."""

    def __init__(self, *, teacher_config, **kwargs):
        if teacher_config is None:
            raise ValueError("ModelREPATeacher requires a teacher_config")

        super().__init__(**kwargs)

        self.teacher = self._build_teacher(teacher_config)
        self._cached_images = None

    def _build_teacher(self, teacher_config):
        cfg = teacher_config
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, dict):
            cfg = dict(cfg)
        else:
            raise TypeError("teacher_config must be a mapping with target/params")

        if not isinstance(cfg, dict) or "target" not in cfg:
            raise ValueError("teacher_config must contain 'target' and optional 'params'")

        teacher = instantiate_from_config(cfg)

        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
        return teacher

    def _teacher_output_to_latents(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        B, T, N, D = tokens.shape
        patch = max(1, int(self.teacher.patch_size))
        h = max(1, height // patch)
        w = max(1, width // patch)
        if h * w != N:
            h = max(1, int(math.floor(math.sqrt(N))))
            w = max(1, int(math.ceil(N / h)))

        latents = tokens.view(B, T, h, w, D).permute(0, 1, 4, 2, 3).contiguous()
        target_dtype = next(self.vit.parameters()).dtype
        return latents.to(device=self.device, dtype=target_dtype)

    def get_input(self, batch, k):
        if k == 'images':
            images, meta = super().get_input(batch, k)
            self._cached_images = images
            return images, meta

        if k == 'latents':
            if isinstance(batch, dict) and batch.get('latents') is not None:
                return super().get_input(batch, k)

            images = self._cached_images
            if images is None:
                images, _ = super().get_input(batch, 'images')

            teacher_imgs = images
            if teacher_imgs.device != self.teacher.device:
                teacher_imgs = teacher_imgs.to(self.teacher.device, non_blocking=True)

            with torch.no_grad():
                teacher_tokens = self.teacher(teacher_imgs)
            self._cached_images = None

            height, width = int(images.shape[-2]), int(images.shape[-1])
            latents = self._teacher_output_to_latents(teacher_tokens, height, width)
            return latents, None

        return super().get_input(batch, k)

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        if self.teacher is not None:
            self.teacher.to(self.teacher.device)
        return model


class ModelREPAIFTeacher(ModelREPAIF):
    """`ModelREPAIF` variant that distills SR targets from a frozen teacher."""

    def __init__(self, *, teacher_config, **kwargs):
        if teacher_config is None:
            raise ValueError("ModelREPAIFTeacher requires a teacher_config")

        super().__init__(**kwargs)

        self.teacher = self._build_teacher(teacher_config)
        self._cached_images = None

    def _build_teacher(self, teacher_config):
        cfg = teacher_config
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, dict):
            cfg = dict(cfg)
        else:
            raise TypeError("teacher_config must be a mapping with target/params")

        if not isinstance(cfg, dict) or "target" not in cfg:
            raise ValueError("teacher_config must contain 'target' and optional 'params'")

        teacher = instantiate_from_config(cfg)

        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
        return teacher

    def _teacher_output_to_latents(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        B, T, N, D = tokens.shape
        patch = max(1, int(self.teacher.patch_size))
        h = max(1, height // patch)
        w = max(1, width // patch)
        if h * w != N:
            h = max(1, int(math.floor(math.sqrt(N))))
            w = max(1, int(math.ceil(N / h)))

        latents = tokens.view(B, T, h, w, D).permute(0, 1, 4, 2, 3).contiguous()
        target_dtype = next(self.vit.parameters()).dtype
        return latents.to(device=self.device, dtype=target_dtype)

    def get_input(self, batch, k):
        if k == 'images':
            images, meta = super().get_input(batch, k)
            self._cached_images = images
            return images, meta

        if k == 'latents':
            if isinstance(batch, dict) and batch.get('latents') is not None:
                return super().get_input(batch, k)

            images = self._cached_images
            if images is None:
                images, _ = super().get_input(batch, 'images')

            teacher_imgs = images
            if teacher_imgs.device != self.teacher.device:
                teacher_imgs = teacher_imgs.to(self.teacher.device, non_blocking=True)

            with torch.no_grad():
                teacher_tokens = self.teacher(teacher_imgs)
            self._cached_images = None

            height, width = int(images.shape[-2]), int(images.shape[-1])
            latents = self._teacher_output_to_latents(teacher_tokens, height, width)
            return latents, None

        return super().get_input(batch, k)

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        if self.teacher is not None:
            self.teacher.to(self.teacher.device)
        return model
