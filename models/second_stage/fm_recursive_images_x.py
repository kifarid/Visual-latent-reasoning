import os
import math
import random
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
import itertools

import time

start_time = time.time() 

from util import instantiate_from_config

def rand_visible_grid_mask(B, S, H, W, max_visible=5):
    # x: [B, S, C, H, W]  -> mask: [B, S, H, W] (True=keep)
   # B, S, _, H, W = x.shape
    HW = H * W
    kmax = min(max_visible, HW)

    k = torch.randint(0, kmax + 1, (B, 1)) #, device=x.device)          # [B,1]
    k[0] = 0  # for debugging, ensure at least one is fully masked
    order = torch.rand(B, HW).argsort(dim=1)         # [B,HW]
    ranks = order.new_empty(order.shape)
    ranks.scatter_(1, order, torch.arange(HW).expand(B, HW))

    mask = (ranks < k).view(B, 1, H, W).expand(B, S, H, W)            # [B,S,H,W]
    return mask, k.squeeze(1)                                         # also returns how many kept


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
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, sigma_min=1e-5, timescale=1.0, enc_scale=4, warmup_steps=5000, min_lr_multiplier=0.1,
                sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                sr_loss_weight=1.0,
                ctx_noise_prob=0.5,
                tube_ctx_mask = True,
                max_patch_size_second_stage = (6, 16),
                loss_type = "cos_sim",
                ctx_noise_exp = 1.0,
                num_ctx_frames = 4,
                num_pred_frames = 1,
                num_thinking_frames = 4, 
                num_thinking_steps = 4,
                future_noise_exp = 0.5,
                repr_loss_weight = 1.0,
                tnk_loss_weight = 1.0,
                ema_thoughts = True, 
                predictor_only_sr = False,
                time_sampler_cfg = None,
                #learning_rate = 1e-4,
                #num_iters_per_epoch = 1,
                
    ):
        super().__init__()

        self.num_pred_frames = num_pred_frames
        self.num_prediction_frames = num_pred_frames
        self.num_context_frames = num_ctx_frames
        self.num_thinking_frames = num_thinking_frames
        self.num_thinking_steps = num_thinking_steps
        self.enc_scale = enc_scale
        self.has_context = self.num_context_frames > 0
        self.has_thinking = True and self.num_thinking_steps > 0 and self.num_thinking_frames > 0

        # Training parameters
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.loss_type = loss_type
        #self.learning_rate = learning_rate
        #self.num_iters_per_epoch = num_iters_per_epoch

        
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
        self.repr_loss_weight = repr_loss_weight
        self.tnk_loss_weight = tnk_loss_weight
        self.ctx_noise_exp =ctx_noise_exp
        self.future_noise_exp = future_noise_exp
        self.max_patch_size_second_stage = max_patch_size_second_stage

        # block intervals 
        if self.has_thinking:
            span = self.vit.depth//self.num_thinking_steps
            self.block_intervals = [(step* span, (step+1)* span -1 ) for step in range(self.num_thinking_steps)]
            self.reg_latents = [end for (_, end) in self.block_intervals]
        else:
            self.block_intervals = []
            self.reg_latents = [-1]

        # Build the predictor for the SL loss if needed
        self.build_projector(generator_config.params.hidden_size)
        self.ctx_noise_prob = ctx_noise_prob
        self.tube_ctx_mask = tube_ctx_mask

        # EMA thoughts
        self.ema_thoughts = ema_thoughts
        self.thoughts_vit = self.ema_vit if ema_thoughts else self.vit
        self.predictor_only_sr = predictor_only_sr

        self.time_sampler_cfg = time_sampler_cfg
        self.time_sampler = None
        self.time_sampler_resolution = None

        if self.time_sampler_cfg is not None:
            cfg = OmegaConf.to_container(self.time_sampler_cfg, resolve=True)
            params = cfg.setdefault("params", {})

            in_h, in_w = self.vit.input_size
            if isinstance(self.vit.patch_size, (tuple, list)):
                p_h, p_w = self.vit.patch_size[:2]
            else:
                p_h = p_w = self.vit.patch_size

            assert in_h % p_h == 0 and in_w % p_w == 0, "input_size must be divisible by patch_size"
            params["resolution"] = (in_h // p_h, in_w // p_w)

            self.time_sampler = instantiate_from_config(cfg)
            self.time_sampler_resolution = params["resolution"]

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
        #check if path exists 
        ckpt_path_full = os.path.join(tokenizer_folder, ckpt_path)
        #if 
        #checkpoint = torch.load(os.path.join(tokenizer_folder, ckpt_path), map_location="cpu", weights_only=True)["state_dict"]
        #model.load_state_dict(checkpoint, strict=False)
        #turn off require grads
        requires_grad(model, False)
        model.eval()
        return model
    
    def build_generator(self, generator_config):
        """
        Instantiate the denoising backbone model from the config.
        """
        model = instantiate_from_config(generator_config)
        return model
    
    def build_projector(self, generator_hidden_dim):
        if self.add_projector and self.has_thinking and len(self.reg_latents) > 0:
            if self.add_norm:
                self.predictor_norm = [instantiate_from_config(self.layer_norm) for _ in range(len(self.reg_latents)) ]
            else:
                #identity layers
                self.predictor_norm = [nn.Identity() for _ in range(len(self.reg_latents)) ]
            self.predictor_proj = [build_mlp(generator_hidden_dim, generator_hidden_dim, generator_hidden_dim) for _ in range(len(self.reg_latents)) ]
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
#        assert len(x.shape) == 5, 'input must be 5D tensor'
        x = x[:, 0:1, ...]
        return x, frame_rate
    
    def patch_second_stage(self, x, patch_size=None):
        """
        Turn a 5D latent grid into per-patch tensors. If `tube_ctx_mask` is True, we
        keep temporal tubes per spatial patch (shape: (b*h_p*w_p, f, e, p, p)).
        Otherwise, we create independent patches per frame (shape: (b*f*h_p*w_p, e, p, p)).
        Returns the patched tensor *and* a meta tuple used by `unpatch_second_stage`.
        """
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
            patch_size_h = vit_patch_h
            patch_size_w = vit_patch_w

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
        (b, f, e, h_p, w_p, patch_size_h, patch_size_w)
        """
        b, f, e, h_p, w_p, patch_size_h, patch_size_w = meta
        if self.tube_ctx_mask:
            # x: (b*h_p*w_p, f, e, p, p) -> (b, f, e, h, w)
            x = rearrange(
                x,
                '(b hp wp) f e p1 p2 -> b f e (hp p1) (wp p2)',
                b=b, hp=h_p, wp=w_p, p1=patch_size_h, p2=patch_size_w
            )
        else:
            # x: (b*f*h_p*w_p, e, p, p) -> (b, f, e, h, w)
            x = rearrange(
                x,
                '(b f hp wp) e p1 p2 -> b f e (hp p1) (wp p2)',
                b=b, f=f, hp=h_p, wp=w_p, p1=patch_size_h, p2=patch_size_w
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
            elif max_patch is None:
                max_patch_h = vit_patch_h
                max_patch_w = vit_patch_w
            else:
                max_patch_h = max_patch_w = max_patch
            max_patch_h = max(max_patch_h, vit_patch_h)
            max_patch_w = max(max_patch_w, vit_patch_w)
            candidates_h = [p for p in range(vit_patch_h, max_patch_h + 1) if h % p == 0] or [vit_patch_h]
            candidates_w = [p for p in range(vit_patch_w, max_patch_w + 1) if w % p == 0] or [vit_patch_w]
            patch_size_h = random.choice(candidates_h)
            patch_size_w = random.choice(candidates_w)

        # Patch
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
        x_t_noised = torch.where(mask.view(-1, 1, 1, 1, 1), x_t_noised, x_patched)
        noise = torch.where(mask.view(-1, 1, 1, 1, 1), noise, torch.zeros_like(noise))

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
        x = self.ae.encode(images)
        x = x['continuous'] if isinstance(x, dict) else x
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

    def x_loss(self, target, pred):
        """
        Docstring for x_loss
        
        :param self: Description
        :param target: The actual signal X_0 to predict
        :param pred: Xhat_0
        :param t: Description
        """
        loss = ((pred.float()-target.float())**2)
        return loss 

    def x_to_v(self, x, z, t, eps=1e-6):
        v_pred = (x - z)/ (1 -t + eps)
        return  v_pred
        
    
    def sr_loss(self, pred_latents, post_latents, detach_pred= True, detach_post= True, detach_pre_pred= False):
        if len(self.reg_latents) == 0 or len(pred_latents) == 0:
            return 0.0, {}
        sr_loss = 0
        sr_loss_layer = {}
        for i in range(len(pred_latents)):
            pred_latent, post_latent = pred_latents[i], post_latents[i]
            pred_latent = pred_latent.detach() if detach_pre_pred else pred_latent
            if self.predictor_proj is not None and self.predictor_norm is not None:
                pred_latent = self.predictor_norm[i](self.predictor_proj[i]((pred_latent)))
                post_latent = self.predictor_norm[i](post_latent)
            
            post_latent = post_latent.detach() if detach_post else post_latent
            pred_latent = pred_latent.detach() if detach_pred else pred_latent
            sr_loss_layer_unred = 0
            # Compute MSE loss if requested
            if self.loss_type in ["mse", "both"]:
                sr_loss_layer_unred = sr_loss_layer_unred + F.mse_loss(pred_latent, post_latent, reduction='none')
            # Compute cosine similarity loss if requested
            if self.loss_type in ["cos_sim", "both"]:
                sr_loss_layer_unred = sr_loss_layer_unred + (1 - F.cosine_similarity(pred_latent, post_latent, dim=-1, eps=1e-8))
            sr_loss_layer_red = sr_loss_layer_unred.mean()
            sr_loss_layer[f"layer_{self.reg_latents[i]}"] = sr_loss_layer_red
            sr_loss += sr_loss_layer_red

        sr_loss = sr_loss / len(self.reg_latents)
        return sr_loss, sr_loss_layer

    def create_context_thinking_pred(self, images):
        '''
        Split a video clip into:
          • context: first `num_context_frames` frames (shape: b, ctx_len, e, h, w)
          • thinking: `thinking_steps` random chunks of length `num_thinking_frames`
          • next_frame: the frame immediately after the context window
          • future_frame: a frame sampled either from the middle region (ctx_len+1 … last-1) or the final frame

        The thinking chunks are sampled uniformly (with replacement when necessary) from frames at indices ≥ ctx_len.
        The function returns `(context, thinking_chunks, next_frame, future_frame)` where `thinking_chunks` is a list
        of length `thinking_steps` and each entry has shape `(b, num_thinking_frames, e, h, w)`.
        the function also returns frame ids, with shape b, .. for each of the predictions 
        '''
        #TODO return the frame ids for each of the returned arrays 
        if images.ndim == 4:
            images = images.unsqueeze(1)  # (b, 1, C, H, W)
        b, f, e, h, w = images.size()

        device = images.device
        ctx_len = self.num_context_frames if self.has_context and f > 1 else 0
        tnk_len = self.num_thinking_frames
        tnk_steps = self.num_thinking_steps
        frames_after_ctx = max(f - ctx_len, 0)

        context = images[:, :ctx_len] if self.has_context else None

        next_idx = min(ctx_len, f - 1)
        next_ids = torch.full((b, 1), next_idx, device=device, dtype=torch.long)
        next_frame = images[:, next_idx:next_idx + 1]

        future_idx = torch.full((b,), f - 1, device=device, dtype=torch.long)
        if frames_after_ctx > 2:
            mid_samples = torch.randint(ctx_len + 1, f - 1, (b,), device=device)
            use_mid = torch.rand(b, device=device) < 0.5
            future_idx = torch.where(use_mid, mid_samples, future_idx)
            future_ids = future_idx.unsqueeze(-1)
            future_frame = images[torch.arange(b, device=device), future_idx].unsqueeze(1)
        else: 
            future_frame = None
            future_ids = None

        if (not self.has_thinking) or tnk_len == 0 or tnk_steps == 0 or frames_after_ctx == 0:
            thinking_frames = None
            thinking_ids = None
        else:
            idx = torch.randint(ctx_len, f, (b, tnk_steps, tnk_len), device=device)
            flat_idx = idx.view(b, -1)
            batch_ids = torch.arange(b, device=device).unsqueeze(-1)
            gathered = images[batch_ids, flat_idx].view(b, tnk_steps, tnk_len, e, h, w)
            thinking_frames = list(torch.unbind(gathered, dim=1))
            thinking_ids = list(torch.unbind(idx, dim=1))

        if self.has_context:
            context_ids = torch.arange(context.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        else:
            context_ids = None
        

        frame_ids = {
            "context": context_ids,
            "thinking": thinking_ids,
            "next": next_ids,
            "future": future_ids,
        }

        return context, thinking_frames, next_frame, future_frame, frame_ids 

        
    def training_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        context, thinking_frames, next_frame, future_frame, frame_ids = self.create_context_thinking_pred(images)
        thinking_frames = thinking_frames or []
        segments = [context, *thinking_frames, next_frame, future_frame]
        lengths  = [s.shape[1] if s is not None else 0 for s in segments]
        starts = list(itertools.accumulate([0, *lengths]))[:-1]

        images_filtered = torch.cat([s for s in segments if s is not None], dim=1)

        x = self.encode_frames(images_filtered)

        pieces = [None if L == 0 else x[:, s:s+L] for s, L in zip(starts, lengths)]

        context = pieces[0]
        thinking_frames = pieces[1:1+len(thinking_frames)]
        next_frame = pieces[1+len(thinking_frames)]
        future_frame = pieces[2+len(thinking_frames)]
        b, f, e, h, w = x.size()
        f = 1

        v_nf_loss_mean = None
        v_ff_loss_mean = None
        v_tff_loss_mean = None
        pred_regs_ff = None
        repr_loss_val_mean = 0.0
        tnk_loss_val_mean = 0.0
        repr_loss_layers = {}

        # Next frame prediction step 
        target = next_frame
        t = self.time_sampler.get_time(b).to(x.device)   #torch.rand((x.shape[0],), device=x.device)[:,None, None, None].expand(b, f, *self.vit.x_embedder.grid_size)
        interm_mask = torch.ones((b, f, *self.vit.x_embedder.grid_size), device=x.device, dtype=torch.bool)
        t_masked = torch.where(interm_mask, t, torch.zeros_like(t))
        #target_t, noise = self.add_noise(target, t)
        target_t, (noise, _, _, _, _) = self.add_noise_ctx(target, t=t_masked.reshape(-1), mask=interm_mask.reshape(-1))
        
        context_noised, ctx_noise = (None, None)

        frame_ids_nf = torch.ones((b, f), device=x.device, dtype=torch.long) 
        pred, pred_regs_nf = self.vit(target_t, context_noised, t, frame_idxs=frame_ids_nf, return_regs=self.reg_latents)

        v_loss = self.x_loss(target, pred) #, noise, t)
        v_nf_loss_mean = v_loss.mean()
        #loss_recon_nf, loss_sem_nf = map(lambda x: x.mean(), torch.chunk(v_loss, 2, dim=2))

        # Future frame prediction step 
        future_frame = next_frame
        if future_frame is not None:
            target = next_frame
            t = torch.rand((x.shape[0],), device=x.device).pow(self.future_noise_exp)
            target_t, noise = self.add_noise(target, t)

            context_noised, ctx_noise =  (None, None)

            frame_ids_ff = torch.cat([frame_ids["context"], frame_ids["future"]], dim=1) if context is not None else frame_ids["future"]
            pred, pred_regs_ff = self.vit(target_t, context_noised, t, frame_idxs=frame_ids_ff, return_regs=self.reg_latents)
            v_ff_loss = self.x_loss(target, pred) #, noise, t)
            v_ff_loss_mean = v_ff_loss.mean()
    



        # recursive prediction step on the thinking frames
        tnk_enabled = self.has_thinking and thinking_frames and pred_regs_ff is not None
        v_losses = [v_nf_loss_mean]
        if v_ff_loss_mean is not None:
            v_losses.append(v_ff_loss_mean)
        if v_tff_loss_mean is not None:
            v_losses.append(v_tff_loss_mean)


        if tnk_enabled: 
                tnk_regs = []
                v_tnk_losses = []
                curr_tnk_reg = pred_regs_ff[-1].detach() if torch.rand(1).item() < 0.5 else None
                
                for i in range(self.num_thinking_steps):
                    t = self.time_sampler.get_time(b).to(x.device) #torch.rand((x.shape[0],), device=x.device)[:,None, None, None].expand(b, f, *self.vit.x_embedder.grid_size)
                    interm_mask = torch.ones((b, f, *self.vit.x_embedder.grid_size), device=x.device, dtype=torch.bool)
                    context_noised, ctx_noise = (None, None)
                    block_start, block_end = self.block_intervals[i]
                    t_masked = torch.where(interm_mask, t, torch.zeros_like(t))
                    frame_ids_tnk = torch.ones((b, f), device=x.device, dtype=torch.long) #* (mask_sequence.shape[1] - 1 -  frame_ids["thinking"][reverse_i])
                    target_t, (noise, _,_, _, _) = self.add_noise_ctx(target, t=t_masked.reshape(-1), mask=interm_mask.reshape(-1))
                    pred, curr_tnk_reg = self.thoughts_vit(target_t, context_noised, t_masked, frame_idxs=frame_ids_tnk, reg_tokens = curr_tnk_reg, return_regs=self.reg_latents, block_start = block_start, block_end = block_end)
                    curr_tnk_reg = curr_tnk_reg[0]
                    tnk_regs.append(curr_tnk_reg)
                    if i//2 == 1:
                        curr_tnk_reg = curr_tnk_reg.detach() if torch.rand(1).item() < 0.5 else None
                    v_tnk_loss = self.x_loss(target, pred) #, noise, t)
                    v_tnk_losses.append(v_tnk_loss.mean())
        
                v_losses += v_tnk_losses
                v_losses_mean = sum(v_losses) / len(v_losses)


                assert len(pred_regs_ff) == len(tnk_regs), "intermediate generated thinking tokens shpould match in length to the feature repr tokens"
                #repr loss 
                repr_loss_val, repr_loss_layers = self.sr_loss(pred_regs_ff, tnk_regs, detach_post=False, detach_pred=True)
                repr_loss_val_mean = repr_loss_val.mean() if hasattr(repr_loss_val, "mean") else repr_loss_val
                # thinking loss 
                tnk_loss_val, tnk_loss_layers = self.sr_loss(pred_regs_ff, tnk_regs, detach_post=True, detach_pred=False, detach_pre_pred=self.predictor_only_sr) 
                tnk_loss_val_mean = tnk_loss_val.mean() if hasattr(tnk_loss_val, "mean") else tnk_loss_val
        else:
            v_losses_mean = sum(v_losses) / len(v_losses)

        # Total loss
        loss = self.recon_loss_weight * v_losses_mean + self.repr_loss_weight * repr_loss_val_mean + self.tnk_loss_weight * tnk_loss_val_mean

        # Log losses as scalars
        self.log("train/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", v_nf_loss_mean.item() if hasattr(v_nf_loss_mean, "item") else v_nf_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        if v_ff_loss_mean is not None:
            self.log("train/future_recon_loss", v_ff_loss_mean.item() if hasattr(v_ff_loss_mean, "item") else v_ff_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if v_tff_loss_mean is not None:
            self.log("train/mfuture_recon_loss", v_tff_loss_mean.item() if hasattr(v_tff_loss_mean, "item") else v_tff_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        if tnk_enabled:
            self.log("train/sr_loss", repr_loss_val_mean.item() if hasattr(repr_loss_val_mean, "item") else repr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            # Log per-layer sr losses
            for layer_name, layer_val in repr_loss_layers.items():
                scalar_val = layer_val.item() if hasattr(layer_val, "item") else layer_val
                self.log(f"train/sr/{layer_name}", scalar_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA at the end of each epoch as fallback"""
        update_ema(self.ema_vit, self.vit)



    def validation_step(self, batch, batch_idx):
        images, frame_rate = self.get_input(batch, 'images')
        context, thinking_frames, next_frame, future_frame, frame_ids = self.create_context_thinking_pred(images)
        thinking_frames = thinking_frames or []
        segments = [context, *thinking_frames, next_frame, future_frame]
        lengths  = [s.shape[1] if s is not None else 0 for s in segments]
        starts = list(itertools.accumulate([0, *lengths]))[:-1]

        images_filtered = torch.cat([s for s in segments if s is not None], dim=1)

        x = self.encode_frames(images_filtered)

        pieces = [None if L == 0 else x[:, s:s+L] for s, L in zip(starts, lengths)]

        context = pieces[0]
        thinking_frames = pieces[1:1+len(thinking_frames)]
        next_frame = pieces[1+len(thinking_frames)]
        future_frame = pieces[2+len(thinking_frames)]
        b, f, e, h, w = x.size()
        f = 1

        v_ff_loss_mean = None
        v_tff_loss_mean = None
        pred_regs_ff = None
        repr_loss_val_mean = 0.0
        tnk_loss_val_mean = 0.0
        repr_loss_layers = {}

        # Next frame prediction step 
        target = next_frame
        t = self.time_sampler.get_time(b).to(x.device) #torch.rand((x.shape[0],), device=x.device)[:,None, None, None].expand(b, f, *self.vit.x_embedder.grid_size)
        interm_mask = torch.ones((b, f, *self.vit.x_embedder.grid_size), device=x.device, dtype=torch.bool)
        t_masked = torch.where(interm_mask, t, torch.zeros_like(t))
        #target_t, noise = self.add_noise(target, t)
        target_t, (noise, _, _, _, _) = self.add_noise_ctx(target, t=t_masked.reshape(-1), mask=interm_mask.reshape(-1))
        
        context_noised, ctx_noise = (None, None)

        frame_ids_nf = torch.ones((b, f), device=x.device, dtype=torch.long) 
        pred, pred_regs_nf = self.vit(target_t, context_noised, t_masked, frame_idxs=frame_ids_nf, return_regs=self.reg_latents)

        v_loss = self.x_loss(target, pred) #, noise, t_masked)
        v_nf_loss_mean = v_loss.mean()
        #loss_recon_nf, loss_sem_nf = map(lambda x: x.mean(), torch.chunk(v_loss, 2, dim=2))

        # Future frame prediction step 
        future_frame = next_frame
        if future_frame is not None:
            target = future_frame
            t = torch.rand((x.shape[0],), device=x.device) #.pow(self.future_noise_exp)
            target_t, noise = self.add_noise(target, t)

            context_noised, ctx_noise = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

            frame_ids_ff =  torch.ones((b, f), device=x.device, dtype=torch.long)  #torch.cat([frame_ids["context"], frame_ids["future"]], dim=1) if context is not None else frame_ids["future"]
            pred, pred_regs_ff = self.vit(target_t, context_noised, t, frame_idxs=frame_ids_ff, return_regs=self.reg_latents)
            v_ff_loss = self.x_loss(target, pred) #, noise, t)
            v_ff_loss_mean = v_ff_loss.mean()
    

        # recursive prediction step on the thinking frames
        tnk_enabled = self.has_thinking and thinking_frames and pred_regs_ff is not None
        v_losses = [v_nf_loss_mean]
        if v_ff_loss_mean is not None:
            v_losses.append(v_ff_loss_mean)
        if v_tff_loss_mean is not None:
            v_losses.append(v_tff_loss_mean)

                #reverse thinking steps
        if tnk_enabled: 
            tnk_regs = []
            v_tnk_losses = []
            curr_tnk_reg = pred_regs_ff[-1].detach() if torch.rand(1).item() < 0.5 else None
            
            for i in range(self.num_thinking_steps):
                t = self.time_sampler.get_time(b).to(x.device) #torch.rand((x.shape[0],), device=x.device)[:,None, None, None].expand(b, f, *self.vit.x_embedder.grid_size)
                interm_mask = torch.ones((b, f, *self.vit.x_embedder.grid_size), device=x.device, dtype=torch.bool)
                context_noised, ctx_noise = (None, None)
                block_start, block_end = self.block_intervals[i]
                t_masked = torch.where(interm_mask, t, torch.zeros_like(t))
                frame_ids_tnk = torch.ones((b, f), device=x.device, dtype=torch.long) #* (mask_sequence.shape[1] - 1 -  frame_ids["thinking"][reverse_i])
                target_t, (noise, _, _, _, _) = self.add_noise_ctx(target, t=t_masked.reshape(-1), mask=interm_mask.reshape(-1))
                pred, curr_tnk_reg = self.thoughts_vit(target_t, context_noised, t_masked, frame_idxs=frame_ids_tnk, reg_tokens = curr_tnk_reg, return_regs=self.reg_latents, block_start = block_start, block_end = block_end)
                curr_tnk_reg = curr_tnk_reg[0]
                tnk_regs.append(curr_tnk_reg)
                v_tnk_loss = self.x_loss(target, pred) #, noise, t)
                v_tnk_losses.append(v_tnk_loss.mean())
    
            v_losses += v_tnk_losses
            v_losses_mean = sum(v_losses) / len(v_losses)
            assert len(pred_regs_ff) == len(tnk_regs), "intermediate generated thinking tokens shpould match in length to the feature repr tokens"
            #repr loss 
            repr_loss_val, repr_loss_layers = self.sr_loss(pred_regs_ff, tnk_regs, detach_post=True, detach_pred=True)
            repr_loss_val_mean = repr_loss_val.mean() if hasattr(repr_loss_val, "mean") else repr_loss_val
        else:
            v_losses_mean = sum(v_losses) / len(v_losses)

        # Total loss
        loss = self.recon_loss_weight * v_losses_mean + self.repr_loss_weight * repr_loss_val_mean + self.tnk_loss_weight * tnk_loss_val_mean
        # Log losses as scalars
        self.log("val/loss", loss.item() if hasattr(loss, "item") else loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", v_nf_loss_mean.item() if hasattr(v_nf_loss_mean, "item") else v_nf_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if v_ff_loss_mean is not None:
            self.log("val/future_recon_loss", v_ff_loss_mean.item() if hasattr(v_ff_loss_mean, "item") else v_ff_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if v_tff_loss_mean is not None:
            self.log("val/mfuture_recon_loss", v_tff_loss_mean.item() if hasattr(v_tff_loss_mean, "item") else v_tff_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        if tnk_enabled:
            self.log("val/sr_loss", repr_loss_val_mean.item() if hasattr(repr_loss_val_mean, "item") else repr_loss_val_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            # Log per-layer sr losses
            for layer_name, layer_val in repr_loss_layers.items():
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
    def sample(self, images=None, mask=None, frame_ids=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None, return_sample=False):
        
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
        t_steps = t_steps[:, None, None, None, None].repeat(t_steps.size(0), num_samples, self.num_pred_frames, *self.vit.x_embedder.grid_size)
        t_steps = torch.where(torch.ones_like(mask[None, :]), t_steps, torch.zeros_like(t_steps)) 
        
        with torch.no_grad():
            for i in range(NFE):
                t = t_steps[i]#.repeat(target_t.shape[0]) #t_steps_masked[i] 
                x_pred = net(target_t, context, t=t * self.timescale, frame_idxs=frame_ids)
                t_up = F.interpolate(t.float(), size=(input_h, input_w), mode='nearest').unsqueeze(2)
                v_pred = - (x_pred - target_t) / (1 - t_up).clamp_min(0.005)
                neg_v = - v_pred
                dt = t_steps[i] - t_steps[i+1] #t_steps_masked[i] - t_steps_masked[i+1] # 
                dt_up = F.interpolate(dt.float(), size=(input_h, input_w), mode='nearest').unsqueeze(2)
                dw = torch.randn(target_t.size()).to(target_t.device) * torch.sqrt(dt_up.float())
                diffusion = dt_up
                target_t  = target_t + neg_v * dt_up + eta *  torch.sqrt(2 * diffusion) * dw
        if return_sample:
            images = self.decode_frames(target_t.clone())
            return target_t, images
        else:
            return target_t

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        images, frame_rate = self.get_input(batch, 'images')
        images = images.unsqueeze(1)
        N = min(4, images.size(0))
        images = images[:N]

        total_frames = images.size(1)
        if total_frames == 0:
            raise ValueError("Expected at least one frame per sample for logging.")

        ctx_len = self.num_context_frames
        context = images[:N//2, :ctx_len] if ctx_len > 0 else None
        next_idx = ctx_len
        next_frame = images[:N//2, next_idx:next_idx + 1]     # [N/2, 1, C, H, W]
        future_frame = images[:N//2, -1:]                     # [N/2, 1, C, H, W]

        targets = torch.cat([next_frame, future_frame], dim=0) # [N, 1, C, H, W]
        if context is not None:
            context = torch.cat([context, context], dim=0)     # [N, ctx, C, H, W]
            images_for_sample = torch.cat([context, targets], dim=1)  # [N, ctx+1, C, H, W]
        else:
            images_for_sample = targets                        # [N, 1, C, H, W]

        # frame ids
        b, S = images_for_sample.size(0), images_for_sample.size(1)
        frame_ids = torch.arange(S, device=images_for_sample.device).unsqueeze(0).expand(b, -1)
        frame_ids[:N//2, -1] = next_idx
        frame_ids[N//2:, -1] = total_frames - 1

        if images.ndim == 4:
            images_for_sample = images_for_sample.unsqueeze(1)
        
        b, f, e, h, w = images_for_sample.size()
        
        # sample mask with random ratio of 0 to 20 numbers seen/unmasked 
        mask, _ = rand_visible_grid_mask(N, f, *self.vit.x_embedder.grid_size, max_visible=5)
        mask = mask.to(device=images_for_sample.device)
        mask_up = F.interpolate(mask.float(), size=(h, w), mode='nearest')
        mask_overlay = (1.0 - 2.0 * mask_up).unsqueeze(2).to(dtype=images_for_sample.dtype)
        masked_images_for_sample = torch.clamp(images_for_sample*mask_overlay, -1.0, 1.0)

        images_ctx = images_for_sample[:, :-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        frame_ids_ctx = torch.ones((N, f), device=images_for_sample.device, dtype=torch.long) #frame_ids[:, :-self.num_pred_frames] if images_ctx is not None else None

        l_visual_recon = [masked_images_for_sample[:, frame_idx] for frame_idx in range(f)]
        l_visual_recon_ema = [masked_images_for_sample[:, frame_idx] for frame_idx in range(f)]

        # sample
        samples = self.sample(images_ctx, mask=mask, eta=0.0, NFE=30, frame_ids=frame_ids_ctx, sample_with_ema=False, num_samples=N, return_sample=True)[1]

        # Only keep the first generated frame
        samples = samples[:N]
        samples_masked = torch.clamp(samples*mask_overlay, -1.0, 1.0)

        for i in range(samples.size(1)):
            l_visual_recon.append(samples_masked[:,i])

        l_visual_recon = torch.cat(l_visual_recon, dim=0)
        chunks = torch.chunk(l_visual_recon, 2 + 2, dim=0)
        sampled = torch.cat(chunks, 0)
        sampled = vutils.make_grid(sampled, nrow=N, padding=2, normalize=False,)
        
        # sample
        samples_ema = self.sample(images_ctx, mask=mask, eta=0.0, NFE=30, frame_ids=frame_ids_ctx, sample_with_ema=True, num_samples=N, return_sample=True)[1]
        # Only keep the first generated frame
        samples_ema = samples_ema[:N]
        samples_ema_masked = torch.clamp(samples_ema*mask_overlay, -1.0, 1.0)

        for i in range(samples_ema.size(1)):
            l_visual_recon_ema.append(samples_ema_masked[:,i])

        l_visual_recon_ema = torch.cat(l_visual_recon_ema, dim=0)
        chunks_ema = torch.chunk(l_visual_recon_ema, 2 + 2, dim=0)
        sampled_ema = torch.cat(chunks_ema, 0)
        sampled_ema = vutils.make_grid(sampled_ema, nrow=N, padding=2, normalize=False)

        log["ema_sampled"] = sampled_ema
        log["sampled"] = sampled
        self.vit.train()
        return log
    
