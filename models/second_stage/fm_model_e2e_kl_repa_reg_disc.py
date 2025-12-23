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
import random
from omegaconf import OmegaConf, ListConfig, DictConfig

import time
import contextlib



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
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, sigma_min=1e-5, timescale=1.0, enc_scale=1, warmup_steps=5000, min_lr_multiplier=0.1,num_pred_frames=1, sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                ae_loss_weight=1.0,
                sr_loss_weight=0.1,
                sr_yol_loss_weight=0.1,
                ctx_noise_prob=0.8,
                tube_ctx_mask = True,
                loss_type = "cos_sim",
                ctx_noise_exp = 1.0,
                latent_dim= 768,
                from_scratch=True,
                grad_acc_steps=2,
                grad_acc_steps_ae=2,
                nan_debug=False,
                only_decoder=False,
                max_patch_size_second_stage=(6,16),
                detach_ae_mean_loss=False,
                dino_predictor_only=True
    ):
        super().__init__()

        self.num_pred_frames = num_pred_frames
        self.enc_scale = enc_scale

        # Training parameters
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.loss_type = loss_type
        self.from_scratch = from_scratch 
        self.only_decoder = only_decoder
        self.sr_yol_loss_weight = sr_yol_loss_weight
        
        # denoising backbone
        self.vit = self.build_generator(generator_config)
        
        # Tokenizer
        self.ae = self.build_tokenizer(tokenizer_config)

        # EMA
        self.ema_vit = init_ema_model(self.vit)
        self.ema_ae = init_ema_model(self.ae)
        self.sigma_min = sigma_min
        self.timescale = timescale

        self.sr_latents = sr_latents
        self.add_projector = add_projector
        self.add_norm = add_norm
        self.layer_norm = layer_norm
        self.recon_loss_weight = recon_loss_weight
        self.sr_loss_weight = sr_loss_weight
        self.ae_loss_weight = ae_loss_weight
        self.ctx_noise_exp =ctx_noise_exp
        self.dino_predictor_only = dino_predictor_only
        # Gradient accumulation
        self.grad_acc_steps = max(int(grad_acc_steps), 1)
        self.grad_acc_steps_ae = max(int(grad_acc_steps_ae), 1)

        # Optional manual gradient clipping (disabled by default). You can set
        # these from outside (e.g., in main.py after model creation):
        #   model.grad_clip = 1.0
        #   model.gradient_clip_algorithm = "norm"  # or "value"
        self.grad_clip = getattr(self, "grad_clip", 0.5)
        self.gradient_clip_algorithm = getattr(self, "gradient_clip_algorithm", "norm")

        # Build the predictor for the SL loss if needed
        self.predictor_proj, self.predictor_norm = self.build_projector(generator_config.params.hidden_size, hidden_dim=latent_dim)
        self.predictor_proj_yol, self.predictor_norm_yol = self.build_projector(generator_config.params.hidden_size, hidden_dim=generator_config.params.hidden_size)
        self.ctx_noise_prob = ctx_noise_prob
        self.tube_ctx_mask = tube_ctx_mask
        self.max_patch_size_second_stage = max_patch_size_second_stage
        # Debug toggles
        self.nan_debug = nan_debug
        #neccessary for REPA-E style of optimizing things 
        self.automatic_optimization = False 
        self._ctx_noise_log_epoch = -1
        self.detach_ae_mean_loss = detach_ae_mean_loss

        # Optional: enable autograd anomaly detection when debugging NaNs
        if self.nan_debug:
            try:
                torch.autograd.set_detect_anomaly(True)
                self.print("Enabled autograd anomaly detection")
                self.setup_hooks()
            except Exception:
                pass

    # -------------------- Debug helpers --------------------
    def _log_nan_issue(self, name: str, t: torch.Tensor):
        try:
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            msg = f"[NaNCheck] {name}: NaN={n_nan} Inf={n_inf} shape={tuple(t.shape)} dtype={t.dtype}"
        except Exception:
            msg = f"[NaNCheck] {name}: non-finite detected"
        # Lightning-friendly print
        try:
            self.print(msg)
        except Exception:
            print(msg)
        # Best-effort logs (won't crash if not in trainer context)
        try:
            if 'n_nan' in locals():
                self.log(f"nan/{name}_nan", float(n_nan), on_step=True, prog_bar=False, logger=True)
            if 'n_inf' in locals():
                self.log(f"nan/{name}_inf", float(n_inf), on_step=True, prog_bar=False, logger=True)
        except Exception:
            pass

    def _check_tensor_finite(self, name: str, t, raise_on_nan: bool = None):
        if not self.nan_debug:
            return
        if t is None:
            return
        # Handle lists/tuples/dicts recursively
        if isinstance(t, (list, tuple)):
            for idx, item in enumerate(t):
                self._check_tensor_finite(f"{name}[{idx}]", item, raise_on_nan)
            return
        if isinstance(t, dict):
            for k, v in t.items():
                self._check_tensor_finite(f"{name}.{k}", v, raise_on_nan)
            return
        if not torch.is_tensor(t):
            return
        fin = torch.isfinite(t)
        if not bool(fin.all()):
            self._log_nan_issue(name, t)
            do_raise = self.nan_debug if raise_on_nan is None else raise_on_nan
            if do_raise:
                raise RuntimeError(f"Non-finite detected in {name}")


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
        model.learning_rate = tokenizer_config.model.base_learning_rate
        model.only_decoder = self.only_decoder
        print(f"Tokenizer only_decoder: {self.only_decoder}")
        if self.from_scratch:
            print("Training tokenizer from scratch")
            return model
        
        checkpoint = torch.load(os.path.join(tokenizer_folder, ckpt_path), map_location="cpu", weights_only=True)["state_dict"]
        model.load_state_dict(checkpoint, strict=False)

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
                predictor_norm = [instantiate_from_config(self.layer_norm) for _ in range(len(self.sr_latents)) ]
            else:
                #identity layers
                predictor_norm = [nn.Identity() for _ in range(len(self.sr_latents)) ]
            predictor_proj = [build_mlp(generator_hidden_dim, hidden_dim, hidden_dim) for _ in range(len(self.sr_latents)) ]
            #move to GPU
            predictor_proj = nn.ModuleList(predictor_proj).to(device=self.device)
            predictor_norm = nn.ModuleList(predictor_norm).to(device=self.device)
        else:
            predictor_norm = None
            predictor_proj = None
        return predictor_proj, predictor_norm

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

    def _accum_no_sync(self, step_now: bool):
        """Compatibility context manager to skip DDP/FSDP gradient sync on non-final
        micro-batches during manual gradient accumulation.

        - Uses `trainer.strategy.no_backward_sync` if available (PL >= 1.8/2.x).
        - Falls back to `self.no_sync()` when running under DDP wrappers.
        - Otherwise, returns a no-op context manager.
        """
        # Only block sync when we are NOT stepping this micro-batch
        if step_now:
            return contextlib.nullcontext()
        strat = getattr(self.trainer, "strategy", None)
        if strat is not None and hasattr(strat, "no_backward_sync"):
            return strat.no_backward_sync(self, enabled=True)
        # Fallback for older PL where LightningModule gets DDP's no_sync
        if hasattr(self, "no_sync"):
            return self.no_sync()
        return contextlib.nullcontext()
    
    def _nan_hook(self, module, inp, out):
        tensors = []
        if isinstance(out, (tuple, list)):
            tensors = [o for o in out if torch.is_tensor(o)]
        elif torch.is_tensor(out):
            tensors = [out]
        for o in tensors:
            if not torch.isfinite(o).all():
                raise RuntimeError(f"NaN/Inf after {module.__class__.__name__}")
        
    def setup_hooks(self):
        for m in self.modules():
            m.register_forward_hook(lambda m, i, o: self._nan_hook(m, i, o))


    def configure_optimizers(self):
        params = list(self.vit.parameters())
        if self.predictor_proj is not None:
            # add each projector exactly once
            for proj in self.predictor_proj:
                params += list(proj.parameters())
            # likewise for any norms
            if self.predictor_norm is not None:
                for norm in self.predictor_norm:
                    params += list(norm.parameters())

        if self.predictor_proj_yol is not None:
            # add each projector exactly once
            for proj in self.predictor_proj_yol:
                params += list(proj.parameters())
            # likewise for any norms
            if self.predictor_norm_yol is not None:
                for norm in self.predictor_norm_yol:
                    params += list(norm.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01, eps=1e-8)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        
        self.ae.trainer=self.trainer
        self.ae.num_iters_per_epoch = self.num_iters_per_epoch
        [opt_ae, opt_disc], [scheduler_ae_warmup, scheduler_disc_warmup] = self.ae.configure_optimizers()
        
        return  [optimizer, opt_ae, opt_disc], [{"scheduler": scheduler, "interval": "step"},
                                                 {"scheduler": scheduler_ae_warmup, "interval": "step"},
                                                 {"scheduler": scheduler_disc_warmup, "interval": "step"}]


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
    
    def encode_frames(self, images, return_posterior=False, ema=False):
        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1

        x = self.ae.encode(images) if not ema else self.ema_ae.encode(images)
        x, posterior = x["continuous"], x["posterior"]
        x = x * (self.enc_scale)
        x = rearrange(x, '(b f) e h w -> b f e h w',b=b, f=f)

        if return_posterior:
            #posterior = rearrange(posterior, '(b f) e h w -> b f e h w',b=b, f=f)
            return x, posterior
        return x

    def decode_frames(self, x, return_distill_conv_out=False):

        B, T, C, H, W = x.shape
        x = x / (self.enc_scale)
        x = x.reshape(B * T, C, H, W)

        if return_distill_conv_out:
            distill_conv_out = self.ae.post_quant_conv_distill(x)
            distill_conv_out = distill_conv_out.reshape(B, T, *distill_conv_out.shape[1:])

        x = self.ae.post_quant_conv(x)
        x = self.ae.decoder(x)

        x = x.reshape(B, T, *x.shape[1:])

        if return_distill_conv_out:
            return x, distill_conv_out
        
        return x

    def v_loss(self, target, pred, noise, t):
        # -dxt/dt
        v = self.A(t) * target + self.B(t) * noise

        loss = ((pred.float() - v.float()) ** 2)
        return loss

    def sr_loss(self, pred_latents, post_latents, pred_yol=False):
        # Debug prints
        #print(f"post_latents_shapes: {tuple(post_latents.shape)}")
        # post is (B, F, C, H, W); each pred[i] is  (B, F, N, D)
        if pred_yol:
            Bp, Fp, N, D = post_latents[0].shape
        else:
            Bp, Fp, C, H, W = post_latents.shape
        predictor_proj, predictor_norm = (self.predictor_proj, self.predictor_norm) if not pred_yol else (self.predictor_proj_yol, self.predictor_norm_yol)

        sr_loss = 0.0
        sr_loss_layer = {}
        post_bases = post_latents
        for i in range(len(pred_latents)):
            pred = pred_latents[i]                          # (B, F, C, H, W)
            B, F_, N, D = pred.shape
            assert B == Bp and F_ == Fp, "Batch/frames mismatch"
            post_base = post_bases if not pred_yol else post_bases[i]
            # --- downsample (H,W) -> (~h_n, ~w_n) so that h_n*w_n ≈ N
            if not pred_yol:
                aspect_ratio = W / H
                w_n = max(1, int(math.floor(math.sqrt(N * aspect_ratio))))
                h_n = max(1, int(math.ceil(N / w_n)))

                post_latents_p = F.adaptive_avg_pool2d(post_base.view(B*F_, C, H, W), (h_n, w_n))
                #pred2d = F.adaptive_avg_pool2d(pred2d, (h_n, w_n))          # (B*F, C, h_n, w_n)
                post_seq = post_latents_p.view(B, F_, C, h_n * w_n).transpose(-1, -2)  # (B, F, N', C)

                # trim/pad sequence to exactly N
                Np = post_seq.shape[-2]
                if Np < N:
                    post_seq = F.pad(post_seq, (0, 0, 0, N - Np))  # pad along N
                elif Np > N:
                    post_seq = post_seq[..., :N, :]
            else:
                post_seq = post_base

            if self.dino_predictor_only and not pred_yol:
                pred = pred.detach()

            # project channels to D if needed if C!=D

            if post_seq.shape[-1] != D and (predictor_proj is None or predictor_proj[i] is None):
                raise ValueError(f"Need predictor_proj[{i}] to map {D} -> {post_seq.shape[-1]}")
            elif predictor_proj is not None and predictor_proj[i] is not None:
                pred = predictor_proj[i](pred)    # (..., D)->(..., C)

            # optional norm
            if predictor_norm is not None and predictor_norm[i] is not None:
                pred = predictor_norm[i](pred)
                post = predictor_norm[i](post_seq)
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
    
    def _get_grad_scaler(self):
        # Returns GradScaler if fp16 mixed precision is active, else None
        scaler = None
        # PL ≥ 2.x: sometimes on strategy directly
        strat = getattr(self.trainer, "strategy", None)
        if strat is not None:
            # 1) direct
            scaler = getattr(strat, "scaler", None)
            # 2) precision_plugin (common in 2.x)
            if scaler is None:
                pp = getattr(strat, "precision_plugin", None)
                scaler = getattr(pp, "scaler", None) if pp is not None else None
            # 3) rarer: nested precision object
            if scaler is None:
                prec = getattr(strat, "precision", None)
                scaler = getattr(prec, "scaler", None) if prec is not None else None
        # PL 1.x: on trainer
        if scaler is None:
            pp = getattr(self.trainer, "precision_plugin", None)
            scaler = getattr(pp, "scaler", None) if pp is not None else None

        return scaler if isinstance(scaler, torch.cuda.amp.GradScaler) else None

    def training_step(self, batch, batch_idx):
        opt_fm, opt_ae, opt_disc = self.optimizers()
        sched_fm, sched_ae, sched_disc = self.lr_schedulers()
        scaler = self._get_grad_scaler()

        acc_steps = getattr(self, "grad_acc_steps", 1)
        acc_steps_ae = getattr(self, "grad_acc_steps_ae", 1)
        #print(f"Using grad acc steps: fm={acc_steps}, ae={acc_steps_ae}")
        step_now = ((batch_idx + 1) % acc_steps == 0) or ((batch_idx + 1) == self.trainer.num_training_batches)
        step_now_ae = ((batch_idx + 1) % acc_steps_ae == 0) or ((batch_idx + 1) == self.trainer.num_training_batches)
    
        images, _ = self.get_input(batch, 'images')
        post_latents, _ = self.get_input(batch, 'latents')
        

        #set models in eval and train modes
        requires_grad(self.ae, True)
        requires_grad(self.vit, False)
        requires_grad(self.predictor_proj, False) if self.predictor_proj is not None else None
        requires_grad(self.predictor_norm, False) if self.predictor_norm is not None else None
        requires_grad(self.predictor_proj_yol, False) if self.predictor_proj_yol is not None else None
        requires_grad(self.predictor_norm_yol, False) if self.predictor_norm_yol is not None else None
        self.ae.train()
        self.vit.eval()
        
        x, posterior = self.encode_frames(images, return_posterior=True)  # requires_grad=True
        b, f, e, h, w = x.size()

        # Split latents
        context = x[:, :-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target  = x[:, -self.num_pred_frames:].detach()

        t = torch.rand((x.shape[0],), device=x.device)
        
        # get the ema latents
        with torch.no_grad():
            x_ema = self.encode_frames(images, return_posterior=False, ema=True)  # requires_grad=True
            context_ema = x_ema[:, :-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
            target_ema  = x_ema[:, -self.num_pred_frames:].detach()
            _, post_latents_yol = self.ema_vit(target_ema, context_ema, t, return_latents=self.sr_latents, early_exit=True)
        
        
        target_t, noise = self.add_noise(target, t)
        context_noised, (noise_ctx, mask_ctx, batch_mask_ctx, t_ctx, patch_size_ctx) = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        # ---------- Pass A: AE loss for SR and also RECON

        #  ---------- SR for VAE losses  ----------
        _, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)


        sr_loss_teacher_val, sr_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_yol_val, sr_yol_layers = self.sr_loss(pred_latents, post_latents_yol, pred_yol=True)

        sr_loss_teacher_mean = sr_loss_teacher_val.mean() if hasattr(sr_loss_teacher_val, "mean") else sr_loss_teacher_val
        sr_loss_yol_mean = sr_loss_yol_val.mean() if hasattr(sr_loss_yol_val, "mean") else sr_loss_yol_val
        # Total sr_loss
        #sr_loss_mean = sr_loss_teacher_mean + self.sr_yol_loss_weight * sr_loss_yol_mean

        # ---------- AE losses  ----------
        x_rec = self.decode_frames(x.detach() if self.only_decoder else x, return_distill_conv_out=False)
        # First-stage outputs (decoder)
        distill_loss = torch.tensor(0.0, device=images.device) #self.ae.distill_loss(self.ae.get_distill_gt(images.reshape(b*f, *images.shape[2:])),
                        #                    dec_conv.reshape(b*f, *dec_conv.shape[2:]))

        aeloss, log_dict_ae = self.ae.loss(
            posterior
            , distill_loss,
            images.reshape(b*f, *images.shape[2:]),
            x_rec.reshape(b*f, *x_rec.shape[2:]),
            0,
            self.global_step,
             last_layer=self.ae.get_last_layer(),
              split="train"
        )

        loss_vae =  self.sr_loss_weight * sr_loss_teacher_mean + self.sr_yol_loss_weight * sr_loss_yol_mean + self.ae_loss_weight * aeloss

        # ---------- Backward (accumulate) ----------
        if (batch_idx % acc_steps_ae) == 0:
            #zero at the start of the accumulation window
            opt_ae.zero_grad(set_to_none=True)
        # Avoid all-reduce on non-final micro-batches (compat across PL versions)
        with self._accum_no_sync(step_now_ae):
            self.manual_backward(loss_vae / acc_steps_ae)  # grads on AE only

        # --------- Optimizer Step (when window completes) ----------
        if step_now_ae:
            if scaler is not None:
                scaler.unscale_(opt_ae)
            self.clip_gradients(opt_ae, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            if scaler is not None:
                scaler.step(opt_ae); scaler.update()
            opt_ae.step()
            opt_ae.zero_grad(set_to_none=True)
            sched_ae.step()

        # ---------- Pass A.1 discriminator training ----------
        optimizer_idx = 1
        discloss, log_dict_disc = self.ae.loss(
            posterior,
            distill_loss,
            images.reshape(b * f, *images.shape[2:]),
            x_rec.reshape(b * f, *x_rec.shape[2:]),
            optimizer_idx,
            self.global_step,
            last_layer=self.ae.get_last_layer(),
            split="train",
        )

        if (batch_idx % acc_steps_ae) == 0:
            opt_disc.zero_grad(set_to_none=True)
        with self._accum_no_sync(step_now_ae):
            self.manual_backward(discloss / acc_steps_ae)

        if step_now_ae:
            if scaler is not None:
                scaler.unscale_(opt_disc)
            opt_disc.step()
            opt_disc.zero_grad(set_to_none=True)
            sched_disc.step()

        # ---------- Pass B: ViT for v_loss and sr_loss (AE detached) ----------
        # Detach both target and context so v_loss is ViT-only
        requires_grad(self.vit, True)
        requires_grad(self.predictor_proj, True) if self.predictor_proj is not None else None
        requires_grad(self.predictor_norm, True) if self.predictor_norm is not None else None
        requires_grad(self.predictor_proj_yol, True) if self.predictor_proj_yol is not None else None
        requires_grad(self.predictor_norm_yol, True) if self.predictor_norm_yol is not None else None
        requires_grad(self.ae, False)

        #set models to eval and train modes
        self.vit.train()
        self.ae.eval()
        
        tgt_det = target.detach()
        #ctx_det = context.detach() if context is not None else None

        # requires grad to true for vit and dit to false 

        target_t_v = target_t.detach()  # target detached → no AE grad via v_loss
        context_noised_v = context_noised.detach() if context_noised is not None else None

        pred_v, pred_latents = self.vit(target_t_v, context_noised_v, t, return_latents=self.sr_latents)
        v_loss = self.v_loss(target=tgt_det, pred=pred_v, noise=noise, t=t)  # no AE grad by construction
        v_loss_mean = v_loss.mean()

        sr_loss_teacher_val, sr_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_yol_val, sr_yol_layers = self.sr_loss(pred_latents, post_latents_yol, pred_yol=True)
        sr_loss_teacher_mean = sr_loss_teacher_val.mean() if hasattr(sr_loss_teacher_val, "mean") else sr_loss_teacher_val
        sr_loss_yol_mean = sr_loss_yol_val.mean() if hasattr(sr_loss_yol_val, "mean") else sr_loss_yol_val
        # Total sr_loss
        #sr_loss_mean = sr_loss_teacher_mean + self.sr_yol_loss_weight * sr_loss_yol_mean
        # Total loss
        loss_fm = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_teacher_mean + self.sr_yol_loss_weight * sr_loss_yol_mean

        # Accumulate grads for ViT
        if (batch_idx % acc_steps) == 0:
           opt_fm.zero_grad(set_to_none=True)
        # Avoid all-reduce on non-final micro-batches (compat across PL versions)
        with self._accum_no_sync(step_now):
            self.manual_backward(loss_fm / acc_steps)  # grads on ViT only

        # AE does not get any grad from this path
        # Zero grad of AE to be sure
        
        
        # ---------- Step ----------
        if step_now:
            # Clip ViT grads before stepping (manual optimization path)
            #self._maybe_clip_grad(self.vit.parameters())
            if scaler is not None:
                scaler.unscale_(opt_fm)
            self.clip_gradients(opt_fm, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            if scaler is not None:
                scaler.step(opt_fm); scaler.update()
            opt_fm.step()
            opt_fm.zero_grad(set_to_none=True)
            sched_fm.step()

        # ---------- Logging ----------
        total_loss = self.recon_loss_weight * v_loss_mean + self.sr_loss_weight * sr_loss_teacher_mean + self.ae_loss_weight * aeloss
        self.log("train/loss", total_loss.item() if hasattr(total_loss, "item") else total_loss
                 , prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sr_loss", sr_loss_teacher_mean.item() if hasattr(sr_loss_teacher_mean, "item") else sr_loss_teacher_mean, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sr_loss_yol", sr_loss_yol_mean.item() if hasattr(sr_loss_yol_mean, "item") else sr_loss_yol_mean, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in sr_layers.items():
            self.log(f"train/sr/{k}", v.item() if hasattr(v, "item") else v, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in sr_yol_layers.items():
            self.log(f"train/sr_yol/{k}", v.item() if hasattr(v, "item") else v, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        self.log("train/aeloss", aeloss.item() if hasattr(aeloss, "item") else aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        for k, v in log_dict_ae.items():
            self.log(f"train/aeloss/{k}", v.item() if hasattr(v, "item") else v, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        self.log("train/aeloss/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        for k, v in log_dict_disc.items():
            #TODO fix this 
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return total_loss

    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA at the end of each epoch as fallback"""
        acc_steps = getattr(self, "grad_acc_steps", 1)
        step_now = ((batch_idx + 1) % acc_steps == 0) or ((batch_idx + 1) == self.trainer.num_training_batches)
        if step_now:
            update_ema(self.ema_vit, self.vit)
            update_ema(self.ema_ae, self.ae)


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # --- modes (redundant with Lightning, but explicit)
        self.vit.eval()
        self.ae.eval()
        self.ema_vit.eval()

        # ---------- Input & latents ----------
        images, frame_rate = self.get_input(batch, 'images')
        post_latents, frame_rate = self.get_input(batch, 'latents')

        x, posterior = self.encode_frames(images, return_posterior=True)  # no grad by decorator
        b, f, e, h, w = x.size()

        # Split latents: context/target (note: no training detaches needed here)
        context = x[:, :-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target  = x[:, -self.num_pred_frames:]

        # Noise schedule for validation.
        # For deterministic validation across epochs, consider fixed t (e.g., linspace) or cached t.
        t = torch.rand((x.shape[0],), device=x.device)
        
        # get the ema latents
        with torch.no_grad():
            x_ema = self.encode_frames(images, return_posterior=False, ema=True)  # requires_grad=True
            context_ema = x_ema[:, :-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
            target_ema  = x_ema[:, -self.num_pred_frames:].detach() 
            _, post_latents_yol = self.ema_vit(target_ema, context_ema, t, return_latents=self.sr_latents, early_exit=True)
        
        
        target_t, noise = self.add_noise(target, t)
        context_noised, _ = self.add_noise_ctx(context, noise=None) if context is not None else (None, None)

        # ---------- (1) ViT forward for v_loss + SR loss ----------
        pred_v, pred_latents = self.vit(target_t, context_noised, t, return_latents=self.sr_latents)
        # SR (student vs EMA teacher)
        sr_loss_teacher_val, sr_layers = self.sr_loss(pred_latents, post_latents)
        sr_loss_yol_val, sr_yol_layers = self.sr_loss(pred_latents, post_latents_yol, pred_yol=True)
        sr_loss_teacher_mean = sr_loss_teacher_val.mean() if hasattr(sr_loss_teacher_val, "mean") else sr_loss_teacher_val
        sr_loss_yol_mean = sr_loss_yol_val.mean() if hasattr(sr_loss_yol_val, "mean") else sr_loss_yol_val
        # Total sr_loss
        #sr_loss_mean = sr_loss_teacher_mean + self.sr_yol_loss_weight * sr_loss_yol_mean

        # V recon loss (prediction target is the clean target)
        v_loss = self.v_loss(target=target, pred=pred_v, noise=noise, t=t)
        v_loss_mean = v_loss.mean()

        # ---------- (2) AE reconstruction & distill ----------
        x_rec = self.decode_frames(x, return_distill_conv_out=False)
        distill_loss = torch.tensor(0.0, device=images.device)
        #self.ae.distill_loss(
        #    self.ae.get_distill_gt(images.reshape(b*f, *images.shape[2:])),
        #    dec_conv.reshape(b*f, *dec_conv.shape[2:]),
        #)
        aeloss, log_dict_ae = self.ae.loss(
            posterior,
            distill_loss,
            images.reshape(b*f, *images.shape[2:]),
            x_rec.reshape(b*f, *x_rec.shape[2:]),
            0, 
            self.global_step,
            last_layer=self.ae.get_last_layer(),
            split="val"
        )

        # ---------- (3) Combine losses ----------
        val_total = (
            self.recon_loss_weight * v_loss_mean
            + self.sr_loss_weight * sr_loss_teacher_mean
            + self.sr_yol_loss_weight * sr_loss_yol_mean
            + self.ae_loss_weight * aeloss
        )

        # ---------- Logging (per step; Lightning will epoch-average) ----------
        self.log("val/loss", val_total.item() if hasattr(val_total, "item") else val_total, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", v_loss_mean.item() if hasattr(v_loss_mean, "item") else v_loss_mean, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/sr_loss", sr_loss_teacher_mean.item() if hasattr(sr_loss_teacher_mean, "item") else sr_loss_teacher_mean, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/sr_loss_yol", sr_loss_yol_mean.item() if hasattr(sr_loss_yol_mean, "item") else sr_loss_yol_mean, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss.item() if hasattr(aeloss, "item") else aeloss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        # Log SR per-layer diagnostics as scalars
        for k, v in sr_layers.items():
            v_tensor = v.item() if hasattr(v, "item") else v
            self.log(f"val/sr/{k}", v_tensor, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in sr_yol_layers.items():
            v_tensor = v.item() if hasattr(v, "item") else v
            self.log(f"val/sr_yol/{k}", v_tensor, on_step=False, on_epoch=True, sync_dist=True)
        # Log AE per-layer diagnostics as scalars
        for k,v in log_dict_ae.items():
            v_tensor = v.item() if hasattr(v, "item") else v
            self.log(f"val/ae/{k}", v_tensor, on_step=False, on_epoch=True, sync_dist=True)
        return val_total



    def roll_out(self, x_0, num_gen_frames=25, latent_input=True, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, encdec_batch_size=8):
        if encdec_batch_size is not None and encdec_batch_size <= 0:
            raise ValueError("encdec_batch_size must be positive.")
        
        b, f = x_0.size(0), x_0.size(1)
        chunk_size = encdec_batch_size if encdec_batch_size is not None else b
        
        if latent_input:
            x_c = x_0.clone()
        else:
            if chunk_size >= b:
                x_c = self.encode_frames(x_0, ema=sample_with_ema)
            else:
                encoded_chunks = []
                for start in range(0, b, chunk_size):
                    encoded_chunks.append(self.encode_frames(x_0[start:start + chunk_size], ema=sample_with_ema))
                x_c = torch.cat(encoded_chunks, dim=0)
        
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
        if chunk_size >= x_all.size(0):
            samples = self.decode_frames(x_all)
        else:
            decoded_chunks = []
            for start in range(0, x_all.size(0), chunk_size):
                decoded_chunks.append(self.decode_frames(x_all[start:start + chunk_size]))
            samples = torch.cat(decoded_chunks, dim=0)
        return x_all, samples
    
    @torch.no_grad()
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None, return_sample=False):
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device
        
        if images is not None:
            b, f, e, h, w = images.size()
            if not latent:
                context = self.encode_frames(images, ema=sample_with_ema)
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

        #put model and vae into eval mode
        self.vit.eval()
        self.ae.eval()
        self.ema_vit.eval()

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

        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = images[:,0]
            #images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1

        log_ae = self.ae.log_images({"images": images}, **kwargs)
        for k, v in log_ae.items():
            log[f"ae/{k}"] = v

        return log

    def on_after_backward(self):
        for n, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    raise RuntimeError(f"Non-finite grad in {n}")
    


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
        patch = max(1, int(getattr(self.teacher, "patch_size", 1)))
        h = max(1, height // patch)
        w = max(1, width // patch)
        if h * w != N:
            h = max(1, int(math.floor(math.sqrt(N))))
            w = max(1, int(math.ceil(N / h)))

        latents = tokens.view(B, T, h, w, D).permute(0, 1, 4, 2, 3).contiguous()
        target_param = next(self.vit.parameters())
        return latents.to(device=target_param.device, dtype=target_param.dtype)

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


class ModelREPAIF(ModelREPA):
    def __init__(self, *, tokenizer_config, generator_config, adjust_lr_to_batch_size=False, 
                 sigma_min=1e-5, timescale=1.0, enc_scale=1.0, enc_scale_dino=1.0, warmup_steps=5000, min_lr_multiplier=0.1, num_pred_frames=1,
                 sr_latents=[],
                add_projector=False,
                add_norm=True,
                layer_norm=None,
                recon_loss_weight=1.0,
                ae_loss_weight=1.0,
                sr_loss_weight=0.1,
                sr_yol_loss_weight=0.1,
                ctx_noise_prob=0.8,
                tube_ctx_mask = True,
                loss_type = "cos_sim",
                ctx_noise_exp = 1.0,
                from_scratch = True,
                latent_dim = 1024,
                dino_predictor_only=True,
                base_feature_channels=None,
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
            sr_yol_loss_weight=sr_yol_loss_weight,
            ctx_noise_prob=ctx_noise_prob,
            tube_ctx_mask=tube_ctx_mask,
            loss_type=loss_type,
            ctx_noise_exp=ctx_noise_exp,
            ae_loss_weight=ae_loss_weight,
            from_scratch=from_scratch,
            latent_dim=latent_dim,
            dino_predictor_only=dino_predictor_only,
        )
        self.enc_scale_dino = enc_scale_dino
        self.base_feature_channels = base_feature_channels

    def _resolve_base_feature_channels(self, total_channels: int) -> int:
        base_ch = self.base_feature_channels
        if base_ch is None:
            base_ch = (total_channels + 1) // 2
        if base_ch <= 0 or base_ch >= total_channels:
            raise ValueError(
                f"base_feature_channels must be in [1, {total_channels - 1}], got {base_ch}"
            )
        return base_ch

    def encode_frames(self, images, return_posterior=False):
        if images.ndim == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1

        x = self.ae.encode(images)
        x, posterior = x["continuous"], x["posterior"]
        x0, x1 = x[0] * self.enc_scale, x[1] * self.enc_scale_dino
        x = torch.cat([x0, x1], dim=1)
        x = rearrange(x, '(b f) e h w -> b f e h w', b=b, f=f)
        
        if return_posterior:
            posterior = rearrange(posterior, '(b f) e h w -> b f e h w',b=b, f=f)   
            return x, posterior
        
        return x

    def decode_frames(self, x, return_distill_conv_out=False):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        base = self._resolve_base_feature_channels(C)

        x = torch.cat([
            x[:, :, :base] / self.enc_scale,
            x[:, :, base:] / self.enc_scale_dino
        ], dim=2)
        
        # Merge time into batch: [B*T, C, H, W]
        x = x.reshape(B * T, C, H, W)

        if return_distill_conv_out:
            distill_conv_out = self.post_quant_conv_distill(x[:, base:])
            distill_conv_out = distill_conv_out.reshape(B, T, *distill_conv_out.shape[1:])

        # Decode all frames at once
        x_post_quant = self.ae.post_quant_conv(x)
        x_dec = self.ae.decoder(x_post_quant)                        # [B*T, Cout, Hout, Wout]

        # Restore time dimension: [B, T, Cout, Hout, Wout]
        x = x_dec.reshape(B, T, x.size(1), x.size(2), x.size(3))

        if return_distill_conv_out:
            return x, distill_conv_out

        return x
