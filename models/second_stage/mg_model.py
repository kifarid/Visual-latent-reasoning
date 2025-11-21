import os
import math
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torchvision.utils as vutils
import torch.distributions as TD
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


class Model(pl.LightningModule):
    def __init__(self, *, tokenizer_config, loss_config, generator_config, scheduler_config, adjust_lr_to_batch_size=False, enc_scale=4, warmup_steps=5000, min_lr_multiplier=0.1,num_pred_frames=1):
        super().__init__()

        self.num_pred_frames = num_pred_frames
        self.enc_scale = enc_scale

        # Training parameters
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        
        # denoising backbone
        self.vit = self.build_generator(generator_config)
        
        # Tokenizer
        self.ae = self.build_tokenizer(tokenizer_config)

        # EMA
        self.ema_vit = init_ema_model(self.vit)


        # Loss 
        self.criterion = instantiate_from_config(loss_config)

        # Scheduler
        # Sampler
        self.codebook_size = self.vit.codebook_size
        self.scheduler = instantiate_from_config(scheduler_config)


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

        #ensure that the model returns the min indecies in the original shape, avoids shit ton of hassle
        model.quantize.sane_index_shape = True
        model.eval()
        return model
    
    def build_generator(self, generator_config):
        """
        Instantiate the denoising backbone model from the config.
        """
        model = instantiate_from_config(generator_config)
        return model
    

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
        optimizer = torch.optim.AdamW(self.vit.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        return  [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def get_input(self, batch, k):
        if type(batch) == dict:
            x = batch[k]
        else:
            x = batch
        assert len(x.shape) == 5, 'input must be 5D tensor'
        return x
    
    
    def add_noise(self, x, codebook_size, drop_token_prob):
        if drop_token_prob <= 0:
            return x

        mask = torch.rand(x.shape, device=x.device) < drop_token_prob
        if not mask.any():
            return x

        dropped = x.clone()
        dropped.masked_fill_(mask, codebook_size)
        return dropped
    
    @torch.no_grad()
    def encode_frames(self, images):
        if len(images.size()) == 5:
            b, f, e, h, w = images.size()
            images = rearrange(images, 'b f e h w -> (b f) e h w')
        else:
            b, e, h, w = images.size()
            f = 1

        #(TODO: Modify to use discrete vqgan like tokens instead of continuous the output format is found in /work/dlclarge2/faridk-diff_force/orbis/models/first_stage/vqgan.py and the quantize module is modules/quantize.py in current repo I am not sure of the shape so please fix this)
        x = self.ae.encode(images)['indices']
        x = rearrange(x, '(b f) h w -> b f h w',b=b, f=f)
        return x
    
    @torch.no_grad()
    def decode_frames(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected codes with shape (batch, frames, h, w); got {tuple(x.shape)}")

        b, f, h, w = x.shape
        codes = rearrange(x, 'b frame h w -> (b frame) h w').long()
        decoded  = self.ae.decode_code(codes)
        decoded = rearrange(decoded, '(b frame) c H W -> b frame c H W', b=b, frame=f)
        return decoded
  
    
    def training_step(self, batch, batch_idx):
        images = self.get_input(batch, 'images')

        code = self.encode_frames(images)

        b, f, h, w = code.size()

        context = code[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = code[:,-self.num_pred_frames:]

        target_masked, mask = self.scheduler.get_mask_code(target)

        pred = self.vit(target_masked, context)
        
        # CE on recon tokens 
        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), target.reshape(-1))

        
        loss = loss.mean()
            
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA at the end of each epoch as fallback"""
        update_ema(self.ema_vit, self.vit)



    def validation_step(self, batch, batch_idx):
        images = self.get_input(batch, 'images')

        code = self.encode_frames(images)

        b, f, h, w = code.size()

        context = code[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None
        target = code[:,-self.num_pred_frames:]

        target_masked, mask = self.scheduler.get_mask_code(target)

        pred = self.vit(target_masked, context)

        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), target.reshape(-1))

        loss = loss.mean()
            
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)


    def roll_out(
        self,
        x_0,
        num_gen_frames=25,
        latent_input=True,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        encdec_batch_size=8,
    ):
        if encdec_batch_size is not None and encdec_batch_size <= 0:
            raise ValueError("encdec_batch_size must be positive.")

        b, f = x_0.size(0), x_0.size(1)
        chunk_size = encdec_batch_size if encdec_batch_size is not None else b

        if latent_input:
            x_c = x_0.clone()
        else:
            if chunk_size >= b:
                x_c = self.encode_frames(x_0)
            else:
                encoded_chunks = []
                for start in range(0, b, chunk_size):
                    encoded_chunks.append(self.encode_frames(x_0[start:start + chunk_size]))
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
    def sample(self, images=None, latent=False, eta=0.0, NFE=20, sample_with_ema=True, num_samples=8, frame_rate=None, return_sample=False, sm_temp=1.0, r_temp=1.0):
        """
        """
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device
        
        context = None

        if images is not None:
            if latent:
                context = images.clone()
            else:
                context = self.encode_frames(images)

        input_h, input_w = self.vit.input_size[0], self.vit.input_size[1]

        if context is not None:
            _, _, h, w = context.size()
        elif images is not None and not latent:
            _, _, _, h, w = images.size()
        else:
            h, w = input_h, input_w

        if context is not None and (context.size(-2) != h or context.size(-1) != w):
            raise ValueError("Context spatial size does not match inferred dimensions.")

        target_masked = torch.full((num_samples, self.num_pred_frames, h, w), self.scheduler.mask_value, device=device)
        mask = torch.ones_like(target_masked, device=device).bool()
        tokens_per_frame = h * w
        total_tokens = self.num_pred_frames * h * w
        mask_flat = mask.view(num_samples, -1)
        batch_indices = torch.arange(num_samples, device=device).unsqueeze(1)
        scheduler = self.scheduler.adap_sche(NFE) 

        with torch.no_grad():
            for indice, t in enumerate(scheduler):
                #t = t_steps[i].repeat(target_masked.shape[0])

                total_remaining = int(mask_flat.sum().item())
                t_int = int(t.item()) if torch.is_tensor(t) else int(t)
                if total_remaining < t_int:
                    t_int = total_remaining

                if total_remaining == 0:
                    break

                if t_int <= 0:
                    continue

                
                logit = net(target_masked, context)
                logit = logit.reshape(num_samples, self.num_pred_frames*tokens_per_frame, -1)
                prob = torch.softmax(logit * sm_temp, dim=-1)
                prob_flat = prob.reshape(num_samples * self.num_pred_frames * tokens_per_frame, -1)
                distri = TD.Categorical(probs=prob_flat)
                pred_code = distri.sample()
                conf = torch.gather(prob_flat, -1, pred_code.unsqueeze(-1)).squeeze(-1)
                pred_code = pred_code.reshape(num_samples, self.num_pred_frames, h, w)
                conf = conf.reshape(num_samples, self.num_pred_frames, h, w)

                #linearly randomize the conf 
                if len(scheduler) <= 1:
                    ratio = 1.0
                else:
                    ratio = indice / (len(scheduler) - 1)
                rand = -torch.log(-torch.log(torch.rand(num_samples, total_tokens, device=device)))
                conf_flat = conf.reshape(num_samples, -1)
                conf_scores = torch.log(conf_flat.clamp_min(1e-12)) + r_temp * (1 - ratio) * rand
                # do not predict on already predicted tokens
                conf_scores.masked_fill_(~mask_flat, -math.inf)

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf_scores, k=t_int, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                selected = (conf_scores >= tresh_conf.unsqueeze(-1)).reshape(num_samples, self.num_pred_frames, h, w)
                f_mask = mask & selected
                target_masked[f_mask] = pred_code[f_mask]


                # update the mask
                if indice_mask.numel() > 0:
                    mask_flat[batch_indices, indice_mask] = False

        
        _target_masked = torch.clamp(target_masked, 0, self.codebook_size - 1).long()
        if return_sample:
            images = self.decode_frames(_target_masked.clone())
            return _target_masked, images
        else:
            return _target_masked

    
    @torch.no_grad()
    def reco(self, images, latent=False):
        '''
        Reconstruct the input images
        from the input images or from the input latent code
        '''
        l_visual = [images[:,f] for f in range(images[:, : -self.num_pred_frames].size(1))]


        if latent:
            code = images.clone()
        else:
            code = self.encode_frames(images)
        
        b, fs, h, w = code.size()


        # first get the decoded target frames from stage 1
        context = code[:,:-self.num_pred_frames] if fs - self.num_pred_frames > 0 else None
        target = code[:,-self.num_pred_frames:]
        target_decoded = self.decode_frames(target)#.view(b * self.num_pred_frames, *images.shape[-3:])
        #move the frames to first 
        l_visual.extend([target_decoded[:,f] for f in range(self.num_pred_frames)])


        # second get the masked version of these frames frame 
        target_masked, mask = self.scheduler.get_mask_code(target)
        mask_interpolated = torch.nn.functional.interpolate(mask.float(), size=images.size()[-2:], mode='nearest').bool().unsqueeze(2)
        target_decoded_masked = target_decoded * mask_interpolated 
        l_visual.extend([target_decoded_masked[:,f] for f in range(self.num_pred_frames)])

        # finally predict these frames and append
        pred = self.vit(target_masked, context)
        unmasked_code = torch.softmax(pred, -1).max(-1)[1]
        target_pred_decoded = self.decode_frames(unmasked_code)
        l_visual.extend([target_pred_decoded[:,f] for f in range(self.num_pred_frames)])

        return torch.cat(l_visual, dim=0)
    

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        images = self.get_input(batch, 'images')
        N = min(5, images.size(0))
        images = images[:N]
        b, f, e, h, w = images.size()

        l_visual_recon = [images[:,f] for f in range(images.size(1))]
        l_visual_recon_ema = [images[:,f] for f in range(images.size(1))]


        recon = self.reco(images)
        recon = vutils.make_grid(recon, nrow=N, padding=2, normalize=False)
        log["recon"] = recon

        
        images = images[:,:-self.num_pred_frames] if f - self.num_pred_frames > 0 else None

        # sample
        # with self.vit.summon_full_params(self.vit, recurse=True):
        samples = self.sample(images, eta=0.0, NFE=30, sample_with_ema=False, num_samples=N, return_sample=True)[1]

        # Only keep the first generated frame
        samples = samples[:N]

        for i in range(samples.size(1)):
            l_visual_recon.append(samples[:,i])

        l_visual_recon = torch.cat(l_visual_recon, dim=0)
        chunks = torch.chunk(l_visual_recon, 2 + 2, dim=0)
        sampled = torch.cat(chunks, 0)
        sampled = vutils.make_grid(sampled, nrow=N, padding=2, normalize=False)
        
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

        # reconstruction with main vit
        log["ema_sampled"] = sampled_ema
        log["sampled"] = sampled
        self.vit.train()
        return log
    

    
class ModelIF(Model):
    def __init__(
        self,
        *,
        tokenizer_config,
        loss_config,
        generator_config,
        scheduler_config,
        adjust_lr_to_batch_size=False,
        enc_scale=4,
        warmup_steps=5000,
        min_lr_multiplier=0.1,
        num_pred_frames=1,
    ):
        super().__init__(
            tokenizer_config=tokenizer_config,
            loss_config=loss_config,
            generator_config=generator_config,
            scheduler_config=scheduler_config,
            adjust_lr_to_batch_size=adjust_lr_to_batch_size,
            enc_scale=enc_scale,
            warmup_steps=warmup_steps,
            min_lr_multiplier=min_lr_multiplier,
            num_pred_frames=num_pred_frames,
        )

        if not hasattr(self.ae, "quantize2"):
            raise AttributeError(
                "Tokenizer for ModelIF must expose a second quantizer (quantize2) for semantics."
            )

        self.num_code_streams = 2
        quantize_sem = getattr(self.ae, "quantize2")
        self.sem_codebook_size = getattr(quantize_sem, "n_e", self.codebook_size)

        if hasattr(self.scheduler, "num_predicted_frames"):
            self.scheduler.num_predicted_frames = self.num_pred_frames * self.num_code_streams
            self.scheduler.total_tokens = (
                self.scheduler.num_tokens
                * self.scheduler.num_tokens
                * self.scheduler.num_predicted_frames
            )

    def _flatten_streams(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 4:
            return codes
        if codes.dim() != 5:
            raise ValueError(
                f"Expected latent codes with 5 dims (b, t, streams, h, w); got {codes.shape}"
            )
        return rearrange(codes, "b t s h w -> b (t s) h w")

    def _unflatten_streams(self, codes: torch.Tensor, frames: Optional[int] = None) -> torch.Tensor:
        if codes.dim() == 5:
            return codes
        if codes.dim() != 4:
            raise ValueError(
                f"Expected flattened codes with shape (b, t, h, w); got {codes.shape}"
            )

        if frames is None:
            if codes.size(1) % self.num_code_streams != 0:
                raise ValueError(
                    "Number of tokens is not divisible by the number of code streams."
                )
            frames = codes.size(1) // self.num_code_streams

        return rearrange(codes, "b (t s) h w -> b t s h w", t=frames, s=self.num_code_streams)

    def _clamp_code_streams(self, codes: torch.Tensor) -> torch.Tensor:
        codes_split = self._unflatten_streams(codes).clone()
        codes_split[..., 0].clamp_(0, self.codebook_size - 1)
        codes_split[..., 1].clamp_(0, self.sem_codebook_size - 1)
        return self._flatten_streams(codes_split).long()

    @torch.no_grad()
    def encode_frames(self, images):
        if images.ndim == 5:
            b, f, c, h, w = images.size()
            flat_images = rearrange(images, "b f c h w -> (b f) c h w")
        else:
            b, c, h, w = images.size()
            f = 1
            flat_images = images

        encoded = self.ae.encode(flat_images)
        indices = encoded.get("indices")
        if not isinstance(indices, (list, tuple)) or len(indices) < 2:
            raise ValueError("Tokenizer must return a tuple of (recon_indices, sem_indices) for IF mode.")

        recon_idx, sem_idx = indices[0], indices[1]
        recon_idx = rearrange(recon_idx.long(), "(b f) h w -> b f h w", b=b, f=f)
        sem_idx = rearrange(sem_idx.long(), "(b f) h w -> b f h w", b=b, f=f)
        stacked = torch.stack((recon_idx, sem_idx), dim=2)
        return self._flatten_streams(stacked)

    @torch.no_grad()
    def decode_frames(self, codes):
        codes_split = self._unflatten_streams(codes)
        b, t, s, h, w = codes_split.shape
        if s != self.num_code_streams:
            raise ValueError(f"Expected {self.num_code_streams} code streams, received {s}.")

        recon_codes = rearrange(codes_split[:, :, 0].long(), "b t h w -> (b t) h w")
        sem_codes = rearrange(codes_split[:, :, 1].long(), "b t h w -> (b t) h w")
        decoded = self.ae.decode_code((recon_codes, sem_codes))
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return rearrange(decoded, "(b t) c H W -> b t c H W", b=b, t=t)

    def _split_context_target(self, codes_flat: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        frames = codes_flat.size(1) // self.num_code_streams
        pred_tokens = self.num_pred_frames * self.num_code_streams
        context_len = max(frames - self.num_pred_frames, 0) * self.num_code_streams
        context = codes_flat[:, :context_len] if context_len > 0 else None
        target = codes_flat[:, -pred_tokens:]
        return context, target

    def training_step(self, batch, batch_idx):
        images = self.get_input(batch, "images")
        code_flat = self.encode_frames(images)
        context, target = self._split_context_target(code_flat)
        target_masked, _ = self.scheduler.get_mask_code(target, value=self.mask_value)
        pred = self.vit(target_masked, context)
        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code_flat.view(-1))
        loss = loss.mean()
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = self.get_input(batch, "images")
        code_flat = self.encode_frames(images)
        context, target = self._split_context_target(code_flat)
        target_masked, _ = self.scheduler.get_mask_code(target, value=self.mask_value)
        pred = self.vit(target_masked, context)
        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code_flat.view(-1))
        loss = loss.mean()
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def reco(self, images, latent=False):
        l_visual = [images[:, f_idx] for f_idx in range(images[:, : -self.num_pred_frames].size(1))]

        if latent:
            code_flat = images.clone()
        else:
            code_flat = self.encode_frames(images)

        context, target = self._split_context_target(code_flat)
        target_split = self._unflatten_streams(target, frames=self.num_pred_frames)
        target_decoded = self.decode_frames(target_split)
        l_visual.append(target_decoded)

        target_masked, mask = self.scheduler.get_mask_code(target, value=self.mask_value)
        mask_split = self._unflatten_streams(mask.float(), frames=self.num_pred_frames)
        mask_recon = mask_split[:, :, 0]
        mask_interpolated = torch.nn.functional.interpolate(
            mask_recon.unsqueeze(2),
            size=images.size()[-2:],
            mode="nearest",
        ).squeeze(2).bool()
        target_decoded_masked = target_decoded * mask_interpolated.unsqueeze(2)
        l_visual.append(target_decoded_masked)

        pred = self.vit(target_masked, context)
        pred_codes = torch.softmax(pred, -1).max(-1)[1]
        pred_codes = self._clamp_code_streams(pred_codes)
        pred_split = self._unflatten_streams(pred_codes, frames=self.num_pred_frames)
        target_pred_decoded = self.decode_frames(pred_split)
        l_visual.append(target_pred_decoded)

        return torch.cat(l_visual, dim=0)

    @torch.no_grad()
    def sample(
        self,
        images=None,
        latent=False,
        eta=0.0,
        NFE=20,
        sample_with_ema=True,
        num_samples=8,
        frame_rate=None,
        return_sample=False,
        sm_temp=1.0,
        r_temp=4.0,
    ):
        net = self.ema_vit if sample_with_ema else self.vit
        device = next(net.parameters()).device

        context = None
        if images is not None:
            if latent:
                context = images.clone()
                if context.dim() == 5:
                    context = self._flatten_streams(context)
            else:
                context = self.encode_frames(images)

        if isinstance(self.vit.input_size, (list, tuple, ListConfig)):
            input_size = list(self.vit.input_size)
            if len(input_size) == 1:
                input_h = input_w = input_size[0]
            else:
                input_h, input_w = input_size[0], input_size[1]
        else:
            input_h = input_w = self.vit.input_size

        if context is not None:
            _, _, h, w = context.size()
        elif images is not None and not latent:
            _, _, _, h, w = images.size()
        else:
            h, w = input_h, input_w

        if context is not None and (context.size(-2) != h or context.size(-1) != w):
            raise ValueError("Context spatial size does not match inferred dimensions.")

        target_masked = torch.full(
            (num_samples, self.num_pred_frames * self.num_code_streams, h, w),
            self.mask_value,
            device=device,
        )
        mask = torch.ones_like(target_masked, dtype=torch.bool, device=device)
        tokens_per_frame = h * w
        total_tokens = self.num_pred_frames * self.num_code_streams * tokens_per_frame
        mask_flat = mask.view(num_samples, -1)
        batch_indices = torch.arange(num_samples, device=device).unsqueeze(1)
        scheduler = self.scheduler.adap_sche(NFE)

        with torch.no_grad():
            for indice, t in enumerate(scheduler):
                total_remaining = int(mask_flat.sum().item())
                t_int = int(t.item()) if torch.is_tensor(t) else int(t)
                if total_remaining < t_int:
                    t_int = total_remaining

                if total_remaining == 0:
                    break

                if t_int <= 0:
                    continue

                logit = net(target_masked, context)
                logit = logit.reshape(num_samples, total_tokens, -1)
                prob = torch.softmax(logit * sm_temp, dim=-1)
                prob_flat = prob.reshape(num_samples * total_tokens, -1)
                distri = TD.Categorical(probs=prob_flat)
                pred_code = distri.sample()
                conf = torch.gather(prob_flat, -1, pred_code.unsqueeze(-1)).squeeze(-1)
                pred_code = pred_code.reshape(num_samples, self.num_pred_frames * self.num_code_streams, h, w)
                conf = conf.reshape(num_samples, self.num_pred_frames * self.num_code_streams, h, w)

                if len(scheduler) <= 1:
                    ratio = 1.0
                else:
                    ratio = indice / (len(scheduler) - 1)
                rand = -torch.log(-torch.log(torch.rand(num_samples, total_tokens, device=device)))
                conf_flat = conf.reshape(num_samples, -1)
                conf_scores = torch.log(conf_flat.clamp_min(1e-12)) + r_temp * (1 - ratio) * rand
                conf_scores.masked_fill_(~mask_flat, -math.inf)

                tresh_conf, indice_mask = torch.topk(conf_scores, k=t_int, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                selected = (conf_scores >= tresh_conf.unsqueeze(-1)).reshape(
                    num_samples, self.num_pred_frames * self.num_code_streams, h, w
                )
                f_mask = mask & selected
                target_masked[f_mask] = pred_code[f_mask]

                if indice_mask.numel() > 0:
                    mask_flat[batch_indices, indice_mask] = False

        clamped = self._clamp_code_streams(target_masked)
        if return_sample:
            decoded = self.decode_frames(self._unflatten_streams(clamped, frames=self.num_pred_frames))
            return clamped, decoded
        return clamped
