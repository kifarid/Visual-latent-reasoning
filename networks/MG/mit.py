# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from omegaconf import ListConfig


# def modulate(x, shift, scale):
#     if len(x.shape) == 3:
#         # x: [B, N, C], shift: [B, C], scale: [B, C]
#         return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
#     elif len(x.shape) == 4:
#         # x: [B, T, N, C], shift: [B, C], scale: [B, C]
#         return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
#     else:
#         raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3D or 4D tensor.")


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
   
#################################################################################
#                   Sine/Cosine Frequency Embedding Functions                  #
#################################################################################

class FrequencyEncoder:
    def __init__(self, embed_dim, freq_min=1, freq_max=5):
        """
        Deterministic frequency encoder with fixed normalization.

        Args:
            embed_dim (int): Dimensionality of the token embeddings.
            freq_min (float): Minimum frequency value.
            freq_max (float): Maximum frequency value.
        """
        self.embed_dim = embed_dim
        self.freq_min = freq_min
        self.freq_max = freq_max

    def encode(self, frequencies):
        """
        Encodes frequencies into embeddings using sine-cosine features.

        Args:
            frequencies (torch.Tensor): Tensor of shape (batch_size,) containing frequencies.

        Returns:
            torch.Tensor: Encoded frequency embeddings of shape (batch_size, embed_dim).
        """
        batch_size = frequencies.size(0)

        # Fixed normalization: Scale frequencies to [0, 1]
        normalized_freq = (frequencies - self.freq_min) / (self.freq_max - self.freq_min)

        # Generate positional features using sine and cosine
        positions = torch.arange(0, self.embed_dim, dtype=torch.float32, device=frequencies.device)
        scaling_factors = 1 / (10000 ** (2 * (positions // 2) / self.embed_dim))
        frequency_features = normalized_freq.unsqueeze(1) * scaling_factors  # Shape: (batch_size, embed_dim)

        # Apply sine to even indices and cosine to odd indices
        encoded_freq = torch.zeros(batch_size, self.embed_dim, device=frequencies.device)
        encoded_freq[:, 0::2] = torch.sin(frequency_features[:, 0::2])  # Sine for even indices
        encoded_freq[:, 1::2] = torch.cos(frequency_features[:, 1::2])  # Cosine for odd indices

        return encoded_freq


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                 Cross Attention                               #
#################################################################################

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism that computes attention between target (x) and context (y).
    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to include bias terms.
        attn_drop (float): Dropout rate for attention weights.
        proj_drop (float): Dropout rate after projection.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False, bias=False)

    def forward(self, x, y):
        B, N, C = x.shape
        B, M, _ = y.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(attn_output))




class STBlock(nn.Module):
    # Used for temporal compression in context
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, 
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, 
                                   attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        B, F, N, D = x.shape

        # Spatial attention
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = x + self.space_attn(self.norm1(x))
        x = x + self.space_mlp(self.norm2(x))

        # Temporal attention
        x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
        x = x + self.time_attn(self.norm3(x))
        x = x + self.time_mlp(self.norm4(x))

        # Restore original shape
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)

        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class MiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, conditional=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        if conditional:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None


    def forward(self, x, c=None):
        params = self.adaLN_modulation(c).chunk(6, dim=1) if self.adaLN_modulation is not None else [0, 0, 1, 0, 0, 1]
        # Unpack and broadcast if needed
        p_unsqueeze = lambda p: p.unsqueeze(1) if len(x.shape) == 3 else p.unsqueeze(1).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            p_unsqueeze(p) if isinstance(p, torch.Tensor) else p for p in params
        ]

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class STMiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, conditional = False, **block_kwargs):
        super().__init__()
        
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        if conditional:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None 
        self.causal_time_attn = causal_time_attn

    def forward(self, x, c=None):
        B, F, N, D = x.shape

        if c is None:
            params = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        
        else:
            params = self.adaLN_modulation(c).chunk(9, dim=1)


        # Unpack and broadcast if needed
        p_unsqueeze = lambda p: p.unsqueeze(1) if len(x.shape) == 3 else p.unsqueeze(1).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp_s, scale_mlp_s, gate_mlp_s, shift_mlp_t, scale_mlp_t, gate_mlp_t = [
            p_unsqueeze(p) if isinstance(p, torch.Tensor) else p for p in params
        ]

        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
        x_ = self.space_attn(x_modulated)
        x_ = rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)
        x = x + gate_msa * x_

        x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + gate_mlp_s * self.space_mlp(x_modulated)

        # — temporal attention path —
        x = rearrange(x, 'b f n d -> (b n) f d', b=B, f=F, n=N)
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x = x + self.time_attn(x, attn_mask=time_attn_mask)
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
        x = x + gate_mlp_t * self.time_mlp(x_modulated)  
        return x
    


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, conditional=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        ) if conditional else None

    def forward(self, x, c):
        params = self.adaLN_modulation(c).chunk(2, dim=1) if self.adaLN_modulation is not None else [0, 0]
        p_unsqueeze = lambda p: p.unsqueeze(1) if len(x.shape) == 3 else p.unsqueeze(1).unsqueeze(1)
        shift, scale = [p_unsqueeze(p) if isinstance(p, torch.Tensor) else p for p in params]
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



#
class MiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=16,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        codebook_size=1024,
        mlp_ratio=4.0,
        max_num_frames=6,
        dropout=0.0,
        conditional=False,
        **kwargs,
    ):
        super().__init__()
        self.input_size= input_size if isinstance(input_size, (list, tuple, ListConfig)) else [input_size, input_size]
        
        self.in_channels = in_channels
        self.out_channels = hidden_size 
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.drop_ctx_rate = 0.0
        self.ctx_noise_aug_ratio = 0.0
        self.ctx_noise_aug_prob = 0.0
        self.codebook_size = codebook_size
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        self.tok_pre_emb = nn.Embedding(codebook_size+1, hidden_size)

        self.x_embedder = PatchEmbed(input_size, patch_size, hidden_size, hidden_size, bias=True)
        
        self.num_patches = self.x_embedder.num_patches

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            MiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, conditional=conditional) for _ in range(depth)
        ])        
        
        self.final_layer = FinalLayer(hidden_size, patch_size, hidden_size)
        
        self.max_num_frames = max_num_frames
        self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0., 0.02)
        

        #(TODO: this needs to be verified)
        self.bias = nn.Parameter(torch.zeros(self.input_size[0], self.input_size[1], codebook_size + 1))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size], cls_token=False, extra_tokens=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # (TODO: Initialize token embedding:)
        
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if block.adaLN_modulation is None:
                continue
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.final_layer.adaLN_modulation is not None:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        #zero initialize bias
        nn.init.constant_(self.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]

        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum('bfhwpqc->bfchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, w * p))
        return imgs

    def get_condition_embeddings(self, t=None):
        """
        Get the condition embeddings for the given timesteps.
        t: (N,) tensor of diffusion timesteps
        returns: (N, D) tensor of condition embeddings
        """
        return self.t_embedder(t) if t else None

    def preprocess_inputs(self, target, context, t=None):
        b, f_target = target.size()[:2]
        ctx = context

        if self.training and ctx is not None:
            if self.drop_ctx_rate > 0 and torch.rand(1, device=target.device) < self.drop_ctx_rate:
                ctx = None
            elif (
                self.ctx_noise_aug_prob > 0
                and torch.rand(1, device=target.device) < self.ctx_noise_aug_prob
                and t is not None
            ):
                noise_mask = t >= self.ctx_noise_aug_ratio
                if noise_mask.any():
                    aug_noise = torch.randn_like(ctx[noise_mask])
                    ctx[noise_mask] = ctx[noise_mask] + aug_noise * self.ctx_noise_aug_ratio

        f_context = ctx.size(1) if ctx is not None else 0
        total_frames = f_target + f_context
        if total_frames > self.max_num_frames:
            raise ValueError(
                f"total frames ({total_frames}) exceed configured max_num_frames ({self.max_num_frames})."
            )

        x = torch.cat((ctx, target), dim=1) if ctx is not None else target
        x = rearrange(x, 'b f h w -> (b f) h w')
        
        # suitable for VQ-VAE token embedding and MG style 
        x = self.tok_pre_emb(x)
        x = x.permute(0, 3, 1, 2)  # (b*f, h, w, c) -> (b*f, c, h, w)
        
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        frame_offset = self.max_num_frames - total_frames
        x = x + self.frame_emb[:, frame_offset:].to(x.device)
        return x

    def postprocess_outputs(self, out):
        out_unp = self.unpatchify(out)
        out_unp = out_unp.permute(0, 1, 3, 4, 2)  # (N, T, C, H, W)
        logit = torch.matmul(out_unp, self.tok_pre_emb.weight.t()) + self.bias.unsqueeze(0).unsqueeze(0).to(out_unp.device)  # (N, T, C, H, W, codebook_size+1)

        return logit

    def forward(self, target, context, t=None, y=None):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        num_frames_ctx = context.size(1)
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)
        
        x = self.preprocess_inputs(target, context, t)
        
        x = rearrange(x,  'b f hw c -> b (f hw) c')
        for block in self.blocks:
            x = block(x, c)
        x = rearrange(x,  'b (f hw) c -> b f hw c', f=(num_frames_ctx+num_frames_pred))[:,-num_frames_pred:]
        out = self.final_layer(x, c)
                  
        out = self.postprocess_outputs(out)

        return out



class STMiT(MiT):
    def __init__(
        self,
        input_size=16,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        codebook_size=16384,
        mlp_ratio=4.0,
        max_num_frames=6,
        dropout=0.1,
        conditional=False,
        ctx_noise_aug_ratio=0.1,
        ctx_noise_aug_prob=0.5,
        drop_ctx_rate=0.2,
        causal_time_attn=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            codebook_size=codebook_size,
            mlp_ratio=mlp_ratio,
            max_num_frames=max_num_frames,
            dropout=dropout,
            conditional=conditional,
            **kwargs,
        )
        self.drop_ctx_rate = drop_ctx_rate
        self.ctx_noise_aug_ratio = ctx_noise_aug_ratio
        self.ctx_noise_aug_prob = ctx_noise_aug_prob
        self.blocks = nn.ModuleList([
                STMiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout, causal_time_attn=causal_time_attn, conditional=conditional) for _ in range(depth)
            ])

    def forward(self, target, context, t=None):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        f_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)                   # (N, D)
        
        x = self.preprocess_inputs(target, context, t)  # (B, F, N, D)

        for block in (self.blocks):
            x = block(x, c)

        out = self.final_layer(x[:,-f_pred:], c)                # (N, T, patch_size * out_channels)
        out = self.postprocess_outputs(out)  # (N, T, patch_size ** 2 * out_channels)
        return out
