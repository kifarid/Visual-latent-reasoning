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

from ..swin.swin_free_aspect_ratio import SwinTransformerBlock
from typing import Tuple, Dict
from util import instantiate_from_config, unsqueeze_middle_match, repeat_first_dim


def select_positional_embeddings_with_stride(
    pos_emb: torch.Tensor,  # (1, max_F, 1, C)
    fps_multipliers: torch.Tensor,  # (N,) values in (0, 1]
    target_F: int
) -> torch.Tensor:
    """
    Efficient strided sampling of positional embeddings without for-loops.

    Args:
        pos_emb: (1, max_F, 1, C) learned positional embeddings
        fps_multipliers: (N,) tensor of frame rate scaling factors in (0, 1]
        target_F: int — number of frames (F) in the input
    Returns:
        sampled_pos_emb: (N, target_F, 1, C)
    """
    device = pos_emb.device
    N = fps_multipliers.shape[0]
    max_F = pos_emb.shape[1]
    C = pos_emb.shape[-1]
    
    # Flatten to (max_F, C)
    base = pos_emb[0, :, 0, :]  # (max_F, C)

    # Compute strided indices for each sample
    base_indices = torch.arange(target_F, device=device).float()  # (F,)
    strides = (1.0 / fps_multipliers).unsqueeze(1)  # (N, 1)
    indices = torch.floor(base_indices.unsqueeze(0) * strides).long()  # (N, F)
    indices = torch.clamp(indices, max=max_F - 1)  # (N, F)

    # Gather using advanced indexing
    # Expand indices to shape (N, F, C)
    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, C)  # (N, F, C)
    base_expanded = base.unsqueeze(0).expand(N, -1, -1)     # (N, max_F, C)
    selected = torch.gather(base_expanded, dim=1, index=idx_expanded)  # (N, F, C)

    return selected.unsqueeze(2)  # (N, F, 1, C)


def repeat_positional_embeddings_with_stride(
    pos_emb: torch.Tensor,         # (1, max_F, 1, C)
    fps_multipliers: torch.Tensor, # (N,) integer repeat counts ≥ 1
    target_F: int
) -> torch.Tensor:
    """
    Fully vectorized version to repeat positional embeddings using fps_multipliers.
    Args:
        pos_emb: Tensor of shape (1, max_F, 1, C) containing positional embeddings.
        fps_multipliers: Tensor of shape (N,) containing integer multipliers for each sample.
        target_F: Target number of frames to generate for each sample.
    Returns:
        Tensor of shape (N, target_F, 1, C)
    """
    device = pos_emb.device
    N = fps_multipliers.shape[0]
    C = pos_emb.shape[-1]
    max_F = pos_emb.shape[1]

    # Compute base indices: each row will be [0, 1, ..., target_F - 1]
    base = torch.arange(target_F, device=device).unsqueeze(0).expand(N, -1)  # (N, target_F)

    # Divide to get repeated index pattern (integer division gives correct stride)
    index = base // fps_multipliers  # (N, target_F)
    index = torch.clamp(index, max=max_F - 1)     # Ensure indices stay in bounds

    # Base pos embeddings: (N, max_F, C)
    pos_emb_expanded = pos_emb[..., 0, :].expand(N, -1, -1)   # (N, max_F, C)

    # Expand for gather
    index_expanded = index.unsqueeze(-1).expand(-1, -1, C)  # (N, target_F, C)

    # Gather the embeddings
    selected = torch.gather(pos_emb_expanded, dim=1, index=index_expanded)  # (N, target_F, C)

    return selected.unsqueeze(2)  # (N, target_F, 1, C)


class NPatchEmbed(nn.Module):
    """Wrapper for PatchEmbed to support videos and multi-view videos."""
    def __init__(self, patch_embed_cfg):
        super().__init__()
        self.patch_embed = instantiate_from_config(patch_embed_cfg)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor with shape
                - B x T x C x H x W (video)
                - B x V x T x C x H x W (multi-view video)
        Returns:
            Tensor with shape:
                - B x T x NumPatches x EmbedDim (for video)
                - B x V x T x NumPatches x EmbedDim (for multi-view)
        """
        orig_shape = x.shape
        if x.ndim == 5:
            # Video input: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.patch_embed(x)
            x = x.view(B, T, *x.shape[1:]) #B T C H W
        elif x.ndim == 6:
            # Multi-view video input: (B, V, T, C, H, W)
            B, V, T, C, H, W = x.shape
            x = x.view(B * V * T, C, H, W)
            x = self.patch_embed(x)
            x = x.view(B, V, T, *x.shape[1:])
        else:
            raise ValueError(f"Unsupported input shape: {orig_shape}")

        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Applies shift and scale modulation to input tensor `x`.
    Supports inputs with shape [B, ..., C] where ... are one or more middle dims.
    
    shift and scale must have shape [B, C].
    """
    if shift.shape != scale.shape:
        raise ValueError("shift and scale must have the same shape")

    if x.shape[0] != shift.shape[0] or x.shape[-1] != shift.shape[-1]:
        raise ValueError("shift/scale must match x on first and last dimensions")

    scale_ = unsqueeze_middle_match(scale, x)
    shift_ = unsqueeze_middle_match(shift, x)
    return x * (1 + scale_) + shift_


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
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
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

        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm,
                                    attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm,
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
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################
# TODO : try:  # self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=ffn_bias)
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class CDiTBlock(nn.Module):
    """
    Changes in comparison to the original CDiTBlock('https://github.com/facebookresearch/nwm/blob/main/models.py'):
    - If conditioning is not provided, the block uses self-attention instead of cross-attention.
    - using CrossAttention instead of nn.MultiheadAttention.
    - qk normalization is applied to the query and key vectors.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_ctx = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        
        self.target_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm,  **block_kwargs)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout_rate)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

    def forward(self, target, ctx, c):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)                
        target = target + gate_msa.unsqueeze(1) * self.target_attn(modulate(self.norm1(target), shift_msa, scale_msa))
        target_modulated = modulate(self.norm2(target), shift_ca_x, scale_ca_x)
        if ctx is not None:
            ctx_norm = modulate(self.norm_ctx(ctx), shift_ca_xcond, scale_ca_xcond)
            target = target + gate_ca_x.unsqueeze(1) * self.attn(target_modulated, ctx_norm)
        else:
            target = target + gate_ca_x.unsqueeze(1) * self.attn(target_modulated, target_modulated)

        target = target + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(target), shift_mlp, scale_mlp))
        return target, ctx
    


class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def _apply_mod(self, x, norm, block, shift, scale, gate):
        
        shift_e = shift 
        scale_e = scale 
        gate_e  = gate 
        return x + gate_e * block(modulate(norm(x), shift_e, scale_e))


    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=-1)


        # Spatial attention path: [B, F, N, D] -> [B*F, N, D]
        x = rearrange(x, 'b f n d -> (b f) n d', b=B, f=F)


        # - spatial path chunks adjust -
        shift_msa, scale_msa, gate_msa = map(lambda x: rearrange(x, 'b f n d -> (b f) n d'),  #N T C H W
                                             (shift_msa, scale_msa, gate_msa))
        shift_mlp_s, scale_mlp_s, gate_mlp_s = map(lambda x: rearrange(x, 'b f n d -> (b f) n d'), 
                                                   (shift_mlp_s, scale_mlp_s, gate_mlp_s))
        # - temporal path chunks adjust -
        shift_mlp_t, scale_mlp_t, gate_mlp_t = map(lambda x: rearrange(x, 'b f n d -> (b n) f d'), 
                                                   (shift_mlp_t, scale_mlp_t, gate_mlp_t))
        

        x = self._apply_mod(x, self.norm1, self.space_attn,
                             shift_msa,   scale_msa,   gate_msa)
        x = self._apply_mod(x, self.norm2, self.space_mlp,
                             shift_mlp_s, scale_mlp_s, gate_mlp_s)

        # — temporal attention path —
        x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
        # repeat first dim for the temporal path
        shift_mlp_t, scale_mlp_t, gate_mlp_t = map(lambda y: repeat_first_dim(y, x),
                                                    (shift_mlp_t, scale_mlp_t, gate_mlp_t))
        x = x + self.time_attn(x)   # same as before
        x = self._apply_mod(x, self.norm3, self.time_mlp,
                             shift_mlp_t, scale_mlp_t, gate_mlp_t)

        # restore
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        return x
  


    
class SwinSTDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, input_shape, layer_idx, mlp_ratio=4.0, window_size=[6, 4], dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.space_attn = SwinTransformerBlock(
                hidden_size,
                input_resolution=input_shape,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=(0,0) if (layer_idx % 2 == 0) else [ws//2 for ws in window_size],
                mlp_ratio=mlp_ratio,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                )
        self.time_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.space_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.time_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def _apply_mod(self, x, norm, block, shift, scale, gate):
        
        reshaping_fn = lambda x,y: x.view(x.shape[0] * x.shape[1], *x.shape[2:])
        shift_e = shift #reshaping_fn(shift, x) 
        scale_e = scale #reshaping_fn(scale, x)
        gate_e  = gate #reshaping_fn(gate,  x)
        #gate_e  = unsqueeze_middle_match(gate_e,  x)
        return x + gate_e * block(modulate(norm(x), shift_e, scale_e))

    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=-1)


        # Spatial attention path: [B, F, N, D] -> [B*F, N, D]
        x = rearrange(x, 'b f n d -> (b f) n d', b=B, f=F)
        

        # - spatial path chunks adjust -
        shift_msa, scale_msa, gate_msa = map(lambda x: rearrange(x, 'b f n d -> (b f) n d'), 
                                             (shift_msa, scale_msa, gate_msa))
        shift_mlp_s, scale_mlp_s, gate_mlp_s = map(lambda x: rearrange(x, 'b f n d -> (b f) n d'), 
                                                   (shift_mlp_s, scale_mlp_s, gate_mlp_s))
        # - temporal path chunks adjust -
        shift_mlp_t, scale_mlp_t, gate_mlp_t = map(lambda x: rearrange(x, 'b f n d -> (b n) f d'), 
                                                   (shift_mlp_t, scale_mlp_t, gate_mlp_t))
        

        x = self._apply_mod(x, self.norm1, self.space_attn,
                             shift_msa,   scale_msa,   gate_msa)
        x = self._apply_mod(x, self.norm2, self.space_mlp,
                             shift_mlp_s, scale_mlp_s, gate_mlp_s)

        # — temporal attention path —
        x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F, n=N)
        # repeat first dim for the temporal path
        shift_mlp_t, scale_mlp_t, gate_mlp_t = map(lambda y: repeat_first_dim(y, x),
                                                    (shift_mlp_t, scale_mlp_t, gate_mlp_t))
        x = x + self.time_attn(x)   # same as before
        x = self._apply_mod(x, self.norm3, self.time_mlp,
                             shift_mlp_t, scale_mlp_t, gate_mlp_t)

        # restore
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MVDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        max_num_frames: int = 6,
        dropout: float = 0.0,
        ctx_noise_aug_ratio: float = 0.1,
        ctx_noise_aug_prob: float = 0.5,
        drop_ctx_rate: float = 0.2,
        frequency_range: Tuple[int, int] = (2, 15),
        learn_sigma: bool = False,
        cond_signals: Dict = {}
    ):
        '''
        Args:
            input_size: Size of the input images (assumed square).
            patch_size: Size of the patches to be extracted from the input images.
            in_channels: Number of input channels (e.g., 3 for RGB images).
            hidden_size: Dimensionality of the hidden representations in the Transformer.
            depth: Number of Transformer blocks in the model.
            num_heads: Number of attention heads in the Transformer.
            mlp_ratio: Ratio of the hidden dimension in the MLP compared to the hidden size.
            max_num_frames: Maximum number of frames in the input video sequence.
            dropout: Dropout rate for the Transformer blocks.
            ctx_noise_aug_ratio: Ratio of noise augmentation applied to context frames.
            ctx_noise_aug_prob: Probability of applying noise augmentation to context frames.
            drop_ctx_rate: Probability of dropping the context frame during training.
            frequency_range: Range of frequencies for the frequency encoder.
            learn_sigma: Whether to learn the sigma parameter in the final layer.
            cond_signals: Dictionary of additional conditioning signals and their configurations.
            '''
        super().__init__()
        self.input_size=input_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.ctx_noise_aug_ratio = ctx_noise_aug_ratio
        self.ctx_noise_aug_prob = ctx_noise_aug_prob
        self.drop_ctx_rate = drop_ctx_rate

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        self.num_patches = self.x_embedder.num_patches

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout) for _ in range(depth)
        ])        
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.max_num_frames = max_num_frames
        self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0., 0.02)

        self.frame_rate_encoder = FrequencyEncoder(hidden_size//patch_size**2, freq_min=frequency_range[0], freq_max=frequency_range[1])

        self.cond_signals_encoders = nn.ModuleDict()
        for signal_name, c_cfg in cond_signals.items():
            if signal_name == 'frame_rate':
                continue
            self.cond_signals_encoders[signal_name] = instantiate_from_config(c_cfg)


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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]
        t = x.shape[1]


        x = x.reshape(shape=(x.shape[0], t, h, w, p, p, c))
        x = torch.einsum('nthwpqc->ntchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], t, c, h * p, w * p))
        return imgs

    def forward(self, x, t, frame_rate, cond_signals=None, view_indicies=None):
        """
        Forward pass of DiT.
        x: (N, V, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        b, f, l, h, w = x.size()

        c = self.t_embedder(t)                   # (N, D)
        enc_cond_signals = {}
        c_highest_dim = c
        if cond_signals is not None:
            for signal_name, signal in cond_signals.items():
                if signal_name == 'frame_rate':
                    continue
                if signal_name not in self.cond_signals_encoders:
                    raise ValueError(f"Conditioning signal '{signal_name}' is not supported.")
                
                enc_cond_signals[signal_name] = self.cond_signals_encoders[signal_name](signal)
                c_highest_dim = enc_cond_signals[signal_name] if enc_cond_signals[signal_name].ndim > c_highest_dim.ndim else c_highest_dim

        #sum all cond signals while using unsqueeze_middle_match to ensure consistent shape
        if len(enc_cond_signals) > 0:
            c = unsqueeze_middle_match(c, c_highest_dim)
            c = c + sum([unsqueeze_middle_match(enc_cond_signals[signal_name], c_highest_dim) for signal_name in enc_cond_signals])

        if self.training:
            # Drop the context frame
            if torch.rand(1, device=x.device)<self.drop_ctx_rate:
                x = x[:, -1:]
                f=1
            elif torch.rand(1, device=x.device) < self.ctx_noise_aug_prob:
                # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
                mask = (t >= self.ctx_noise_aug_ratio)
                aug_noise = torch.randn_like(x[:, :-1])
                x[:, :-1][mask] = x[:, :-1][mask] + aug_noise[mask] * self.ctx_noise_aug_ratio



        frame_embeddings = self.frame_rate_encoder.encode(frame_rate)
        #frame_embeddings = frame_embeddings.unsqueeze(1).unsqueeze(1).to(x.device)
        frame_embeddings = unsqueeze_middle_match(frame_embeddings, x).to(x.device)
        
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', f=f)

        frame_t_emb = self.frame_emb[:, self.max_num_frames-f:].to(x.device)
        x = x + self.frame_emb[:, self.max_num_frames-f:].to(x.device) 
        x = x + frame_embeddings
        
        x = rearrange(x,  'b f hw c -> b (f hw) c')
        for block in self.blocks:
            x = block(x, c)
        x = rearrange(x,  'b (f hw) c -> b f hw c', f=f)[:,-1]
        out = self.final_layer(x, c)                
        out = self.unpatchify(out)
        return out


class MVSTDiT(MVDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, **kwargs)
        self.blocks = nn.ModuleList([
                STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout) for _ in range(depth)
            ])       
        self.frame_rate_encoder = FrequencyEncoder(hidden_size, freq_min=frequency_range[0], freq_max=frequency_range[1])

        
    def forward(self, x, t, frame_rate, cond_signals=None, t_indices=None):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        b, f, l, h, w = x.size()
        
        c = self.t_embedder(t)                   # (N, D)


        enc_cond_signals = {}
        c_highest_dim = c
        if cond_signals is not None:
            for signal_name, signal in cond_signals.items():
                if signal_name == 'frame_rate':
                    continue
                if signal_name not in self.cond_signals_encoders:
                    raise ValueError(f"Conditioning signal '{signal_name}' is not supported.")
                
                enc_cond_signals[signal_name] = self.cond_signals_encoders[signal_name](signal)
                c_highest_dim = enc_cond_signals[signal_name] if enc_cond_signals[signal_name].ndim > c_highest_dim.ndim else c_highest_dim

        #sum all cond signals while using unsqueeze_middle_match to ensure consistent shape
        if len(enc_cond_signals) > 0:
            c = unsqueeze_middle_match(c, c_highest_dim)
        
        #add the conditioning signals
        if enc_cond_signals:
            for signal_name, signal in enc_cond_signals.items():
                c = c + unsqueeze_middle_match(signal, c_highest_dim)

        #adding the embedding for frame_time, positional embedding and frame embeddings
        

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device) 
        
        x = rearrange(x, '(b f) hw c -> b f hw c', f=f)
        frame_t_emb = self.frame_emb.to(x.device)[0, t_indices, 0, :].unsqueeze(2)  # (N, T, 1, D)
            
        x = x + frame_t_emb


        #possible C dimensions must start with B and end with C
        for block in (self.blocks):
            x = block(x, c)

        out = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        out = self.unpatchify(out)  # (N, T, patch_size ** 2 * out_channels)
        
        return out    


class MVSwinSTDiT(MVDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, window_size=[6, 4], dropout=0.1, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, **kwargs)
        self.blocks = nn.ModuleList([
                SwinSTDiTBlock(hidden_size=hidden_size, num_heads=num_heads, input_shape=input_size, layer_idx=layer_idx, mlp_ratio=mlp_ratio, window_size=window_size, dropout_rate=dropout) for layer_idx in range(depth)
            ])

        self.frame_rate_encoder = FrequencyEncoder(hidden_size//patch_size**2, freq_min=frequency_range[0], freq_max=frequency_range[1])
    
    def forward(self, x, t, frame_rate, cond_signals=None, t_indices=None):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        cond_signals: dictionary of additional conditioning signals
        t_indices (N, F): frame indices for the current batch, used to gather frame embeddings
        """
        b, f, l, h, w = x.size()
        
        c = self.t_embedder(t)                   # (N, D)


        enc_cond_signals = {}
        c_highest_dim = c
        if cond_signals is not None:
            for signal_name, signal in cond_signals.items():
                if signal_name == 'frame_rate':
                    continue
                if signal_name not in self.cond_signals_encoders:
                    raise ValueError(f"Conditioning signal '{signal_name}' is not supported.")
                
                enc_cond_signals[signal_name] = self.cond_signals_encoders[signal_name](signal)
                c_highest_dim = enc_cond_signals[signal_name] if enc_cond_signals[signal_name].ndim > c_highest_dim.ndim else c_highest_dim

        #sum all cond signals while using unsqueeze_middle_match to ensure consistent shape
        if len(enc_cond_signals) > 0:
            c = unsqueeze_middle_match(c, c_highest_dim)
        
        #add the conditioning signals
        if enc_cond_signals:
            for signal_name, signal in enc_cond_signals.items():
                c = c + unsqueeze_middle_match(signal, c_highest_dim)

        #adding the embedding for frame_time, positional embedding and frame embeddings
        

        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed.to(x.device) 
        
        x = rearrange(x, '(b f) hw c -> b f hw c', f=f)
        frame_t_emb = self.frame_emb.to(x.device)[0, t_indices, 0, :].unsqueeze(2)  # (N, T, 1, D)
            
        x = x + frame_t_emb


        #possible C dimensions must start with B and end with C
        for block in (self.blocks):
            x = block(x, c)

        out = self.final_layer(x, c)                # (N, T, patch_size * out_channels)
        out = self.unpatchify(out)  # (N, T, patch_size ** 2 * out_channels)
        
        return out