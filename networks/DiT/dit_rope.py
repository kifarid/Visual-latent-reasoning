# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Final, Optional, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Mlp
from networks.DiT.dit import FrequencyEncoder, TimestepEmbedder, FinalLayer, modulate



#################################################################################
#                                 ROPE ATTENTION                                #
#################################################################################

def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)



class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm is True'
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class STRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm is True'
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = head_dim # int(2 * ((head_dim // 3) // 2))
        self.h_dim = (head_dim // 2) #int(2 * ((head_dim // 3) // 2))
        self.w_dim = (head_dim // 2) #int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, temporal=False, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        s = 0
        # Rotate depth
        if temporal:
            qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)
            kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
            s += self.d_dim
        
        if not temporal:
            # Rotate height dim
            qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
            kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
            s += self.h_dim
            # Rotate width dim
            qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
            kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
            s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            if temporal:
                q = torch.cat([qd, qr], dim=-1)
                k = torch.cat([kd, kr], dim=-1)
            else:
                q = torch.cat([qh, qw, qr], dim=-1)
                k = torch.cat([kh, kw, kr], dim=-1)
        else:
            if temporal:
                q = qd
                k = kd
            else:
                q = torch.cat([qh, qw], dim=-1)
                k = torch.cat([kh, kw], dim=-1)


        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class SwiGLUFFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        qk_scale=None,
        drop=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            proj_drop=drop,
            norm_layer=nn.LayerNorm,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
    def forward(self, x, c, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    
class STDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        qk_scale=None,
        drop=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        causal_time_attn=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.causal_time_attn = causal_time_attn

        # Spatial attention grid size is input size, so we use relative spatial positional encoding
        self.space_attn = STRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            proj_drop=drop,
            norm_layer=nn.LayerNorm,
        )
        
        # Temporal attention grid size is 1, so we use relative temporal positional encoding
        self.time_attn = STRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=1,
            proj_drop=drop,
            norm_layer=nn.LayerNorm,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.space_mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
            self.time_mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.space_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.time_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )
        
    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
        x_ = self.space_attn(x_modulated)
        x_ = rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)
        x = x + gate_msa.unsqueeze(1).unsqueeze(1) * x_

        x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)

        # — temporal attention path —
        x = rearrange(x, 'b f n d -> (b n) f d', b=B, f=F, n=N)
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x = x + self.time_attn(x, temporal=True, attn_mask=time_attn_mask)
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
        x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated)  
        
        return x
    

class STDiTBlock2(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        qk_scale=None,
        drop=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        causal_time_attn=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.causal_time_attn = causal_time_attn

        # Spatial attention grid size is input size, so we use relative spatial positional encoding
        self.space_attn = STRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            proj_drop=drop,
            norm_layer=nn.LayerNorm,
        )
        
        # Temporal attention grid size is 1, so we use relative temporal positional encoding
        self.time_attn = STRoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=1,
            proj_drop=drop,
            norm_layer=nn.LayerNorm,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.space_mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
            self.time_mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.space_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.time_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 12 * dim, bias=True)
        )
        
    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 12 [B, C] vectors
        (shift_msa_s, scale_msa_s, gate_msa_s,
         shift_msa_t, scale_msa_t, gate_msa_t,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(12, dim=1)

        x_modulated = modulate(self.norm1(x), shift_msa_s, scale_msa_s)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b f) n d', b=B, f=F)
        x_ = self.space_attn(x_modulated)
        x = x + gate_msa_s.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b f) n d -> b f n d', b=B, f=F)

        x_modulated = modulate(self.norm2(x), shift_mlp_s, scale_mlp_s)
        x = x + gate_mlp_s.unsqueeze(1).unsqueeze(1) * self.space_mlp(x_modulated)

        # — temporal attention path —
        x_modulated = modulate(self.norm3(x), shift_msa_t, scale_msa_t)
        x_modulated = rearrange(x_modulated, 'b f n d -> (b n) f d', b=B, f=F, n=N)
        time_attn_mask = torch.tril(torch.ones(F, F, device=x.device)) if self.causal_time_attn else None
        x_ = self.time_attn(x_modulated, temporal=True, attn_mask=time_attn_mask)
        x = x + gate_msa_t.unsqueeze(1).unsqueeze(1) * rearrange(x_, '(b n) f d -> b f n d', b=B, f=F)
        x_modulated = modulate(self.norm4(x), shift_mlp_t, scale_mlp_t)
        x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated)  
        
        return x

class DiT(nn.Module):
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
        mlp_ratio=4.0,
        max_num_frames=6,
        dropout=0.0,
        ctx_noise_aug_ratio=0.1,
        ctx_noise_aug_prob = 0.5,
        drop_ctx_rate=0.2,
        frequency_range=(2, 15),
        learn_sigma=False,
    ):
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

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=True) for _ in range(depth)
        ])        
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.max_num_frames = max_num_frames
        # self.frame_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, self.max_num_frames, 1, hidden_size)), 0., 0.02)
        self.frame_rate_encoder = FrequencyEncoder(hidden_size, freq_min=frequency_range[0], freq_max=frequency_range[1])

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

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

    def get_condition_embeddings(self, t):
        """
        Get the condition embeddings for the given timesteps.
        t: (N,) tensor of diffusion timesteps
        returns: (N, D) tensor of condition embeddings
        """
        return self.t_embedder(t)

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

    def preprocess_inputs(self, target, context, t, frame_rate):
        b = target.size(0)
        
        if self.training:
            # Drop the context frame
            if torch.rand(1, device=target.device)<self.drop_ctx_rate:
                context = None
            elif torch.rand(1, device=target.device) < self.ctx_noise_aug_prob:
                # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
                mask = (t >= self.ctx_noise_aug_ratio)
                aug_noise = torch.randn_like(context)
                context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio
                
        frame_rate_embeddings = self.frame_rate_encoder.encode(frame_rate)
        frame_rate_embeddings = frame_rate_embeddings.unsqueeze(1).unsqueeze(1).to(target.device)

        x = torch.cat((context, target), dim=1) if context is not None else target
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        x = x + frame_rate_embeddings
        return x
    
    def postprocess_outputs(self, out):
        return self.unpatchify(out)

    def forward(self, target, context, t, frame_rate):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)
        
        total_frames = x.size(1)
        x = rearrange(x,  'b f hw c -> b (f hw) c')
        for block in self.blocks:
            x = block(x, c)
        x = rearrange(x,  'b (f hw) c -> b f hw c', f=(total_frames))[:,-num_frames_pred:]
        out = self.final_layer(x, c)
                  
        out = self.postprocess_outputs(out)
        return out


class STDiT(DiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.0, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), causal_time_attn=False, **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, frequency_range=frequency_range, **kwargs)
        self.blocks = nn.ModuleList([
            STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=True, causal_time_attn=causal_time_attn) for _ in range(depth)
        ])      

    def forward(self, target, context, t, frame_rate):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)
        
        x = self.preprocess_inputs(target, context, t, frame_rate)
        
        for block in self.blocks:
            x= block(x, c)                               
        out = self.final_layer(x[:,-num_frames_pred:], c)     
                  
        out = self.postprocess_outputs(out)
        return out
    
    
class STDiT2(STDiT):
    def __init__(self, input_size=16, patch_size=2, in_channels=32, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, max_num_frames=6, dropout=0.0, ctx_noise_aug_ratio=0.1,ctx_noise_aug_prob=0.5, drop_ctx_rate=0.2, frequency_range=(2, 15), causal_time_attn=False, **kwargs):
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, max_num_frames=max_num_frames, dropout=dropout, ctx_noise_aug_ratio=ctx_noise_aug_ratio, ctx_noise_aug_prob=ctx_noise_aug_prob, drop_ctx_rate=drop_ctx_rate, frequency_range=frequency_range, causal_time_attn=causal_time_attn, **kwargs)
        self.blocks = nn.ModuleList([
            STDiTBlock2(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_norm=True, causal_time_attn=causal_time_attn) for _ in range(depth)
        ])      