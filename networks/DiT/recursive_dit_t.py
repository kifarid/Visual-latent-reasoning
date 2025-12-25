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

from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers.attention import AttentionRope
from omegaconf import ListConfig

def broadcast_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Insert singleton dims in x so that it is broadcastable to target.
    Assumes:
      - dim 0 is batch, dim -1 is channels
      - x and target are broadcast-compatible.
    """
    t_inner = list(target.shape[1:-1])      # target inner dims
    x_inner = list(x.shape[1:-1])           # x inner dims

    i = 0
    positions = []
    # greedily match x_inner dims into t_inner
    for j, td in enumerate(t_inner):
        if i < len(x_inner) and (x_inner[i] == td or x_inner[i] == 1):
            positions.append(j)
            i += 1
    # assume all x_inner are matched (you guaranteed this by construction)

    used = set(positions)
    offset = 0
    # insert 1-dims at the remaining positions
    for j in range(len(t_inner)):
        if j not in used:
            dim = 1 + j + offset          # +1 to skip batch dim
            x = x.unsqueeze(dim)
            offset += 1

    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Applies adaLN-Zero modulation to input tensor x."""
    # x: [B, T, N, C], shift: [B, C], scale: [B, C]
    return x * (1 + scale) + shift


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """
    Numerically stable truncated normal initializer matching JAX's default.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
            return tensor

        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2

        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(
            1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
        )

        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clamp_(lower * comp_std, upper * comp_std)
    return tensor

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

# --- small RoPE helpers ---
def _rotate_half(x):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_nd(q, k, cos, sin):
    """
    q,k: (..., D)
    cos,sin: broadcastable to q,k
    """
    q_ = (q * cos) + (_rotate_half(q) * sin)
    k_ = (k * cos) + (_rotate_half(k) * sin)
    return q_, k_


class RotaryEmbedding3D(nn.Module):
    """
    Axial 3D RoPE (t, y, x). Compact per-axis caches.
    Forward supports axis ablations via `mode`: {"st","t","s","none"}.
    """
    def __init__(self, dim: int, time: int, height: int, width: int,
                 base: float = 10000.0, device=None, dtype=torch.float32):
        super().__init__()
        assert dim % 6 == 0, "dim must be divisible by 6 for 3D axial RoPE"
        self.dim, self.time, self.height, self.width = dim, time, height, width
        self.hw = height * width

        dim_axis = dim // 6  # per axis (t, y, x)
        inv = 1.0 / (base ** (torch.arange(0, dim_axis, dtype=dtype, device=device) / dim_axis))

        t_pos = torch.arange(time,   dtype=dtype, device=device)
        y_pos = torch.arange(height, dtype=dtype, device=device)
        x_pos = torch.arange(width,  dtype=dtype, device=device)

        ft = torch.outer(t_pos, inv)   # (T, D/6)
        fy = torch.outer(y_pos, inv)   # (H, D/6)
        fx = torch.outer(x_pos, inv)   # (W, D/6)

        # prebuild spatial block [y|x] → (HW, 2*D/6)
        freqs_s = torch.cat([
            fy[:, None, :].expand(height, width, -1),
            fx[None, :, :].expand(height, width, -1),
        ], dim=-1).reshape(self.hw, 2 * dim_axis)

        # compact per-axis caches
        self.register_buffer("cos_t", ft.cos(), persistent=False)     # (T, D/6)
        self.register_buffer("sin_t", ft.sin(), persistent=False)
        self.register_buffer("cos_s", freqs_s.cos(), persistent=False)  # (HW, 2*D/6)
        self.register_buffer("sin_s", freqs_s.sin(), persistent=False)

    @torch.no_grad()
    def forward(self, t_idxs: torch.Tensor, mode: str = "global", flatten: bool = True):
        """
        t_idxs: (B, S) long frame indices
        mode: "global" | "t" | "s" | "none"
        returns cos, sin shaped:
          (B, S*HW, D) if flatten else (B, S, HW, D)
        """
        B, S = t_idxs.shape
        D = self.dim
        d6 = D // 6
        d_spatial_half = 2 * d6  # (y|x) block size
        HW = self.hw

        # gather time half: (B,S,D/6)
        ct = self.cos_t[t_idxs]  # cos(time)
        st = self.sin_t[t_idxs]  # sin(time)

        # spatial half tiled once: (B,S,HW, 2*D/6)
        cs = self.cos_s.view(1, 1, HW, d_spatial_half).expand(B, S, HW, -1)
        ss = self.sin_s.view(1, 1, HW, d_spatial_half).expand(B, S, HW, -1)

        # expand time along HW: (B,S,HW,D/6)
        ct_exp = ct.unsqueeze(2).expand(-1, -1, HW, -1)
        st_exp = st.unsqueeze(2).expand(-1, -1, HW, -1)

        # --- Axis masking via (cos=1, sin=0) ---
        if mode in ("t", "none"):
            # disable spatial rotations
            cs = torch.ones_like(cs)
            ss = torch.zeros_like(ss)
        if mode in ("s", "none"):
            # disable temporal rotations
            ct_exp = torch.ones_like(ct_exp)
            st_exp = torch.zeros_like(st_exp)
        # ---------------------------------------

        # assemble axial half: [t | (y|x)] → (B,S,HW,D/2)
        cos_half = torch.cat([ct_exp, cs], dim=-1)
        sin_half = torch.cat([st_exp, ss], dim=-1)

        # duplicate across halves to match rotate_half → (B,S,HW,D)
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)

        if flatten:
            cos = cos.reshape(B, S * HW, D)
            sin = sin.reshape(B, S * HW, D)
        return cos, sin



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
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, attn_drop=dropout_rate, proj_drop=dropout_rate, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class STDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, causal_time_attn=False, **block_kwargs):
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.causal_time_attn = causal_time_attn

    def forward(self, x, c):
        B, F, N, D = x.shape

        # chunk into 9 [B, C] vectors
        (shift_msa, scale_msa, gate_msa,
         shift_mlp_s, scale_mlp_s, gate_mlp_s,
         shift_mlp_t, scale_mlp_t, gate_mlp_t) = self.adaLN_modulation(c).chunk(9, dim=-1)
        
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
        x = x + self.time_attn(x, attn_mask=time_attn_mask)
        x = rearrange(x, '(b n) f d -> b f n d', b=B, n=N, f=F)
        x_modulated = modulate(self.norm3(x), shift_mlp_t, scale_mlp_t)
        x = x + gate_mlp_t.unsqueeze(1).unsqueeze(1) * self.time_mlp(x_modulated)  
        
        return x
    


class FlexDiTBlock(nn.Module):
    """
    Minimal DiT-style block with LayerNorm + adaLN-Zero conditioning.
    Modes:
      - "global": attention over all tokens (F*N)
      - "t"     : temporal attention across frames per spatial token
      - "s"     : spatial attention across tokens per frame
      - "none"  : like "t" path but RoPE disabled (via rope.mode="none")
    Inputs:
      x: [B, F, N, D]
      t_idxs: [B, F]
      rope: RotaryEmbedding3D(dim=D, time=F, height=H, width=W, ...)
      c: [B, D] conditioner vector for adaLN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                dropout_rate=0.0, causal_time_attn=False,
                num_prefix_tokens: int = 0,
                **attn_kwargs):
        super().__init__()
        eps = 1e-6

        # Pre-norms used with adaLN (no affine inside the norms)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)  # for attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)  # for MLP

        # Shared Attention (your implementation)
        self.attn = AttentionRope(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qkv_fused=True,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=dropout_rate,
            proj_drop=dropout_rate,
            norm_layer=nn.LayerNorm,
            qk_norm=True,
            scale_norm=False,
            proj_bias=True,
            rotate_half=False,
            **attn_kwargs,
        )

        # Lightweight pointwise MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu, drop=dropout_rate)

        # adaLN head: (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Identity-at-init for adaLN-Zero
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.causal_time_attn = causal_time_attn

    @torch.no_grad()
    def _causal_mask(self, L, device):
        # boolean lower-triangular mask (broadcasted by Attention)
        return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

    def _rope_embed(self, rope3d, t_idxs, mode, flatten, Bprime=None, L=None):
        """Convert RotaryEmbedding3D output into AttentionRope format [B', 1, L, 2 * dim]."""
        cos, sin = rope3d(t_idxs, mode=mode, flatten=flatten)
        if not flatten:
            if Bprime is None or L is None:
                raise ValueError("Bprime and L must be provided when flatten=False.")
            cos = cos.reshape(Bprime, L, -1)
            sin = sin.reshape(Bprime, L, -1)
        rope = torch.cat([sin, cos], dim=-1)
        return rope.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        t_idxs: torch.Tensor,
        rope3d,
        mode: str,
        c: torch.Tensor,
        reg_tokens: torch.Tensor | None = None,
    ):
        """
        Returns:
            x: [B, F, N, D] tensor
            reg_tokens: updated register tokens if provided
        """
        assert mode in ("global", "t", "s", "none")
        B, F, N, D = x.shape

        # adaLN params
        (shift_attn, scale_attn, gate_attn,
         shift_mlp,  scale_mlp,  gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        gate_attn = torch.tanh(gate_attn)
        gate_mlp  = torch.tanh(gate_mlp)

        # --- Attention path ---
        x_attn_in = modulate(self.norm1(x), shift_attn, scale_attn)

        if mode == "global":
            # [B, F*N, D]
            xin = rearrange(x_attn_in, 'b f n d -> b (f n) d')
            L = xin.shape[1]
            if reg_tokens is not None:
                if self.attn.num_prefix_tokens != reg_tokens.shape[1]:
                    raise ValueError("reg_tokens length must equal AttentionRope.num_prefix_tokens.")
                xin = torch.cat([reg_tokens, xin], dim=1)
            rope = self._rope_embed(
                rope3d, t_idxs, mode="global",
                flatten=True,
            )
            y = self.attn(xin, rope=rope, attn_mask=None)
            if reg_tokens is not None:
                reg_tokens, y = torch.split(y, [reg_tokens.shape[1], L], dim=1)
            y = rearrange(y, 'b (f n) d -> b f n d', f=F, n=N)

        elif mode in ("t", "none"):
            # temporal attention per spatial token: [(B*N), F, D]
            assert N == rope3d.hw, "For 't'/'none', N must match rope3d.hw (H*W)."
            xin = rearrange(x_attn_in, 'b f n d -> (b n) f d')
            Bp, L = xin.shape[0], F
            if reg_tokens is not None:
                regs = reg_tokens.repeat_interleave(N, dim=0)
                xin = torch.cat([regs, xin], dim=1)
            rope = self._rope_embed(
                rope3d, t_idxs, mode=("t" if mode == "t" else "none"),
                flatten=False, Bprime=Bp, L=L
            )
            attn_mask = None
            if self.causal_time_attn:
                m = self._causal_mask(F, xin.device)
                attn_mask = m.view(1, 1, F, F)
            y = self.attn(xin, rope=rope, attn_mask=attn_mask)
            if reg_tokens is not None:
                regs, y = torch.split(y, [reg_tokens.shape[1], L], dim=1)
                reg_tokens = regs.view(B, N, reg_tokens.shape[1], D).mean(dim=1)
            y = rearrange(y, '(b n) f d -> b f n d', b=B, n=N)

        else:  # mode == "s"
            # spatial attention per frame: [(B*F), N, D]
            assert N == rope3d.hw, "For 's', N must match rope3d.hw (H*W)."
            xin = rearrange(x_attn_in, 'b f n d -> (b f) n d')
            Bp, L = xin.shape[0], N
            if reg_tokens is not None:
                regs = reg_tokens.repeat_interleave(F, dim=0)
                xin = torch.cat([regs, xin], dim=1)
            rope = self._rope_embed(
                rope3d, t_idxs, mode="s",
                flatten=False, Bprime=Bp, L=L
            )
            y = self.attn(xin, rope=rope, attn_mask=None)
            if reg_tokens is not None:
                regs, y = torch.split(y, [reg_tokens.shape[1], L], dim=1)
                reg_tokens = regs.view(B, F, reg_tokens.shape[1], D).mean(dim=1)
            y = rearrange(y, '(b f) n d -> b f n d', b=B, f=F)

        # gated residual (attention)
        x = x + gate_attn * y

        # --- MLP path ---
        x_mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        z = self.mlp(x_mlp_in)
        x = x + gate_mlp * z
        return x, reg_tokens


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
        learn_sigma=False,
    ):
        super().__init__()
        self.input_size= input_size if isinstance(input_size, (list, tuple, ListConfig)) else [input_size, input_size]
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
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size], cls_token=False, extra_tokens=0)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum('bfhwpqc->bfchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, w * p))
        return imgs

    def get_condition_embeddings(self, t):
        """
        Get the condition embeddings for the given timesteps.
        t: (N, ...) tensor of diffusion timesteps
        returns: (N, ..., D) tensor of condition embeddings
        """
        t_shape = t.shape
        t = t.reshape(-1)
        t_emb = self.t_embedder(t)
        t_emb = t_emb.reshape(*t_shape, -1)
        return t_emb

    def preprocess_inputs(self, target, context, t):
        b, f_target = target.size()[:2]
        f_context = context.size(1) if context is not None else 0
        
        if self.training:
            # Drop the context frame
            if torch.rand(1, device=target.device)<self.drop_ctx_rate:
                context = None
            elif torch.rand(1, device=target.device) < self.ctx_noise_aug_prob:
                # Add noise to context frames (if t is less than ctx_noise_aug_ratio, we do not add noise)
                mask = (t >= self.ctx_noise_aug_ratio)
                aug_noise = torch.randn_like(context)
                context[mask] = context[mask] + aug_noise[mask] * self.ctx_noise_aug_ratio
        x = torch.cat((context, target), dim=1) if context is not None else target
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) #+ self.pos_embed.to(x.device)
        x = rearrange(x, '(b f) hw c -> b f hw c', b=b)
        #x = x + self.frame_emb[:, self.max_num_frames-(f_target+f_context):].to(x.device) 
        return x

    def postprocess_outputs(self, out):
        return self.unpatchify(out)

    def forward(self, target, context, t, return_latents=[], reg_tokens=None, early_exit=False):
        """
        Forward pass of DiT.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        return_latents: (list) of indices of the intermediate DiT latents to return

        return: (N, T, patch_size ** 2 * out_channels) tensor of output images
        if return_latents:
            return (N, T, patch_size ** 2 * hidden_channels) tensor of intermediate latents
        """
        
        num_frames_ctx = context.size(1)
        num_frames_pred = target.size(1)
        
        c = self.get_condition_embeddings(t)
        
        x = self.preprocess_inputs(target, context, t)
        
        x = rearrange(x,  'b f hw c -> b (f hw) c')
        l_latents = []
        exit_after = max(return_latents) if return_latents else None
        for idx, block in enumerate(self.blocks):
            x = block(x, c)
            if idx in return_latents:
                l_latents.append(x)
            if early_exit and exit_after is not None and idx >= exit_after:
                break
        x = rearrange(x,  'b (f hw) c -> b f hw c', f=(num_frames_ctx+num_frames_pred))[:,-num_frames_pred:]
        out = self.final_layer(x, c)
                  
        out = self.postprocess_outputs(out)

        if return_latents:
            return out, l_latents
        return out


class FlexDiT(DiT):
    """
    FlexDiT stitches FlexDiTBlocks into a structured schedule:
      1. A main attention (mode in {"global","t","s"}) defined per depth slot.
      2. A spatial attention block (mode='s').
      3. A temporal attention block (mode='t').
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
        ctx_noise_aug_prob=0.5,
        drop_ctx_rate=0.2,
        main_block_type="global",
        num_spatial_per_main=1,
        num_temporal_per_main=1,
        num_reg_tokens=0,
        causal_time_attn=False,
        inner_depth=None,
        **kwargs,
    ):
        """
        Args mirror DiT/STDiT with the addition of:
            main_block_type: str describing the main attention mode {"global", "t", "s"}.
            num_spatial_per_main: number of spatial-only FlexDiTBlocks to run after each main block.
            num_temporal_per_main: number of temporal-only FlexDiTBlocks to run after each main block.
            num_reg_tokens: number of register tokens prepended in main blocks (uses AttentionRope prefix slots).
            inner_depth: number of unique blocks; total depth must be a multiple of this value.
                If None, defaults to depth (no repetition). Effective repeats = depth // inner_depth.
        """
        self.hidden_size = hidden_size
        self.num_reg_tokens = int(num_reg_tokens)
        if self.num_reg_tokens < 0:
            raise ValueError("num_reg_tokens must be >= 0.")
        self.reg_token = None
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_num_frames=max_num_frames,
            dropout=dropout,
            ctx_noise_aug_ratio=ctx_noise_aug_ratio,
            ctx_noise_aug_prob=ctx_noise_aug_prob,
            drop_ctx_rate=drop_ctx_rate,
            **kwargs,
        )

        # Discard the base DiT block stack (we build custom blocks below).
        self.blocks = nn.ModuleList()

        self.depth = int(depth)
        if inner_depth is None:
            self.inner_depth = self.depth
        else:
            self.inner_depth = int(inner_depth)  # number of unique layers
        if self.inner_depth < 1:
            raise ValueError("inner_depth must be >= 1.")
        if self.depth % self.inner_depth != 0:
            raise ValueError("depth must be divisible by inner_depth.")
        self.block_repeats = self.depth // self.inner_depth  # effective repeat count
        grid_h, grid_w = self.x_embedder.grid_size
        self.rope = RotaryEmbedding3D(
            dim=hidden_size//num_heads,
            time=self.max_num_frames,
            height=grid_h,
            width=grid_w,
        )

        allowed_modes = {"global", "t", "s"}
        mode_key = str(main_block_type).lower()
        if mode_key not in allowed_modes:
            raise ValueError(f"Unsupported main block mode '{main_block_type}'. Expected one of {allowed_modes}.")
        self.main_block_mode = mode_key

        if num_spatial_per_main < 0 or num_temporal_per_main < 0:
            raise ValueError("num_spatial_per_main and num_temporal_per_main must be >= 0.")
        self.num_spatial_per_main = int(num_spatial_per_main)
        self.num_temporal_per_main = int(num_temporal_per_main)

        def _make_block(num_prefix_tokens: int = 0):
            return FlexDiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout,
                causal_time_attn=causal_time_attn,
                num_prefix_tokens=num_prefix_tokens,
            )

        self.main_blocks = nn.ModuleList(
            [_make_block(num_prefix_tokens=self.num_reg_tokens) for _ in range(self.inner_depth)]
        )
        self.spatial_blocks = (
            nn.ModuleList(
                [nn.ModuleList([_make_block(num_prefix_tokens=self.num_reg_tokens) for _ in range(self.num_spatial_per_main)]) for _ in range(self.inner_depth)]
            )
            if self.num_spatial_per_main > 0
            else None
        )
        self.temporal_blocks = (
            nn.ModuleList(
                [nn.ModuleList([_make_block(num_prefix_tokens=self.num_reg_tokens) for _ in range(self.num_temporal_per_main)]) for _ in range(self.inner_depth)]
            )
            if self.num_temporal_per_main > 0
            else None
        )

    def _block_indices(self):
        """
        Returns iterable of block indices to execute. Subclasses can override.
        """
        return range(self.depth)

    def _build_time_indices(self, batch_size, seq_len, device):
        if seq_len > self.max_num_frames:
            raise ValueError(f"Sequence of {seq_len} frames exceeds max_num_frames={self.max_num_frames}.")
        start = 0 #self.max_num_frames - seq_len
        frame_ids = torch.arange(start, seq_len, device=device)
        return frame_ids.unsqueeze(0).expand(batch_size, -1)

    def forward(self, target, context, t, frame_idxs = None, reg_tokens=None, return_latents=[], return_regs=[], early_exit=False):
        f_pred = target.size(1)
        #  (B, F, H, W) -> (B, F, HW)
        if t.ndim == 1:
            t = t[:, None, None, None]
            t = t.expand(-1, f_pred, *self.x_embedder.grid_size)
        t = rearrange(t, 'B F H W -> B F (H W)')
        
        c = self.get_condition_embeddings(t) # (B, F, HW) -> (B, F, HW, D) 
        x = self.preprocess_inputs(target, context, t)  # (B, F, HW, D)
        batch_size, seq_len, num_tokens, _ = x.shape
        if num_tokens != self.rope.hw:
            raise ValueError(f"Token count {num_tokens} does not match RoPE HW={self.rope.hw}.")

        if frame_idxs is None:
            frame_indices = self._build_time_indices(batch_size, seq_len, x.device)
        else:
            frame_indices = frame_idxs
        latents = []
        regs = []
        exit_targets = []
        if return_latents:
            exit_targets.append(max(return_latents))
        if return_regs:
            exit_targets.append(max(return_regs))
        exit_after = max(exit_targets) if exit_targets else None
        if reg_tokens is None and self.reg_token is not None:
            reg_tokens = self.reg_token.expand(batch_size, -1, -1)
        elif reg_tokens is not None:
            if reg_tokens.size(0) != batch_size:
                raise ValueError("Provided reg_tokens batch dimension must match inputs.")
            expected = self.main_blocks[0].attn.num_prefix_tokens
            if reg_tokens.size(1) != expected:
                raise ValueError(f"Provided reg_tokens must have {expected} tokens.")

        for idx in self._block_indices():
            block_idx = idx % self.inner_depth  # reuse inner_depth block set across repeats
            x, reg_tokens = self.main_blocks[block_idx](x, frame_indices, self.rope, self.main_block_mode, c, reg_tokens)

            if self.num_spatial_per_main > 0:
                for block in self.spatial_blocks[block_idx]:
                    x, reg_tokens = block(x, frame_indices, self.rope, "s", c, reg_tokens)

            if self.num_temporal_per_main > 0:
                for block in self.temporal_blocks[block_idx]:
                    x, reg_tokens = block(x, frame_indices, self.rope, "t", c, reg_tokens)

            if idx in return_latents:
                latents.append(x)
            if idx in return_regs: 
                regs.append(reg_tokens)
            if early_exit and exit_after is not None and idx >= exit_after:
                break

        x = x[:, -f_pred:]
        out = self.final_layer(x, c)
        out = self.postprocess_outputs(out)

        if return_latents:
            return (out, latents, regs) if return_regs else  (out, latents)
        return (out, regs) if return_regs else out 
   
    def initialize_weights(self):
        super().initialize_weights()
        if self.num_reg_tokens > 0:
            if self.reg_token is None:
                param = nn.Parameter(torch.empty(1, self.num_reg_tokens, self.hidden_size))
                self.reg_token = param
            trunc_normal_init_(self.reg_token, std=0.02)


class RecursiveDiT(FlexDiT):
    """
    FlexDiT variant that runs only a contiguous slice of blocks specified at forward time.
    Enables recursive scheduling strategies where different calls reuse shared weights.
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
        ctx_noise_aug_prob=0.5,
        drop_ctx_rate=0.2,
        main_block_type="global",
        num_spatial_per_main=1,
        num_temporal_per_main=1,
        num_reg_tokens=0,
        causal_time_attn=False,
        inner_depth=None,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_num_frames=max_num_frames,
            dropout=dropout,
            ctx_noise_aug_ratio=ctx_noise_aug_ratio,
            ctx_noise_aug_prob=ctx_noise_aug_prob,
            drop_ctx_rate=drop_ctx_rate,
            main_block_type=main_block_type,
            num_spatial_per_main=num_spatial_per_main,
            num_temporal_per_main=num_temporal_per_main,
            num_reg_tokens=num_reg_tokens,
            causal_time_attn=causal_time_attn,
            inner_depth=inner_depth,
            **kwargs,
        )
        self._default_block_range = (0, self.depth - 1)
        self._forward_block_range = None

    def _block_indices(self):
        start, end = self._forward_block_range or self._default_block_range
        return range(start, end + 1)

    def forward(
        self,
        target,
        context,
        t,
        frame_idxs = None,
        reg_tokens=None,
        return_latents=[],
        block_start=None,
        block_end=None,
        return_regs=[],
        early_exit=False,
    ):
        start = self._default_block_range[0] if block_start is None else int(block_start)
        end = self._default_block_range[1] if block_end is None else int(block_end)

        if not (0 <= start < self.depth):
            raise ValueError(f"block_start must be within [0, {self.depth - 1}]. Got {block_start}.")
        if not (0 <= end < self.depth):
            raise ValueError(f"block_end must be within [0, {self.depth - 1}]. Got {block_end}.")
        if start > end:
            raise ValueError("block_start must be <= block_end.")

        prev_range = self._forward_block_range
        self._forward_block_range = (start, end)
        try:
            return super().forward(
                target,
                context,
                t,
                frame_idxs = frame_idxs,
                reg_tokens=reg_tokens,
                return_latents=return_latents,
                return_regs=return_regs,
                early_exit=early_exit,
            )
        finally:
            self._forward_block_range = prev_range
