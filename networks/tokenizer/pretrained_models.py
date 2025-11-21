import torch
import timm
from torch import nn
from omegaconf import ListConfig
from einops import rearrange


from typing import Tuple, Union

class Encoder(nn.Module):
    def __init__(
        self, 
        resolution: Union[Tuple[int, int], int], 
        channels: int = 3, 
        pretrained_encoder = 'MAE',
        patch_size: int = 16,
        z_channels: int = 768,
        e_dim: int = 8,
        normalize_embedding: bool = True,
        use_pretrained_weights: bool = True
        # **ignore_kwargs
    ) -> None:
        # Initialize parent class with the first patch size
        super().__init__()
        self.image_size = resolution
        self.patch_size = patch_size
        self.channels = channels
        self.normalize_embedding = normalize_embedding
        self.z_channels = z_channels
        self.e_dim = e_dim
        self.use_pretrained_weights = use_pretrained_weights
        self.pretrained_encoder = pretrained_encoder
        
        self.init_transformer(pretrained_encoder)

    def init_transformer(self, pretrained_encoder):
        if pretrained_encoder == 'VIT_DINO':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_encoder == 'VIT_DINOv3':
            pretrained_encoder_model = 'timm/vit_base_patch16_dinov3_qkvb.lvd1689m'
        elif pretrained_encoder == 'VIT_DINOv2':
            pretrained_encoder_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_encoder == 'MAE':
            pretrained_encoder_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_encoder == 'MAE_VIT_L':
            pretrained_encoder_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_encoder == 'VIT':
            pretrained_encoder_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_encoder == 'CLIP32':
            pretrained_encoder_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_encoder == 'CLIP':
            pretrained_encoder_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_encoder == 'base':
            pretrained_encoder_model = 'timm/vit_base_patch16_224'
        elif pretrained_encoder == 'large':
            pretrained_encoder_model = 'timm/vit_large_patch16_224'
       

        self.encoder = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=False, dynamic_img_size=True).train()

        if self.use_pretrained_weights:
            pretrained_model = timm.create_model(pretrained_encoder_model, img_size=self.image_size, patch_size=self.patch_size, pretrained=True)
            """Initialize weights of target_model with weights from source_model."""

            with torch.no_grad():
                for target_param, source_param in zip(self.encoder.parameters(), pretrained_model.parameters()):
                    target_param.data.copy_(source_param.data)

            # Clean up
            del pretrained_model
    
    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        if self.pretrained_encoder ==  'VIT_DINOv3':
            h = self.encoder.forward_features(img)[:,5:]
        else:
            h = self.encoder.forward_features(img)[:,1:]
        h = h.permute(0, 2, 1).contiguous()
        h = h.reshape(h.shape[0], -1, img.size(2)//self.patch_size, img.size(3)//self.patch_size)
        return h


class _PassthroughPatchEmbed(nn.Module):
    """Minimal patch embed replacement that forwards latent tokens."""

    def __init__(self, embed_dim: int, patch_size: Union[int, Tuple[int, int]], grid_size: Tuple[int, int]) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim != 4:
            raise ValueError(f'Expected tensor with 4 dims (B, C, H, W); got {x.shape}')
        b, c, h, w = x.shape
        if c != self.embed_dim:
            raise ValueError(f'Channel mismatch. Expected {self.embed_dim}, got {c}')
        if (h, w) != tuple(self.grid_size):
            self.grid_size = (h, w)
            self.num_patches = h * w
        return x.permute(0, 2, 3, 1).contiguous()


class Decoder(nn.Module):
    def __init__(
        self,
        resolution: Union[Tuple[int, int], int],
        channels: int = 3,
        pretrained_decoder: str = 'MAE',
        patch_size: int = 16,
        use_pretrained_weights: bool = True,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        if ignore_kwargs:
            ignore_kwargs.clear()
        if isinstance(resolution, int):
            self.image_size = (resolution, resolution)
        else:
            self.image_size = resolution
        self.pretrained_decoder = pretrained_decoder
        self.channels = channels
        self.patch_size = patch_size
        self.use_pretrained_weights = use_pretrained_weights

        self.init_transformer(pretrained_decoder)
        self.embed_dim = getattr(self.decoder, 'embed_dim', None)
        if self.embed_dim is None:
            raise AttributeError('Decoder backbone must expose embed_dim attribute.')

        patch_h, patch_w = self._patch_size
        self.patch_area = patch_h * patch_w
        self.conv_out = nn.Linear(self.embed_dim, self.patch_area * self.channels)

    def init_transformer(self, pretrained_decoder: str) -> None:
        if pretrained_decoder == 'VIT_DINO':
            pretrained_model = 'timm/vit_base_patch16_224.dino'
        elif pretrained_decoder == 'VIT_DINOv3':
            pretrained_model = 'timm/vit_base_patch16_dinov3_qkvb.lvd1689m'
        elif pretrained_decoder == 'VIT_DINOv2':
            pretrained_model = 'timm/vit_base_patch14_dinov2.lvd142m'
        elif pretrained_decoder == 'MAE':
            pretrained_model = 'timm/vit_base_patch16_224.mae'
        elif pretrained_decoder == 'MAE_VIT_L':
            pretrained_model = 'timm/vit_large_patch16_224.mae'
        elif pretrained_decoder == 'VIT':
            pretrained_model = 'timm/vit_large_patch32_224.orig_in21k'
        elif pretrained_decoder == 'CLIP32':
            pretrained_model = 'timm/vit_base_patch32_clip_224.openai'
        elif pretrained_decoder == 'CLIP':
            pretrained_model = 'timm/vit_base_patch16_clip_224.openai'
        elif pretrained_decoder == 'base':
            pretrained_model = 'timm/vit_base_patch16_224'
        elif pretrained_decoder == 'large':
            pretrained_model = 'timm/vit_large_patch16_224'
        else:
            raise ValueError(f'Unsupported decoder backbone: {pretrained_decoder}')

        decoder = timm.create_model(
            pretrained_model,
            img_size=self.image_size,
            patch_size=self.patch_size,
            pretrained=False,
            dynamic_img_size=True,
        ).train()

        if self.use_pretrained_weights:
            pretrained = timm.create_model(
                pretrained_model,
                img_size=self.image_size,
                patch_size=self.patch_size,
                pretrained=True,
            )
            with torch.no_grad():
                for target_param, source_param in zip(decoder.parameters(), pretrained.parameters()):
                    target_param.data.copy_(source_param.data)
            del pretrained

        original_patch_embed = decoder.patch_embed
        self._patch_size = original_patch_embed.patch_size if hasattr(original_patch_embed, 'patch_size') else (
            self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        )
        grid_size = getattr(
            original_patch_embed,
            'grid_size',
            (self.image_size[0] // self._patch_size[0], self.image_size[1] // self._patch_size[1]),
        )
        decoder.patch_embed = _PassthroughPatchEmbed(decoder.embed_dim, self._patch_size, grid_size)
        self.original_patch_embed = original_patch_embed

        self.decoder = decoder

    def unpatchify(self, tokens: torch.FloatTensor) -> torch.FloatTensor:
        grid_h, grid_w = self.decoder.patch_embed.grid_size
        patch_h, patch_w = self._patch_size
        return rearrange(
            tokens,
            'b (h w) (ph pw c) -> b c (h ph) (w pw)',
            h=grid_h,
            w=grid_w,
            ph=patch_h,
            pw=patch_w,
            c=self.channels,
        )

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if z.ndim != 4:
            raise ValueError(f'Expected latent tensor with shape (B, C, H, W); got {z.shape}')
        if z.shape[1] != self.embed_dim:
            raise ValueError(f'Latent channel dimension {z.shape[1]} does not match decoder embed dim {self.embed_dim}.')
        tokens = self.decoder.forward_features(z)
        spatial_tokens = tokens[:, self.decoder.num_prefix_tokens:]
        spatial_tokens = self.conv_out(spatial_tokens)
        decoded = self.unpatchify(spatial_tokens).contiguous()
        return decoded
