from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from data.latent_cache.utils import auto_device, select_dtype


def _is_config_like(obj: Any) -> bool:
    if OmegaConf is not None and isinstance(obj, DictConfig):
        return True
    return isinstance(obj, dict)


def _to_container(cfg: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    if not path:
        raise ValueError("Empty module path")
    module: nn.Module = model
    for segment in path.split('.'):
        if hasattr(module, segment):
            module = getattr(module, segment)
            continue
        try:
            idx = int(segment)
            module = list(module.children())[idx]
        except Exception as exc:  # pragma: no cover
            raise AttributeError(f"Segment '{segment}' not found when resolving '{path}'") from exc
    return module


@dataclass
class LayerSpec:
    type: str
    spec: str
    keep_tokens: bool = True


class DistillationTeacher(nn.Module):
    """Fast feature extractor used as a distillation teacher."""

    def __init__(
        self,
        *,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        revision: str = "main",
        modality: str = "image",
        device: str = "cuda",
        dtype: str = "bf16",
        class_tokens: int = 1,
        reg_tokens: int = 4,
        patch_size: int = 16,
        shortest_edge: Optional[int] = None,
        crop_size: Optional[int] = None,
        layer: Optional[Mapping[str, Any]] = None,
        runtime: Optional[Mapping[str, Any]] = None,
        offline: Optional[bool] = None,
    ):
        super().__init__()
        if modality != "image":
            raise ValueError(f"Only 'image' modality is supported for teacher, got '{modality}'")

        def _as_dict(section: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
            if section is None:
                return {}
            if _is_config_like(section):
                return dict(_to_container(section))
            if isinstance(section, Mapping):
                return dict(section)
            raise TypeError(f"Expected mapping-like object, got {type(section).__name__}")

        layer_cfg = {"type": "head", "spec": "", "keep_tokens": True}
        layer_cfg.update(_as_dict(layer))

        runtime_cfg = _as_dict(runtime)
        offline_flag = offline if offline is not None else runtime_cfg.get("offline", False)

        if offline_flag:
            os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_id = model_id
        self.revision = revision
        self.device = auto_device(device)
        self.dtype = select_dtype(dtype)
        self.cls_tokens = int(class_tokens)
        self.reg_tokens = int(reg_tokens)
        self.patch_size = int(patch_size)
        self._size = {"shortest_edge": int(shortest_edge)} if shortest_edge else None
        self._crop_size = int(crop_size) if crop_size else None

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            size=self._size,
            do_resize=bool(self._size),
            do_center_crop=bool(self._crop_size),
            crop_size=self._crop_size,
            use_fast=True,
        )

        self.model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        self.model.eval().to(self.device, dtype=self.dtype)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.layer_spec = LayerSpec(
            type=layer_cfg.get("type", "head"),
            spec=layer_cfg.get("spec", ""),
            keep_tokens=bool(layer_cfg.get("keep_tokens", True)),
        )

        self._hook_tensor: Optional[torch.Tensor] = None
        self._register_hook_if_needed()

        self.hidden_size: Optional[int] = None

    def _register_hook_if_needed(self) -> None:
        if self.layer_spec.type != "hook":
            return
        module = _resolve_module(self.model, self.layer_spec.spec)

        def _capture(_, __, output):
            self._hook_tensor = output

        module.register_forward_hook(_capture)


    def _extract_tokens(self, outputs: Any) -> torch.Tensor:
        extra_tokens = self.cls_tokens + self.reg_tokens
        if self.layer_spec.type == "hook":
            if self._hook_tensor is None:
                raise RuntimeError("Hook did not capture any tensor")
            hidden = self._hook_tensor
        elif self.layer_spec.type == "head":
            hidden = outputs.last_hidden_state
        elif self.layer_spec.type == "pooled":
            hidden = getattr(outputs, "pooler_output", None)
            if hidden is None:
                hidden = outputs.last_hidden_state
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_spec.type}")

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)

        hidden = hidden[:, extra_tokens:]
        if hidden.dim() > 2 and not self.layer_spec.keep_tokens:
            hidden = hidden.mean(dim=1, keepdim=True)
        return hidden

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 5:
            raise ValueError(f"Teacher expects images shaped (B, T, C, H, W), got {tuple(images.shape)}")
        #convert from (-1, 1) to (0, 1)
        images = (images + 1) / 2
        b, t, c, h, w = images.shape
        device = self.device
        tensor = images
        tensor = tensor.flatten(0, 1).to(device=device, dtype=torch.float32, non_blocking=True)

        processed = self.processor(images=tensor, return_tensors="pt", do_rescale=False)
        processed = {k: v.to(device) for k, v in processed.items()}

        with torch.autocast(device_type=device.type, enabled=self.dtype in {torch.float16, torch.bfloat16}):
            outputs = self.model(**processed)
        tokens = self._extract_tokens(outputs)
        tokens = tokens.to(device=device, dtype=self.dtype)

        if tokens.dim() == 3:
            tokens = tokens  # (B*T, L, D)
        elif tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")

        seq_len = tokens.shape[1]
        tokens = tokens.view(b, t, seq_len, tokens.shape[-1])
        if self.hidden_size is None:
            self.hidden_size = int(tokens.shape[-1])
        return tokens.detach()

    def to(self, *args, **kwargs):  # type: ignore[override]
        # Teacher parameters remain frozen on their configured device.
        return self
