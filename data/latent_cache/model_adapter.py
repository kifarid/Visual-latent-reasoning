from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoImageProcessor#, AutoVideoProcessor
from .utils import auto_device, select_dtype
from .logging_utils import log

class HookCapture:
    def __init__(self):
        self.last: Optional[torch.Tensor] = None
    def __call__(self, module, inp, out):
        self.last = out

def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    parts = path.split(".")
    m = model
    for p in parts:
        if hasattr(m, p):
            m = getattr(m, p)
        else:
            try:
                idx = int(p)
                m = list(m.children())[idx]
            except Exception as e:
                raise AttributeError(f"Path segment '{p}' not found under {m._get_name()}") from e
    return m

@dataclass
class LayerSpec:
    type: str  # hook | head | pooled
    spec: str
    keep_tokens: bool = True


class ModelAdapter:
    def __init__(self, cfg: Any):
        model_id = cfg.get("model", "model_id")
        revision = cfg.get("model", "revision")
        se = cfg.get("model", "shortest_edge")
        self.size = {"shortest_edge": int(se)} if se else None
        cs = cfg.get("model", "crop_size")
        self.cls_toks = cfg.get("model", "class_toks", default=1)
        self.reg_toks = cfg.get("model", "reg_toks", default=0)
        self.crop_size = int(cs) if cs else None
        self.patch_size = cfg.get("model", "patch_size", default=14)
        self.modality = cfg.get("model", "modality")
        self.device = auto_device(cfg.get("model", "device", default="auto"))
        self.dtype = select_dtype(cfg.get("model", "dtype", default="fp32"))
        off = bool(cfg.get("runtime", "offline", default=False))
        if off:
            import os; os.environ["HF_HUB_OFFLINE"] = "1"

        log(f"Loading model {model_id} (rev={revision}) on {self.device} with dtype={self.dtype}")
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, 
                                                            size=self.size,
                                                            do_resize=bool(self.size),
                                                            do_center_crop=bool(self.crop_size),
                                                            crop_size=self.crop_size,
                                                            use_fast=True
                                                            ) if self.modality == "image" else AutoVideoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision)
        self.model.eval().to(self.device, dtype=self.dtype)

        # --- in __init__: pass keep_tokens from cfg (default False preserves current behavior) ---
        ls = LayerSpec(
            type=cfg.get("model", "layer", "type"),
            spec=cfg.get("model", "layer", "spec"),
            keep_tokens=bool(cfg.get("model", "layer", "keep_tokens", default=False)),
        )
        self.layer_spec = ls
        self.hook: Optional[HookCapture] = None
        if ls.type == "hook":
            try:
                module = resolve_module(self.model, ls.spec)
            except Exception:
                names = [name for name, _ in self.model.named_modules()]
                raise ValueError(f"Layer path '{ls.spec}' not found. Example names: {names[:20]} ...")
            self.hook = HookCapture()
            module.register_forward_hook(self.hook)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        '''
        images (B,C,H,W) or (T,C,H,W) or (B,T,C,H,W) in range [-1,1] - > (B, L, D)
        '''
        img_shp = images.shape
        vid = False
        if images.dim() > 4:
            images = images.flatten(0, 1)
            vid = True

        images = (images + 1) / 2
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device.type, enabled=self.dtype in (torch.float16, torch.bfloat16)):
            out = self.model(**inputs)
        emb = self._extract(out)

        emb = emb.permute(0, 2, 1)  # [B,D,L] or [T,D] if B=1
        emb = emb.reshape(emb.shape[0], -1, img_shp[-2] // self.patch_size, img_shp[-1] // self.patch_size)
        
        if vid:
            emb = emb.reshape(img_shp[0], img_shp[1], *emb.shape[1:])  # [B,,D] or [T,D] if B=1
        
        return emb, {"hidden_size": emb.shape[-1]}

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        videos = (videos + 1) / 2 
        batch_list = []
        for vid in videos:  # [B,T,C,H,W]
            frames = [f for f in vid]
            batch_list.append(frames)
        inputs = self.processor(videos=batch_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device.type, enabled=self.dtype in (torch.float16, torch.bfloat16)):
            out = self.model(**inputs)
        emb = self._extract(out)
        return emb, {"hidden_size": emb.shape[-1]}

    def _extract(self, out) -> torch.Tensor:
        extra_toks = self.cls_toks + self.reg_toks
        if self.layer_spec.type == "hook":
            assert self.hook and self.hook.last is not None, "Hook did not capture output"
            x = self.hook.last[:,extra_toks:]
        elif self.layer_spec.type == "head":
            print(out.last_hidden_state.shape)
            x = out.last_hidden_state[:, extra_toks:]  # [B,D] or [T,D] if B=1
        else:  # pooled
            x = getattr(out, "pooler_output", None)
            if x is None:
                x = out.last_hidden_state[:, 0]

        # NEW: only pool if user didn't request tokens-as-is
        if x.dim() > 2 and not self.layer_spec.keep_tokens:
            x = x.mean(dim=tuple(range(1, x.dim()-1)))

        return x.detach().to("cpu", dtype=torch.float16)
