from __future__ import annotations
import importlib
import math
import os
from pathlib import Path
import torch
from typing import Any, Dict

def auto_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

def select_dtype(dtype_str: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_str]

def import_from_path(dotted: str):
    mod, _, attr = dotted.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid import path: {dotted}")
    m = importlib.import_module(mod)
    return getattr(m, attr)

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    if n == 0:
        return "0 B"
    i = int(math.floor(math.log(n, 1024)))
    return f"{n / (1024 ** i):.1f} {units[i]}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def env_offline(off: bool) -> None:
    if off:
        os.environ["HF_HUB_OFFLINE"] = "1"


def model_signature(import_path: str, cfg: Dict[str, Any]) -> str:
    m = cfg.get("model", {})
    layer = m.get('layer', 'last')
    if isinstance(layer, dict):
        layer = layer.get('type', 'last')
    parts = [
        f"id={m.get('model_id', 'unknown')}",
        f"rev={m.get('revision', 'none')}",
        f"layer={layer}",
    ]
    #f"{import_path}|" +
    return  "|".join(map(str, parts))