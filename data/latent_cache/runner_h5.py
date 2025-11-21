#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, hashlib
from pathlib import Path
from typing import Any, Dict, List

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py
import numpy as np
from filelock import FileLock

from data.latent_cache.utils import model_signature

# Logging (uses your existing helper if present)
try:
    from latent_cache.logging_utils import log as _log, log_jsonl as _log_jsonl  # type: ignore
except Exception:
    def _log(msg: str) -> None: print(msg)
    def _log_jsonl(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f: f.write(json.dumps({"ts": None, **record}) + "\n")

def sha1_bytes_from_tensor(x: torch.Tensor) -> str:
    arr = x.detach().cpu()
    try: b = arr.numpy().tobytes(order="C")
    except Exception: b = arr.contiguous().cpu().numpy().tobytes()
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def _written_key(key: str) -> str: return f"{key}_written"

def _ensure_latent_dataset(hf: h5py.File, key: str, N: int, C: int, H: int, W: int, np_dtype: np.dtype) -> h5py.Dataset:
    if key in hf:
        ds = hf[key]
        ok = (tuple(ds.shape) == (N, C, H, W)) and (ds.dtype == np_dtype)
        if not ok:
            del hf[key]
            ds = hf.create_dataset(key, shape=(N, C, H, W), dtype=np_dtype, chunks=(1, C, H, W), compression="lzf")
        return ds
    return hf.create_dataset(key, shape=(N, C, H, W), dtype=np_dtype, chunks=(1, C, H, W), compression="lzf")

def _mark_written(hf: h5py.File, key: str, start: int, T: int) -> None:
    wkey = _written_key(key)
    N = int(hf[key].shape[0])
    chunk_len = int(min(max(1, N), 1024))
    if wkey in hf:
        wds = hf[wkey]
        if (wds.dtype != np.bool_) or (tuple(wds.shape) != (N,)):
            del hf[wkey]
            wds = hf.create_dataset(wkey, shape=(N,), dtype=np.bool_, chunks=(chunk_len,), compression="lzf")
    else:
        wds = hf.create_dataset(wkey, shape=(N,), dtype=np.bool_, chunks=(chunk_len,), compression="lzf")
    wds[start:start+T] = True

def write_latents_h5(latent_h5_path: Path, key: str, start: int, tensor: torch.Tensor,
                     full_len: int, c: int, h: int, w: int, safe: bool=True) -> None:
    np_dtype = np.float16 if tensor.dtype == torch.float16 else np.float32
    lock = FileLock(str(latent_h5_path) + ".lock")
    with lock:
        with h5py.File(latent_h5_path, "a", libver="latest") as hf:
            if not hf.attrs.get("cache_target"):
                hf.attrs["cache_target"] = "latents"
            ds = _ensure_latent_dataset(hf, key, full_len, c, h, w, np_dtype)
            ds[start:start+tensor.shape[0]] = tensor.detach().cpu().numpy().astype(np_dtype, copy=False)
            _mark_written(hf, key, start, int(tensor.shape[0]))
            hf.flush()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batched latent cache builder (H5 latents, runner-side adapter)")
    # dataset / cache config
    p.add_argument("--hdf5-paths-file", type=str, required=True)
    p.add_argument("--size", type=str, required=True, help="Output size: single int for square (e.g., '224') or H,W (e.g., '360,640').")
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--caching-path", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-safe-writes", action="store_true")
    # model adapter
    p.add_argument("--adapter-import-path", type=str, required=True)
    p.add_argument("--adapter-cfg-json", type=str, required=True)
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--modality", type=str, choices=("image", "video"), default=None)
    return p

def build_adapter(import_path: str, cfg: Dict[str, Any]):
    mod_path, _, cls_name = import_path.rpartition(".")
    if not mod_path: raise ValueError(f"Invalid adapter_import_path: {import_path}")
    module = importlib.import_module(mod_path)
    AdapterCls = getattr(module, cls_name)
    class _Shim:
        def __init__(self, raw: Dict[str, Any]) -> None: self.raw = raw
        def get(self, *keys, default=None):
            d: Any = self.raw
            for k in keys:
                if isinstance(d, dict) and k in d: d = d[k]
                else: return default
            return d
    return AdapterCls(_Shim(cfg))

def infer_modality(cfg: Dict[str, Any], override: str | None) -> str:
    if override: return override
    return cfg.get("model", {}).get("modality", "image")


def main(args: argparse.Namespace) -> None:
    from data.dataset_cache_wrapper_h5 import cacheable_dataset  # new wrapper

    with open(args.adapter_cfg_json, "r") as f:
        adapter_cfg = json.load(f)

    # Base dataset
    from data.custom_multiframe import MultiHDF5DatasetMultiFrame  # type: ignore
    Cached = cacheable_dataset(MultiHDF5DatasetMultiFrame)

    # Build adapter & signature ONCE
    adapter = build_adapter(args.adapter_import_path, adapter_cfg)
    sig = model_signature(args.adapter_import_path, adapter_cfg)

    # parse --size: "224" -> (224,224), "H,W" -> (H,W)
    _parts = args.size.replace(" ", "").split(",")
    if len(_parts) == 1:
        n = int(_parts[0]); size = (n, n)
    elif len(_parts) == 2:
        size = (int(_parts[0]), int(_parts[1]))
    else:
        raise ValueError("--size must be a single int or H,W")

    # Construct dataset (build mode) — dataset does NOT encode
    ds = Cached(
        size=size,
        hdf5_paths_file=args.hdf5_paths_file,
        num_frames=args.num_frames,
        cache_mode="build",
        caching_path=args.caching_path,
        split=args.split,
        model_cfg=adapter_cfg,
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        drop_last=False, collate_fn=lambda b: b,
    )

    modality = infer_modality(adapter_cfg, args.modality)
    encode_bs = max(1, int(args.encode_batch_size))
    safe_writes = not bool(args.no_safe_writes)

    # Manifest path (reuse dataset’s if any)
    manifest_path: Path | None = getattr(ds, "_manifest_path", None)
    if manifest_path is None:
        manifest_path = Path(args.caching_path) / args.split / "index_manifest.jsonl"

    # timing
    encode_total_time = 0.0
    encode_total_calls = 0

    total = len(ds); cached_total = 0; written_total = 0; failed_total = 0

    pbar = tqdm(loader, total=(total + args.batch_size - 1)//args.batch_size, desc="Caching latents (H5)")
    for batch in pbar:
        try:
            todo = [it for it in batch if not it.get("cached", False)]
            cached_total += (len(batch) - len(todo))
            if not todo:
                avg = (encode_total_time / encode_total_calls) if encode_total_calls else 0.0
                pbar.set_postfix(cached=cached_total, written=written_total, failed=failed_total, enc_avg_s=round(avg, 4))
                continue

            frames_list: List[torch.Tensor] = [it["frames"] for it in todo]  # [T,C,H,W]
            lengths: List[int] = [int(fr.shape[0]) for fr in frames_list]

            # Encode
            if modality == "image":
                imgs = torch.cat(frames_list, dim=0)  # [sum_T,C,H,W]
                outs: List[torch.Tensor] = []
                for s in range(0, imgs.shape[0], encode_bs):
                    _sl = imgs[s:s+encode_bs]
                    t0 = time.perf_counter()
                    e, _ = adapter.encode_image(_sl)
                    encode_total_time += (time.perf_counter() - t0)
                    encode_total_calls += 1
                    outs.append(e.detach().cpu())
                emb_all = torch.cat(outs, dim=0)
                per_item: List[torch.Tensor] = []
                off = 0
                for L in lengths:
                    per_item.append(emb_all[off:off+L]); off += L
            else:
                vids = torch.stack(frames_list, dim=0)  # [B,T,C,H,W]
                outs: List[torch.Tensor] = []
                for s in range(0, vids.shape[0], encode_bs):
                    _sl = vids[s:s+encode_bs]
                    t0 = time.perf_counter()
                    e, _ = adapter.encode_video(_sl)
                    encode_total_time += (time.perf_counter() - t0)
                    encode_total_calls += 1
                    outs.append(e.detach().cpu())
                per_item = list(torch.cat(outs, dim=0))  # [B,...] -> list

            # Write + manifest
            for lat, it in zip(per_item, todo):
                try:
                    start, stop = it["latent_slice"]; C, H_e, W_e = map(int, lat.shape[1:4])
                    N = int(it["identifiers"].get("video_length", start + lat.shape[0]))
                    write_latents_h5(Path(it["latent_h5_path"]), it["latent_key"], start, lat
                                     , N, C, H_e, W_e, safe=safe_writes)
                    rec = {
                        "idx": it["idx"],
                        "latent_h5_path": it["latent_h5_path"],
                        "latent_key": it["latent_key"],
                        "slice": [start, stop],
                        "shape": list(lat.shape),
                        "dtype": str(lat.dtype).replace("torch.", ""),
                        "checksum": sha1_bytes_from_tensor(lat),
                        "cache_target": "latents",
                        "adapter_import_path": args.adapter_import_path,
                        "model_signature": sig,
                    }
                    _log_jsonl(manifest_path, rec)
                    written_total += 1
                except Exception as e:
                    failed_total += 1
                    _log(f"[cache] write failed for {it['latent_h5_path']}:{it['latent_key']} @ {start}:{stop} → {e}")

        except Exception as e:
            failed_total += len(batch)
            _log(f"[cache] batch failed: {e}")

        avg = (encode_total_time / encode_total_calls) if encode_total_calls else 0.0
        pbar.set_postfix(cached=cached_total, written=written_total, failed=failed_total, enc_avg_s=round(avg, 4))

    _log(f"[cache] Done. total={total}, cached={cached_total}, newly_written={written_total}, failed={failed_total}")

if __name__ == "__main__":
    main(build_argparser().parse_args())