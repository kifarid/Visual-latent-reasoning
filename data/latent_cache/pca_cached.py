#!/usr/bin/env python3
# viz_dino_pca.py
import argparse, os, json, sys, warnings
from pathlib import Path
import numpy as np
import torch
import h5py
from PIL import Image, ImageDraw

# --- your codebase imports ---
from data.custom_multiframe import MultiHDF5DatasetMultiFrame  # type: ignore
from data.dataset_cache_wrapper_h5 import cacheable_dataset     # type: ignore

# ---------- tiny helpers (no sklearn) ----------
def pca3_rgb(feat_chw: np.ndarray) -> np.ndarray:
    """feat_chw: (C,H,W) -> returns (H,W,3) in [0,255] uint8"""
    C, H, W = feat_chw.shape
    X = feat_chw.reshape(C, H*W).T  # (HW, C)
    X = X - X.mean(0, keepdims=True)
    # SVD for PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = (X @ Vt[:3].T).reshape(H, W, 3)
    # per-channel min-max to [0,1]
    mn = Y.min((0,1), keepdims=True); mx = Y.max((0,1), keepdims=True)
    Y = (Y - mn) / (mx - mn + 1e-8)
    return (Y * 255.0).clip(0,255).astype(np.uint8)

def pca3_basis_chunk(feats_tchw: np.ndarray):
    """
    Compute a single PCA basis over a temporal chunk for consistent colors.
    feats_tchw: (T,C,H,W) -> returns (Vt3, S3, meanC)
    """
    assert feats_tchw.ndim == 4, "expected (T,C,H,W)"
    T, C, H, W = feats_tchw.shape
    X = feats_tchw.reshape(T * H * W, C).astype(np.float32)
    meanC = X.mean(0, keepdims=False)
    Xc = X - meanC
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:3].astype(np.float32), S[:3].astype(np.float32), meanC.astype(np.float32)

def pca3_apply_with_basis(feat_chw: np.ndarray, Vt3: np.ndarray, meanC: np.ndarray) -> np.ndarray:
    C, H, W = feat_chw.shape
    X = feat_chw.reshape(C, H * W).T.astype(np.float32)
    X = X - meanC[None, :]
    Y = X @ Vt3.T
    return Y.reshape(H, W, 3)

def pca3_rgb_chunk(feats_tchw: np.ndarray, norm: str = "global"):
    """
    Apply a single chunk PCA to each frame and return uint8 RGB images.
    """
    Vt3, S3, meanC = pca3_basis_chunk(feats_tchw)
    T, C, H, W = feats_tchw.shape
    Ys = [pca3_apply_with_basis(feats_tchw[t], Vt3, meanC) for t in range(T)]
    if norm == "global":
        Ystack = np.stack(Ys, 0)
        mn = Ystack.min(axis=(0, 1, 2), keepdims=True)
        mx = Ystack.max(axis=(0, 1, 2), keepdims=True)
        Ystack = (Ystack - mn) / (mx - mn + 1e-8)
        Ys = [ (Ystack[t] * 255.0).clip(0, 255).astype(np.uint8) for t in range(T) ]
    else:
        out = []
        for Y in Ys:
            mn = Y.min((0, 1), keepdims=True); mx = Y.max((0, 1), keepdims=True)
            Yn = (Y - mn) / (mx - mn + 1e-8)
            out.append( (Yn * 255.0).clip(0, 255).astype(np.uint8) )
        Ys = out
    return Ys, (Vt3, S3, meanC)

def principal_angles_deg(Bf: np.ndarray, Bc: np.ndarray):
    M = Bf @ Bc.T
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.arccos(s) * (180.0 / np.pi)
    return np.sort(ang)[::-1]

def pca_chunk_metrics(feats_tchw: np.ndarray, Vt3_chunk: np.ndarray, S3_chunk: np.ndarray):
    T, C, H, W = feats_tchw.shape
    X = feats_tchw.reshape(T * H * W, C).astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    _, S_all, _ = np.linalg.svd(Xc, full_matrices=False)
    denom = (S_all ** 2).sum() + 1e-12
    chunk_var_ratio = ((S3_chunk ** 2) / denom).tolist()

    frame_max_angles = []
    frame_eigengap = []
    for t in range(T):
        Xt = feats_tchw[t].reshape(C, H * W).T.astype(np.float32)
        Xt = Xt - Xt.mean(0, keepdims=True)
        _, St, Vtt = np.linalg.svd(Xt, full_matrices=False)
        Bf = Vtt[:3].astype(np.float32)
        ang = principal_angles_deg(Bf, Vt3_chunk)
        frame_max_angles.append(float(ang[0]))
        gap = float((St[0] - St[1]) / max(St[0], 1e-6)) if St.size >= 2 else 0.0
        frame_eigengap.append(gap)

    return {
        "chunk_var_ratio": chunk_var_ratio,
        "frame_max_angles_deg": frame_max_angles,
        "frame_eigengap": frame_eigengap,
    }

def draw_label(img_hw3: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img_hw3)
    draw = ImageDraw.Draw(im)
    pad = 4
    w = 6 * len(text) + 2 * pad
    h = 10 + 2 * pad
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return np.asarray(im)

def to_uint8_rgb(img_t: torch.Tensor) -> np.ndarray:
    """img_t: (C,H,W) in [-1,1] or [0,1] -> (H,W,3) uint8"""
    if img_t.ndim == 3 and img_t.shape[0] in (1,3):
        x = img_t.detach().cpu().float()
        if x.min() < 0: x = (x + 1) * 0.5
        x = x.clamp(0,1)
        if x.shape[0] == 1: x = x.repeat(3,1,1)
        return (x.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    raise ValueError("Expected image tensor (C,H,W)")

def resize_np(img_hw3: np.ndarray, size_hw: tuple[int,int], nearest: bool=True) -> np.ndarray:
    pil = Image.fromarray(img_hw3)
    r = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(pil.resize((size_hw[1], size_hw[0]), r))

def hstack_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] != h: a = resize_np(a, (h, int(a.shape[1] * h / a.shape[0])))
    if b.shape[0] != h: b = resize_np(b, (h, int(b.shape[1] * h / b.shape[0])))
    return np.concatenate([a, b], axis=1)

def hstack_list(imgs: list[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        if im.shape[0] != h:
            im = resize_np(im, (h, int(im.shape[1] * h / im.shape[0])))
        out.append(im)
    return np.concatenate(out, axis=1)

def vstack_list(imgs: list[np.ndarray]) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    out = []
    for im in imgs:
        if im.shape[1] != w:
            im = resize_np(im, (int(im.shape[0] * w / im.shape[1]), w))
        out.append(im)
    return np.concatenate(out, axis=0)

# Fallback if base_out doesn't expose frames cleanly
def read_frames_from_h5(info: dict, T: int, K: int) -> list[np.ndarray]:
    frames = []
    with h5py.File(info["h5_path"], "r") as hf:
        key = info["key"]; start = info["start_frame"]
        for i in range(T):
            fidx = start + i * K
            arr = hf[key][fidx]  # (H,W,3) uint8
            frames.append(arr)
    return frames

# Try to extract frames from base_out with common patterns
def extract_frames(base_out) -> torch.Tensor | None:
    # direct tensor
    if torch.is_tensor(base_out):
        return base_out
    # tuple/list: pick first tensor
    if isinstance(base_out, (list, tuple)):
        for x in base_out:
            if torch.is_tensor(x): return x
    # dict: look for common keys
    if isinstance(base_out, dict):
        for k in ("frames", "images", "imgs", "x", "image", "img"):
            if k in base_out and torch.is_tensor(base_out[k]):
                return base_out[k]
        # search values
        for v in base_out.values():
            if torch.is_tensor(v): return v
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("PCA(3D)->RGB viz for cached DINO latents (read mode)")
    ap.add_argument("--list", default='/p/scratch/nxtaim-1/farid1/datasets/caching/h5/caching_train.txt', help="Text file with HDF5 paths (one per line)")
    ap.add_argument("--cache-root", default="/e/project1/multiscale-wm/farid1/data/cached_latents/dinov3l", help="Root of latent cache")
    ap.add_argument("--split", default="train_check")
    ap.add_argument("--outdir", default="viz_dino_pca")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--T", type=int, default=8, help="num frames (must match dataset)")
    ap.add_argument("--K", type=int, default=1, help="frame interval (must match dataset)")
    # model signature bits (defaults match typical DINOv2-giant head cache)
    ap.add_argument("--model-cfg-json", default='data/latent_cache/dinov3l.json', help="Optional JSON with model_cfg to match cache signature")
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--size", default="350,630", help="Size of output frames, e.g. 224 or H,W")
    ap.add_argument("--norm", choices=["global", "perframe"], default="global",
                    help="Normalization for PCA RGB. 'global' keeps brightness consistent across the chunk.")
    ap.add_argument("--report-metrics", action="store_true",
                    help="Print diagnostics: explained variance, max principal angles, eigengaps.")
    ap.add_argument("--annotate", action="store_true",
                    help="Overlay simple per-frame metrics on the PCA images.")
    args = ap.parse_args()
    # parse --size: "224" -> (224,224), "H,W" -> (H,W)
    _parts = args.size.replace(" ", "").split(",")
    if len(_parts) == 1:
        n = int(_parts[0]); size = (n, n)
    elif len(_parts) == 2:
        size = (int(_parts[0]), int(_parts[1]))
    else:
        raise ValueError("--size must be a single int or H,W")

    # with open(args.model_cfg_json, "r") as f:
    #     model_cfg = json.load(f)

    # wrap base dataset
    Cached = cacheable_dataset(MultiHDF5DatasetMultiFrame)
    # NOTE: adjust kwargs below if your base dataset uses different names.
    ds = Cached(
        size=size,
        hdf5_paths_file=args.list,
        num_frames=args.T,
        caching_path=args.cache_root,
        cache_mode="read",
        split=args.split,
        model_cfg_path=args.model_cfg_json,
    )

    # DataLoader: keep it simple, avoid default collate (we need the raw tuple)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=lambda x: x)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    samples_done = 0
    for batch in dl:
        for sample in batch:
            try:
                latents, base_out = sample["latents"], sample["images"]  # latents: (T,C,Hf,Wf)
            except Exception as e:
                warnings.warn(f"Unexpected sample format: {type(sample)} ({e}). Skipping.")
                continue

            idx = samples_done
            info = ds.index_of(idx)  # has h5_path, key, start_frame, etc.

            # --- get RGB frames ---
            frames_t = extract_frames(base_out)
            frames_rgb = []
            if frames_t is not None:
                # unify to (T,C,H,W)
                if frames_t.ndim == 3: frames_t = frames_t.unsqueeze(0)
                if frames_t.shape[1] not in (1,3):
                    warnings.warn("Frame tensor has unexpected channel count; will fallback to HDF5 read.")
                    frames_t = None
            if frames_t is None:
                raw = read_frames_from_h5(info, T=args.T, K=args.K)  # list of HxWx3 uint8
                for im in raw: frames_rgb.append(im)
            else:
                for t in range(min(args.T, frames_t.shape[0])):
                    frames_rgb.append(to_uint8_rgb(frames_t[t]))

            # --- PCA->RGB for each time step ---
            if isinstance(latents, torch.Tensor): latents = latents.detach().cpu().float().numpy()
            T = min(args.T, latents.shape[0], len(frames_rgb))
            rows = []
            # compute chunk PCA once
            pca_chunk_imgs, (Vt3, S3, meanC) = pca3_rgb_chunk(latents[:T], norm=args.norm)
            metrics = None
            if args.report_metrics:
                metrics = pca_chunk_metrics(latents[:T], Vt3, S3)
                vr = metrics["chunk_var_ratio"]
                print(f"[chunk PCA] var_ratio top3: {vr[0]:.3f}, {vr[1]:.3f}, {vr[2]:.3f}")
                print(f"[chunk PCA] max principal angle per frame (deg):",
                      ", ".join(f"{a:.1f}" for a in metrics["frame_max_angles_deg"]))
                print(f"[chunk PCA] per-frame eigengap (s1-s2)/s1:",
                      ", ".join(f"{g:.2f}" for g in metrics['frame_eigengap']))

            for t in range(T):
                feat = latents[t]  # (C,Hf,Wf)
                f = frames_rgb[t]

                # per-frame PCA (may drift)
                pca_frame = pca3_rgb(feat)
                pca_frame_up = resize_np(pca_frame, f.shape[:2], nearest=True)

                # chunk PCA (stable)
                pca_chunk = pca_chunk_imgs[t]
                pca_chunk_up = resize_np(pca_chunk, f.shape[:2], nearest=True)

                if args.annotate:
                    # simple eigengap for per-frame
                    C, Hf, Wf = feat.shape
                    X = feat.reshape(C, Hf * Wf).T.astype(np.float32)
                    X = X - X.mean(0, keepdims=True)
                    _, S_local, _ = np.linalg.svd(X, full_matrices=False)
                    gap = float((S_local[0] - S_local[1]) / max(S_local[0], 1e-6)) if S_local.size >= 2 else 0.0
                    pca_frame_up = draw_label(pca_frame_up, f"gap={gap:.2f}")

                    if metrics is not None:
                        ang = metrics["frame_max_angles_deg"][t]
                        pca_chunk_up = draw_label(pca_chunk_up, f"θmax={ang:.1f}°")

                row = hstack_list([f, pca_frame_up, pca_chunk_up])
                rows.append(row)

            grid = vstack_list(rows)
            name = f"idx{idx:06d}_key={Path(info['h5_path']).stem}:{info['key']}_s{info['start_frame']}.png"
            Image.fromarray(grid).save(outdir / name)

            print(f"[OK] {name}")
            samples_done += 1
            if samples_done >= args.max_samples:  # guard for typos
                break
            if samples_done >= args.max_samples:
                break
        if samples_done >= args.max_samples:
            break

    print(f"Done. Wrote {samples_done} samples to {str(outdir)}")

if __name__ == "__main__":
    main()