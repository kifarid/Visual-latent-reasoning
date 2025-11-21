# stdlib
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# third-party
import torch
import cv2
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset

from evaluate.semantic_common import (  # noqa: E402
    add_shared_arguments,
    build_cityscapes_dataset,
    encode_segmap,
    mkdir,
    str2bool,
    train_linear_probe,
)

# # project
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
from util import instantiate_from_config  # noqa: E402

# from evaluate.semantic_common import (  # noqa: E402
#     add_shared_arguments,
#     build_cityscapes_dataset,
#     encode_segmap,
#     mkdir,
#     str2bool,
#     train_linear_probe,
# )


CACHE_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_cache_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in CACHE_DTYPE_MAP:
        valid = ", ".join(sorted(CACHE_DTYPE_MAP))
        raise ValueError(f"Unsupported cache dtype '{name}'. Valid options: {valid}")
    return CACHE_DTYPE_MAP[key]


class CachedFeatureDataset(Dataset):
    """Dataset that serves cached feature tensors and encoded labels."""

    def __init__(self, files: Sequence[Path]) -> None:
        super().__init__()
        self.files: List[Path] = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = torch.load(self.files[idx])
        return item["feat"], item["label"]


@torch.inference_mode()
def prepare_feature_cache(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    rebuild: bool,
    t_noise: float = 0.0,
    frame_rate: int = 5,
    feature_depth_idx: int = -1,
) -> List[Path]:
    """Precompute and store backbone features for a dataloader split."""

    dtype_name = dtype_name.lower()
    cache_root = cache_dir / split
    cache_root.mkdir(parents=True, exist_ok=True)
    meta_file = cache_root / "_meta.json"

    feature_files = sorted(cache_root.glob("*.pt"))
    if feature_files and meta_file.exists() and not rebuild:
        print(f"[cache] Using existing '{split}' cache from {cache_root}")
        return feature_files

    if rebuild:
        print(f"[cache] Rebuilding '{split}' cache at {cache_root}")
    else:
        print(f"[cache] Creating '{split}' cache at {cache_root}")

    # Clean stale cache contents
    for path in cache_root.glob("*.pt"):
        path.unlink()
    if meta_file.exists():
        meta_file.unlink()

    saved_paths: List[Path] = []
    feature_shape: Tuple[int, ...] | None = None
    model.eval()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels_encoded = encode_segmap(labels).cpu()

        x = model.encode_frames(imgs)
        if feature_depth_idx == -1: 
            feat = x
        else:
            t = torch.full((x.shape[0],), t_noise, device=device)
            target_t, _ = model.add_noise(x, t)
            fr = torch.full((x.shape[0],), frame_rate, device=device)
            _, feats = model.vit(target_t, None, t, return_latents=[feature_depth_idx])
            feat = feats[0].squeeze(1)
            feat = rearrange(feat, "b (h w) c -> b c h w", h=18, w=32)
        
        feat = feat.detach().to(dtype=dtype).cpu()

        if feature_shape is None and feat.shape[0] > 0:
            feature_shape = tuple(feat[0].shape)

        for local_idx in range(feat.shape[0]):
            path = cache_root / f"{batch_idx:05d}_{local_idx:02d}.pt"
            torch.save({"feat": feat[local_idx], "label": labels_encoded[local_idx]}, path)
            saved_paths.append(path)

    feature_files = sorted(saved_paths)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump({"count": len(feature_files), "dtype": dtype_name, "feature_shape": list(feature_shape) if feature_shape else None}, f)

    return feature_files


# Model heads

class SegProbe(nn.Module):
    """1×1 conv classifier head for dense features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.LazyConv2d(num_classes, kernel_size=1, bias=True)

    def forward(self, feats: torch.Tensor, target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        logits_lr = self.classifier(feats)  # (B,C,h,w)
        if target_size is None:
            return logits_lr
        return F.interpolate(logits_lr, size=target_size, mode="bilinear", align_corners=False)


# Train / Eval

def calculate_semantic(args: argparse.Namespace, unknown_args: Sequence[str]) -> None:
    """Train a linear probe and optionally dump colored predictions/GT."""

    # Determinism (optional)
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    device = torch.device(args.device)

    # Load backbone from Hydra config
    cfg_model = OmegaConf.load(args.config)
    cfg_model = OmegaConf.merge(cfg_model, OmegaConf.from_dotlist(list(unknown_args)))
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    dataset_train = build_cityscapes_dataset(
        args.data_path,
        split="train",
        size=tuple(args.input_size),
        mode="fine",
        target_type="semantic",
    )
    dataset_val = build_cityscapes_dataset(
        args.data_path,
        split="val",
        size=tuple(args.input_size),
        mode="fine",
        target_type="semantic",
    )

    num_workers = 4 #,min(4, os.cpu_count() or 1)
    use_cache = bool(args.cache_dir)
    cache_dir_path = Path(args.cache_dir) if use_cache else None
    cache_dtype = resolve_cache_dtype(args.cache_dtype) if use_cache else None

    if cache_dir_path is not None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        build_loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        cache_train_loader = DataLoader(dataset_train, **build_loader_kwargs)
        cache_val_loader = DataLoader(dataset_val, **build_loader_kwargs)

        train_cache_files = prepare_feature_cache(
            model=model,
            loader=cache_train_loader,
            split="train",
            cache_dir=cache_dir_path,
            device=device,
            dtype=cache_dtype,
            dtype_name=args.cache_dtype,
            rebuild=args.rebuild_cache,
            t_noise=args.t_noise,
            frame_rate=args.frame_rate,
            feature_depth_idx=args.feature_depth_idx,
        )
        val_cache_files = prepare_feature_cache(
            model=model,
            loader=cache_val_loader,
            split="val",
            cache_dir=cache_dir_path,
            device=device,
            dtype=cache_dtype,
            dtype_name=args.cache_dtype,
            rebuild=args.rebuild_cache,
            t_noise=args.t_noise,
            frame_rate=args.frame_rate,
            feature_depth_idx=args.feature_depth_idx,
        )

        dataset_train = CachedFeatureDataset(train_cache_files)
        dataset_val = CachedFeatureDataset(val_cache_files)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    def feature_step(batch, linear_probe, train):  # noqa: ARG001
        if use_cache:
            feats, labels = batch
            feats = feats.to(
                device=device,
                dtype=linear_probe.classifier.weight.dtype,
                non_blocking=True,
            )
            labels = labels.to(device)
            logits = linear_probe(feats, target_size=labels.shape[-2:])
            return logits, labels

        imgs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = encode_segmap(labels.to(device))

        x = model.encode_frames(imgs)
        if args.feature_depth_idx == -1: 
            feat = x.squeeze(1)
        else:
            t = torch.full((x.shape[0],), args.t_noise, device=device)
            target_t, _ = model.add_noise(x, t)
            _, feats = model.vit(target_t, None, t, return_latents=[args.feature_depth_idx])
            feat = feats[0].squeeze(1)
            feat = rearrange(
                feat,
                "b (h w) c -> b c h w",
                h=model.vit.input_size[0],
                w=model.vit.input_size[1],
            ).detach()

        logits = linear_probe(feat, target_size=labels.shape[-2:])
        return logits, labels

    train_linear_probe(
        args=args,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        device=device,
        feature_step=feature_step,
    )

def parse_args(argv: Iterable[str] | None = None) -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(description="Train/eval a linear segmentation probe on Cityscapes features.")
    parser.add_argument("--t_noise", type=float, default=0.0, help="Noise timestep to use for feature extraction")
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[288, 512],
        help="Resize (H W) for Cityscapes images",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=5,
        help="Frame rate conditioning value for feature extraction",
    )
    parser.add_argument(
        "--feature_depth_idx",
        type=int,
        default=-4,
        help="Index of DiT feature depth to use for probing (negative values count from the end)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory to cache backbone features (relative to exp_dir)",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="float16",
        help="Storage dtype for cached features",
    )
    parser.add_argument(
        "--rebuild_cache",
        type=str2bool,
        default=False,
        help="Force rebuilding cached features even if present",
    )

    add_shared_arguments(
        parser,
        data_path_default="/p/scratch/nxtaim-1/farid1/datasets/Cityscapes/",
        num_epoch_default=100,
        batch_size_default=16,
        eval_every_default=5,
    )

    args, unknown = parser.parse_known_args(argv)

    # Expand checkpoint/config/frames paths relative to exp_dir
    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)
    if args.cache_dir:
        args.cache_dir = os.path.join(args.exp_dir, args.cache_dir)

    # Output dirs (created later if dump_vis)
    args.seq_fake = os.path.join(args.frames_dir, "fake_images")
    args.seq_real = os.path.join(args.frames_dir, "real_images")

    return args, unknown


def main() -> None:
    args, unknown = parse_args()

    print(">>> Checkpoint:", args.ckpt)
    print(">>> Config:    ", args.config)
    print(">>> Data root: ", args.data_path)
    if args.cache_dir:
        print(">>> Cache dir: ", args.cache_dir)
        print(">>> Cache dtype:", args.cache_dtype)
        if args.rebuild_cache:
            print("[info] Cache rebuild requested.")

    # Prepare output dirs if we're dumping
    if args.dump_vis:
        mkdir(args.seq_fake)
        mkdir(args.seq_real)
        print("[info] Visualization enabled – images may overwrite existing files.")

    calculate_semantic(args=args, unknown_args=unknown)


if __name__ == "__main__":
    main()
