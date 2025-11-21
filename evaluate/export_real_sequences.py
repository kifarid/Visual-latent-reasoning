#!/usr/bin/env python3
import argparse
import json
import random
import os
import re
import shutil
import sys
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import torch
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import instantiate_from_config  # noqa: E402
os.environ["TK_WORK_DIR"] = "/p/scratch/nxtaim-1/farid1/orbis/logs_tk"
os.environ["WM_WORK_DIR"] = "/p/scratch/nxtaim-1/farid1/orbis/logs_wm"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def natural_key(path: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", os.path.basename(path))]


def _slugify(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "-", value)
    return cleaned.strip("-").lower()


def _collapse_repeated(value):
    if isinstance(value, list) and value:
        first = value[0]
        if all(item == first for item in value):
            return first
    return value


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def analyze_existing_sequences(frames_dir: str, expected_frames: int, split: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    real_dir = os.path.join(frames_dir, "real_images")
    if not os.path.isdir(real_dir):
        return {}, {}
    complete, incomplete = {}, {}
    prefix = f"{split}_sequence_"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(?:_.*)?$")
    for entry in os.listdir(real_dir):
        match = pattern.match(entry)
        if not match:
            continue
        idx = int(match.group(1))
        seq_path = os.path.join(real_dir, entry)
        if not os.path.isdir(seq_path):
            continue
        frame_files = [f for f in os.listdir(seq_path) if f.startswith("frame_") and f.endswith(".jpg")]
        if len(frame_files) >= expected_frames:
            complete[idx] = seq_path
        else:
            incomplete[idx] = seq_path
    return complete, incomplete


def cleanup_partial_sequences(frames_dir: str, partial: Dict[int, str]) -> None:
    if not partial:
        return
    for seq_path in partial.values():
        if os.path.isdir(seq_path):
            shutil.rmtree(seq_path, ignore_errors=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_close_dataset(dataset) -> None:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    close_fn = getattr(base, "close", None)
    if callable(close_fn):
        close_fn()


def enable_return_metadata(dataset) -> None:
    stack = [dataset]
    seen = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if hasattr(current, "return_metadata"):
            try:
                setattr(current, "return_metadata", True)
            except Exception:
                pass
        nested = getattr(current, "dataset", None)
        if nested is not None:
            stack.append(nested)
        if isinstance(current, ConcatDataset):
            stack.extend(current.datasets)


def to_serializable(value):
    if torch.is_tensor(value):
        value_cpu = value.detach().cpu()
        if value_cpu.ndim == 0:
            return value_cpu.item()
        return value_cpu.numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, Mapping):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def extract_labels_for_sample(labels_batch, sample_idx: int):
    if labels_batch is None:
        return None
    if isinstance(labels_batch, (str, bytes, bytearray)):
        return labels_batch
    if isinstance(labels_batch, Mapping):
        result = {}
        for key, value in labels_batch.items():
            try:
                value_slice = value[sample_idx]
            except Exception:
                value_slice = value
            result[key] = to_serializable(value_slice)
        return result
    if hasattr(labels_batch, "__getitem__"):
        try:
            value_slice = labels_batch[sample_idx]
        except Exception:
            value_slice = labels_batch
        return to_serializable(value_slice)
    return to_serializable(labels_batch)



def build_sequence_dir_name(split: str, dataset_idx: int, labels: Optional[Mapping]) -> str:
    base = f"{split}_sequence_{dataset_idx:04d}"
    if not labels:
        return base
    suffixes = []
    subsplit = labels.get("subsplit") if isinstance(labels, Mapping) else None
    if isinstance(subsplit, str):
        slug = _slugify(subsplit)
        if slug:
            suffixes.append(slug)
    video_name = labels.get("video_name") if isinstance(labels, Mapping) else None
    if isinstance(video_name, str):
        slug = _slugify(video_name)
        if slug:
            suffixes.append(slug)
    if suffixes:
        return f"{base}_{'_'.join(suffixes)}"
    return base


def prepare_data_config(
    config: DictConfig,
    overrides: Optional[DictConfig],
    val_config_path: Optional[str],
    split: str,
    num_gen_frames: int,
    num_pred_frames: int,
    save_real: bool,
) -> Tuple[DictConfig, int]:
    if val_config_path is not None:
        data_cfg = OmegaConf.load(val_config_path)
        if overrides is not None:
            data_cfg = OmegaConf.merge(data_cfg, overrides)
    else:
        data_cfg = clone_config(config.data)

    data_cfg = clone_config(data_cfg)
    params = data_cfg.get("params")
    if params is None or "validation" not in params:
        raise ValueError("Data config must define params.validation.")
    val_cfg = params.validation
    if isinstance(val_cfg, (list, ListConfig)):
        raise ValueError("List-valued validation configs are not supported; please consolidate before running.")
    val_params = val_cfg.get("params")
    if val_params is None:
        raise ValueError("Validation config must expose a params node.")
    if "split" in val_params:
        val_params.split = split
    else:
        raise ValueError(f"Validation params do not expose 'split'; cannot set split='{split}'.")
    if "num_frames" not in val_params:
        raise ValueError("Validation params must define num_frames.")
    num_frames_cfg = int(val_params.num_frames)
    num_condition_frames = num_frames_cfg - num_pred_frames
    if num_condition_frames <= 0:
        raise ValueError(
            f"num_frames ({num_frames_cfg}) must exceed num_pred_frames ({num_pred_frames}) for split '{split}'."
        )
    if save_real:
        val_params.num_frames = num_condition_frames + num_gen_frames
    return data_cfg, num_condition_frames


def instantiate_val_loader(data_cfg: DictConfig) -> Tuple[object, DataLoader]:
    print("Instantiating validation dataloader... with config:")
    print(OmegaConf.to_yaml(data_cfg))
    datamodule = instantiate_from_config(data_cfg)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.val_dataloader()
    if loader is None:
        raise ValueError("datamodule.val_dataloader() returned None.")
    return datamodule, loader


def build_subset_loader(loader: DataLoader, indices: Sequence[int]) -> Tuple[DataLoader, Subset]:
    subset = Subset(loader.dataset, indices)
    subset_loader = DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=getattr(loader, "collate_fn", None),
    )
    return subset_loader, subset


def export_real_split(
    config: DictConfig,
    overrides: Optional[DictConfig],
    val_config_path: Optional[str],
    split: str,
    frames_dir: str,
    args,
    num_pred_frames: int,
) -> Tuple[int, int, int]:
    ensure_dir(frames_dir)
    ensure_dir(os.path.join(frames_dir, "real_images"))

    data_cfg, num_condition_frames = prepare_data_config(
        config=config,
        overrides=overrides,
        val_config_path=val_config_path,
        split=split,
        num_gen_frames=args.num_gen_frames,
        num_pred_frames=num_pred_frames,
        save_real=True,
    )
    expected_frames = num_condition_frames + args.num_gen_frames

    complete_map, partial_map = analyze_existing_sequences(frames_dir, expected_frames, split)
    cleanup_partial_sequences(frames_dir, partial_map)

    datamodule, val_loader = instantiate_val_loader(data_cfg)
    dataset = val_loader.dataset
    enable_return_metadata(dataset)
    dataset_len = len(dataset)
    if dataset_len == 0:
        print(f"   [{split}] validation dataset is empty; skipping.")
        return 0, 0, 0

    desired_total = args.num_videos if args.num_videos and args.num_videos > 0 else dataset_len
    desired_total = min(desired_total, dataset_len)

    existing_set = {idx for idx in complete_map if idx < dataset_len}
    existing_count = min(len(existing_set), desired_total)
    if existing_count >= desired_total:
        print(f"   [{split}] already satisfied ({existing_count}/{desired_total}); skipping.")
        return 0, desired_total, existing_count

    remaining_pool = [idx for idx in range(dataset_len) if idx not in existing_set]
    if args.shuffle:
        random.shuffle(remaining_pool)

    missing_indices = remaining_pool[: desired_total - existing_count]
    if not missing_indices:
        print(f"   [{split}] no remaining sequences available; skipping.")
        return 0, desired_total, existing_count

    target_total = desired_total
    newly_exported = 0
    try:
        subset_loader, subset = build_subset_loader(val_loader, missing_indices)

        progress = tqdm(subset_loader, desc=f"{split}: {len(missing_indices)} pending", leave=False)

        with torch.no_grad():
            pointer = 0
            for batch in progress:
                if isinstance(batch, Mapping):
                    images = batch["images"]
                    labels_batch = batch.get("labels")
                    video_batch = batch.get("video_id")
                    frame_idx_batch = batch.get("frame_idx")
                else:
                    images = batch
                    labels_batch = None
                    video_batch = None
                    frame_idx_batch = None

                if not torch.is_tensor(images):
                    raise TypeError(f"Expected tensor for images, received {type(images)}.")
                batch_size = images.size(0)
                batch_indices = missing_indices[pointer : pointer + batch_size]
                pointer += batch_size

                real_batch = images.detach().cpu()

                for local_idx, dataset_idx in enumerate(batch_indices):
                    labels_dict: Optional[Dict[str, object]] = None
                    if labels_batch is not None:
                        labels_payload = extract_labels_for_sample(labels_batch, local_idx)
                        if isinstance(labels_payload, Mapping):
                            labels_dict = to_serializable(labels_payload)
                        elif labels_payload is not None:
                            labels_dict = {"value": to_serializable(labels_payload)}

                    entry_split = split
                    if labels_dict is not None:
                        split_value = labels_dict.get("split")
                        if isinstance(split_value, str) and split_value:
                            entry_split = split_value

                    dir_name = build_sequence_dir_name(entry_split, dataset_idx, labels_dict)
                    real_dir = os.path.join(frames_dir, "real_images", dir_name)
                    ensure_dir(real_dir)

                    real_clip = real_batch[local_idx]
                    for frame_idx in range(real_clip.shape[0]):
                        save_image(
                            (real_clip[frame_idx] + 1.0) / 2.0,
                            os.path.join(real_dir, f"frame_{frame_idx:04d}.jpg"),
                        )

                    real_uint8 = ((real_clip + 1.0) / 2.0).clamp(0, 1)
                    real_uint8 = (real_uint8 * 255).byte().permute(0, 2, 3, 1).numpy()
                    real_gif_path = os.path.join(real_dir, f"{dir_name}.gif")
                    imageio.mimsave(real_gif_path, list(real_uint8), fps=args.gif_fps, loop=0)

                    payload: Dict[str, object] = {"split": entry_split}

                    if labels_dict is not None:
                        labels_for_json: Optional[Dict[str, object]]
                        if args.label_keys is None:
                            labels_for_json = dict(labels_dict)
                        else:
                            keep = {str(k) for k in args.label_keys}
                            labels_for_json = {k: v for k, v in labels_dict.items() if k in keep}
                        if labels_for_json:
                            payload["labels"] = labels_for_json

                        subsplit = labels_dict.get("subsplit")
                        if isinstance(subsplit, str) and subsplit:
                            payload.setdefault("subsplit", subsplit)
                        video_name = labels_dict.get("video_name")
                        if isinstance(video_name, str) and video_name:
                            payload.setdefault("video_name", video_name)

                    if video_batch is not None:
                        video_value = to_serializable(video_batch[local_idx])
                        payload["video_id"] = _collapse_repeated(video_value)
                    if frame_idx_batch is not None:
                        frame_ids = to_serializable(frame_idx_batch[local_idx])
                        payload["frame_ids"] = frame_ids

                    if payload:
                        labels_path = os.path.join(real_dir, "labels.json")
                        with open(labels_path, "w") as f:
                            json.dump(payload, f, indent=2)

                    newly_exported += 1
                    progress.set_postfix_str(f"{existing_count + newly_exported}/{target_total}")

        maybe_close_dataset(subset)
    finally:
        maybe_close_dataset(dataset)
        teardown = getattr(datamodule, "teardown", None)
        if callable(teardown):
            teardown(stage="validate")

    return newly_exported, target_total, existing_count


def infer_num_pred_frames(config: DictConfig, fallback: Optional[int]) -> int:
    if fallback is not None:
        return fallback
    params = getattr(config.model, "params", None)
    if params is not None and hasattr(params, "num_pred_frames"):
        value = params.num_pred_frames
        if value is not None:
            return int(value)
    try:
        maybe_value = config.model.get("num_pred_frames")  # type: ignore[attr-defined]
        if maybe_value is not None:
            return int(maybe_value)
    except Exception:
        pass
    print("[warn] Could not infer num_pred_frames from config; defaulting to 0.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Export only real videos from the validation splits.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing configs and data logs.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Model config (relative to exp_dir unless absolute).")
    parser.add_argument("--val_config", type=str, default=None, help="Optional data config to override the one in config.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "intermixed"], help="Dataset splits to export.")
    parser.add_argument("--num_videos", type=int, default=100, help="Target number of videos per split (<= dataset size).")
    parser.add_argument("--num_gen_frames", type=int, default=20, help="Number of future frames to keep when saving real videos.")
    parser.add_argument("--num_pred_frames", type=int, default=None, help="Number of frames the model predicts (used to infer conditioning length).")
    parser.add_argument("--gif_fps", type=int, default=7, help="FPS for generated GIFs.")
    parser.add_argument("--frames_root", type=str, default="gen_rollout_real_only", help="Root folder (inside exp_dir) for outputs.")
    parser.add_argument("--data_tag", type=str, default=None, help="Override the data sub-folder name (default: from val_config).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; negative disables seeding.")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle dataset order before exporting.")
    parser.add_argument(
        "--label_keys",
        nargs="*",
        default=None,
        help="Optional subset of label keys to keep in the exported JSON payloads.",
    )
    args, unknown = parser.parse_known_args()

    exp_dir = os.path.abspath(args.exp_dir)
    config_path = resolve_path(args.config, exp_dir)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)

    overrides = OmegaConf.from_dotlist(unknown) if unknown else None
    if overrides is not None:
        config = OmegaConf.merge(config, overrides)

    val_config_path = resolve_path(args.val_config, exp_dir) if args.val_config else None
    if val_config_path and not os.path.isfile(val_config_path):
        raise FileNotFoundError(f"Validation config file not found: {val_config_path}")

    data_tag = args.data_tag or (os.path.splitext(os.path.basename(val_config_path))[0] if val_config_path else "default_data")

    if args.seed >= 0:
        seed_everything(args.seed)

    num_pred_frames = infer_num_pred_frames(config, args.num_pred_frames)

    splits = [s.strip() for s in args.splits if s.strip()]
    if not splits:
        raise ValueError("No dataset splits specified after parsing.")

    frames_root = os.path.join(exp_dir, args.frames_root)
    summary = []

    for split in splits:
        frames_dir = os.path.join(frames_root, split, data_tag, "real_only")
        print(f"\nExporting real videos for split '{split}' into {frames_dir}")
        exported, target_total, existing = export_real_split(
            config=config,
            overrides=overrides,
            val_config_path=val_config_path,
            split=split,
            frames_dir=frames_dir,
            args=args,
            num_pred_frames=num_pred_frames,
        )
        summary.append((split, exported, existing, target_total))

    print("\nExport summary:")
    for split, exported, existing, target in summary:
        print(f" - {split}: {existing}+{exported}/{target} sequences ready")


if __name__ == "__main__":
    main()
