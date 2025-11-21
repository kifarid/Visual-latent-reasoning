#!/usr/bin/env python3
import argparse
import glob
import inspect
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch
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


def _collapse_repeated(value):
    if isinstance(value, list) and value:
        first = value[0]
        if all(item == first for item in value):
            return first
    return value


def slugify_name(name: str) -> str:
    if not name:
        return "sequence"
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "sequence"


def flatten_frame_ids(frame_ids) -> List[int]:
    values: List[int] = []
    stack = [frame_ids]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if isinstance(current, (np.integer, int)):
            values.append(int(current))
        elif isinstance(current, float) and current.is_integer():
            values.append(int(current))
    return values


def compute_frame_range(frame_ids) -> Optional[Tuple[int, int]]:
    if frame_ids is None:
        return None
    if not isinstance(frame_ids, (list, tuple)):
        try:
            value = int(frame_ids)
        except (TypeError, ValueError):
            return None
        return value, value
    values = flatten_frame_ids(frame_ids)
    if not values:
        return None
    return min(values), max(values)


def build_sequence_name(
    dataset_idx: int,
    *,
    video_name: Optional[str] = None,
    frame_range: Optional[Tuple[int, int]] = None,
) -> str:
    parts = [slugify_name(video_name)]
    if frame_range is not None:
        parts.append(f"{frame_range[0]:04d}-{frame_range[1]:04d}")
    parts.append(f"{dataset_idx:04d}")
    return "_".join(parts)


def extract_dataset_index(sequence_dir: str) -> Optional[int]:
    name = os.path.basename(sequence_dir)
    match = re.search(r"_(\d+)$", name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    labels_path = os.path.join(sequence_dir, "labels.json")
    if os.path.isfile(labels_path):
        try:
            with open(labels_path, "r") as f:
                payload = json.load(f)
            idx = payload.get("dataset_index")
            if isinstance(idx, int):
                return idx
        except Exception:
            pass
    return None


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def analyze_existing_sequences(
    split_dir: str,
    expected_frames: int,
) -> Tuple[Dict[int, Dict[str, object]], Dict[int, str], Dict[str, int]]:
    if not os.path.isdir(split_dir):
        return {}, {}, {}
    complete: Dict[int, Dict[str, object]] = {}
    incomplete: Dict[int, str] = {}
    subsplit_counts: Dict[str, int] = {}
    for subsplit in os.listdir(split_dir):
        subsplit_path = os.path.join(split_dir, subsplit)
        if not os.path.isdir(subsplit_path):
            continue
        for entry in os.listdir(subsplit_path):
            seq_path = os.path.join(subsplit_path, entry)
            if not os.path.isdir(seq_path):
                continue
            dataset_idx = extract_dataset_index(seq_path)
            if dataset_idx is None:
                continue
                gen_dir = os.path.join(seq_path, "gen")
                frame_files = (
                    [f for f in os.listdir(gen_dir) if f.startswith("frame_") and f.endswith(".jpg")]
                    if os.path.isdir(gen_dir)
                    else []
                )
                if len(frame_files) >= expected_frames:
                    complete[dataset_idx] = {"path": seq_path, "subsplit": subsplit}
                    subsplit_counts[subsplit] = subsplit_counts.get(subsplit, 0) + 1
                else:
                    incomplete[dataset_idx] = seq_path
    return complete, incomplete, subsplit_counts


def cleanup_partial_sequences(
    split_dir: str,
    partial: Dict[int, str],
) -> None:
    if not partial:
        return
    for seq_path in partial.values():
        if os.path.isdir(seq_path):
            shutil.rmtree(seq_path, ignore_errors=True)
            parent = os.path.dirname(seq_path)
            try:
                if not os.listdir(parent):
                    os.rmdir(parent)
            except Exception:
                pass


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
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
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


def prepare_data_config(
    config: DictConfig,
    overrides: Optional[DictConfig],
    val_config_path: Optional[str],
    split: str,
    num_gen_frames: int,
    num_pred_frames: int,
    save_real: bool,
    batch_size: Optional[int] = None,
    dataset_seed: Optional[int] = None,
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
            f"num_frames ({num_frames_cfg}) must exceed model.num_pred_frames ({num_pred_frames}) for split '{split}'."
        )
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        params.batch_size = batch_size
        params.val_batch_size = batch_size
    if save_real:
        val_params.num_frames = num_condition_frames + num_gen_frames
    if dataset_seed is not None:
        try:
            val_params.rng_seed = int(dataset_seed)
        except (TypeError, ValueError):
            pass
    return data_cfg, num_condition_frames


def instantiate_val_loader(data_cfg: DictConfig) -> Tuple[object, DataLoader]:
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
        num_workers=0,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=getattr(loader, "collate_fn", None),
    )
    return subset_loader, subset


def rollout_split(
    model,
    config: DictConfig,
    overrides: Optional[DictConfig],
    val_config_path: Optional[str],
    split: str,
    output_root: str,
    args,
    device: torch.device,
) -> Tuple[int, int, int]:
    ensure_dir(output_root)
    split_dir = os.path.join(output_root, split)
    ensure_dir(split_dir)

    complete_map, partial_map, subsplit_existing_counts = analyze_existing_sequences(
        split_dir, args.num_gen_frames
    )
    cleanup_partial_sequences(split_dir, partial_map)

    if getattr(args, "seed", None) is not None and args.seed >= 0:
        seed_everything(args.seed, workers=True)

    data_cfg, num_condition_frames = prepare_data_config(
        config=config,
        overrides=overrides,
        val_config_path=val_config_path,
        split=split,
        num_gen_frames=args.num_gen_frames,
        num_pred_frames=getattr(model, "num_pred_frames", None) or 0,
        save_real=args.save_real,
        batch_size=args.batch_size,
        dataset_seed=args.seed if args.seed >= 0 else None,
    )

    datamodule, val_loader = instantiate_val_loader(data_cfg)
    dataset = val_loader.dataset
    enable_return_metadata(dataset)
    dataset_len = len(dataset)

    per_subsplit_limit = args.num_videos if args.num_videos and args.num_videos > 0 else None

    complete_indices = set(complete_map.keys())
    missing_indices = [idx for idx in range(dataset_len) if idx not in complete_indices]

    initial_existing_total = len(complete_indices)

    target_counts: Dict[str, int] = {}
    generated_by_subsplit: Dict[str, int] = {}

    if per_subsplit_limit is None:
        target_total = dataset_len
    else:
        subsplit_names = getattr(dataset, "available_subsplits", None)
        if subsplit_names:
            for name in subsplit_names:
                target_counts[name] = per_subsplit_limit
        for subsplit_name, count in subsplit_existing_counts.items():
            target_counts.setdefault(subsplit_name, per_subsplit_limit)
        target_total = sum(target_counts.values()) if target_counts else initial_existing_total

    def subsplit_total(subsplit_name: str) -> int:
        existing = subsplit_existing_counts.get(subsplit_name, 0)
        generated = generated_by_subsplit.get(subsplit_name, 0)
        return existing + generated

    def subsplit_has_capacity(subsplit_name: str) -> bool:
        if per_subsplit_limit is None:
            return True
        target = target_counts.setdefault(subsplit_name, per_subsplit_limit)
        return subsplit_total(subsplit_name) < target

    def all_targets_satisfied() -> bool:
        if per_subsplit_limit is None:
            return False
        if not target_counts:
            return False
        return all(subsplit_total(name) >= target for name, target in target_counts.items())

    newly_generated = 0
    completed_targets = False
    try:
        if dataset_len == 0:
            print(f"   [{split}] validation dataset is empty; skipping.")
            return newly_generated, target_total, initial_existing_total
        if not missing_indices:
            if per_subsplit_limit is None:
                print(
                    f"   [{split}] already satisfied ({initial_existing_total}/{target_total}); skipping."
                )
                return newly_generated, target_total, initial_existing_total
            existing_targets_met = False
            if target_counts:
                existing_targets_met = all(
                    subsplit_existing_counts.get(name, 0) >= target for name, target in target_counts.items()
                )
            final_target_total = sum(target_counts.values()) if target_counts else initial_existing_total
            if existing_targets_met:
                print(
                    f"   [{split}] already satisfied ({initial_existing_total}/{final_target_total}); skipping."
                )
            else:
                print(
                    f"   [{split}] no remaining dataset indices to sample; generated {initial_existing_total}/{final_target_total}."
                )
            return newly_generated, final_target_total, initial_existing_total

        subset_loader, subset = build_subset_loader(val_loader, missing_indices)
        if model is None:
            raise ValueError("Model must be provided for rollout.")
        rollout_params = inspect.signature(model.roll_out).parameters
        supports_raymaps = "raymaps" in rollout_params
        encdec_supported = "encdec_batch_size" in rollout_params
        if args.encdec_batch_size is not None and not encdec_supported:
            print(f"   [{split}] encdec_batch_size requested but model.roll_out has no such argument; ignoring.")
        pointer = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        progress = tqdm(subset_loader, desc=f"{split}: {len(missing_indices)} pending", leave=False)

        with torch.no_grad():
            for batch in progress:
                labels_batch = None
                video_batch = None
                frame_idx_batch = None
                if isinstance(batch, Mapping):
                    images = batch["images"]
                    raymaps = batch.get("raymaps")
                    labels_batch = batch.get("labels")
                    video_batch = batch.get("video_id")
                    frame_idx_batch = batch.get("frame_idx")
                else:
                    images = batch
                    raymaps = None

                if not torch.is_tensor(images):
                    raise TypeError(f"Expected tensor for images, received {type(images)}.")
                batch_size = images.size(0)
                batch_indices = missing_indices[pointer : pointer + batch_size]
                pointer += batch_size

                images = images.to(device, non_blocking=True)
                if raymaps is not None and hasattr(raymaps, "to"):
                    raymaps = raymaps.to(device, non_blocking=True)

                cond_x = images[:, :num_condition_frames]
                autocast_cm = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_cm:
                    rollout_kwargs = dict(
                        x_0=cond_x,
                        num_gen_frames=args.num_gen_frames,
                        latent_input=False,
                        eta=args.eta,
                        NFE=args.num_steps,
                        sample_with_ema=args.evaluate_ema,
                        num_samples=cond_x.size(0),
                    )
                    if supports_raymaps and raymaps is not None:
                        rollout_kwargs["raymaps"] = raymaps
                    if args.encdec_batch_size is not None and encdec_supported:
                        rollout_kwargs["encdec_batch_size"] = args.encdec_batch_size
                    _, gen_frames = model.roll_out(**rollout_kwargs)

                gen_frames = gen_frames.detach().cpu()
                real_batch = images.detach().cpu() if args.save_real else None

                for local_idx, dataset_idx in enumerate(batch_indices):
                    labels_dict: Optional[Dict[str, object]] = None
                    if labels_batch is not None:
                        labels_payload = extract_labels_for_sample(labels_batch, local_idx)
                        if isinstance(labels_payload, Mapping):
                            labels_dict = dict(labels_payload)
                        elif labels_payload is not None:
                            labels_dict = {"value": labels_payload}

                    entry_split = split
                    if labels_dict is not None:
                        split_value = labels_dict.get("split")
                        if isinstance(split_value, str) and split_value:
                            entry_split = split_value

                    payload: Dict[str, object] = {"split": entry_split, "dataset_index": dataset_idx}

                    subsplit_name: Optional[str] = None
                    video_name: Optional[str] = None

                    if labels_dict is not None:
                        payload["labels"] = labels_dict
                        subsplit = labels_dict.get("subsplit")
                        if isinstance(subsplit, str) and subsplit:
                            subsplit_name = subsplit
                            payload["subsplit"] = subsplit
                        video_name_label = labels_dict.get("video_name")
                        if isinstance(video_name_label, str) and video_name_label:
                            video_name = video_name_label
                            payload.setdefault("video_name", video_name_label)

                    if video_batch is not None:
                        video_value = to_serializable(video_batch[local_idx])
                        payload["video_id"] = _collapse_repeated(video_value)
                        if video_name is None and isinstance(payload["video_id"], str):
                            video_name = payload["video_id"]

                    frame_ids = None
                    if frame_idx_batch is not None:
                        frame_ids = to_serializable(frame_idx_batch[local_idx])
                        payload["frame_ids"] = frame_ids

                    frame_range = compute_frame_range(frame_ids)
                    if frame_range is not None:
                        payload["frame_range"] = {"start": frame_range[0], "end": frame_range[1]}

                    split_output_dir = split_dir
                    subsplit_dir_name = subsplit_name if subsplit_name else "default"
                    payload.setdefault("subsplit", subsplit_dir_name)
                    subsplit_dir = os.path.join(split_output_dir, subsplit_dir_name)
                    ensure_dir(subsplit_dir)

                    if not subsplit_has_capacity(subsplit_dir_name):
                        if per_subsplit_limit is not None and all_targets_satisfied():
                            completed_targets = True
                            break
                        continue

                    sequence_name = build_sequence_name(
                        dataset_idx,
                        video_name=video_name,
                        frame_range=frame_range,
                    )
                    payload["sequence_name"] = sequence_name

                    sequence_root = os.path.join(subsplit_dir, sequence_name)
                    ensure_dir(sequence_root)

                    gen_dir = os.path.join(sequence_root, "gen")
                    gen_gif_dir = os.path.join(sequence_root, "gen_gifs")
                    ensure_dir(gen_dir)
                    ensure_dir(gen_gif_dir)

                    clip = gen_frames[local_idx]
                    for frame_idx in range(clip.shape[0]):
                        save_image(
                            (clip[frame_idx] + 1.0) / 2.0,
                            os.path.join(gen_dir, f"frame_{frame_idx:04d}.jpg"),
                        )

                    clip_uint8 = ((clip + 1.0) / 2.0).clamp(0, 1)
                    clip_uint8 = (clip_uint8 * 255).byte().permute(0, 2, 3, 1).numpy()
                    gen_gif_path = os.path.join(gen_gif_dir, "generated.gif")
                    imageio.mimsave(gen_gif_path, list(clip_uint8), fps=args.gif_fps, loop=0)

                    if args.save_real and real_batch is not None:
                        real_dir = os.path.join(sequence_root, "real")
                        real_gif_dir = os.path.join(sequence_root, "real_gifs")
                        ensure_dir(real_dir)
                        ensure_dir(real_gif_dir)
                        real_clip = real_batch[local_idx]
                        for frame_idx in range(real_clip.shape[0]):
                            save_image(
                                (real_clip[frame_idx] + 1.0) / 2.0,
                                os.path.join(real_dir, f"frame_{frame_idx:04d}.jpg"),
                            )

                        real_uint8 = ((real_clip + 1.0) / 2.0).clamp(0, 1)
                        real_uint8 = (real_uint8 * 255).byte().permute(0, 2, 3, 1).numpy()
                        real_gif_path = os.path.join(real_gif_dir, "real.gif")
                        imageio.mimsave(real_gif_path, list(real_uint8), fps=args.gif_fps, loop=0)

                    labels_path = os.path.join(sequence_root, "labels.json")
                    with open(labels_path, "w") as f:
                        json.dump(payload, f, indent=2)

                    newly_generated += 1
                    current_target = (
                        sum(target_counts.values())
                        if per_subsplit_limit is not None and target_counts
                        else target_total
                    )
                    progress.set_postfix_str(
                        f"{newly_generated + initial_existing_total}/{current_target}"
                    )

                    if per_subsplit_limit is not None:
                        generated_by_subsplit[subsplit_dir_name] = (
                            generated_by_subsplit.get(subsplit_dir_name, 0) + 1
                        )
                        if all_targets_satisfied():
                            completed_targets = True
                            break

                if completed_targets:
                    break

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            print(f"   [{split}] peak GPU memory {peak_mem:.02f} GB")

        maybe_close_dataset(subset)
    finally:
        maybe_close_dataset(dataset)
        teardown = getattr(datamodule, "teardown", None)
        if callable(teardown):
            teardown(stage="validate")

    if per_subsplit_limit is not None:
        final_target_total = sum(target_counts.values()) if target_counts else initial_existing_total + newly_generated
    else:
        final_target_total = target_total

    return newly_generated, final_target_total, initial_existing_total


def main():
    parser = argparse.ArgumentParser(description="Holistic rollout over checkpoints and dataset splits.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing checkpoints.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Model config (relative to exp_dir unless absolute).")
    parser.add_argument("--val_config", type=str, default=None, help="Optional data config to override the one in config.")
    parser.add_argument("--ckpt_pattern", type=str, default="*.ckpt", help="Glob pattern (inside checkpoints) selecting checkpoints.")
    parser.add_argument("--ckpt_stride", type=int, default=1, help="Use every Nth checkpoint after sorting.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to roll out.")
    parser.add_argument("--num_videos", type=int, default=50, help="Target number of videos per split (<= dataset size).")
    parser.add_argument("--num_gen_frames", type=int, default=2, help="Number of frames to roll out.")
    parser.add_argument("--num_steps", type=int, default=32, help="Sampling steps (NFE).")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--evaluate_ema", type=str2bool, default=False, help="Use EMA weights during sampling.")
    parser.add_argument("--save_real", type=str2bool, default=True, help="Also store ground-truth frames.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for rollout (if applicable).")
    parser.add_argument("--frames_root", type=str, default="gen_rollout", help="Root folder (inside exp_dir) for outputs.")
    parser.add_argument("--data_tag", type=str, default=None, help="Override the data sub-folder name (default: from val_config).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda, cuda:1, cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; negative disables seeding.")
    parser.add_argument("--gif_fps", type=int, default=7, help="FPS for generated GIFs.")
    parser.add_argument(
        "--encdec_batch_size",
        type=int,
        default=2,
        help="Chunk size to use inside model.roll_out for encode/decode operations.",
    )
    args, unknown = parser.parse_known_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.encdec_batch_size is not None and args.encdec_batch_size <= 0:
        raise ValueError("--encdec_batch_size must be positive.")

    exp_dir = os.path.abspath(args.exp_dir)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found at {ckpt_dir}")

    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, args.ckpt_pattern)), key=natural_key, reverse=True)
    epoch_pattern = re.compile(r"^epoch-(\d+)\.ckpt$")
    ckpt_matches = []
    for ckpt_path in ckpt_paths:
        match = epoch_pattern.match(os.path.basename(ckpt_path))
        if match:
            ckpt_matches.append((int(match.group(1)), ckpt_path))
    ckpt_paths = [path for _, path in sorted(ckpt_matches, key=lambda item: item[0], reverse=True)]
    ckpt_paths = ckpt_paths[::args.ckpt_stride]
    #keep only ckpt with epoch 18 
    #ckpt_paths = [path for path in ckpt_paths if "epoch-18.ckpt" in path]
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints matched the required 'epoch-<number>.ckpt' format in {ckpt_dir}"
        )
    splits = [s.strip() for s in args.splits if s.strip()]
    if not splits:
        raise ValueError("No dataset splits specified after parsing.")

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
    frames_root = args.frames_root

    if args.seed >= 0:
        seed_everything(args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    summary = []
    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\nProcessing checkpoint {ckpt_name}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"'state_dict' missing in checkpoint {ckpt_path}")
        epoch = checkpoint.get("epoch")
        global_step = checkpoint.get("global_step")
        if epoch is None or global_step is None:
            raise KeyError(f"'epoch' or 'global_step' missing in checkpoint {ckpt_path}")
        ckpt_tag = f"ep{epoch}iter{global_step}_{args.num_steps}steps"

        model = instantiate_from_config(config.model)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device)
        model.eval()

        output_root = os.path.join(exp_dir, frames_root, data_tag, ckpt_tag)

        for split in splits:
            generated, target_total, existing = rollout_split(
                model=model,
                config=config,
                overrides=overrides,
                val_config_path=val_config_path,
                split=split,
                output_root=output_root,
                args=args,
                device=device,
            )
            summary.append((ckpt_name, split, generated, existing, target_total))

        del model
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nRollout summary:")
    for ckpt_name, split, generated, existing, target in summary:
        print(f" - {ckpt_name} [{split}]: {existing}+{generated}/{target} sequences ready")


if __name__ == "__main__":
    main()
