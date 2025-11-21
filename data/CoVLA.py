import argparse
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pytz
import torch
from astral import LocationInfo
from astral.sun import sun
from torch.utils.data import Dataset
from torchvision import transforms
from custom_multiframe_mp4_sa import MP4ClipDataset

from tqdm import tqdm


try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

import matplotlib.pyplot as plt


def mark_intermixed_transitions(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """
    Identify rows that should be flagged as 'intermixed' because any of the provided
    categorical columns transition between consecutive frames of the same video.

    Parameters
    ----------
    df:
        DataFrame that already contains the columns listed in *columns* plus
        'video_id' and 'idx'. The frame order should be per-video chronological.
    columns:
        Iterable of column names whose transitions should trigger the intermixed flag.

    Returns
    -------
    pd.Series[bool]
        Boolean mask aligned with ``df`` where True indicates an intermixed frame.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    # Work on a sorted copy to ensure chronological order per video.
    sorted_df = df.sort_values(["video_id", "idx"]).reset_index()
    group_ids = sorted_df["video_id"]
    intermixed = pd.Series(False, index=sorted_df.index)

    # Precompute cumcount to avoid marking the first frame of each video.
    frame_index_in_video = sorted_df.groupby("video_id").cumcount()

    for col in columns:
        if col not in sorted_df.columns:
            continue
        current = sorted_df[col]
        previous = sorted_df.groupby("video_id")[col].shift(1)

        # Detect transitions, treating None/NaN carefully.
        cur_na = current.isna()
        prev_na = previous.isna()
        differ = current != previous
        na_mismatch = cur_na ^ prev_na
        transitions = (frame_index_in_video > 0) & (differ | na_mismatch)
        transitions = transitions.fillna(False)

        # Mark both sides of the boundary (current and previous frame).
        prev_transitions = (
            transitions.groupby(group_ids).shift(-1).fillna(False).infer_objects(copy=False)
        )
        intermixed |= transitions | prev_transitions

    # Map back to original indices
    mask = pd.Series(False, index=df.index)
    mask.loc[sorted_df.loc[intermixed, "index"]] = True
    return mask


#- `--latitude` / `--longitude`: location coordinates (default Tokyo 35.6895, 139.6917)
def time_of_day_3(
    timestamp_ms: int,
    tz_name: str,
    latitude: float = 35.6895,
    longitude: float = 139.6917,
    _cache: Optional[dict] = None,
) -> str:
    """
    Classify timestamp as 'night' | 'dawn' | 'day' | 'dusk' using Astral, matching tutorial.ipynb.
    We cache dawn/sunrise/sunset/dusk per date for performance.
    """ 
    tz = pytz.timezone(tz_name)  # type: ignore[attr-defined]
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz)
    date_key = dt.date()
    if _cache is None:
        _cache = {}
    if date_key not in _cache:
        loc = LocationInfo("Tokyo", "Japan", tz_name, latitude=latitude, longitude=longitude)  # type: ignore[name-defined]
        s = sun(loc.observer, date=date_key, tzinfo=tz)  # type: ignore[name-defined]
        _cache[date_key] = (
            s["dawn"],
            s["sunrise"],
            s["sunset"],
            s["dusk"],
        )
    dawn, sunrise, sunset, dusk = _cache[date_key]
    if dawn <= dt < sunrise:
        return "dawn"
    elif sunrise <= dt < sunset:
        return "day"
    elif sunset <= dt < dusk:
        return "dusk"
    else:
        return "night"

def _iter_jsonl_records(path: Path) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or len(obj) != 1:
                raise ValueError(f"Unexpected record at {path} line {line_no}: {obj!r}")
            key, payload = next(iter(obj.items()))
            idx = int(key)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected dict payload at {path} line {line_no}")
            yield idx, payload


class RandomShiftCropTensorBatch:
    def __init__(self, size: int | Tuple[int, int], max_shift_horizontal: int = 0, max_shift_vertical: int = 0):
        self.size = (size, size) if isinstance(size, int) else size
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical

    def _get_params(self, tensor_batch: torch.Tensor) -> Tuple[int, int]:
        width, height = tensor_batch.shape[-1], tensor_batch.shape[-2]
        crop_h, crop_w = self.size
        center_left = (width - crop_w) // 2
        center_top = (height - crop_h) // 2
        shift_h = torch.randint(-self.max_shift_horizontal, self.max_shift_horizontal + 1, (1,)).item()
        shift_v = torch.randint(-self.max_shift_vertical, self.max_shift_vertical + 1, (1,)).item()
        left = max(0, min(center_left + shift_h, width - crop_w))
        top = max(0, min(center_top + shift_v, height - crop_h))
        return left, top

    def __call__(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        left, top = self._get_params(tensor_batch)
        crop_h, crop_w = self.size
        return tensor_batch[..., top:top + crop_h, left:left + crop_w]


class DecordToTensorBatch:
    def __call__(self, video_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(video_array).permute(0, 3, 1, 2).float() / 255.0


class CoVLAMultiFrameDataset(MP4ClipDataset):
    def __init__(
        self,
        covla_root: Path,
        mp4_paths_file: Path,
        num_frames: int,
        metadata_cache_csv: Optional[Path] = None,
        regenerate_cache: bool = False,
        stored_data_frame_rate: int = 20,
        frame_rate: int = 5,
        split: str = "train",
        size: int | Tuple[int, int] = 256,
        aug: str = "resize_center",
        backend: Optional[str] = None,
        future_horizon: Optional[int] = None,
        tz_name: str = "Asia/Tokyo",
        angle_unit: str = "deg",
        straight_threshold: float = 12.0,
        boundary_threshold: float = 30.0,
        min_turn_speed_ms: float = 0.05,
        video_length: int = 600
    ) -> None:
        mp4_paths_path = Path(os.path.expandvars(str(mp4_paths_file)))
        mp4_paths = [Path(p) for p in mp4_paths_path.read_text().splitlines() if p.strip()]

        super().__init__(
            mp4_paths=mp4_paths,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            video_length=video_length,
            size=size,
            aug=aug,
            backend=backend,

        )

        self.selected_split = split.lower() if split else "all"
        valid_splits = {"train", "val", "test", "intermixed", "all"}
        if self.selected_split not in valid_splits:
            warnings.warn(f"Unknown split '{split}'; defaulting to 'all'.")
            self.selected_split = "all"
        self._clip_frame_offsets = [i * self.frame_interval for i in range(self.num_frames)]
        self.covla_root = Path(os.path.expandvars(str(covla_root)))
        self.states_dir = self.covla_root / "states"
        self.future_horizon = future_horizon
        cache_path = (
            Path(metadata_cache_csv)
            if metadata_cache_csv is not None
            else self.covla_root / "analysis_cache" / "covla_features.pkl"
        )
        cache_path = Path(os.path.expandvars(str(cache_path)))
        if cache_path.suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError(f"CoVLA metadata cache must end with .pkl or .pickle, got {cache_path}")
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.tz_name = tz_name
        self.angle_unit = angle_unit
        self.straight_threshold = straight_threshold
        self.boundary_threshold = boundary_threshold
        self.min_turn_speed_ms = min_turn_speed_ms
        self._frame_features = self._load_or_build_feature_cache(regenerate_cache)
    
    def _load_or_build_feature_cache(self, regenerate_cache: bool) -> pd.DataFrame:
        df: Optional[pd.DataFrame] = None
        if not regenerate_cache and self.cache_path.exists():
            try:
                df = self._read_cached_dataframe(self.cache_path)
            except Exception as exc:
                warnings.warn(f"Failed to load CoVLA cache from {self.cache_path}: {exc}")

        if df is None:
            df = self._build_feature_dataframe()
            self._write_dataframe_cache(df)
        else:
            df = self._finalize_dataframe(df)

        indexed_df = df.set_index(["video_id", "frame_idx"]).sort_index()
        clip_ok_series = indexed_df["clip_ok"]

        start_pairs = list(self.index_to_starting_frame_map)
        start_keys = [(path.stem, start) for path, start in start_pairs]
        clip_ok_mask = clip_ok_series.reindex(start_keys).fillna(False).to_numpy(dtype=bool)

        filtered_pairs: List[Tuple[Path, int]] = []

        for (path, start), ok in zip(start_pairs, clip_ok_mask):
            if not ok:
                continue
            filtered_pairs.append((path, start))

        if not filtered_pairs:
            warnings.warn(
                f"No clips available after applying split filter '{self.selected_split}'. "
                "Try regenerating the cache or selecting a different split."
            )

        self.index_to_starting_frame_map = filtered_pairs
        return indexed_df

    def _get_clip_feature_rows(self, video_id: str, start_frame: int) -> pd.DataFrame:
        if self._clip_frame_offsets:
            frame_indices = [start_frame + offset for offset in self._clip_frame_offsets]
        else:
            frame_indices = [start_frame]
        multi_index = pd.MultiIndex.from_arrays(
            [[video_id] * len(frame_indices), frame_indices],
            names=self._frame_features.index.names,
        )
        clip_rows = self._frame_features.reindex(multi_index)
        missing_mask = clip_rows.isnull().all(axis=1)
        if bool(missing_mask.any()):
            missing = [frame_indices[i] for i, missing_row in enumerate(missing_mask.to_numpy()) if missing_row]
            raise KeyError(f"Missing cached metadata for {video_id} frames {missing}")
        return clip_rows.reset_index(drop=True)

    def _write_dataframe_cache(self, df: pd.DataFrame) -> None:
        df.to_pickle(self.cache_path, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_cached_dataframe(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported cache file extension for {path}")

    def _build_feature_dataframe(self) -> pd.DataFrame:
        required_keys: Dict[str, set[int]] = {}
        clip_offsets = self._clip_frame_offsets if self._clip_frame_offsets else [0]
        for path, start in self.index_to_starting_frame_map:
            video_id = path.stem
            frame_indices = {start + offset for offset in clip_offsets}
            required_keys.setdefault(video_id, set()).update(frame_indices)

        records: List[Dict[str, object]] = []
        state_files = list(self.states_dir.glob("*.jsonl"))
        for state_file in tqdm(state_files, desc="Building CoVLA features", unit="file"):
            video_id = state_file.stem
            needed_frames = required_keys.get(video_id)
            if not needed_frames:
                continue

            for frame_idx, payload in _iter_jsonl_records(state_file):
                if frame_idx not in needed_frames:
                    continue
                traj = payload.get("trajectory") or []
                if traj:
                    if self.frame_interval > 1:
                        traj = traj[self.frame_interval - 1 :: self.frame_interval]
                    if self.future_horizon is not None:
                        traj = traj[: self.future_horizon]

                records.append(
                    {
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "idx": frame_idx,
                        "timestamp_ms": payload.get("timestamp"),
                        "speed_ms": payload.get("vEgo"),
                        "acceleration_ms2": payload.get("aEgo"),
                        "steering_deg": payload.get("steeringAngleDeg"),
                        "camera_intrinsic": payload.get("intrinsic_matrix") or [],
                        "camera_extrinsic": payload.get("extrinsic_matrix") or [],
                        "future_traj": traj,
                    }
                )

        if not records:
            raise RuntimeError("No CoVLA metadata collected; check mp4 paths vs states directory")

        df = pd.DataFrame(records)
        df["time3"] = df["timestamp_ms"].apply(lambda ts: time_of_day_3(int(ts), tz_name=self.tz_name) if pd.notna(ts) else None)
        df["is_dark"] = df["time3"] == "night"

        split_base = pd.Series("train", index=df.index, dtype="object")
        df["is_left"] = (df["steering_deg"] >= 12) & (df["speed_ms"] > 0.05)
        split_base[(df["is_dark"]) & df["is_left"]] = "val"
        split_base[df["time3"] == "dusk"] = "test"
        df["split_base"] = split_base
        self._frame_summary = {
            "val_frames": int((split_base == "val").sum()),
            "steering_ge_12": int((df["steering_deg"] >= 12).sum()),
            "is_dark": int(df["is_dark"].sum()),
            "is_left": int(df["is_left"].sum()),
        }
        print(f"summary: { self._frame_summary }")

        transition_columns = ["split_base", "time3", "is_dark", "is_left"]
        intermixed_mask = mark_intermixed_transitions(df, columns=transition_columns)
        df["split"] = df["split_base"]
        if intermixed_mask.any():
            df.loc[intermixed_mask, "split"] = "intermixed"
        df.drop(columns=["split_base", "is_dark", "is_left"], inplace=True)

        return self._finalize_dataframe(df)

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "clip_ok" in df.columns:
            df = df.drop(columns=["clip_ok"])

        df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
        split_norm = df["split"].fillna("").str.lower()
        offsets = self._clip_frame_offsets if self._clip_frame_offsets else [0]

        if self.selected_split == "all":
            frame_ok = split_norm != "intermixed"
        elif self.selected_split == "intermixed":
            frame_ok = split_norm == "intermixed"
        else:
            frame_ok = (split_norm == self.selected_split) & (split_norm != "intermixed")
        frame_ok &= split_norm != ""

        df["_frame_ok_tmp"] = frame_ok
        clip_ok = frame_ok.copy()
        grouped = df.groupby("video_id")["_frame_ok_tmp"]
        for offset in offsets[1:]:
            clip_ok &= grouped.shift(-offset).fillna(False).infer_objects(copy=False)
        df["clip_ok"] = clip_ok
        df.drop(columns=["_frame_ok_tmp"], inplace=True)
        return df

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frames_tensor, path, start_frame = self._get_clip(idx)
        video_id = path.stem
        frame_offsets = self._clip_frame_offsets if self._clip_frame_offsets else [0]
        clip_rows = self._get_clip_feature_rows(video_id, start_frame)
        frame_indices = [start_frame + offset for offset in frame_offsets]

        def _stack_matrices(series: pd.Series, name: str) -> torch.Tensor:
            matrices = []
            for value, frame_idx in zip(series.tolist(), frame_indices):
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tensor.numel() == 0:
                    raise ValueError(f"Empty {name} for {video_id} frame {frame_idx}")
                matrices.append(tensor)
            return torch.stack(matrices, dim=0)

        speeds = torch.tensor(
            clip_rows["speed_ms"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        accelerations = torch.tensor(
            clip_rows["acceleration_ms2"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        steerings = torch.tensor(
            clip_rows["steering_deg"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        camera_intrinsic = _stack_matrices(clip_rows["camera_intrinsic"], "camera_intrinsic")
        camera_extrinsic = _stack_matrices(clip_rows["camera_extrinsic"], "camera_extrinsic")

        future_row = clip_rows.iloc[-1]
        future_arr = torch.as_tensor(future_row["future_traj"], dtype=torch.float32)
        if self.future_horizon is not None and future_arr.numel() > 0:
            future_arr = future_arr[: self.future_horizon]
        split_value = future_row["split"] if isinstance(future_row["split"], str) else None

        sample = {
            "frames": frames_tensor,
            "video_id": video_id,
            "start_frame": start_frame,
            "speed_ms": speeds,
            "acceleration_ms2": accelerations,
            "steering_deg": steerings,
            "camera_intrinsic": camera_intrinsic,
            "camera_extrinsic": camera_extrinsic,
            "future_traj": future_arr,
            "split": split_value,
            "time3": clip_rows["time3"].tolist(),
        }
        return sample


def _parse_size(value: str) -> int | Tuple[int, int]:
    value = value.strip().lower()
    for separator in ("x", ","):
        if separator in value:
            parts = value.split(separator)
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Invalid size '{value}'; expected '<h>{separator}<w>'.")
            return int(parts[0]), int(parts[1])
    return int(value)


def _tensor_summary(t: torch.Tensor) -> str:
    if t.ndim == 0:
        return f"{t.dtype} value={t.item()}"
    return f"{t.dtype} shape={tuple(t.shape)}"


def _maybe_scalar(value: object) -> object:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    return value


def _plot_trajectory_on_image(
    frame: np.ndarray,
    trajectory: np.ndarray,
    extrinsic_matrix: np.ndarray,
    intrinsic_matrix: np.ndarray,
    out_path: Path,
) -> Optional[Path]:
    if plt is None:
        warnings.warn("matplotlib is not available; skipping trajectory plot.")
        return None

    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.size == 0:
        return None
    if traj.ndim == 1:
        if traj.size % 3 != 0:
            return None
        traj = traj.reshape(-1, 3)
    else:
        if traj.shape[-1] < 3:
            return None
        traj = traj.reshape(-1, traj.shape[-1])[:, :3]

    extrinsic = np.asarray(extrinsic_matrix, dtype=np.float32)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
    if extrinsic.shape != (4, 4):
        return None

    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        return None

    ones = np.ones((traj.shape[0], 1), dtype=np.float32)
    traj_h = np.hstack([traj, ones])
    camera_h = (extrinsic @ traj_h.T).T
    camera = camera_h[:, :3]

    forward_mask = camera[:, 2] > 0.0
    if not np.any(forward_mask):
        return None
    camera = camera[forward_mask]

    proj = (intrinsic @ camera.T).T
    depth = proj[:, 2:3]
    valid_depth = depth > 0.0
    proj = proj[valid_depth.squeeze()]
    if proj.size == 0:
        return None

    coords = proj[:, :2] / proj[:, 2:3]
    if coords.size == 0:
        return None
    print(f"current coords are: {coords}")
    image_height, image_width = frame.shape[:2]
    in_bounds = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < image_width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < image_height)
    )
    coords = coords[in_bounds]
    print(f"current inbound coords are: {coords}")

    fig, ax = plt.subplots(figsize=(image_width / 100.0, image_height / 100.0), dpi=100)
    ax.imshow(frame)
    ax.axis("off")
    if coords.size:
        ax.plot(coords[:, 0], coords[:, 1], marker="o", color="red", linestyle="-", linewidth=1, markersize=3)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quick CLI to inspect CoVLAMultiFrameDataset samples.")
    parser.add_argument("--covla-root", type=Path, default="/data/nxtaimraid02/datasets/CoVLA", help="Path to CoVLA root containing states/ and captions/.")
    parser.add_argument("--mp4-paths-file", type=Path, default="/work/dlclarge2/faridk-diff_force/orbis/data/covla_videos.txt", help="Text file listing MP4 clip paths (one per line).")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per clip.")
    parser.add_argument("--metadata-cache-csv", type=Path, default="/work/dlclarge2/faridk-diff_force/orbis/data/covla_features.pkl", help="Optional path for the metadata cache (.pkl).")
    parser.add_argument("--regenerate-cache", action="store_true", help="Regenerate the metadata cache even if it exists.")
    parser.add_argument("--stored-frame-rate", type=int, default=20, help="Frame rate the MP4 files were encoded with.")
    parser.add_argument("--frame-rate", type=int, default=5, help="Target frame rate when sampling clips.")
    parser.add_argument("--size", type=str, default="1208x1928", help="Output spatial size. Accepts single int or HxW, e.g. 224x384.")
    parser.add_argument("--aug", type=str, default="resize_center", help="Augmentation pipeline to use (resize_center | random_shift).")
    parser.add_argument("--backend", type=str, choices=["decord", "torchcodec"], help="Video decoding backend override.")
    parser.add_argument("--future-horizon", type=int, help="Optional slice for future trajectory length.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (train | val | test | intermixed | all).")
    parser.add_argument("--tz-name", type=str, default="Asia/Tokyo", help="Timezone name for time-of-day bucketing.")
    parser.add_argument("--angle-unit", type=str, default="deg", help="Unit used for steering angles.")
    parser.add_argument("--straight-threshold", type=float, default=12.0, help="Threshold for straight label in degrees.")
    parser.add_argument("--boundary-threshold", type=float, default=30.0, help="Threshold for boundary turn in degrees.")
    parser.add_argument("--min-turn-speed-ms", type=float, default=0.05, help="Minimum speed to consider a turn.")
    parser.add_argument("--sample-indices", type=int, nargs="*", default=[0,1000,2000], help="Dataset indices to inspect.")
    args = parser.parse_args(argv)

    dataset = CoVLAMultiFrameDataset(
        covla_root=args.covla_root,
        mp4_paths_file=args.mp4_paths_file,
        num_frames=args.num_frames,
        metadata_cache_csv=args.metadata_cache_csv,
        regenerate_cache=args.regenerate_cache,
        stored_data_frame_rate=args.stored_frame_rate,
        frame_rate=args.frame_rate,
        size=_parse_size(args.size),
        aug=args.aug,
        backend=args.backend,
        future_horizon=args.future_horizon,
        split=args.split,
        tz_name=args.tz_name,
        angle_unit=args.angle_unit,
        straight_threshold=args.straight_threshold,
        boundary_threshold=args.boundary_threshold,
        min_turn_speed_ms=args.min_turn_speed_ms,
    )

    total = len(dataset)
    print(f"Loaded CoVLAMultiFrameDataset with {total} clips.")
    print("Metadata cache:", dataset.cache_path)
    for idx in args.sample_indices:
        if idx < 0 or idx >= total:
            print(f"- Index {idx} out of range (0 <= idx < {total}); skipping.")
            continue
        print(f"\nSample @ index {idx}")
        sample = dataset[idx]
        frames = sample["frames"]
        gif_path = Path(f"covla_sample_{idx}_{sample['video_id']}.gif")
        frames_cpu = frames.detach().cpu().clamp(-1, 1)
        frames_uint8 = torch.clamp(((frames_cpu + 1) * 127.5).round(), 0, 255).to(torch.uint8)
        gif_array = frames_uint8.permute(0, 2, 3, 1).contiguous().numpy()
        fps = args.frame_rate if args.frame_rate and args.frame_rate > 0 else None
        gif_kwargs = {"loop": 0}
        if fps:
            gif_kwargs["fps"] = fps
        else:
            gif_kwargs["duration"] = 0.1
        imageio.mimsave(str(gif_path), gif_array, **gif_kwargs)
        traj_image_path: Optional[Path] = None
        future_traj_np = sample["future_traj"].detach().cpu().numpy()
        try:
            last_frame = gif_array[-1]
            intrinsic_np = sample["camera_intrinsic"][-1].detach().cpu().numpy()
            extrinsic_np = sample["camera_extrinsic"][-1].detach().cpu().numpy()
            traj_image_path = _plot_trajectory_on_image(
                frame=last_frame,
                trajectory=future_traj_np,
                extrinsic_matrix=extrinsic_np,
                intrinsic_matrix=intrinsic_np,
                out_path=Path(f"covla_traj_{idx}_{sample['video_id']}.png"),
            )
        except Exception as exc:  # pragma: no cover - diagnostic aid
            warnings.warn(f"Failed to plot trajectory for sample {idx}: {exc}")
        print(f"  gif_path: {gif_path}")
        if traj_image_path:
            print(f"  traj_image: {traj_image_path}")
        elif future_traj_np.size == 0:
            print("  traj_image: skipped (empty future trajectory)")
        else:
            print("  traj_image: skipped (projection unavailable)")
        print(f"  frames: {_tensor_summary(frames)}")
        print(f"  video_id: {sample['video_id']} start_frame: {sample['start_frame']}")
        print(
            "  motion:",
            f"speed={_maybe_scalar(sample['speed_ms'])}",
            f"accel={_maybe_scalar(sample['acceleration_ms2'])}",
            f"steering={_maybe_scalar(sample['steering_deg'])}",
        )
        #print(f"  time3={sample['time3']} split={sample['split']}")
        print(f"  camera_intrinsic: {_tensor_summary(sample['camera_intrinsic'])}")
        print(f"  camera_extrinsic: {_tensor_summary(sample['camera_extrinsic'])}")
        print(f"  future_traj: {_tensor_summary(sample['future_traj'])}")


if __name__ == "__main__":
    main()
