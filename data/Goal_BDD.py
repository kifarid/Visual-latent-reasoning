import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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

try:
    from astral import LocationInfo
    from astral.sun import sun

    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False


MP4PathInput = str | os.PathLike[str]


def _load_mp4_paths_from_txt(list_path: Path) -> List[Path]:
    """Read MP4 paths from a text file, ignoring blanks and comments."""
    resolved_file = Path(os.path.expandvars(str(list_path))).expanduser()
    if not resolved_file.is_file():
        raise FileNotFoundError(f"MP4 list file {resolved_file} does not exist")
    mp4_paths: List[Path] = []
    for line_no, raw_line in enumerate(resolved_file.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        mp4_paths.append(Path(os.path.expandvars(stripped)).expanduser())
    if not mp4_paths:
        raise ValueError(f"No MP4 paths found in {resolved_file}")
    return mp4_paths


def _normalize_mp4_inputs(mp4_inputs: Sequence[MP4PathInput]) -> List[Path]:
    """Expand environment variables and flatten optional txt lists into concrete MP4 paths."""
    normalized: List[Path] = []
    for entry in mp4_inputs:
        entry_path = Path(os.path.expandvars(str(entry))).expanduser()
        if entry_path.suffix.lower() == ".txt":
            normalized.extend(_load_mp4_paths_from_txt(entry_path))
        else:
            normalized.append(entry_path)
    if not normalized:
        raise ValueError("mp4_paths resolved to an empty list")
    return normalized


def _log_goal_bdd(message: str) -> None:
    print(f"[BDDGoalDataset] {message}", flush=True)


class RandomShiftCropTensorBatch:
    """Apply a random crop centered around the frame center to every clip in the batch."""

    def __init__(self, size: int | Tuple[int, int], max_shift_horizontal: int = 60, max_shift_vertical: int = 60):
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
    """Convert a Decord numpy batch (T,H,W,C) in [0,255] to a float tensor (T,C,H,W) in [0,1]."""

    def __call__(self, video_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(video_array).permute(0, 3, 1, 2).float() / 255.0


class MP4ClipDataset(Dataset):
    """Standalone MP4 reader with clip indexing, augmentation, and backend (Decord/TorchCodec) handling."""

    def __init__(
        self,
        mp4_paths: Sequence[MP4PathInput],
        num_frames: int,
        stored_data_frame_rate: int = 20,
        frame_rate: int = 5,
        video_length: int = None,
        size: int | Tuple[int, int] = 256,
        aug: str = "resize_center",
        backend: Optional[str] = None,
    ) -> None:
        self.mp4_paths = _normalize_mp4_inputs(mp4_paths)
        self.num_frames = num_frames
        self.stored_data_frame_rate = stored_data_frame_rate
        self.video_length = video_length
        self.frame_rate = frame_rate
        self.frame_interval = max(1, stored_data_frame_rate // frame_rate)
        self.size = (size, size) if isinstance(size, int) else size

        if backend is not None:
            if backend not in {"decord", "torchcodec"}:
                raise ValueError(f"Unknown backend {backend}")
            if backend == "decord" and not DECORD_AVAILABLE:
                raise ImportError("Decord backend requested but not available")
            if backend == "torchcodec" and not TORCHCODEC_AVAILABLE:
                raise ImportError("TorchCodec backend requested but not available")
            self.backend = backend
        else:
            if DECORD_AVAILABLE:
                self.backend = "decord"
            elif TORCHCODEC_AVAILABLE:
                self.backend = "torchcodec"
            else:
                raise ImportError("Install decord or torchcodec to decode MP4 clips")

        if aug == "resize_center":
            self.transform = transforms.Compose(
                [
                    DecordToTensorBatch() if self.backend == "decord" else transforms.Lambda(lambda x: torch.from_numpy(x).float() / 255.0),
                    transforms.Resize(min(self.size)),
                    transforms.CenterCrop(self.size),
                ]
            )
        elif aug == "random_shift":
            random_crop = RandomShiftCropTensorBatch(self.size, max_shift_horizontal=60, max_shift_vertical=30)
            self.transform = transforms.Compose(
                [
                    DecordToTensorBatch() if self.backend == "decord" else transforms.Lambda(lambda x: torch.from_numpy(x).float() / 255.0),
                    transforms.Resize(min(self.size)),
                    random_crop,
                ]
            )
        else:
            raise ValueError(f"Unsupported augmentation {aug}")

        self.index_to_starting_frame_map: List[Tuple[Path, int]] = []
        self._scan_mp4_files()

    def _scan_mp4_files(self) -> None:
        for path in tqdm(self.mp4_paths, desc="Scanning"):
            if self.video_length and self.stored_data_frame_rate:
                fps = self.stored_data_frame_rate
                length = self.video_length
            elif self.backend == "decord":
                reader = VideoReader(str(path), ctx=decord_cpu(0))
                fps = reader.get_avg_fps()
                length = len(reader)
            else:
                reader = VideoDecoder(str(path))
                fps = reader.metadata.average_fps
                length = reader.metadata.num_frames

            if int(round(fps)) != self.stored_data_frame_rate:
                raise ValueError(f"{path} reports {fps} fps; expected {self.stored_data_frame_rate}")

            max_start = length - self.num_frames * self.frame_interval
            for start in range(0, max_start + 1):
                self.index_to_starting_frame_map.append((path, start))

    def __len__(self) -> int:
        return len(self.index_to_starting_frame_map)

    def _read_frames(self, path: Path, start_frame: int) -> np.ndarray:
        indices = list(range(start_frame, start_frame + self.num_frames * self.frame_interval, self.frame_interval))
        if self.backend == "decord":
            reader = VideoReader(str(path), ctx=decord_cpu(0))
            frames = reader.get_batch(indices).asnumpy()
        else:
            reader = VideoDecoder(str(path))
            frames = reader[start_frame:start_frame + self.num_frames * self.frame_interval:self.frame_interval]
        return frames

    def _get_clip(self, idx: int) -> Tuple[torch.Tensor, Path, int]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        path, start_frame = self.index_to_starting_frame_map[idx]
        frames = self._read_frames(path, start_frame)
        frames_tensor = self.transform(frames)
        frames_tensor = frames_tensor * 2 - 1
        return frames_tensor, path, start_frame


EARTH_RADIUS_M = 6_371_000.0
TIME_OF_DAY_LABELS = ("night", "dawn", "day", "dusk", "unknown")
TIME_OF_DAY_TO_ID = {label: idx for idx, label in enumerate(TIME_OF_DAY_LABELS)}
_LOCATION_CACHE: Dict[Tuple[int, int], object] = {}
_LOCATION_PRECISION = 2  # degrees precision for caching LocationInfo instances


@dataclass(frozen=True)
class FrameMetadata:
    timestamp_ms: int
    lat: float
    lon: float
    x: float
    y: float
    speed_ms: float
    yaw_rad: float
    time_of_day_id: int

    @property
    def time_of_day_label(self) -> str:
        return TIME_OF_DAY_LABELS[self.time_of_day_id]


def _latlon_to_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = (lon_rad - lon0_rad) * np.cos((lat_rad + lat0_rad) / 2.0) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x.astype(np.float64), y.astype(np.float64)


def _compute_yaw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float64)
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.unwrap(np.arctan2(dy, dx))
    yaw[~np.isfinite(yaw)] = 0.0
    return yaw.astype(np.float64)


def _get_location_info(lat: float, lon: float):
    if not ASTRAL_AVAILABLE:
        return None
    key = (int(round(lat * 10**_LOCATION_PRECISION)), int(round(lon * 10**_LOCATION_PRECISION)))
    loc = _LOCATION_CACHE.get(key)
    if loc is None:
        loc = LocationInfo(name="bdd", region="bdd", timezone="UTC", latitude=lat, longitude=lon)
        _LOCATION_CACHE[key] = loc
    return loc


def _classify_time_of_day(lat: float, lon: float, timestamp_ms: int) -> str:
    if not ASTRAL_AVAILABLE:
        return "unknown"
    loc = _get_location_info(lat, lon)
    if loc is None:
        return "unknown"
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    try:
        sun_times = sun(loc.observer, date=dt.date(), tzinfo=timezone.utc)
    except Exception:
        return "unknown"
    if dt < sun_times["dawn"] or dt >= sun_times["dusk"]:
        return "night"
    if dt < sun_times["sunrise"]:
        return "dawn"
    if dt < sun_times["sunset"]:
        return "day"
    return "dusk"


def _encode_time_of_day(label: str) -> int:
    return TIME_OF_DAY_TO_ID.get(label, TIME_OF_DAY_TO_ID["unknown"])


def _video_has_gps(info_root: Path, video_id: str) -> bool:
    json_path = info_root / f"{video_id}.json"
    if not json_path.exists():
        return False
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    gps_records = payload.get("gps")
    return bool(gps_records)


class _BDDMetadataCache:
    """Lazy loader for per-video GPS/timestamp metadata with optional on-disk caching."""

    def __init__(self, info_root: Path, cache_dir: Path, target_hz: int) -> None:
        self.info_root = Path(info_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory: Dict[str, List[FrameMetadata]] = {}
        self.target_hz = max(1, int(target_hz))

    def get(self, video_id: str) -> List[FrameMetadata]:
        if video_id not in self._in_memory:
            self._in_memory[video_id] = self._load_or_build(video_id)
        return self._in_memory[video_id]

    def _cache_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}_{self.target_hz}hz.npz"

    def _load_or_build(self, video_id: str) -> List[FrameMetadata]:
        cache_path = self._cache_path(video_id)
        if cache_path.exists():
            return self._load_from_disk(cache_path)
        metadata = self._build_from_json(video_id)
        self._save_to_disk(cache_path, metadata)
        return metadata

    def _build_from_json(self, video_id: str) -> List[FrameMetadata]:
        json_path = self.info_root / f"{video_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No metadata JSON for video {video_id} at {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        gps_records = payload.get("gps") or []
        if not gps_records:
            raise RuntimeError(f"Metadata file {json_path} contains no GPS entries")
        gps_records = sorted(gps_records, key=lambda rec: rec.get("timestamp", 0))
        timestamps = np.array([int(rec.get("timestamp", 0)) for rec in gps_records], dtype=np.int64)
        lat = np.array([float(rec.get("latitude", 0.0)) for rec in gps_records], dtype=np.float64)
        lon = np.array([float(rec.get("longitude", 0.0)) for rec in gps_records], dtype=np.float64)
        speed = np.array([float(rec.get("speed", 0.0) or 0.0) for rec in gps_records], dtype=np.float64)
        target_step_ms = max(1, int(round(1000 / self.target_hz)))
        if len(timestamps) == 1:
            uniform_t = timestamps.copy()
            lat_interp = lat.copy()
            lon_interp = lon.copy()
            speed_interp = speed.copy()
        else:
            uniform_t = np.arange(timestamps[0], timestamps[-1] + target_step_ms, target_step_ms, dtype=np.int64)
            base_t = timestamps.astype(np.float64)
            lat_interp = np.interp(uniform_t.astype(np.float64), base_t, lat)
            lon_interp = np.interp(uniform_t.astype(np.float64), base_t, lon)
            speed_interp = np.interp(uniform_t.astype(np.float64), base_t, speed)
        x, y = _latlon_to_xy(lat_interp, lon_interp, float(lat_interp[0]), float(lon_interp[0]))
        yaw = _compute_yaw(x, y)
        metadata: List[FrameMetadata] = []
        for idx in range(len(uniform_t)):
            tod_label = _classify_time_of_day(float(lat_interp[idx]), float(lon_interp[idx]), int(uniform_t[idx]))
            metadata.append(
                FrameMetadata(
                    timestamp_ms=int(uniform_t[idx]),
                    lat=float(lat_interp[idx]),
                    lon=float(lon_interp[idx]),
                    x=float(x[idx]),
                    y=float(y[idx]),
                    speed_ms=float(speed_interp[idx]),
                    yaw_rad=float(yaw[idx]),
                    time_of_day_id=_encode_time_of_day(tod_label),
                )
            )
        return metadata

    def _save_to_disk(self, cache_path: Path, metadata: List[FrameMetadata]) -> None:
        np.savez_compressed(
            cache_path,
            timestamp_ms=np.array([entry.timestamp_ms for entry in metadata], dtype=np.int64),
            lat=np.array([entry.lat for entry in metadata], dtype=np.float64),
            lon=np.array([entry.lon for entry in metadata], dtype=np.float64),
            x=np.array([entry.x for entry in metadata], dtype=np.float64),
            y=np.array([entry.y for entry in metadata], dtype=np.float64),
            speed_ms=np.array([entry.speed_ms for entry in metadata], dtype=np.float32),
            yaw_rad=np.array([entry.yaw_rad for entry in metadata], dtype=np.float32),
            time_of_day_id=np.array([entry.time_of_day_id for entry in metadata], dtype=np.int8),
        )

    def _load_from_disk(self, cache_path: Path) -> List[FrameMetadata]:
        data = np.load(cache_path, allow_pickle=False)
        timestamps = data["timestamp_ms"]
        lat = data["lat"]
        lon = data["lon"]
        x = data["x"]
        y = data["y"]
        speed = data["speed_ms"]
        yaw = data["yaw_rad"]
        tod = data["time_of_day_id"]
        metadata: List[FrameMetadata] = []
        for idx in range(len(timestamps)):
            metadata.append(
                FrameMetadata(
                    timestamp_ms=int(timestamps[idx]),
                    lat=float(lat[idx]),
                    lon=float(lon[idx]),
                    x=float(x[idx]),
                    y=float(y[idx]),
                    speed_ms=float(speed[idx]),
                    yaw_rad=float(yaw[idx]),
                    time_of_day_id=int(tod[idx]),
                )
            )
        return metadata


class BDDGoalDataset(MP4ClipDataset):
    """
    Dataset that augments MP4 clips with driving context information derived from the BDD100K metadata JSON.
    Each sample returns the clip tensor along with context/goal positions relative to the clip start,
    a direction vector computed from the first few steps, and time-of-day labels inferred via Astral.
    """

    def __init__(
        self,
        mp4_paths: Sequence[MP4PathInput],
        info_root: Path,
        metadata_cache_dir: Path,
        num_frames: int,
        ctx_frame: int = 0,
        direction_window: int = 5,
        stored_data_frame_rate: int = 10,
        frame_rate: int = 5,
        video_length: Optional[int] = None,
        size: int | Tuple[int, int] = 256,
        aug: str = "resize_center",
        backend: Optional[str] = None,
        drop_missing_metadata: bool = True,
    ) -> None:
        init_start = time.perf_counter()
        _log_goal_bdd(f"Initializing dataset from {len(mp4_paths)} mp4 list entries...")
        resolve_start = time.perf_counter()
        info_root = Path(info_root)
        metadata_cache_dir = Path(metadata_cache_dir)
        processed_paths = _normalize_mp4_inputs(mp4_paths)
        _log_goal_bdd(f"Resolved {len(processed_paths)} MP4 files in {time.perf_counter() - resolve_start:.1f}s")
        dropped: List[Path] = []
        if drop_missing_metadata:
            _log_goal_bdd(f"Filtering {len(processed_paths)} videos for GPS metadata...")
            filter_start = time.perf_counter()
            processed_paths, dropped = self._filter_paths_with_gps(processed_paths, info_root, show_progress=True)
            _log_goal_bdd(
                f"GPS metadata filtering completed in {time.perf_counter() - filter_start:.1f}s "
                f"(kept {len(processed_paths)}, dropped {len(dropped)})."
            )
            if dropped:
                print(f"Dropping {len(dropped)} videos without GPS metadata: {[p.stem for p in dropped[:5]]}{'...' if len(dropped) > 5 else ''}")
        if not processed_paths:
            raise RuntimeError("No MP4 clips remain after filtering for GPS metadata")
        self.metadata_cache = _BDDMetadataCache(
            info_root=info_root,
            cache_dir=metadata_cache_dir,
            target_hz=stored_data_frame_rate,
        )
        scan_start = time.perf_counter()
        super().__init__(
            mp4_paths=processed_paths,
            num_frames=num_frames,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            video_length=video_length,
            size=size,
            aug=aug,
            backend=backend,
        )
        dropped_clip_count, dropped_short_videos = self._prune_clips_without_metadata()
        self.dropped_short_metadata_videos = dropped_short_videos
        if not self.index_to_starting_frame_map:
            raise RuntimeError("No clips remain after pruning those without sufficient metadata coverage")
        if dropped_clip_count:
            extra = (
                f" Removed {len(dropped_short_videos)} entire videos."
                if dropped_short_videos
                else ""
            )
            _log_goal_bdd(
                f"Pruned {dropped_clip_count} clips whose metadata ended before the goal frame.{extra}"
            )
        _log_goal_bdd(
            f"Indexed {len(self.index_to_starting_frame_map)} clips from {len(self.mp4_paths)} videos in "
            f"{time.perf_counter() - scan_start:.1f}s."
        )
        _log_goal_bdd(f"Dataset initialization finished in {time.perf_counter() - init_start:.1f}s.")
        if ctx_frame < 0 or ctx_frame >= num_frames:
            raise ValueError(f"ctx_frame must be within [0, {num_frames - 1}], got {ctx_frame}")
        if direction_window < 1:
            raise ValueError("direction_window must be >= 1")
        self.ctx_frame = ctx_frame
        self.direction_window = direction_window
        self.dropped_videos = dropped

    def _direction_vector(self, metadata: List[FrameMetadata], start_frame: int) -> np.ndarray:
        if not metadata:
            return np.zeros(2, dtype=np.float32)
        last_idx = min(len(metadata) - 1, start_frame + self.direction_window)
        if last_idx <= start_frame:
            return np.zeros(2, dtype=np.float32)
        deltas = []
        for idx in range(start_frame, last_idx):
            current = metadata[idx]
            nxt = metadata[idx + 1]
            deltas.append([nxt.x - current.x, nxt.y - current.y])
        direction = np.mean(deltas, axis=0)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def _heading_matrix(self, metadata: List[FrameMetadata], start_frame: int) -> np.ndarray:
        if not metadata:
            return np.eye(2, dtype=np.float32)
        end_idx = min(len(metadata), start_frame + self.direction_window + 1)
        if end_idx <= start_frame + 1:
            yaw = float(metadata[start_frame].yaw_rad)
        else:
            yaw_samples = np.array([metadata[i].yaw_rad for i in range(start_frame, end_idx)], dtype=np.float64)
            cos_mean = np.mean(np.cos(yaw_samples))
            sin_mean = np.mean(np.sin(yaw_samples))
            yaw = float(np.arctan2(sin_mean, cos_mean))
        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            return np.eye(2, dtype=np.float32)
        heading /= norm
        right = np.array([heading[1], -heading[0]], dtype=np.float32)
        return np.stack([right, heading], axis=0)

    @staticmethod
    def _filter_paths_with_gps(mp4_paths: Sequence[Path], info_root: Path, show_progress: bool = False) -> Tuple[List[Path], List[Path]]:
        valid: List[Path] = []
        dropped: List[Path] = []
        iterator: Sequence[Path] | Iterable[Path] = mp4_paths
        progress = None
        if show_progress:
            progress = tqdm(mp4_paths, desc="Checking GPS metadata", unit="video")
            iterator = progress
        for path in iterator:
            video_id = path.stem
            if _video_has_gps(info_root, video_id):
                valid.append(path)
            else:
                dropped.append(path)
        if progress is not None:
            progress.close()
        return valid, dropped

    def _prune_clips_without_metadata(self) -> Tuple[int, List[str]]:
        kept: List[Tuple[Path, int]] = []
        valid_videos: set[str] = set()
        video_meta_len: Dict[str, int] = {}
        dropped_clips = 0
        for path, start_frame in self.index_to_starting_frame_map:
            video_id = path.stem
            if video_id not in video_meta_len:
                video_meta_len[video_id] = len(self.metadata_cache.get(video_id))
            meta_len = video_meta_len[video_id]
            goal_idx = start_frame + self.num_frames - 1
            if meta_len == 0 or goal_idx >= meta_len:
                dropped_clips += 1
                continue
            kept.append((path, start_frame))
            valid_videos.add(video_id)
        dropped_videos = sorted(set(p.stem for p in self.mp4_paths) - valid_videos)
        if dropped_clips:
            self.index_to_starting_frame_map = kept
            if valid_videos:
                self.mp4_paths = [p for p in self.mp4_paths if p.stem in valid_videos]
            else:
                self.mp4_paths = []
        return dropped_clips, dropped_videos

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frames_tensor, path, start_frame = self._get_clip(idx)
        video_id = path.stem
        metadata = self.metadata_cache.get(video_id)
        ctx_idx = start_frame + self.ctx_frame
        goal_idx = start_frame + self.num_frames - 1
        for frame_idx in (start_frame, ctx_idx, goal_idx):
            if frame_idx >= len(metadata):
                raise IndexError(f"Metadata for video {video_id} is shorter than required frame {frame_idx}")
        start_meta = metadata[start_frame]
        ctx_meta = metadata[ctx_idx]
        goal_meta = metadata[goal_idx]

        origin_xy = np.array([start_meta.x, start_meta.y], dtype=np.float32)
        rotation = self._heading_matrix(metadata, start_frame)
        ctx_delta = np.array([ctx_meta.x, ctx_meta.y], dtype=np.float32) - origin_xy
        goal_delta = np.array([goal_meta.x, goal_meta.y], dtype=np.float32) - origin_xy
        ctx_xy = rotation @ ctx_delta
        goal_xy = rotation @ goal_delta
        relative_goal = goal_xy - ctx_xy
        direction_vec = self._direction_vector(metadata, start_frame)
        direction_vec = rotation @ direction_vec
        sample = {
            "images": frames_tensor,
            "video_id": video_id,
            "start_frame": start_frame,
            "timestamp_ms": torch.tensor(ctx_meta.timestamp_ms, dtype=torch.int64),
            "context_position_xy": torch.from_numpy(ctx_xy),
            "goal_position_xy": torch.from_numpy(goal_xy),
            "relative_goal_xy": torch.from_numpy(relative_goal),
            "context_latlon": torch.tensor([ctx_meta.lat, ctx_meta.lon], dtype=torch.float32),
            "goal_latlon": torch.tensor([goal_meta.lat, goal_meta.lon], dtype=torch.float32),
            "start_latlon": torch.tensor([start_meta.lat, start_meta.lon], dtype=torch.float32),
            "direction": torch.from_numpy(direction_vec),
            "time_of_day_id": ctx_meta.time_of_day_id,
            "time_of_day_label": ctx_meta.time_of_day_label,
            "speed_ms": torch.tensor(ctx_meta.speed_ms, dtype=torch.float32),
            "yaw_rad": torch.tensor(ctx_meta.yaw_rad, dtype=torch.float32),
        }
        return sample
