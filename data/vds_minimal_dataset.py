import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F_nn
from torch.utils.data import Dataset, get_worker_info

from .custom_multiframe import _default_transform_tensor


@dataclass
class _SubsplitInfo:
    name: str
    frames_path: str
    latents_path: Optional[str]
    runs: np.ndarray
    clip_counts: np.ndarray
    cumulative: np.ndarray
    components: Dict[str, str]
    video_ids_path: Optional[str]
    frame_idxs_path: Optional[str]
    source_indices_path: Optional[str]
    run_video_ids: Optional[np.ndarray] = None
    run_source_indices: Optional[np.ndarray] = None

    @property
    def total_clips(self) -> int:
        if self.cumulative.size == 0:
            return 0
        return int(self.cumulative[-1])


class _VideoNameLookup:
    """Lazy video-name lookup that avoids loading all strings into RAM."""

    __slots__ = ("_keys", "_indices", "_names")

    def __init__(self, videos_group: h5py.Group) -> None:
        source_ds = videos_group.get("source_index")
        id_ds = videos_group.get("id")
        name_ds = videos_group.get("name")
        if source_ds is None or id_ds is None or name_ds is None:
            raise KeyError("videos group must contain 'source_index', 'id', and 'name'")

        source_idx = np.asarray(source_ds[...], dtype=np.int64)
        video_ids = np.asarray(id_ds[...], dtype=np.int64)
        if source_idx.shape != video_ids.shape:
            raise ValueError("source_index and id datasets must share the same shape")

        if source_idx.size == 0:
            raise ValueError("videos group does not contain any entries")

        # Build a structured array of keys for lexicographic search while tracking dataset indices.
        key_dtype = np.dtype([("source", np.int64), ("video", np.int64)])
        keys = np.empty(source_idx.shape[0], dtype=key_dtype)
        keys["source"] = source_idx
        keys["video"] = video_ids

        indices = np.arange(source_idx.shape[0], dtype=np.int64)
        order = np.argsort(keys, order=("source", "video"))
        self._keys = keys[order]
        self._indices = indices[order]

        try:
            self._names = name_ds.asstr()
        except AttributeError:
            self._names = name_ds

    def get(self, key: Tuple[int, int], default=None):
        if not isinstance(key, tuple) or len(key) != 2:
            return default
        source_idx, video_id = int(key[0]), int(key[1])
        query = np.zeros((), dtype=self._keys.dtype)
        query["source"] = source_idx
        query["video"] = video_id
        right = int(np.searchsorted(self._keys, query, side="right"))
        pos = right - 1
        if pos < 0:
            return default
        match = self._keys[pos]
        if match["source"] != source_idx or match["video"] != video_id:
            return default
        dataset_idx = int(self._indices[pos])
        try:
            raw = self._names[dataset_idx]
        except Exception:
            return default
        return MinimalVDSDataset._decode_str(raw)


class MinimalVDSDataset(Dataset):
    """Minimal VDS dataset with optional latent support and uniform subsplit sampling."""

    def __init__(
        self,
        *,
        vds_path: str,
        split: str = "train",
        num_frames: int = 8,
        size: Optional[Sequence[int]] = None,
        transform=None,
        return_metadata: bool = False,
        load_frames: bool = True,
        load_latents: bool = False,
        latents_dtype: Union[str, torch.dtype] = "bf16",
        subsplit_names: Optional[Sequence[str]] = None,
        rng_seed: Optional[int] = None,
        frame_rate: float = 5.0,
        h5_cache_nbytes: Optional[int] = None,
        h5_cache_nslots: Optional[int] = None,
        h5_cache_w0: Optional[float] = None,
        resolve_video_name: bool = False,
    ) -> None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not load_frames and not load_latents:
            raise ValueError("At least one of load_frames or load_latents must be True")

        self.vds_path = vds_path
        self.split = split
        self.num_frames = int(num_frames)
        self.transform = transform
        self.return_metadata = return_metadata
        self.load_frames = load_frames
        self.load_latents = load_latents
        self.frame_rate = float(frame_rate)
        self.size = self._normalize_size(size)
        self._target_latent_dtype = (
            self._parse_latent_dtype(latents_dtype) if load_latents else None
        )
        self.resolve_video_name = bool(resolve_video_name)

        self._base_seed = int(rng_seed if rng_seed is not None else random.randint(0, 2**31 - 1))
        self._epoch = 0

        self._h5_open_kwargs = {}
        # Use a larger default raw-data chunk cache to reduce HDF5 thrash on memory-rich nodes.
        # 256 MiB sits well below the available per-worker headroom we observed (~1.5 TiB host RAM).
        default_rdcc_nbytes = 256 * 1024 * 1024
        default_rdcc_nslots = 4096
        default_rdcc_w0 = 0.90
        self._h5_open_kwargs["rdcc_nbytes"] = int(h5_cache_nbytes) if h5_cache_nbytes is not None else default_rdcc_nbytes
        self._h5_open_kwargs["rdcc_nslots"] = int(h5_cache_nslots) if h5_cache_nslots is not None else default_rdcc_nslots
        self._h5_open_kwargs["rdcc_w0"] = float(h5_cache_w0) if h5_cache_w0 is not None else default_rdcc_w0

        self._file: Optional[h5py.File] = None
        self._dataset_cache: Dict[str, h5py.Dataset] = {}

        with self._open_file() as h5_file:
            if split not in h5_file:
                raise KeyError(f"Split '{split}' not found in {vds_path}")
            split_group = h5_file[split]

            self._video_name_lookup = (
                self._load_video_name_lookup(h5_file) if self.resolve_video_name else None
            )

            self._subsplits = self._build_subsplit_map(h5_file, split_group, subsplit_names)
        if not self._subsplits:
            with self._open_file() as h5_file:
                split_group = h5_file[split]
                info = self._build_subsplit_info(h5_file, split_group, name=split, components={})
            if info is None:
                raise ValueError(f"No runs with at least {self.num_frames} frames in split '{split}'")
            self._subsplits = {split: info}

        self._subsplit_names: List[str] = list(self._subsplits.keys())
        self._total_clips = sum(info.total_clips for info in self._subsplits.values())
        if self._total_clips == 0:
            raise ValueError(f"No clips available for split '{split}'")
        # File handle remains closed until first access to avoid duplicating caches across workers.
        self._file = None

    @staticmethod
    def _normalize_size(size: Optional[Sequence[int]]) -> Optional[Sequence[int]]:
        if size is None:
            return None
        if isinstance(size, int):
            return (int(size), int(size))
        if len(size) != 2:
            raise ValueError("size must be None, an int, or a length-2 sequence")
        return (int(size[0]), int(size[1]))

    def _open_file(self) -> h5py.File:
        return h5py.File(self.vds_path, "r", **self._h5_open_kwargs)

    def _ensure_file(self) -> h5py.File:
        if self._file is None:
            self._file = self._open_file()
            self._dataset_cache = {}
        return self._file

    def _close_file_handle(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        self._dataset_cache.clear()

    def _get_dataset(self, h5_file: h5py.File, path: str) -> h5py.Dataset:
        try:
            return self._dataset_cache[path]
        except KeyError:
            dataset = h5_file[path]
            self._dataset_cache[path] = dataset
            return dataset

    def _build_subsplit_map(
        self,
        h5_file: h5py.File,
        split_group: h5py.Group,
        subsplit_names: Optional[Sequence[str]],
    ) -> Dict[str, _SubsplitInfo]:
        subsplit_root = split_group.get("subsplits")
        if subsplit_root is None:
            return {}

        available_names = self._discover_subsplit_names(h5_file, subsplit_root)
        if subsplit_names is not None:
            requested = set(subsplit_names)
            names = [name for name in available_names if name in requested]
            if not names:
                raise ValueError(
                    f"Requested subsplits {sorted(requested)} not available in split '{self.split}'"
                )
        else:
            names = available_names

        subsplits: Dict[str, _SubsplitInfo] = {}
        for name in names:
            if name not in subsplit_root:
                continue
            components = self._parse_subsplit_components(name)
            info = self._build_subsplit_info(h5_file, subsplit_root[name], name=name, components=components)
            if info is not None and info.total_clips > 0:
                subsplits[name] = info
        return subsplits

    def _discover_subsplit_names(self, h5_file: h5py.File, subsplit_root: h5py.Group) -> List[str]:
        options_group = h5_file.get("subsplit_options")
        names: List[str] = []
        if options_group is not None and self.split in options_group:
            raw = options_group[self.split][...]
            if raw.size > 0:
                names = [self._decode_str(item) for item in raw]
        if not names:
            names = list(subsplit_root.keys())
        return names

    def _load_video_name_lookup(self, h5_file: h5py.File) -> Optional[_VideoNameLookup]:
        videos_group = h5_file.get("videos")
        if videos_group is None:
            return None
        try:
            return _VideoNameLookup(videos_group)
        except Exception:
            return None

    @staticmethod
    def _decode_str(value) -> str:
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _parse_subsplit_components(name: str) -> Dict[str, str]:
        components: Dict[str, str] = {}
        for part in name.split("/"):
            if not part:
                continue
            if "=" in part:
                key, val = part.split("=", 1)
                components[key] = val
        return components

    def _build_subsplit_info(
        self,
        h5_file: h5py.File,
        group: h5py.Group,
        *,
        name: str,
        components: Dict[str, str],
    ) -> Optional[_SubsplitInfo]:
        if "frames" not in group or "index" not in group:
            return None
        frames = group["frames"]
        index_group = group["index"]
        if "contiguous_runs" not in index_group:
            return None
        runs = np.asarray(index_group["contiguous_runs"][...], dtype=np.int64)
        if runs.ndim != 2 or runs.shape[1] < 2:
            return None

        starts = runs[:, 0].astype(np.int64)
        lengths = runs[:, 1].astype(np.int64)
        valid_mask = lengths >= self.num_frames
        if not np.any(valid_mask):
            return None

        valid_starts = starts[valid_mask]
        valid_lengths = lengths[valid_mask]
        clip_counts = (valid_lengths - self.num_frames + 1).astype(np.int64)
        positive_mask = clip_counts > 0
        if not np.any(positive_mask):
            return None

        clip_counts = clip_counts[positive_mask]
        valid_starts = valid_starts[positive_mask]
        valid_lengths = valid_lengths[positive_mask]

        cumulative = np.cumsum(clip_counts)
        valid_runs = np.stack((valid_starts, valid_lengths), axis=1)

        latents = None
        if self.load_latents:
            if "latents" not in group:
                raise KeyError(
                    f"Latents dataset missing in '{name}' for split '{self.split}' in {self.vds_path}"
                )
            latents = group["latents"]
            if latents.shape[0] != frames.shape[0]:
                raise ValueError(
                    f"Latents length {latents.shape[0]} does not match frames length {frames.shape[0]}"
                )

        video_ids = index_group.get("video_id")
        frame_idxs = index_group.get("frame_idx")
        source_indices = index_group.get("source_index")

        run_video_ids = None
        run_source_indices = None
        if self.resolve_video_name:
            if source_indices is not None:
                # Cache per-run metadata so we can avoid random single-element HDF5 reads later.
                run_source_indices = np.asarray(source_indices[valid_starts], dtype=np.int64)
                run_source_indices = run_source_indices[positive_mask]
            if video_ids is not None:
                run_video_ids = np.asarray(video_ids[valid_starts], dtype=np.int64)
                run_video_ids = run_video_ids[positive_mask]

        return _SubsplitInfo(
            name=name,
            frames_path=frames.name,
            latents_path=latents.name if latents is not None else None,
            runs=valid_runs,
            clip_counts=clip_counts,
            cumulative=cumulative,
            components=components,
            video_ids_path=video_ids.name if video_ids is not None else None,
            frame_idxs_path=frame_idxs.name if frame_idxs is not None else None,
            source_indices_path=source_indices.name if source_indices is not None else None,
            run_video_ids=run_video_ids,
            run_source_indices=run_source_indices,
        )

    @property
    def available_subsplits(self) -> List[str]:
        return list(self._subsplit_names)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._total_clips

    def _make_rng(self, index: int) -> random.Random:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = (
            self._base_seed
            + int(index)
            + worker_id * 1_000_003
            + self._epoch * 1_000_000_007
        ) & 0xFFFFFFFF
        return random.Random(seed)

    def _sample_from_subsplit(self, rng: random.Random) -> _SubsplitInfo:
        name = rng.choice(self._subsplit_names)
        return self._subsplits[name]

    def __getitem__(self, index: int):
        rng = self._make_rng(index)
        info = self._sample_from_subsplit(rng)
        total_clips = info.total_clips
        if total_clips <= 0:
            raise RuntimeError(f"No clips available in subsplit '{info.name}'")

        clip_offset = rng.randrange(total_clips)
        run_idx = int(np.searchsorted(info.cumulative, clip_offset, side="right"))
        prev = int(info.cumulative[run_idx - 1]) if run_idx > 0 else 0
        offset_in_run = clip_offset - prev

        run = info.runs[run_idx]
        start = int(run[0] + offset_in_run)
        end = start + self.num_frames

        h5_file = self._ensure_file()

        sample = {}
        if self.load_frames:
            frames_np = self._get_dataset(h5_file, info.frames_path)[start:end]
            clip = _default_transform_tensor(frames_np)
            if self.size is not None and clip.shape[-2:] != self.size:
                clip = F_nn.interpolate(clip, size=self.size, mode="bilinear", align_corners=False)
            if self.transform is not None:
                clip = self.transform(clip)
            sample["images"] = clip

        if self.load_latents and info.latents_path is not None:
            latents_np = self._get_dataset(h5_file, info.latents_path)[start:end]
            latents = torch.from_numpy(latents_np)
            if self._target_latent_dtype is not None:
                latents = latents.to(self._target_latent_dtype)
            sample["latents"] = latents

        sample["frame_rate"] = self.frame_rate
        labels = {"split": self.split, "subsplit": info.name}
        if info.components:
            labels.update(info.components)
        if self.resolve_video_name:
            video_name = self._lookup_video_name(h5_file, info, start, run_idx)
            if video_name is not None:
                labels["video_name"] = video_name
        sample["labels"] = labels

        if self.return_metadata:
            if info.video_ids_path is not None:
                video_ids = np.asarray(self._get_dataset(h5_file, info.video_ids_path)[start:end]).astype(
                    np.int64, copy=False
                )
                sample["video_id"] = torch.from_numpy(video_ids)
            if info.frame_idxs_path is not None:
                frame_idxs = np.asarray(self._get_dataset(h5_file, info.frame_idxs_path)[start:end]).astype(
                    np.int64, copy=False
                )
                sample["frame_idx"] = torch.from_numpy(frame_idxs)

        return sample

    @staticmethod
    def _parse_latent_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(value, torch.dtype):
            if value not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError("latents_dtype must be bf16, float16, or float32")
            return value
        if isinstance(value, str):
            key = value.lower()
            if key == "bf16":
                return torch.bfloat16
            if key in {"fp16", "float16", "half", "f16"}:
                return torch.float16
            if key in {"fp32", "float32", "single", "f32"}:
                return torch.float32
        raise ValueError(f"Unsupported latents_dtype: {value}")

    def _lookup_video_name(
        self,
        h5_file: h5py.File,
        info: _SubsplitInfo,
        frame_index: int,
        run_idx: Optional[int] = None,
    ) -> Optional[str]:
        if (
            self._video_name_lookup is None
            or info.source_indices_path is None
            or info.video_ids_path is None
        ):
            return None
        if not self.resolve_video_name:
            return None
        if run_idx is not None and info.run_source_indices is not None and info.run_video_ids is not None:
            try:
                source_idx = int(info.run_source_indices[run_idx])
                video_id = int(info.run_video_ids[run_idx])
            except (IndexError, ValueError, TypeError):
                pass
            else:
                return self._video_name_lookup.get((source_idx, video_id))
        try:
            source_idx = int(
                np.asarray(self._get_dataset(h5_file, info.source_indices_path)[frame_index]).item()
            )
        except Exception:
            return None
        try:
            video_id = int(
                np.asarray(self._get_dataset(h5_file, info.video_ids_path)[frame_index]).item()
            )
        except Exception:
            return None
        return self._video_name_lookup.get((source_idx, video_id))

    def close(self) -> None:
        self._close_file_handle()
        self._subsplits = {}
        self._subsplit_names = []
        self._video_name_lookup = None

    def __del__(self):
        self.close()


class MinimalVDSLatentDataset(MinimalVDSDataset):
    def __init__(self, **kwargs) -> None:
        defaults = {"load_frames": True, "load_latents": True}
        defaults.update(kwargs)
        super().__init__(**defaults)
