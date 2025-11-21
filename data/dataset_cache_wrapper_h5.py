# orbis/data/dataset_cache_wrapper_h5.py
from __future__ import annotations
import json, os, hashlib
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

from data.latent_cache.utils import model_signature

import torch
from PIL import Image
import h5py
import numpy as np
from filelock import FileLock
import re

PathLike = Union[str, Path]
CacheMode = Literal["off", "build", "read"]
TDS = TypeVar("TDS")

def _sha1_bytes(x: bytes) -> str:
    h = hashlib.sha1(); h.update(x); return h.hexdigest()

def _atomic_append_line(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try: os.write(fd, text.encode("utf-8"))
    finally: os.close(fd)


def cacheable_dataset(base_cls: Type[TDS]) -> Type[TDS]:
    class CachedDataset(base_cls):  # type: ignore[misc]
        def __init__(
            self,
            *args: Any,
            caching_path: Optional[PathLike] = None,
            cache_mode: CacheMode = "off",
            safe_writes: bool = True,
            index_manifest_name: str = "index_manifest.jsonl",
            split: str = "unspecified",
            model_cfg: Dict = None,
            model_cfg_path: str = None,
            **kwargs: Any,
        ) -> None:
            self._mode: CacheMode = cache_mode
            self._safe_writes = bool(safe_writes)
            self._split = split
            self._cache_root: Optional[Path] = Path(caching_path) if caching_path else None
            self._manifest_name = index_manifest_name
            self.model_cfg = model_cfg

            assert model_cfg or model_cfg_path, "Either model_cfg or model_cfg_path must be provided."

            # Normalize model_cfg: allow passing a JSON file path; default to {}
            if isinstance(model_cfg_path, (str, _Path)):
                try:
                    with open(str(model_cfg_path), "r") as _f:
                        model_cfg = json.load(_f)
                except Exception:
                    model_cfg = {}
            self.model_cfg = model_cfg if model_cfg is not None else {}

            self._model_patch_size: int = int(self.model_cfg.get("model", {}).get("patch_size", 14))
            self._model_signature: str = model_signature(None, self.model_cfg)
            self._safe_sig = re.sub(r'[\\/:\*\?"<>\|\s]+', '_', self._model_signature).strip('_')[:120]
            m =  self.model_cfg.get("model", {})
            se = m.get("shortest_edge")
            self.model_shortest_edge = {"shortest_edge": int(se)} if se else None
            cs = m.get("crop_size")
            self.model_crop_size = int(cs) if cs else None



            super().__init__(*args, **kwargs)

            self._size: Tuple[int, int] = tuple(getattr(self, "size", (0, 0)))  # (H,W)
            self._num_frames: int = int(getattr(self, "num_frames", 0))
            self._frame_interval: int = int(getattr(self, "frame_interval", 1))
            self._transform_name: str = getattr(self, "aug", "resize_center") if hasattr(self, "aug") else "resize_center"
            self._ds_version: str = getattr(base_cls, "__version__", "0")

            if self._mode in ("build", "read"):
                if not self._cache_root:
                    raise ValueError("caching_path must be provided when cache_mode is 'build' or 'read'")
                self._cache_root = self._cache_root / self._split
                self._cache_root.mkdir(parents=True, exist_ok=True)
                self._manifest_path = self._cache_root / self._manifest_name
                self._latents_root: Path = self._cache_root / "latents"
                self._latents_root.mkdir(parents=True, exist_ok=True)
            else:
                self._manifest_path = None  # type: ignore[assignment]

            self._seq_lengths: Dict[Tuple[str, str], int] = {}
            self._mapping: List[Tuple[str, str, int]] = []
            self._mapping = self._enumerate_mapping_from_hdf5()

            if self._mode in ("build", "read"):
                self._length = len(self._mapping)

        @property
        def building(self) -> bool: return self._mode == "build"

        def index_of(self, idx: int) -> Dict[str, Any]:
            if self._mode == "build":
                h5, key, start = self._mapping[idx]
                return {
                    "h5_path": h5, "key": key, "start_frame": start, "frame_interval": self._frame_interval,
                    "video_length": int(self._seq_lengths.get((h5, key), 0)),
                    "latent_h5_path": str(self._latent_h5_path_for(h5)), "latent_key": key
                }
            if self._mode == "read":
                #use the get indices function
                h5_index, key, indicies = super().get_indices()
                files = getattr(self, "files", None)
                h5 = files[h5_index].filename if hasattr(files[h5_index], "filename") else files[h5_index].name
                start = indicies[0] if indicies else 0
                return {
                    "h5_path": h5, "key": key, "start_frame": start, "frame_interval": self._frame_interval,
                    "video_length": int(self._seq_lengths.get((h5, key), 0)),
                    "latent_h5_path": str(self._latent_h5_path_for(h5)), "latent_key": key
                }
            raise RuntimeError("index_of() requires build/read mode.")


        def _enumerate_mapping_from_hdf5(self) -> List[Tuple[str, str, int]]:
            files = getattr(self, "files", None)
            file_keys = getattr(self, "file_keys", None)
            lengths = getattr(self, "lengths", None)
            if files is None or file_keys is None or lengths is None:
                raise RuntimeError("Base dataset must expose files/file_keys/lengths.")

            mapping: List[Tuple[str, str, int]] = []
            for f_idx, file in enumerate(files):
                if file is None: continue
                h5_path = getattr(file, "filename", "") or getattr(file, "name", "")
                for key in file_keys[f_idx]:
                    if "meta_data" in key: continue
                    vlen = int(lengths[f_idx][key])
                    self._seq_lengths[(h5_path, key)] = vlen
                    max_start = vlen - self._num_frames * self._frame_interval
                    if max_start < 0: continue
                    for start in range(0, max_start + 1):
                        mapping.append((h5_path, key, start))
            return mapping
        
        def _latent_h5_path_for(self, src: str) -> Path:
            """
            Put latents under caching_path/<split>/latents/<source_h5_path>.latents.<safe_sig>.h5
            """
            p = Path(src)
            if not self._cache_root:
                raise RuntimeError("Cache root not set. Use caching_path in constructor.")
            
            dataset_name = p.parent.name
            os.makedirs(self._latents_root / dataset_name, exist_ok=True)
            return self._latents_root / dataset_name / f"{p.stem}.latents.{self._safe_sig}.h5"

        def _infer_source_from_latent_path(self, latent_path: Path) -> str:
            name = latent_path.name
            stem, suffix = os.path.splitext(name)
            marker = f".latents.{self._safe_sig}"
            if stem.endswith(marker):
                stem = stem[: -len(marker)]
            return str(latent_path.with_name(stem + suffix))

        def _ensure_latent_dataset(
            self,
            hf: h5py.File,
            key: str,
            N: int,
            C: int,
            H: int,
            W: int,
            np_dtype: np.dtype,
            *,
            source_h5_path: Optional[str] = None
        ) -> h5py.Dataset:
            created = False
            if key in hf:
                ds = hf[key]
                if (tuple(ds.shape) != (N, C, H, W)) or (ds.dtype != np_dtype):
                    del hf[key]
                    ds = hf.create_dataset(
                        key, shape=(N, C, H, W), dtype=np_dtype,
                        chunks=(64, C, H, W), compression="lzf"
                    )
                    created = True
            else:
                ds = hf.create_dataset(
                    key, shape=(N, C, H, W), dtype=np_dtype,
                    chunks=(64, C, H, W), compression="lzf"
                )
                created = True
            # Attach per-dataset metadata on first creation
            if created:
                if source_h5_path is not None:
                    ds.attrs["source_h5_path"] = source_h5_path
                ds.attrs["seq_length"] = int(N)
            return ds

        def _written_key(self, key: str) -> str: return f"{key}_written"

        def _ensure_written_ds(self, hf: h5py.File, key: str, N: int) -> h5py.Dataset:
            wkey = self._written_key(key)
            if wkey in hf:
                ds = hf[wkey]
                if (tuple(ds.shape) != (N,)) or (ds.dtype != np.bool_):
                    del hf[wkey]
                    ds = hf.create_dataset(wkey, shape=(N,), dtype=np.bool_, chunks=(1024,), compression="lzf")
                return ds
            return hf.create_dataset(wkey, shape=(N,), dtype=np.bool_, chunks=(1024,), compression="lzf")

        def _mark_written(self, hf: h5py.File, key: str, start: int, T: int) -> None:
            wds = self._ensure_written_ds(hf, key, int(hf[key].shape[0])); wds[start:start+T] = True

        def write_latents(self, latent_h5_path: str, key: str, start: int, tensor: torch.Tensor) -> None:
            path = Path(latent_h5_path)
            # Best-effort N (prefer precomputed)

            # Determine true source path and sequence length
            source_h5 = None
            true_len = None
            # If file exists and dataset has attrs, trust them
            if path.exists():
                try:
                    with h5py.File(path, "r", libver="latest", swmr=True) as _hf:
                        if key in _hf:
                            _ds = _hf[key]
                            source_h5 = _ds.attrs.get("source_h5_path", None)
                            true_len = _ds.attrs.get("seq_length", None)
                except Exception:
                    pass
            # Otherwise infer and look up seq length from in-memory index
            if source_h5 is None:
                source_h5 = self._infer_source_from_latent_path(path)
            if true_len is None:
                true_len = self._seq_lengths.get((source_h5, key))
            # Last resort: make it just large enough for this write
            if true_len is None:
                true_len = start + int(tensor.shape[0])


            C, H, W = map(int, tensor.shape[1:4])
            np_dtype = np.float16 if tensor.dtype == torch.float16 else np.float32
            lock = FileLock(str(path) + ".lock")
            with lock:
                with h5py.File(path, "a", libver="latest") as hf:
                    if not hf.attrs.get("cache_target"):
                        hf.attrs["cache_target"] = "latents"
                        hf.attrs["model_signature"] = self._model_signature
                        hf.attrs["ds_class"] = base_cls.__name__
                        hf.attrs["ds_version"] = self._ds_version
                        hf.attrs["size"] = json.dumps(list(self._size))
                        hf.attrs["num_frames"] = self._num_frames
                        hf.attrs["frame_interval"] = self._frame_interval
                        hf.attrs["transform"] = self._transform_name
                        hf.attrs["model_crop_size"] = self._model_crop_size
                        hf.attrs["model_shortest_edge"] = self._model_shortest_edge
                        hf.attrs["model_patch_size"] = self._model_patch_size
                    
                    ds = self._ensure_latent_dataset(
                        hf, key, int(true_len), C, H, W, np_dtype, source_h5_path=source_h5
                    )
                    self._mark_written(hf, key, start, int(tensor.shape[0]))
                    hf.flush()

        def _materialize_from_hdf5(self, idx: int) -> torch.Tensor:
            h5_path, key, start = self._mapping[idx]
            return self._extract_frames(h5_path, key, start)

        def _extract_frames(self, h5_path, key, start):
            transform = getattr(self, "transform", None)
            if transform is None: raise RuntimeError("Base dataset missing transform(img).")
            frames: List[torch.Tensor] = []
            with h5py.File(h5_path, "r") as hf: # rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000
                for i in range(self._num_frames):
                    fidx = start + i * self._frame_interval
                    img = Image.fromarray(hf[key][fidx])
                    frames.append(transform(img))
            return torch.stack(frames, dim=0) * 2 - 1

        def _append_manifest_line(self, record: Dict[str, Any]) -> None:
            if self._manifest_path: _atomic_append_line(json.dumps(record) + "\n", self._manifest_path)

        def __len__(self) -> int:  # type: ignore[override]
            if self._mode == "build": return self._length
            return super().__len__()  # type: ignore[misc]

        def __getitem__(self, idx: int):
            if self._mode == "off": return super().__getitem__(idx)  # type: ignore[misc]

            identifiers = self.index_of(idx)
            h5_path = identifiers["h5_path"]; key = identifiers["key"]
            start = identifiers["start_frame"]; stop = start + self._num_frames * self._frame_interval
            latent_h5_path = self._latent_h5_path_for(h5_path)

            cached = False
            if Path(latent_h5_path).exists():
                try:
                    with h5py.File(latent_h5_path, "r", libver="latest", swmr=True) as hf:
                        wkey = self._written_key(key)
                        if key in hf and wkey in hf:
                            w = hf[wkey][start:stop:self._frame_interval]
                            need = max(0, (stop - start + self._frame_interval - 1)//self._frame_interval)
                            cached = (w.size == need) and bool(np.all(w))
                except Exception:
                    cached = False

            if self._mode == "build":
                if cached:
                    return {
                        "cached": True,
                        "latent_h5_path": str(latent_h5_path),
                        "latent_key": key,
                        "latent_slice": (start, stop),
                        "dtype": "float16",
                        "shape": [self._num_frames, *list(self._size)],
                        "identifiers": identifiers,
                        "idx": idx
                    }
                frames = self._materialize_from_hdf5(idx)
                T, C, H, W = frames.shape
                return {
                    "cached": False,
                    "latent_h5_path": str(latent_h5_path),
                    "latent_key": key,
                    "latent_slice": (start, stop),
                    "dtype": "float16",
                    "shape": [T, C, H, W],
                    "frames": frames,
                    "identifiers": identifiers,
                    "idx": idx
                }

            if self._mode == "read":
                if cached:
                    with h5py.File(latent_h5_path, "r", libver="latest", swmr=True) as hf:
                        ds = hf[key]
                        sl = slice(start, stop, self._frame_interval)
                        arr = ds[sl]
                        t = torch.from_numpy(arr)
                    base_out = self._extract_frames(h5_path, key, start)
                    return {
                        "images": base_out,
                        "latents": t
                    }
                raise FileNotFoundError(f"Cache miss in read mode for idx={idx}: {latent_h5_path} (build latents first)")

    CachedDataset.__name__ = f"{base_cls.__name__}Cached"
    return cast(Type[TDS], CachedDataset)

