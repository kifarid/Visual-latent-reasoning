import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
        mp4_paths: Sequence[Path],
        num_frames: int,
        stored_data_frame_rate: int = 20,
        frame_rate: int = 5,
        video_length: int = None,
        size: int | Tuple[int, int] = 256,
        aug: str = "resize_center",
        backend: Optional[str] = None,
    ) -> None:
        self.mp4_paths = [Path(os.path.expandvars(str(p))) for p in mp4_paths]
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
