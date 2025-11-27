import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Tuple, Literal, Optional
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image


class MnistSudokuGridDataset(Dataset):
    """
    Lazy MNIST Sudoku grid builder mirroring SRM's mnist_sudoku_lazy dataset.
    - Loads the top-N confident MNIST digits per class (from top_5000_values.csv).
    - Uses Sudoku solutions from sudokus.npy to assemble 9x9 grids (28x28 per cell → 252x252).
    - Train split uses all but the last `test_samples_num` puzzles; val/test use the tail.
    - Returns a float tensor in [-1, 1] under the `images` key (C,H,W).
    """

    Split = Literal["train", "val", "test"]

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        top_n: int = 1000,
        test_samples_num: int = 10_000,
        subset_size: Optional[int] = None,
        image_size: Sequence[int] | None = (252, 252),
        to_rgb: bool = True,
        download: bool = True,
        allow_synthetic_if_missing: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.top_n = top_n
        self.test_samples_num = test_samples_num
        self.subset_size = subset_size
        self.image_size = tuple(image_size) if image_size is not None else None
        self.to_rgb = to_rgb
        self.allow_synthetic_if_missing = allow_synthetic_if_missing

        # Load MNIST digits and pre-select the top-N per label using the ranking CSV.
        self.mnist_images = self._load_top_mnist_digits(download=download)

        # Load sudoku solutions and slice according to split.
        sudoku_path = self.root / "sudokus.npy"
        if not sudoku_path.exists():
            if not self.allow_synthetic_if_missing:
                raise FileNotFoundError(f"Missing sudoku solutions at {sudoku_path}")
            fallback_len = self.subset_size if self.subset_size is not None else 128
            warnings.warn(
                f"Missing sudoku solutions at {sudoku_path}; generating {fallback_len} synthetic grids."
            )
            sudoku_grids = np.random.randint(0, 10, size=(fallback_len, 9, 9), dtype=np.int64)
        else:
            sudoku_grids = np.load(sudoku_path)
        if split == "train":
            if test_samples_num > 0:
                sudoku_grids = sudoku_grids[: -test_samples_num]
        else:
            if test_samples_num > 0:
                sudoku_grids = sudoku_grids[-test_samples_num:]

        if subset_size is not None:
            sudoku_grids = sudoku_grids[:subset_size]

        self.sudoku_grids = sudoku_grids

        resize = [transforms.Resize(self.image_size)] if self.image_size is not None else []
        self.transform = transforms.Compose(
            resize
            + [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2.0 - 1.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.sudoku_grids)

    def __repr__(self) -> str:
        return (
            f"MnistSudokuGridDataset(split={self.split}, "
            f"top_n={self.top_n}, len={len(self)})"
        )

    def _load_top_mnist_digits(self, download: bool) -> torch.Tensor:
        ranking_csv = self.root / "top_5000_values.csv"
        mnist_dataset = MNIST(root=self.root, train=True, download=download)

        if not ranking_csv.exists():
            if not self.allow_synthetic_if_missing:
                raise FileNotFoundError(f"Missing ranking CSV at {ranking_csv}")
            warnings.warn(
                f"Missing ranking CSV at {ranking_csv}; using first {self.top_n} samples per digit as fallback."
            )
            labels = mnist_dataset.targets
            selected = []
            for digit in range(10):
                digit_indices = (labels == digit).nonzero(as_tuple=True)[0][: self.top_n]
                digit_images = mnist_dataset.data[digit_indices]
                selected.append(digit_images)
            return torch.stack(selected, dim=0)

        top_df = pd.read_csv(ranking_csv)
        labels = mnist_dataset.targets
        selected = []
        for digit in range(10):
            digit_df = top_df[top_df.label == digit].sort_values(
                by="confidence", ascending=False
            )
            digit_df = digit_df.iloc[: self.top_n]
            indices = digit_df["image_index"].to_numpy()
            digit_images = mnist_dataset.data[labels == digit][indices]
            if len(digit_images) != self.top_n:
                raise ValueError(
                    f"Expected {self.top_n} images for digit {digit}, got {len(digit_images)}"
                )
            selected.append(digit_images)
        selected = torch.stack(selected, dim=0)
        # Shape: (10, top_n, 28, 28)
        return selected

    def _sample_digit(self, digit_label: int, rng: np.random.Generator) -> torch.Tensor:
        idx = rng.integers(0, self.mnist_images.shape[1])
        return self.mnist_images[digit_label, idx]

    def _build_grid_image(self, grid: np.ndarray, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(idx) if self.split != "train" else np.random.default_rng()
        full_image = torch.empty((252, 252), dtype=torch.uint8)
        for r in range(9):
            for c in range(9):
                digit_label = int(grid[r, c])
                digit_img = self._sample_digit(digit_label, rng)
                top, left = r * 28, c * 28
                full_image[top : top + 28, left : left + 28] = digit_img
        return full_image

    def __getitem__(self, idx: int) -> dict:
        grid = self.sudoku_grids[idx]
        image = self._build_grid_image(grid, idx)
        # Convert to PIL for torchvision transforms.
        pil_img = to_pil_image(image)
        if self.to_rgb:
            pil_img = pil_img.convert("RGB")
        tensor_img = self.transform(pil_img)
        return {"images": tensor_img}
