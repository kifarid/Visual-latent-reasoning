import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from data.mnist_sudoku import MnistSudokuGridDataset


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor in [-1,1] range to disk as PNG."""
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0  # [0,1]
    pil_img = to_pil_image(tensor)
    pil_img.save(path, format="PNG")


def main():
    parser = argparse.ArgumentParser(description="Visualize MnistSudokuGridDataset samples.")
    parser.add_argument("--root", type=str, default="datasets/mnist_sudoku", help="Path containing sudokus.npy and top_5000_values.csv")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="outputs/mnist_sudoku_samples", help="Directory to save sample grids")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to save")
    parser.add_argument("--top_n", type=int, default=1000, help="Top-N digits per class to use from ranking CSV")
    parser.add_argument("--test_samples_num", type=int, default=10_000, help="Tail size reserved for val/test splits")
    parser.add_argument("--subset_size", type=int, default=None, help="Optional cap on number of puzzles to load")
    parser.add_argument("--no_rgb", action="store_true", help="Keep grids grayscale instead of converting to RGB")
    parser.add_argument("--no_download", action="store_true", help="Disable MNIST download (will fail if data missing)")
    parser.add_argument(
        "--allow_synthetic_if_missing",
        action="store_true",
        help="Allow synthetic grids/digit ranking if sudoku assets are missing",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MnistSudokuGridDataset(
        root=args.root,
        split=args.split,
        top_n=args.top_n,
        test_samples_num=args.test_samples_num,
        subset_size=args.subset_size,
        to_rgb=not args.no_rgb,
        download=not args.no_download,
        allow_synthetic_if_missing=args.allow_synthetic_if_missing,
    )

    num = min(args.num_samples, len(dataset))
    print(f"Saving {num} samples from split={args.split} to {out_dir}")

    for i in range(num):
        sample = dataset[i]
        img = sample["images"]
        save_tensor_image(img, out_dir / f"{args.split}_{i:04d}.png")

    print("Done.")


if __name__ == "__main__":
    main()
