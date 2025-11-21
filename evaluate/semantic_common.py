# stdlib
import argparse
import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

# third-party
import torch
import cv2  # noqa: F401  # ensure OpenCV/LibJPEG are loaded before Albumentations
import albumentations as A
import imageio.v2 as imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from torchvision.datasets import Cityscapes


IGNORE_INDEX: int = 255

# Cityscapes IDs: https://www.cityscapes-dataset.com/dataset-overview/#fine-annotation-labels
# We keep IGNORE_INDEX as 255 and only map train IDs for valid classes (19 classes).
VALID_LABEL_IDS: List[int] = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

CLASS_NAMES: List[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
N_CLASSES: int = len(VALID_LABEL_IDS)

PALETTE: np.ndarray = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def mkdir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_id_mapping() -> torch.Tensor:
    lut = torch.full((256,), IGNORE_INDEX, dtype=torch.long)
    for train_id, label_id in enumerate(VALID_LABEL_IDS):
        lut[label_id] = train_id
    return lut


ID_LUT: torch.Tensor = build_id_mapping()


def encode_segmap(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.long:
        mask = mask.long()
    lut = ID_LUT.to(mask.device)
    mask = mask.clamp_min(0).clamp_max(255)
    return lut[mask]


def colorize(labels: torch.Tensor) -> np.ndarray:
    arr = labels.detach().cpu().numpy()
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = arr != IGNORE_INDEX
    rgb[valid] = PALETTE[arr[valid]]
    return rgb


class CityScapesDataset(Cityscapes):
    """Cityscapes wrapper with Albumentations transform producing tensors."""

    def __init__(self, data_path: str, *, size: Tuple[int, int], **kwargs: Any) -> None:
        super().__init__(data_path, **kwargs)
        self.transform = A.Compose(
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")

        targets: List[Any] = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        transformed = self.transform(image=np.array(image), mask=np.array(target))
        img_t: torch.Tensor = transformed["image"]
        mask_t: torch.Tensor = transformed["mask"]
        return img_t, torch.as_tensor(mask_t)


def build_cityscapes_dataset(
    data_path: str,
    *,
    split: str,
    size: Tuple[int, int],
    mode: str = "fine",
    target_type: str | Sequence[str] = "semantic",
) -> CityScapesDataset:
    return CityScapesDataset(
        data_path,
        split=split,
        mode=mode,
        target_type=target_type,
        size=size,
    )


class SegProbe(nn.Module):
    """1×1 conv classifier head for dense features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.LazyConv2d(num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        feats: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        logits_lr = self.classifier(feats)
        if target_size is None:
            return logits_lr
        return F.interpolate(logits_lr, size=target_size, mode="bilinear", align_corners=False)


FeatureStep = Callable[[Tuple[torch.Tensor, ...], SegProbe, bool], Tuple[torch.Tensor, torch.Tensor]]


def train_linear_probe(
    args: argparse.Namespace,
    *,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    device: torch.device,
    feature_step: FeatureStep,
) -> None:
    linear_probe = SegProbe(N_CLASSES).to(device)
    opt = optim.AdamW(linear_probe.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    perclass_metric = JaccardIndex(
        task="multiclass",
        num_classes=N_CLASSES,
        ignore_index=IGNORE_INDEX,
        average="none",
    ).to(device)

    miou_metric = JaccardIndex(
        task="multiclass",
        num_classes=N_CLASSES,
        ignore_index=IGNORE_INDEX,
        average="macro",
    ).to(device)

    def run(loader: DataLoader, train: bool) -> float:
        total_loss, total_count = 0.0, 0
        linear_probe.train(train)

        perclass_metric.reset()
        miou_metric.reset()

        no_grad_ctx = torch.enable_grad() if train else torch.inference_mode()
        with no_grad_ctx:
            idx = 0
            for batch in loader:
                logits, labels = feature_step(batch, linear_probe, train)
                loss = crit(logits, labels)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size

                if not train:
                    miou_metric.update(logits, labels)
                    perclass_metric.update(logits, labels)

                    if args.dump_vis:
                        preds = logits.argmax(dim=1)
                        for b in range(batch_size):
                            gt_rgb = colorize(labels[b])
                            pd_rgb = colorize(preds[b])
                            imageio.imwrite(f"{args.seq_real}/gt_{idx:05d}.png", gt_rgb)
                            imageio.imwrite(f"{args.seq_fake}/pred_{idx:05d}.png", pd_rgb)
                            idx += 1

        if not train:
            miou = float(miou_metric.compute().item())
            ious: List[float] = list(map(float, perclass_metric.compute().detach().cpu().tolist()))
            print(f"[eval] mIoU: {miou:.4f}")
            print(f"[eval] per-class IoU: {ious}")

        return total_loss / max(total_count, 1)

    for ep in range(1, args.num_epoch + 1):
        train_loss = run(dataloader_train, train=True)
        should_eval = ep % args.eval_every == 0 or ep == args.num_epoch
        val_loss = run(dataloader_val, train=False) if should_eval else float("nan")
        if np.isnan(val_loss):
            print(f"epoch {ep:03d}: train {train_loss:.3f}")
        else:
            print(f"epoch {ep:03d}: train {train_loss:.3f}  val {val_loss:.3f}")


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in {"yes", "true", "t", "y", "1"}:
        return True
    if val in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_shared_arguments(
    parser: argparse.ArgumentParser,
    *,
    data_path_default: str,
    num_epoch_default: int,
    batch_size_default: int,
    eval_every_default: int,
) -> argparse.ArgumentParser:
    parser.add_argument("--exp_dir", type=str, required=True, help="Root experiment directory")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/last.ckpt",
        help="Checkpoint path (relative to exp_dir)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Model config (relative to exp_dir)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_path_default,
        help="Path to the Cityscapes dataset root",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="vis_semantic",
        help="Where to dump visualizations (relative to exp_dir)",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=num_epoch_default,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size_default,
        help="Global batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=eval_every_default,
        help="Evaluate every N epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (<=0 disables repeatability)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="cpu | cuda",
    )
    parser.add_argument(
        "--dump_vis",
        type=str2bool,
        default=True,
        help="Dump colored GT/pred PNGs during eval",
    )
    return parser


__all__ = [
    "IGNORE_INDEX",
    "N_CLASSES",
    "SegProbe",
    "CityScapesDataset",
    "build_cityscapes_dataset",
    "encode_segmap",
    "colorize",
    "mkdir",
    "train_linear_probe",
    "str2bool",
    "add_shared_arguments",
]
