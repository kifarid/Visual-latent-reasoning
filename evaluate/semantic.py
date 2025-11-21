# stdlib
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

# third-party
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from util import instantiate_from_config  # noqa: E402

from evaluate.semantic_common import (  # noqa: E402
    add_shared_arguments,
    build_cityscapes_dataset,
    encode_segmap,
    mkdir,
    train_linear_probe,
)

STAGE1_RESOLUTION: Tuple[int, int] = (192, 336)


def calculate_semantic(args: argparse.Namespace, unknown_args: Sequence[str]) -> None:
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    device = torch.device(args.device)

    cfg_model = OmegaConf.load(args.config)
    cfg_model = OmegaConf.merge(cfg_model, OmegaConf.from_dotlist(list(unknown_args)))
    model = instantiate_from_config(cfg_model.model)
    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset_train = build_cityscapes_dataset(
        args.data_path,
        split="train",
        size=STAGE1_RESOLUTION,
        mode="fine",
        target_type="semantic",
    )
    dataset_val = build_cityscapes_dataset(
        args.data_path,
        split="val",
        size=STAGE1_RESOLUTION,
        mode="fine",
        target_type="semantic",
    )

    num_workers = min(4, os.cpu_count() or 1)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    def feature_step(batch, linear_probe, train):  # noqa: ARG001
        imgs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = encode_segmap(labels.to(device))
        feats = model.ae.encode(imgs)["continuous"].detach()
        logits = linear_probe(feats, target_size=labels.shape[-2:])
        return logits, labels

    train_linear_probe(
        args=args,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        device=device,
        feature_step=feature_step,
    )


def parse_args(argv: Iterable[str] | None = None) -> Tuple[argparse.Namespace, Sequence[str]]:
    parser = argparse.ArgumentParser(description="Train/eval a linear segmentation probe on Cityscapes features.")
    add_shared_arguments(
        parser,
        data_path_default="/data/lmbraid12/datasets/public/Cityscapes/",
        num_epoch_default=400,
        batch_size_default=64,
        eval_every_default=200,
    )

    args, unknown = parser.parse_known_args(argv)

    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)

    args.seq_fake = os.path.join(args.frames_dir, "fake_images")
    args.seq_real = os.path.join(args.frames_dir, "real_images")

    return args, unknown


def main() -> None:
    args, unknown = parse_args()

    print(">>> Checkpoint:", args.ckpt)
    print(">>> Config:    ", args.config)
    print(">>> Data root: ", args.data_path)

    if args.dump_vis:
        mkdir(args.seq_fake)
        mkdir(args.seq_real)
        print("[info] Visualization enabled – images may overwrite existing files.")

    calculate_semantic(args=args, unknown_args=unknown)


if __name__ == "__main__":
    main()
