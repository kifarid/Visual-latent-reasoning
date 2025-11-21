import argparse
import os
import sys
import shutil
import glob
import warnings
import random
import imageio
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from pytorch_fid import fid_score

# Make project root import-able
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import instantiate_from_config  # noqa: E402


def get_ckpt_epoch_step(ckpt_path: str):
    """Return `(epoch, global_step)` stored inside a Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["epoch"], ckpt["global_step"]


def mkdir(path: os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def reconstruct_images(args: argparse.Namespace, unknown_args: Sequence[str]):
    """Run tokenizer encode→decode on the validation split and dump frames."""
    if args.seed > 0:
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)

    cfg_model = OmegaConf.load(args.config)
    cfg_model = OmegaConf.merge(cfg_model, OmegaConf.from_dotlist(unknown_args))
    model = instantiate_from_config(cfg_model.model)

    state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()

    cfg_data = cfg_model  # default – maybe overridden below
    if args.val_config is not None:
        cfg_data = OmegaConf.merge(
            OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args)
        )

    # We keep the user-defined number of frames. Nothing special for recon.
    data = instantiate_from_config(cfg_data.data)
    data.prepare_data()
    data.setup()
    val_loader = data.val_dataloader()


    sample_idx = 0
    progress_bar = tqdm(range(len(val_loader.dataset) // val_loader.batch_size))
    loader_iter = iter(val_loader)

    for _ in progress_bar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        if args.num_samples is not None and sample_idx >= args.num_samples:
            break

        if isinstance(batch, dict):
            x = batch["images"].to(args.device)
        else:
            x = batch.to(args.device)

        # Expect shape: (bs, 3, H, W)
        assert (
            x.ndim == 4 and x.shape[1] == 3
        ), f"Expected (bs, 3, H, W), got {x.shape}"

        bs, C, H, W = x.shape


        tokens = model.encode(x)['continuous']  # ➜ (bs, …)
        x_hat = model.decode(tokens)  # ➜ (bs*T, 3, H, W) in [-1,1] or [0,1]
        x_hat = x_hat[0] if isinstance(x_hat, tuple) else x_hat  # handle tuple output
        x_hat = torch.tanh(x_hat)  # ensure in [-1,1] if necessary


        for b in range(bs):
            real_img = (x[b] + 1.0) / 2.0  # scale to [0,1]
            fake_img = (x_hat[b] + 1.0) / 2.0
            

            # Then directly save to .jpg files:
            real_img_path = os.path.join(args.seq_real, f"sequence_{sample_idx:04d}.jpg")
            fake_img_path = os.path.join(args.seq_fake, f"sequence_{sample_idx:04d}.jpg")

            save_image(real_img, real_img_path)
            save_image(fake_img, fake_img_path)
            sample_idx += 1

        progress_bar.set_description(
            f"Max memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.02f} GB"
        )
        
        
        
        
def calculate_fid(real_dir, fake_dir, batch_size, device):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048  # Default: InceptionV3 last pooling layer
    )
    print(f"FID between real and fake images: {fid_value:.4f}")
    return fid_value



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct frames with tokenizer and dump for rFID computation."
    )

    # Experiment paths
    parser.add_argument("--exp_dir", type=str, required=True, help="Root experiment directory")
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path *relative* to exp_dir"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Model config *relative* to exp_dir"
    )
    parser.add_argument(
        "--val_config", type=str, default=None, help="Optional validation data-config file"
    )
    parser.add_argument(
        "--skip_reconstruction",
        type=str2bool,
        default=False,
        help="Skip reconstruction and only compute rFID from existing frames",
    )

    # Output
    parser.add_argument(
        "--frames_dir",
        type=str,
        default='vis',
        help="Where to dump frames (relative to exp_dir). If omitted a folder name is automatically generated.",
    )

    # Misc
    parser.add_argument("--num_samples", type=int, default=None, help="Stop after N sequences")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (<=0 disables repeatability)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    parser.add_argument(
        "--save_gifs", type=str2bool, default=False, help="Save reconstructed GIFs alongside JPGs"
    )

    # Catch-all for additional Hydra dot-list overrides
    args, unknown = parser.parse_known_args()

    # Expand checkpoint/config relative paths
    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)

    print(">>> Checkpoint:", args.ckpt)
    print(">>> Config:    ", args.config)
    

    # Initialise output folder
    if args.frames_dir is None:
        # Derive default folder name from checkpoint epoch/step and data split
        epoch, gstep = get_ckpt_epoch_step(args.ckpt)
        split_name = (
            Path(args.val_config).stem if args.val_config is not None else "default_data"
        )
        args.frames_dir = os.path.join(
            "gen_reconstructions", split_name, f"ep{epoch}iter{gstep}"
        )
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)
    mkdir(args.frames_dir)

    if (Path(args.frames_dir) / "fake_images").exists():
        print(
            "[INFO] Folder exists – new images will overwrite existing ones. "
            "Delete it if you want to start from scratch."
        )
        

    args.seq_fake = os.path.join(args.frames_dir, "fake_images")
    args.seq_real = os.path.join(args.frames_dir, "real_images")

    # Create directories if they don't exist
    mkdir(args.seq_fake)
    mkdir(args.seq_real)

    # Check if either directory is non-empty
    fake_files = glob.glob(os.path.join(args.seq_fake, "*.jpg"))
    real_files = glob.glob(os.path.join(args.seq_real, "*.jpg"))

    folders_full = len(fake_files) > 0 or len(real_files) > 0

    # If folders are not empty and reconstruction is not skipped
    if folders_full and not args.skip_reconstruction:
        print(f"[WARNING] Output folders are not empty:")
        print(f"  {args.seq_fake} has {len(fake_files)} files")
        print(f"  {args.seq_real} has {len(real_files)} files")

        response = input("Do you want to empty these folders and regenerate images? [y/N]: ").lower()
        if response == "y":
            shutil.rmtree(args.seq_fake)
            shutil.rmtree(args.seq_real)
            mkdir(args.seq_fake)
            mkdir(args.seq_real)
            print(">>> Cleared existing folders.")
        else:
            print(">>> Skipping reconstruction (set --skip_reconstruction to suppress this prompt).")
            args.skip_reconstruction = True

    if not args.skip_reconstruction:
        print(">>> Reconstructing images...")
        reconstruct_images(args, unknown)
    else:
        print(">>> Skipping reconstruction, using existing frames...")

    calculate_fid(
        real_dir=args.seq_real,
        fake_dir=args.seq_fake,
        batch_size=32,  # Adjust as needed
        device=args.device
    )



if __name__ == "__main__":
    main()
