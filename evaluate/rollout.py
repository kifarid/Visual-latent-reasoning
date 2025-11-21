import argparse
import os
import sys
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf


import torch
from torchvision.utils import save_image

from pytorch_lightning import seed_everything

from util import instantiate_from_config
from evaluate.metrics import *


def get_ckpt_epoch_step(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    return epoch, global_step


@torch.no_grad()
def generate_images(args, unknown_args):
    if os.path.exists(args.frames_dir):
        print("Folder exist, new images will be saved to the same folder, delete it if you want to start from scratch")
    if not os.path.exists(args.frames_dir):
        os.makedirs(args.frames_dir)

    if args.seed > 0:
        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
        # np.random.seed(args.seed)
        # random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
        seed_everything(args.seed)
        
    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown_args))
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(args.ckpt)["state_dict"], strict=True)
    model = model.cuda()
    _ = model.eval()
    
    # Validation data config, if provided
    if args.val_config is not None:
        config = OmegaConf.merge(OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args))
    
    # Get the dataset (for conditioning)
    num_condition_frames = config.data.params.validation.params.num_frames - model.num_pred_frames
    num_frames = num_condition_frames + args.num_gen_frames  # we need to generate more frames for evaluation
    if args.save_real:  # we need to save real frames as well
        config.data.params.validation.params.num_frames = num_frames
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = data.val_dataloader()
    
    sample_idx = 0
    progress_bar = tqdm(range(len(val_loader.dataset)//val_loader.batch_size))
    loader_iter = iter(val_loader)
    for batch_idx, _ in enumerate(progress_bar):
        # try:
        data_batch = next(loader_iter)
        if args.num_videos is not None and sample_idx >= args.num_videos: break                  

        if isinstance(data_batch, dict):
            x = data_batch['images'].cuda()
        elif hasattr(data_batch, 'cuda'):
            x = data_batch.cuda()

        cond_x = x[:, :num_condition_frames]
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            latents, gen_frames = model.roll_out(x_0=cond_x, num_gen_frames=args.num_gen_frames, latent_input=False, eta=args.eta, NFE=args.num_steps, sample_with_ema=args.evaluate_ema, num_samples=cond_x.size(0))

        # save generated images and gifs
        for sample_in_batch_idx in range(x.shape[0]):
            subfolder_path_fake = os.path.join(args.frames_dir, "fake_images", f"sequence_{sample_idx:04d}")
            subfolder_path_gifs = os.path.join(args.frames_dir, f"gen_gifs")
            if not os.path.exists(subfolder_path_fake): os.makedirs(subfolder_path_fake)
            if not os.path.exists(subfolder_path_gifs): os.makedirs(subfolder_path_gifs)
            for f in range(len(gen_frames[sample_in_batch_idx])):
                save_image((gen_frames[sample_in_batch_idx, f]+1.0)/2.0, os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg"))
            # gifs
            imageio.mimsave(os.path.join(subfolder_path_gifs, f"sequence_{sample_idx:04d}.gif"), [np.array(Image.open(os.path.join(subfolder_path_fake, f"frame_{f:04d}.jpg"))) for f in range(len(gen_frames[sample_in_batch_idx]))], fps=7, loop=0)

            if args.save_real:
                subfolder_path_real = os.path.join(args.frames_dir, "real_images", f"sequence_{sample_idx:04d}")
                if not os.path.exists(subfolder_path_real): os.makedirs(subfolder_path_real)
                for f in range(x.shape[1]):
                    save_image((x[sample_in_batch_idx, f]+1.0)/2.0, os.path.join(subfolder_path_real, f"frame_{f:04d}.jpg"))
            sample_idx += 1
        
    progress_bar.set_description(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.02f} GB")


def main(args, unknown_args):
    generate_images(args, unknown_args)


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "y", "1"): return True
        elif v.lower() in ("no", "false", "f", "n", "0"): return False
        else: raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to the experiment directory, where the config and checkpoints are stored")
    parser.add_argument("--ckpt", type=str, default='checkpoints/last.ckpt', help="Path to the checkpoint file, relative to exp_dir")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to the config file, relative to exp_dir")
    parser.add_argument("--val_config", type=str, default=None, help="Path to the validation data config file")
    parser.add_argument("--num_gen_frames", type=int, default=1, help="Number of frames to generate (roll-out length)")
    parser.add_argument("--frames_dir", type=str, default=None, help="Path of the folder for the real and fake frames, relative to exp_dir")
    parser.add_argument("--save_real", type=str2bool, default=True, help="Save real frames in the same folder as fake frames")

    parser.add_argument("--num_videos", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of steps for sampling")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity for sampling")
    parser.add_argument("--evaluate_ema", type=str2bool, default=True, help="If the evaluation happen with ema model")

    args, unknown = parser.parse_known_args()
    
    args.ckpt = os.path.join(args.exp_dir, args.ckpt)
    args.config = os.path.join(args.exp_dir, args.config)

    
    if args.frames_dir is None:
        epoch, global_step = get_ckpt_epoch_step(args.ckpt)
        args.frames_dir = os.path.join('gen_rollout',
                                       os.path.basename(args.val_config).split(".")[0] if args.val_config is not None else "default_data",
                                       f"ep{epoch}iter{global_step}_{args.num_steps}steps",)
    args.frames_dir = os.path.join(args.exp_dir, args.frames_dir)
    print(f">>> Saving generated images to {args.frames_dir}")

    main(args, unknown)