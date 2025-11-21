import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from modules.lpips import LPIPS

def compute_LPIPS_last_frame(args, fake_images_path, real_images_path):
    """
    Compute LPIPS score between the fake and real predicted frames only
    """

    def load_and_prepare(img_path):
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img).unsqueeze(0)
        # value range [0, 1] -> [-1, 1]
        img = img * 2 - 1
        return img.cuda()

    lpips_loss_fn = LPIPS().eval().cuda()

    lpipsdists = [[] for _ in range(args.num_gen_frames)]
    for video_name in tqdm(os.listdir(real_images_path)):
        gen_frame_names = sorted(os.listdir(os.path.join(real_images_path, video_name)))[-args.num_gen_frames:]
        for gen_idx in range(args.num_gen_frames):
            gen_frame_name = gen_frame_names[gen_idx]
            real_last_frame_path = os.path.join(real_images_path, video_name, gen_frame_name)
            fake_last_frame_path = os.path.join(fake_images_path, video_name, gen_frame_name)
            lpipsdists[gen_idx].append(lpips_loss_fn(load_and_prepare(real_last_frame_path), load_and_prepare(fake_last_frame_path)).item())

    return [np.mean(lpipsdists[gen_idx]) for gen_idx in range(args.num_gen_frames)]


def compute_dino_dist_last_frame(args, fake_images_path, real_images_path, dino_arch='dinov2_vitl14'):
    """
    Mean per-patch L2 distance between the DINO embeddings of the last frame of the real and fake videos.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
    ])
    def load_and_prepare(img_path):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).cuda()
        return img.cuda()

    model = torch.hub.load('facebookresearch/dinov2', dino_arch).cuda()

    dinodists = [[] for _ in range(args.num_gen_frames)]
    for video_name in tqdm(os.listdir(real_images_path)):
        gen_frame_names = sorted(os.listdir(os.path.join(real_images_path, video_name)))[-args.num_gen_frames:]
        for gen_idx in range(args.num_gen_frames):
            gen_frame_name = gen_frame_names[gen_idx]
            real_last_frame_path = os.path.join(real_images_path, video_name, gen_frame_name)
            fake_last_frame_path = os.path.join(fake_images_path, video_name, gen_frame_name)
            dino_1 = model.get_intermediate_layers(load_and_prepare(real_last_frame_path))[0]
            dino_2 = model.get_intermediate_layers(load_and_prepare(fake_last_frame_path))[0]
            # compute euclidean distance
            # dinodists.append(torch.nn.functional.pairwise_distance(dino_1, dino_2).mean().item())
            # compute cosine distance
            dinodists[gen_idx].append(torch.nn.functional.cosine_similarity(dino_1, dino_2).mean().item())

    return [float(np.mean(dinodists[gen_idx])) for gen_idx in range(args.num_gen_frames)]



def compute_psnr_last_frame(args, fake_images_path, real_images_path, dino_arch='dinov2_vitl14'):
    """
    Compute PSNR between the last generated and real frames of videos.
    """
    def load_and_prepare(img_path):
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img).unsqueeze(0)
        # value range [0, 1] -> [-1, 1]
        img = img * 2 - 1
        return img.cuda()

    psnr_list = [[] for _ in range(args.num_gen_frames)]
    for video_name in tqdm(os.listdir(real_images_path)):
        gen_frame_names = sorted(os.listdir(os.path.join(real_images_path, video_name)))[-args.num_gen_frames:]
        for gen_idx in range(args.num_gen_frames):
            gen_frame_name = gen_frame_names[gen_idx]
            real_last_frame_path = os.path.join(real_images_path, video_name, gen_frame_name)
            fake_last_frame_path = os.path.join(fake_images_path, video_name, gen_frame_name)
            
            # Compute Mean Squared Error (MSE)
            mse = F.mse_loss(load_and_prepare(real_last_frame_path), load_and_prepare(fake_last_frame_path), reduction='mean')
            # Avoid division by zero
            if mse == 0:
                return float('inf')
            # Compute PSNR
            max_pixel_value = 2.0
            psnr = 10 * torch.log10(max_pixel_value**2 / mse)

            psnr_list[gen_idx].append(psnr.mean().item())

    return [float(np.mean(psnr_list[gen_idx])) for gen_idx in range(args.num_gen_frames)]
    