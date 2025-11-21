import os
import json
import math
import random
import importlib
import time
from collections import OrderedDict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class MultiHDF5DatasetMultiFrameMultiViewViz(Dataset):
    def __init__(self, size, hdf5_paths_file, num_frames=6, views=["CAM_R2", "CAM_R1", "CAM_R0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_B0"], n_views=2, frame_rate_multiplier=1):
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.views = views
        self.n_views = n_views


        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = []
        for path in self.hdf5_paths:
            try:
                file = h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
                self.files.append(file)
            except Exception as e:
                print(f'Error opening file {path}: {e}')
        
        self.chunk_keys = []
        self.chunk_lengths = []
        for file in self.files:
            chunk_keys = []
            chunk_lengths = {}
            for chunk in file.keys():
                try:
                    frame_group = file[chunk]['frames']
                    if 'CAM_F0' not in frame_group:
                        continue
                    num_frames = frame_group['CAM_F0'].shape[0]
                    chunk_keys.append(chunk)
                    chunk_lengths[chunk] = num_frames
                except KeyError:
                    continue
            self.chunk_keys.append(chunk_keys)
            self.chunk_lengths.append(chunk_lengths)

        self.total_length = sum(sum(lengths.values()) for lengths in self.chunk_lengths)
        print(f'Total length: {self.total_length}, {len(self.files)} files.')
        self.transform = transforms.Compose([#transforms.Resize(min(self.size)),
                                    #transforms.CenterCrop(self.size),
                                    transforms.ToTensor(),
                                    ])    
    def __len__(self):
        return self.total_length
    
    def get_indices(self):
        file_index = random.randint(0, len(self.files) - 1)
        if not self.chunk_keys[file_index]:
            return self.get_indices()  # retry

        chunk = random.choice(self.chunk_keys[file_index])
        length = self.chunk_lengths[file_index][chunk]

        frames_needed = (self.num_frames - 1) * self.frame_interval + 1
        if length < frames_needed:
            return self.get_indices()

        start_idx = random.randint(0, length - frames_needed)
        indices = [start_idx + i * self.frame_interval for i in range(self.num_frames)]
        return file_index, chunk, indices



    def __getitem__(self, idx):
        file_index, chunk, indices = self.get_indices()
        h5_file = self.files[file_index]
        frame_group = h5_file[chunk]['frames']
        raymap_group = h5_file[chunk]['raymaps']

        frames = []
        raymaps = []
        target_vector = [0] * self.num_frames 
        

        # Convert and transform frames
        frames = [
            [self.transform(Image.fromarray(frame_group[view][i])) * 2 - 1 for i in indices]
            for view in self.views if view in frame_group
        ]

        # Stack frames horizontally (along width) for each view: [C, H, W * num_frames]
        frames = [torch.cat(view_frames, dim=2) for view_frames in frames]

        # Stack views vertically (along height): [C, H * num_views, W * num_frames]
        video = torch.cat(frames, dim=1)

        raymaps = [
            torch.tensor(raymap_group[view][0], dtype=torch.float32) 
            for view in self.views if view in raymap_group
        ]
        #raymaps = torch.cat(raymaps, dim=1)  # Stack raymaps along width
        return {
            'video': video,
            'raymaps': raymaps
        }
        #return torch.stack(output, dim=0)  # (n_views, num_frames, C, H, W)
    def close(self):
        for file in self.files:
            file.close()


class MultiHDF5DatasetMultiFrameMultiView(Dataset):
    def __init__(self, size, hdf5_paths_file, num_frames=6, views=["CAM_R2", "CAM_R1", "CAM_R0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_B0"], n_views=2, frame_rate_multiplier=1):
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.views = views
        self.n_views = n_views


        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = []
        for path in self.hdf5_paths:
            try:
                file = h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
                self.files.append(file)
            except Exception as e:
                print(f'Error opening file {path}: {e}')
        
        self.chunk_keys = []
        self.chunk_lengths = []
        for file in self.files:
            chunk_keys = []
            chunk_lengths = {}
            for chunk in file.keys():
                try:
                    frame_group = file[chunk]['frames']
                    if 'CAM_F0' not in frame_group:
                        continue
                    num_frames = frame_group['CAM_F0'].shape[0]
                    chunk_keys.append(chunk)
                    chunk_lengths[chunk] = num_frames
                except KeyError:
                    continue
            self.chunk_keys.append(chunk_keys)
            self.chunk_lengths.append(chunk_lengths)

        self.total_length = sum(sum(lengths.values()) for lengths in self.chunk_lengths)
        print(f'Total length: {self.total_length}, {len(self.files)} files.')
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                    transforms.CenterCrop(self.size),
                                    transforms.ToTensor(),
                                    ])    
    def __len__(self):
        return self.total_length
    
    def len_videos(self):
        return sum(len(chunk) for chunk in self.chunk_keys)

    def get_indices(self):
        file_index = random.randint(0, len(self.files) - 1)
        if not self.chunk_keys[file_index]:
            return self.get_indices()  # retry

        chunk = random.choice(self.chunk_keys[file_index])
        length = self.chunk_lengths[file_index][chunk]

        frames_needed = (self.num_frames - 1) * self.frame_interval + 1
        if length < frames_needed:
            return self.get_indices()

        start_idx = random.randint(0, length - frames_needed)
        indices = [start_idx + i * self.frame_interval for i in range(self.num_frames)]
        return file_index, chunk, indices


    def get_train_mix(self, frame_group, raymap_group, indices):
        """
        Returns:
            frames: list of PIL images
            raymaps: list of np arrays
            target_vector: list of 0/1 indicating whether frame is input (1) or target (0)
            selected_views: tuple of used views
        """
        assert len(self.views) >= 2, "At least 2 views are required"
        num_frames = len(indices)
        all_views = [v for v in self.views if v in frame_group]
        if len(all_views) < 2:
            raise ValueError("Not enough valid views in this sample")
        target_vector = [0] * (num_frames)
        mode = random.choice(["single_view", "mix_views"])

        if mode == "single_view":
            view_idx = random.randint(0, len(all_views)-1)
            view = all_views[view_idx]
            frames = [Image.fromarray(frame_group[view][i]) for i in indices]
            raymaps = [raymap_group[view][i] for i in indices]

            target_vector[-1] = 1  # last frame is the target
            view_indices = [view_idx] * num_frames  # all frames are from the same view

            #convert target_vector to bool
            target_vector = [bool(x) for x in target_vector]
            #convert view_indices to int
            view_indices = [int(x) for x in view_indices]

            t_indices = list(range(num_frames))  # all views are used
            
            return frames, raymaps, target_vector, view_indices, t_indices

        elif mode == "mix_views":
            view1 = "CAM_F0"  # always use the reference view
            view2_idx = random.randint(1, len(all_views)-1)
            view2 = all_views[view2_idx]
            split_point = num_frames // 2

            # Alternate strategy: interleave half from view1 and half from view2
            frames = [
                Image.fromarray(frame)
                for pair in zip(
                    [frame_group[view1][i] for i in indices[:split_point]],
                    [frame_group[view2][i] for i in indices[:split_point]]
                )
                for frame in pair
            ]

            raymaps = [
                ray
                for pair in zip(
                    [raymap_group[view1][i] for i in indices[:split_point]],
                    [raymap_group[view2][i] for i in indices[:split_point]]
                )
                for ray in pair
            ]


            # Decide which part is "seen" vs "unseen" either 1,1 or 1,0 or 0,1
            perm = random.choice([[1, 1], [1, 0], [0, 1]])

            # target_vector[split_point-1] = perm[0]
            # target_vector[-1] = perm[1]
            target_vector[-2:] = perm  # last two frames are the target

            #return also a views list like the target_vector with ids of the views used
            # This is useful for debugging and visualization    
            # get view index
            view_indices = [0,view2_idx] * split_point
            # Interleave t_indices in repeat/interleave fashion for each frame
            t_indices = [i for i in range(split_point) for _ in range(2)]
            return frames, raymaps, target_vector, view_indices, t_indices


    def __getitem__(self, idx):
        file_index, chunk, indices = self.get_indices()
        h5_file = self.files[file_index]
        frame_group = h5_file[chunk]['frames']
        raymap_group = h5_file[chunk]['raymaps']

        frames, raymaps, target_vector, view_indices, t_indices = self.get_train_mix(frame_group, raymap_group, indices)
        # Normalize frames by multiplying by 2 and subtracting 1 and applying transforms
        frames = [self.transform(frame) * 2 - 1 for frame in frames]
        # Convert raymaps to tensors
        raymaps = [torch.tensor(raymap, dtype=torch.float32) for raymap in raymaps]
        # Stack frames and raymaps

        return {
            'images': torch.stack(frames, dim=0),  # (num_frames*NV, C, H, W)
            'raymaps': torch.stack(raymaps, dim=0).permute(0, 3, 1, 2),  # (num_frames, 6, H, W)
            'target_vector': torch.tensor(target_vector, dtype=torch.bool), # (num_frames, )
            'view_indices': torch.tensor(view_indices, dtype=torch.int64),  # (num_frames*NV, )
            't_indices': torch.tensor(t_indices, dtype=torch.int64)  # number of views used in this sample
        }
        #return torch.stack(output, dim=0)  # (n_views, num_frames, C, H, W)
    def close(self):
        for file in self.files:
            file.close()


class MultiHDF5DatasetMultiFrameMultiViewBiased(Dataset):
    def __init__(self, size, hdf5_paths_file, num_frames=6, views=["CAM_R2", "CAM_R1", "CAM_R0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_B0"], n_views=2, frame_rate_multiplier=1):
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.views = views
        self.n_views = n_views


        # if data is stored at a higher frame rate than needed, we can skip some frames:
        # we can only reduce frame rate:
        assert frame_rate_multiplier <= 1, 'frame_rate_multiplier must be <= 1'
        # we can only reduce frame rate by integer factor: reciprocal of frame_rate_multiplier must be an integer
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)
        
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = []
        for path in self.hdf5_paths:
            try:
                file = h5py.File(path, 'r', rdcc_nbytes=1024*1024*1024*400, rdcc_nslots=1000)
                self.files.append(file)
            except Exception as e:
                print(f'Error opening file {path}: {e}')
        
        self.chunk_keys = []
        self.chunk_lengths = []
        for file in self.files:
            chunk_keys = []
            chunk_lengths = {}
            for chunk in file.keys():
                try:
                    frame_group = file[chunk]['frames']
                    if 'CAM_F0' not in frame_group:
                        continue
                    num_frames = frame_group['CAM_F0'].shape[0]
                    chunk_keys.append(chunk)
                    chunk_lengths[chunk] = num_frames
                except KeyError:
                    continue
            self.chunk_keys.append(chunk_keys)
            self.chunk_lengths.append(chunk_lengths)

        self.total_length = sum(sum(lengths.values()) for lengths in self.chunk_lengths)
        print(f'Total length: {self.total_length}, {len(self.files)} files.')
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                    transforms.CenterCrop(self.size),
                                    transforms.ToTensor(),
                                    ])    
    def __len__(self):
        return self.total_length
    
    def len_videos(self):
        return sum(len(chunk) for chunk in self.chunk_keys)

    def get_indices(self):
        file_index = random.randint(0, len(self.files) - 1)
        if not self.chunk_keys[file_index]:
            return self.get_indices()  # retry

        chunk = random.choice(self.chunk_keys[file_index])
        length = self.chunk_lengths[file_index][chunk]

        frames_needed = (self.num_frames - 1) * self.frame_interval + 1
        if length < frames_needed:
            return self.get_indices()

        start_idx = random.randint(0, length - frames_needed)
        indices = [start_idx + i * self.frame_interval for i in range(self.num_frames)]
        return file_index, chunk, indices


    def get_train_mix(self, frame_group, raymap_group, indices):
        """
        Returns:
            frames: list of PIL images
            raymaps: list of np arrays
            target_vector: list of 0/1 indicating whether frame is input (1) or target (0)
            selected_views: tuple of used views
        """
        assert len(self.views) >= 2, "At least 2 views are required"
        num_frames = len(indices)
        all_views = [v for v in self.views if v in frame_group]
        if len(all_views) < 2:
            raise ValueError("Not enough valid views in this sample")
        target_vector = [0] * (num_frames)
        mode = random.choices(["single_view", "mix_views"], weights=[0.0, 0.1])[0]

        if mode == "single_view":
            view_idx = random.randint(0, len(all_views)-1)
            view = all_views[view_idx]
            frames = [Image.fromarray(frame_group[view][i]) for i in indices]
            raymaps = [raymap_group[view][i] for i in indices]

            target_vector[-1] = 1  # last frame is the target
            view_indices = [view_idx] * num_frames  # all frames are from the same view

            #convert target_vector to bool
            target_vector = [bool(x) for x in target_vector]
            #convert view_indices to int
            view_indices = [int(x) for x in view_indices]

            t_indices = list(range(num_frames))  # all views are used
            
            return frames, raymaps, target_vector, view_indices, t_indices

        elif mode == "mix_views":
            view1 = "CAM_F0"  # always use the reference view
            view2_idx = random.randint(1, len(all_views)-1)
            view2 = all_views[view2_idx]
            split_point = num_frames // 2

            # Alternate strategy: interleave half from view1 and half from view2
            frames = [
                Image.fromarray(frame)
                for pair in zip(
                    [frame_group[view1][i] for i in indices[:split_point]],
                    [frame_group[view2][i] for i in indices[:split_point]]
                )
                for frame in pair
            ]

            raymaps = [
                ray
                for pair in zip(
                    [raymap_group[view1][i] for i in indices[:split_point]],
                    [raymap_group[view2][i] for i in indices[:split_point]]
                )
                for ray in pair
            ]


            # Decide which part is "seen" vs "unseen" either 1,1 or 1,0 or 0,1
            perm = random.choices([[1, 1], [1, 0], [0, 1]], weights=[0.1, 0.0, 0.0])[0]

            # target_vector[split_point-1] = perm[0]
            # target_vector[-1] = perm[1]
            target_vector[-2:] = perm  # last two frames are the target

            #return also a views list like the target_vector with ids of the views used
            # This is useful for debugging and visualization    
            # get view index
            view_indices = [0,view2_idx] * split_point
            # Interleave t_indices in repeat/interleave fashion for each frame
            t_indices = [i for i in range(split_point) for _ in range(2)]
            return frames, raymaps, target_vector, view_indices, t_indices


    def __getitem__(self, idx):
        file_index, chunk, indices = self.get_indices()
        h5_file = self.files[file_index]
        frame_group = h5_file[chunk]['frames']
        raymap_group = h5_file[chunk]['raymaps']

        frames, raymaps, target_vector, view_indices, t_indices = self.get_train_mix(frame_group, raymap_group, indices)
        # Normalize frames by multiplying by 2 and subtracting 1 and applying transforms
        frames = [self.transform(frame) * 2 - 1 for frame in frames]
        # Convert raymaps to tensors
        raymaps = [torch.tensor(raymap, dtype=torch.float32) for raymap in raymaps]
        # Stack frames and raymaps

        return {
            'images': torch.stack(frames, dim=0),  # (num_frames*NV, C, H, W)
            'raymaps': torch.stack(raymaps, dim=0).permute(0, 3, 1, 2),  # (num_frames, 6, H, W)
            'target_vector': torch.tensor(target_vector, dtype=torch.bool), # (num_frames, )
            'view_indices': torch.tensor(view_indices, dtype=torch.int64),  # (num_frames*NV, )
            't_indices': torch.tensor(t_indices, dtype=torch.int64)  # number of views used in this sample
        }
        #return torch.stack(output, dim=0)  # (n_views, num_frames, C, H, W)
    def close(self):
        for file in self.files:
            file.close()


def main_mv_final():
    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultiHDF5DatasetMultiFrameMultiView(size=256, hdf5_paths_file='/p/scratch/nxtaim-1/farid1/orbis/data/mv_h5_paths.txt', num_frames=8, views=["CAM_F0", "CAM_R0", "CAM_L0"], n_views=2, frame_rate_multiplier=1)
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of videos: {dataset.len_videos()}")
    # Fetch one sample
    for j in tqdm(range(50)):
        sample = dataset[j]

        # Unpack sample
        video = sample['video']            # (num_frames, C, H, W)
        raymaps = sample['raymaps']        # (num_frames, H, W)
        target_vector = sample['target_vector']
        view_indices = sample['view_indices']

        # Save each frame and raymap
        for i in range(video.size(0)):
            # Save image frame
            save_image((video[i] + 1) / 2.0, os.path.join(output_dir, f"vid{j}_frame_{i}_view{view_indices[i].item()}.png"))

            # Save raymap as grayscale image
            plt.imsave(
                os.path.join(output_dir, f"vid{j}_raymap_{i}_view{view_indices[i].item()}.png"),
                raymaps[i][...,-3].numpy(),
                cmap='viridis'
            )

        # Optional: print meta info
        print("Target vector:", target_vector.tolist())
        print("View indices:", view_indices.tolist())

    # Cleanup
    dataset.close()


def main_mv_viz():
    output_dir = 'output_frames_viz'
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultiHDF5DatasetMultiFrameMultiViewViz(size=256, hdf5_paths_file='/p/scratch/nxtaim-1/farid1/orbis/data/mv_h5_paths.txt', num_frames=6, n_views=2, frame_rate_multiplier=0.5)
    print(f"Dataset length: {len(dataset)}")
    # Fetch one sample
    for j in tqdm(range(50)):
        sample = dataset[j]

        # Unpack sample
        video = sample['video']            # (num_frames, C, H, W)
        raymaps = sample['raymaps']        # (C, H, W)
        # Save video as a single image
        save_image((video + 1) / 2.0, os.path.join(output_dir, f"video_sample_{j}.png"))

        
        # Save raymap as grayscale image
        for x, map in enumerate(raymaps):
            plt.imsave(
                os.path.join(output_dir, f"raymap_{x}_sample_{j}.png"),
                map[...,-3].numpy(),
                cmap='viridis'
            )


    # Cleanup
    dataset.close()

if __name__ == "__main__":
    main_mv_final()
    #main_mv_viz()
