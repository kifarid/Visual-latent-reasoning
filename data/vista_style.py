from torch.utils.data import Dataset
import json
import os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.v2.functional as FV2
import cv2
from PIL import Image
import sqlite3
import pandas as pd


class VistaStyleNuScenesLoader(Dataset):
    def __init__(self, *, size, json_path, images_root, num_frames=None, frame_rate_multiplier=1, sample_indices=None):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.json_path = json_path
        self.num_frames = num_frames
        self.images_root = images_root

        assert frame_rate_multiplier <= 1, "Frame rate multiplier should be less than or equal to 1"
        assert 1/frame_rate_multiplier == int(1/frame_rate_multiplier), 'reciprocal of frame_rate_multiplier must be an integer'
        self.frame_interval = int(1/frame_rate_multiplier)        
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.sample_indices = sample_indices
        if sample_indices is not None:
            self.data = [self.data[i] for i in sample_indices]
            
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                             transforms.CenterCrop(self.size),
                                             transforms.ToTensor()])
        
    def __getitem__(self, index):
        sample = self.data[index]
        frame_paths = sample['frames'][::self.frame_interval]
        if self.num_frames is not None:
            if len(frame_paths) < self.num_frames:
                print(f"Warning: Number of frames {len(frame_paths)} is less than the required {self.num_frames}")
                raise ValueError(f"Number of frames {len(frame_paths)} is less than the required {self.num_frames}")
            frame_paths = frame_paths[:self.num_frames]
        images = [cv2.imread(os.path.join(self.images_root, frame_path)) for frame_path in frame_paths]
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        images = [self.transform(image)*2-1 for image in images]
        images = torch.stack(images, dim=0)
        return images
        
    def __len__(self):
        return len(self.data)
    
    
    
def extract_pose_table(db_path):
    """
    Connects to the nuPlan SQLite database and extracts the entire ego_pose table.
    Returns a pandas DataFrame containing the pose metadata.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM ego_pose;"
    pose_df = pd.read_sql_query(query, conn)
    conn.close()
    # Ensure timestamp is numeric and sort by timestamp
    pose_df['timestamp'] = pd.to_numeric(pose_df['timestamp'], errors='coerce')
    pose_df.sort_values("timestamp", inplace=True)
    return pose_df

def get_pose(pose_df, pose_token):
    tk = bytes.fromhex(pose_token)
    pose = pose_df[pose_df['token'] == tk]
    if pose.empty:
        raise ValueError(f"Pose with token {pose_token} not found.")
    return pose


class VistaStyleNuScenesLoaderSteering(VistaStyleNuScenesLoader):
    def __init__(self, *, size, json_path, images_root, dbs_root, num_frames=None, stored_data_frame_rate=10, frame_rate=5, sample_indices=None):
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        frame_rate_multiplier = frame_rate / stored_data_frame_rate
        super().__init__(size=size, json_path=json_path, images_root=images_root, num_frames=num_frames, frame_rate_multiplier=frame_rate_multiplier, sample_indices=sample_indices)
        self.dbs_root = dbs_root
    
    def reconstruct_trajectory_from_speed_and_yaw_rate(self, speeds, yaw_rates, initial_position=(0.0, 0.0), initial_yaw=0.0):
        timestamps = np.arange(0, len(speeds) / self.stored_data_frame_rate, 1 / self.stored_data_frame_rate)
        N = len(speeds)
        dt = np.diff(timestamps)
        
        trajectory = np.zeros((N, 2))
        yaw_angles = np.zeros(N)
        trajectory[0] = np.array(initial_position)
        yaw_angles[0] = initial_yaw

        for i in range(1, N):
            yaw_angles[i] = yaw_angles[i-1] + yaw_rates[i-1] * dt[i-1]
            dx = speeds[i-1] * np.cos(yaw_angles[i-1]) * dt[i-1]
            dy = speeds[i-1] * np.sin(yaw_angles[i-1]) * dt[i-1]
            trajectory[i] = trajectory[i-1] + np.array([dx, dy])

        return trajectory
    
    def __getitem__(self, index):
        images = super().__getitem__(index)
        sample = self.data[index]
        pose_table = extract_pose_table(os.path.join(self.dbs_root, sample['db_name']))
        pose_tokens = sample['pose_tokens']
        poses = [get_pose(pose_table, pose_token) for pose_token in pose_tokens]
        # get steeting signals: 'vx' and 'angular_rate_z'
        speeds = np.array([pose['vx'].values[0] for pose in poses])
        yaw_rates = np.array([pose['angular_rate_z'].values[0] for pose in poses])
        steering = np.stack([speeds, yaw_rates], axis=-1)
        
        # reconstruct trajectory
        trajectory = self.reconstruct_trajectory_from_speed_and_yaw_rate(speeds, yaw_rates)
        
        # apply frame_interval
        steering = steering[::self.frame_interval]
        trajectory = trajectory[::self.frame_interval]
        if self.num_frames is not None:
            if len(steering) < self.num_frames:
                print(f"Warning: Number of steering signals {len(steering)} is less than the required {self.num_frames}")
                raise ValueError(f"Number of steering signals {len(steering)} is less than the required {self.num_frames}")
            steering = steering[:self.num_frames]
            trajectory = trajectory[:self.num_frames]
        
        assert len(steering) == images.shape[0], f"Steering signals {len(steering)} and images {images.shape[0]} do not match"
        assert len(trajectory) == images.shape[0], f"Trajectory {len(trajectory)} and images {images.shape[0]} do not match"
        
        return {
            'images': images,
            'steering': torch.from_numpy(steering).float(),
            'trajectory': torch.from_numpy(trajectory).float(),
            'frame_rate': torch.tensor(self.frame_rate).float(),
        }