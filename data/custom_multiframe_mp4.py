import os
import random
from torchvision import transforms
from data.custom_multiframe import MultiHDF5DatasetMultiFrameIdxMapping
import torch

try:
    from torchcodec.decoders import VideoDecoder, SimpleVideoDecoder
    TORCHCODEC_AVAILABLE = True
    import sys
except ImportError:
    TORCHCODEC_AVAILABLE = False
    TORCHCODEC_VERSION = '-1'
    TORCHCODEC_HASAPPROXIMATE = False

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class RandomShiftCropTensorBatch(object):
    def __init__(self, size, max_shift_horizontal=60, max_shift_vertical=60):
        """
        size: Crop size. If an int is provided, the crop will be (size, size). 
              If a tuple is provided, it should be (crop_width, crop_height).
        max_shift_horizontal: Maximum horizontal shift (in pixels) from the center of the crop.
        max_shift_vertical: Maximum vertical shift (in pixels) from the center of the crop.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical

    def get_params(self, tensor_batch):
        width, height = tensor_batch.shape[-1], tensor_batch.shape[-2]
        crop_height, crop_width = self.size

        # Calculate the center coordinates for the crop
        center_left = (width - crop_width) // 2
        center_top = (height - crop_height) // 2

        # Apply random horizontal and vertical shifts
        shift_horizontal = random.randint(-self.max_shift_horizontal, self.max_shift_horizontal)
        shift_vertical = random.randint(-self.max_shift_vertical, self.max_shift_vertical)

        left = center_left + shift_horizontal
        top = center_top + shift_vertical

        # Clamp the values to ensure the crop is entirely within the image boundaries
        left = max(0, min(left, width - crop_width))
        top = max(0, min(top, height - crop_height))

        return left, top

    def __call__(self, img):
        left, top = self.get_params(img)
        crop_height, crop_width = self.size
        return img[..., top:top + crop_height, left:left + crop_width]


class DecordToTensorBatch:
    def __call__(self, video_array):
        # video_array is a numpy array of shape (num_frames, H, W, C) with values in [0, 255]
        # Convert to torch tensor and permute to (num_frames, C, H, W)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float() / 255.0
        return video_tensor
    

class MultiMP4DatasetMultiFrameIdxMapping(MultiHDF5DatasetMultiFrameIdxMapping):
    def __init__(self, size, mp4_paths_file, num_frames, stored_data_frame_rate=5, frame_rate=5, aug='resize_center', scale_min=0.15, scale_max=0.5, backend=None):

        self.frame_interval = int(stored_data_frame_rate/frame_rate)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        self.mp4_paths_file = mp4_paths_file
        # expand environment variables in path
        with open(os.path.expandvars(mp4_paths_file), 'r') as f:
            self.mp4_paths = f.read().splitlines()
            
        if backend is not None:
            self.backend = backend
            assert self.backend in ['decord', 'torchcodec'], f'Unknown backend {self.backend}'
            assert (self.backend == 'decord' and DECORD_AVAILABLE) or (self.backend == 'torchcodec' and TORCHCODEC_AVAILABLE), f'Specified backend {self.backend} is not available'
        else:
            if DECORD_AVAILABLE:
                self.backend = 'decord'
            elif TORCHCODEC_AVAILABLE:
                self.backend = 'torchcodec'
            else:
                raise ImportError('Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping')
        
        self.scan_mp4_files()
        if self.backend == 'decord':
            self.read_frames = self.read_frames_decord
        elif self.backend == 'torchcodec':
            self.read_frames = self.read_frames_torchcodec
        else:
            raise ImportError('Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping')
        
        self.aug = aug
        if self.aug == 'resize_center':
            self.transform = transforms.Compose([DecordToTensorBatch() if self.backend == 'decord' else transforms.Lambda(lambda x: x/255.0),
                                                 transforms.Resize(min(self.size)),
                                                 transforms.CenterCrop(self.size)])
        elif self.aug == 'random_shift':
            self.custom_crop = RandomShiftCropTensorBatch(size=self.size, max_shift_horizontal=60, max_shift_vertical=30)
            self.transform = transforms.Compose([DecordToTensorBatch() if self.backend == 'decord' else transforms.Lambda(lambda x: x/255.0),
                                                 transforms.Resize(min(self.size)),
                                                 self.custom_crop])
        else:
            raise ValueError(f'Unknown augmentation type: {self.aug}')
    
    def scan_mp4_files(self):
        # for each index we store a reference to (file, starting_frame)
        self.index_to_starting_frame_map = []
        for path in self.mp4_paths:
            if self.backend == 'decord':
                file = VideoReader(path, ctx=decord_cpu(0))
                assert self.stored_data_frame_rate == file.get_avg_fps(), f'Stored data frame rate {self.stored_data_frame_rate} does not match actual frame rate {file.get_avg_fps()} for file {file}'
                video_length = len(file)
            elif self.backend == 'torchcodec':
                file = VideoDecoder(path)
                assert self.stored_data_frame_rate == file.metadata.average_fps, f'Stored data frame rate {self.stored_data_frame_rate} does not match actual frame rate {file.metadata.average_fps} for file {file}'
                video_length = file.metadata.num_frames
            else:
                raise ImportError('Either Decord or TorchCodec must be installed to use MultiMP4DatasetMultiFrameIdxMapping')
            
            # we take every nth frame, as long as we can get num_frames frames after that
            max_frame_index = video_length - self.num_frames*self.frame_interval-1
            for i in range(0, max_frame_index + 1):
                self.index_to_starting_frame_map.append((path, i))
    
    def __str__(self):
        s = f'MultiMP4DatasetMultiFrameIdxMapping({self.mp4_paths_file}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})'
        return s
    
    def apply_transforms(self, images):
        # we just apply the transform to the whole batch tensor
        images = self.transform(images)*2-1
        return images
    
    def read_frames_decord(self, path, start_frame):
        indices = list(range(start_frame, start_frame + self.num_frames*self.frame_interval, self.frame_interval))
        file = VideoReader(path, ctx=decord_cpu(0))
        frames = file.get_batch(indices).asnumpy()
        return frames
    
    def read_frames_torchcodec(self, path, start_frame):
        file = VideoDecoder(path)
        frames = file[start_frame:start_frame + self.num_frames*self.frame_interval:self.frame_interval]
        return frames
    
    def get_images_and_indices(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f'Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}')
        path, start_frame = self.index_to_starting_frame_map[idx]
        frames = self.read_frames(path, start_frame)
        return frames, (None, path, start_frame)