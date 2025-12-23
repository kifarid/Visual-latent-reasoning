import os
from PIL import Image
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ThroughputMonitor
from pytorch_lightning.utilities import rank_zero_only

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        val_batch_frequency=None,
        epoch_frequency=1,
    ):
        super().__init__()
        self.batch_freq_train = batch_frequency
        self.batch_freq_val = val_batch_frequency if val_batch_frequency is not None else batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._testensorboard,
        }
        if wandb is not None:
            self.logger_log_images[pl.loggers.WandbLogger] = self._log_wandb
        self.increase_log_steps = increase_log_steps
        self.log_steps = {
            "train": self._build_log_steps(self.batch_freq_train),
            "val": self._build_log_steps(self.batch_freq_val),
        }
        self.clamp = clamp
        # Gate logging so it only happens every `epoch_frequency` epochs (1 = every epoch).
        self.epoch_frequency = 1 if epoch_frequency is None else max(1, int(epoch_frequency))

    def _build_log_steps(self, freq):
        if freq is None or freq <= 0:
            return []
        if not self.increase_log_steps:
            return [freq]
        max_power = int(np.log2(freq)) if freq > 0 else 0
        return [2 ** n for n in range(max_power + 1)]

    def _should_log_epoch(self, current_epoch: int) -> bool:
        return (current_epoch % self.epoch_frequency) == 0


    @rank_zero_only
    def _testensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = grid.clamp(-1, 1.)
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w            
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _log_wandb(self, pl_module, images, batch_idx, split):
        if wandb is None:
            return
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = grid.clamp(-1., 1.)
            grid = (grid + 1.0) / 2.0
            pl_module.logger.experiment.log(
                {f"{split}/{k}": wandb.Image(grid)},
                step=pl_module.global_step,
            )

    # This code will only be executed by the process with rank 0
    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = grid.clamp(-1., 1.)  # -1,1 -> -1,1; c,h,w
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        current_epoch = getattr(pl_module, "current_epoch", 0)
        if (self._should_log_epoch(current_epoch) and
                self.check_frequency(batch_idx, split) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx, split):
        freq = self.batch_freq_train if split == "train" else self.batch_freq_val
        steps = self.log_steps.get(split, [])
        if freq and freq > 0 and (batch_idx % freq) == 0:
            return True
        if batch_idx in steps:
            try:
                steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class ThroughputMonitorDictKeyBatch(ThroughputMonitor):
    def __init__(self, key='images', **kwargs):
        super().__init__(batch_size_fn=lambda batch: batch[key].size(0), **kwargs)
