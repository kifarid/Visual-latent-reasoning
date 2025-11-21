import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class SaveCheckpointEveryNSteps(Callback):
    def __init__(self, save_step_frequency, dirpath, filename):
        self.save_step_frequency = save_step_frequency
        self.dirpath = dirpath
        self.filename = filename
        self.step_count = 0

    def on_batch_end(self, trainer, pl_module):
        self.step_count += 1
        if self.step_count % self.save_step_frequency == 0:
            filepath = os.path.join(self.dirpath, self.filename.format(step=self.step_count))
            trainer.save_checkpoint(filepath)


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, end: int = None, dirpath: str = None):
        super().__init__()
        self.every = every
        self.dirpath = dirpath
        self.end = end

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0 and pl_module.current_epoch>0:
            assert self.dirpath is not None
            current = f"{self.dirpath}/epoch-{pl_module.current_epoch}.ckpt"
            prev = (
                f"{self.dirpath}/epoch-{pl_module.current_epoch - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            # prev.unlink(missing_ok=True)
        if self.end is not None and pl_module.current_epoch >= self.end:
            trainer.should_stop = True
            return
