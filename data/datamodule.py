from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from omegaconf import ListConfig, DictConfig

from main import logger, instantiate_from_config

from .utils import custom_collate


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size=None, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, dbg=False, shuffle_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.shuffle_val = shuffle_val
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.dbg = dbg
        
        if self.wrap:
            raise NotImplementedError("Wrapped datasets not implemented")

    def setup(self, stage=None):
        self.datasets = dict()
        for k, cfg in self.dataset_configs.items():
            logger.info("Loading dataset: %s", k)
            if isinstance(cfg, (list, ListConfig)):
                datasets = [instantiate_from_config(c) for c in cfg]
                self.datasets[k] = ConcatDataset(datasets)
                [logger.info(d) for d in datasets]
            elif isinstance(cfg, DictConfig):
                ds = instantiate_from_config(cfg)
                self.datasets[k] = ds
                logger.info(ds)
            else:
                raise ValueError(f"Invalid dataset config: {cfg}")

    def _train_dataloader(self):
        if self.dbg:
            #prefetch_kwargs = {"prefetch_factor": 1} if self.num_workers and self.num_workers > 0 else {}
            sampler = DistributedSampler(self.datasets["train"], shuffle=True) # if self.trainer.use_ddp else None
            return DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                #**prefetch_kwargs,
                shuffle=False,
                collate_fn=custom_collate,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
            )
        else:
            #prefetch_kwargs = {"prefetch_factor": 1} if self.num_workers and self.num_workers > 0 else {}
            return DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                #**prefetch_kwargs,
                shuffle=True,
                #collate_fn=custom_collate,
                pin_memory=True,
                drop_last=True,
            )

    def _val_dataloader(self):
        prefetch_kwargs = {"prefetch_factor": 1} if self.num_workers and self.num_workers > 0 else {}
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            **prefetch_kwargs,
            pin_memory=True,
            shuffle=self.shuffle_val,
            collate_fn=custom_collate,
        )

    def _test_dataloader(self):
        prefetch_kwargs = {"prefetch_factor": 1} if self.num_workers and self.num_workers > 0 else {}
        return DataLoader(
            self.datasets["test"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            **prefetch_kwargs,
            pin_memory=True,
            collate_fn=custom_collate,
        )
