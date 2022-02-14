import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10,CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import os
from typing import Optional

from hydra.utils import to_absolute_path, get_original_cwd
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0, 0, 0], [1, 1, 1])
                                    ])
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = CIFAR10(get_original_cwd(), train=True, download=True, transform=self.transform)
        self.test_set = CIFAR10(get_original_cwd(), train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)



class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0, 0, 0], [1, 1, 1])
                                    ])
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = CIFAR100(get_original_cwd(), train=True, download=True, transform=self.transform)
        self.test_set = CIFAR100(get_original_cwd(), train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)