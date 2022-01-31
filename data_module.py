import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import os
from typing import Optional

from hydra.utils import to_absolute_path, get_original_cwd


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = CIFAR10(get_original_cwd(), train=True, download=True, transform=self.transform)
        self.test_set = CIFAR10(get_original_cwd(), train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
