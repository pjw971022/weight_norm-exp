import os

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from config import TrainConfig

from model import SimpleCNN
from data_module import CIFAR10DataModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hydra.utils import to_absolute_path


def train(cfg: TrainConfig):
    model_lists = {"simple_cnn": SimpleCNN}
    optimizer_lists = {"adam": optim.Adam, 'sgd': optim.SGD}
    optimizer = optimizer_lists[cfg.optimizer]

    model_type = model_lists[cfg.model]

    model = model_type(cfg.hidden_dims, cfg.output_dims, cfg.lr, optimizer)
    data_module = CIFAR10DataModule(cfg.batch_size)

    save_dir_path = to_absolute_path(cfg.save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir_path, save_top_k=1, monitor='test_accuracy')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir_path)

    trainer = pl.Trainer(max_epochs=cfg.training_epochs, gpus=cfg.gpus, callbacks=[checkpoint_callback])
    trainer.logger = tb_logger
    trainer.fit(model, data_module)