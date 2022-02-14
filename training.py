import os

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from config import TrainConfig

from simple_cnn import SimpleCNN
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from wide_resnet import wide_resnet28
from data_module import CIFAR10DataModule, CIFAR100DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel as DDP

def train(cfg: TrainConfig):
    model_lists = {"simple_cnn": SimpleCNN, "resnet34": resnet34(cfg.lr), "resnet50": resnet50(cfg.lr), "wide_resnet": wide_resnet28(cfg.lr, cfg.dropout_rate, cfg.num_classes, cfg.dropout_mode, cfg.dropout_rate_bound)}
    optimizer_lists = {"adam": optim.Adam, 'sgd': optim.SGD, 'adamW':optim.AdamW,"RMSprop":optim.RMSprop,"adamax":optim.Adamax,"adadelta":optim.Adadelta}
    optimizer = optimizer_lists[cfg.optimizer]
    data_modules ={"cifar10": CIFAR10DataModule(cfg.batch_size), "cifar100":CIFAR100DataModule(cfg.batch_size) }
    model_type = model_lists[cfg.model]
    # data_module = CIFAR10DataModule(cfg.batch_size)
    data_module = data_modules[cfg.dataset]
    # model = model_type(cfg.hidden_dims, cfg.output_dims, cfg.lr, optimizer)
    model = model_type
    save_dir_path = to_absolute_path(cfg.save_dir_path)
    os.makedirs(save_dir_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir_path, save_top_k=1, monitor='test_accuracy')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir_path,cfg.log_name)

    trainer = pl.Trainer(max_epochs=cfg.training_epochs, gpus=cfg.gpus, strategy=cfg.strategy, callbacks=[checkpoint_callback])
    trainer.logger = tb_logger
    trainer.fit(model, data_module)