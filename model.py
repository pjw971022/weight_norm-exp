import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import pytorch_lightning as pl

from typing import List
from typing import Union
from omegaconf.listconfig import ListConfig


class SimpleCNN(pl.LightningModule):
    def __init__(self,
                 hidden_dims: ListConfig,
                 output_dim: int = 10,
                 lr: float = 1e-3,
                 optimizer: Union[optim.Adam, optim.SGD, optim.AdamW, optim.RMSprop] = optim.Adam):
        # eg: hidden_dims = [521, 256, 128]
        super().__init__()
        self.save_hyperparameters("optimizer")
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        in_dim = 4096
        self.linear = []
        for out_dim in self.hidden_dims:
            self.linear.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.fc1 = nn.Sequential(
            *self.linear
        )
        self.fc2 = nn.Linear(in_dim, self.output_dim)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        train_accuracy = (torch.argmax(output, dim=1) == y).float().mean().item()
        linear_weights = self.get_linear_weights_norm()

        self.log('train_step_loss', loss, on_step=True, prog_bar=True)

        for idx, val in linear_weights.items():
            self.log(idx, val, on_step=True, prog_bar=True)

        outputs = {"loss": loss, "train_accuracy": train_accuracy}

        return outputs

    def training_epoch_end(self, outputs) -> None:
        train_accuracy = 0
        for dic in outputs:
            train_accuracy += dic['train_accuracy']

        train_accuracy /= len(outputs)
        self.log('train_accuracy', train_accuracy, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        accuracy = (torch.argmax(output, dim=1) == y).float().mean().item()
        return accuracy

    def validation_epoch_end(self, outputs):
        accuracy = 0
        for result in outputs:
            accuracy += result
        accuracy /= len(outputs)

        self.log('test_accuracy', accuracy, prog_bar=True)

    def get_linear_weights(self):
        linear_weight_lists = []

        for m in self.modules():
            if isinstance(m, nn.Linear):
                linear_weight_lists.append(m.weight.cpu().detach().numpy())
        return linear_weight_lists

    def get_linear_weights_norm(self):
        # get frobenius norm (전체 element들을 다 더하는거라 mean, variance를 구할 순 없음)
        linear_weight_lists = self.get_linear_weights()
        result = {}

        for idx, weight in enumerate(linear_weight_lists):
            result[f"fc{idx}"] = np.linalg.norm(weight)

        return result





