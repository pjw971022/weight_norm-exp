from dataclasses import dataclass
import os
from typing import List


@dataclass
class TrainConfig:
    model: str = 'simple_cnn'
    hidden_dims: List = 784, 128
    output_dims: int = 10
    optimizer: str = 'adam'
    lr: float = 1e-3
    batch_size: int = 128
    training_epochs: int = 100
    gpus: int = 0

    save_dir_path: str = 'checkpoints'

