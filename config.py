from dataclasses import dataclass
import os
from omegaconf.listconfig import ListConfig



@dataclass
class TrainConfig:
    model: str = 'resnet50'
    hidden_dims: ListConfig = 784, 128
    output_dims: int = 10
    optimizer: str = 'adam'
    lr: float = 1e-3
    batch_size: int = 128
    training_epochs: int = 100
    gpus: int = 0

    save_dir_path: str = 'checkpoints'

