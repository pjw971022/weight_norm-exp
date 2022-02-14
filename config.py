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
    dropout_rate: float = 0.3
    dropout_rate_bound: float = 0.1
    dataset: str = 'cifar10'
    dropout_mode: int = 0
    num_classes: int = 10
    batch_size: int = 128
    training_epochs: int = 100
    gpus: int = 4
    strategy: str = "ddp" 
    save_dir_path: str = 'checkpoints'
    log_name: str = 'baseline'
