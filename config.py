import hydra
from omegaconf import OmegaConf
from dataclasses import dataclass
import logging
import logging.config
import os
import sys

# Set up logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'config.log',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})

@dataclass
class FlowVLAConfig:
    """FlowVLA configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = 'VisualChainOfThought'
    num_layers: int = 6
    num_heads: int = 8
    embedding_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10

@dataclass
class TrainingConfig:
    """Training configuration"""
    optimizer: str = 'Adam'
    scheduler: str = 'StepLR'
    patience: int = 5
    early_stopping: bool = True
    validation_split: float = 0.2
    seed: int = 42

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = 'FlowVLA'
    data_path: str = 'data/FlowVLA'
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

def setup_config(config_path: str) -> FlowVLAConfig:
    """Setup configuration from Hydra"""
    config = OmegaConf.load(config_path)
    return FlowVLAConfig(
        model=ModelConfig(**config.model),
        training=TrainingConfig(**config.training),
        data=DataConfig(**config.data)
    )

def validate_config(config: FlowVLAConfig) -> None:
    """Validate configuration"""
    if not isinstance(config.model, ModelConfig):
        raise ValueError("Invalid model configuration")
    if not isinstance(config.training, TrainingConfig):
        raise ValueError("Invalid training configuration")
    if not isinstance(config.data, DataConfig):
        raise ValueError("Invalid data configuration")

def get_config() -> FlowVLAConfig:
    """Get configuration from Hydra"""
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    config = setup_config(config_path)
    validate_config(config)
    return config

if __name__ == '__main__':
    config = get_config()
    logging.info(config)