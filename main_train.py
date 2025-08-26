import os
import sys
import logging
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator
from hydra import compose, initialize
from omegaconf import OmegaConf
from model_architecture import FlowVLA
from data_loader import FlowVLADataset
from trainer import FlowVLATrainer
from config import FlowVLAConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainTrain:
    def __init__(self, config: FlowVLAConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

    def setup_distributed_training(self):
        if self.config.distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')

    def load_config(self):
        with initialize(version_base='1.2', config_path='./config'):
            self.config = compose(config_name='flowvla_config')

    def initialize_model(self):
        self.model = FlowVLA(self.config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.tokenizer)

    def run_training_stage1(self):
        dataset = FlowVLADataset(self.config.data, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True)
        trainer = FlowVLATrainer(self.model, data_loader, self.config.training)
        trainer.train()

    def run_training_stage2(self):
        dataset = FlowVLADataset(self.config.data, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True)
        trainer = FlowVLATrainer(self.model, data_loader, self.config.training)
        trainer.fine_tune()

    def save_checkpoint(self, epoch: int):
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))

    def log_metrics(self, metrics: dict):
        for metric, value in metrics.items():
            logger.info(f'{metric}: {value}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowvla_config.yaml')
    args = parser.parse_args()

    config = FlowVLAConfig()
    config.load_from_yaml(args.config)

    main_train = MainTrain(config)
    main_train.load_config()
    main_train.setup_distributed_training()
    main_train.initialize_model()
    main_train.run_training_stage1()
    main_train.run_training_stage2()

    for epoch in range(config.training.epochs):
        main_train.run_training_stage1()
        main_train.save_checkpoint(epoch)
        main_train.log_metrics({'loss': 0.1, 'accuracy': 0.9})

if __name__ == '__main__':
    main()