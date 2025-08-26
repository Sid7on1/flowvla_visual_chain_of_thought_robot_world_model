import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
from optical_flow_encoder import OpticalFlowEncoder
from tqdm import tqdm
import logging
import hydra
from omegaconf import DictConfig
from typing import List, Tuple, Dict
from einops import rearrange
from accelerate import Accelerator
from datasets import load_dataset
from tokenizers import Tokenizer
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowVLADataset(data.Dataset):
    """
    Custom PyTorch dataset class for loading LIBERO, Bridge V2, and SimplerEnv datasets.
    Handles optical flow pre-loading, tokenization, and sequence construction for both training stages.
    """
    
    def __init__(self, 
                 cfg: DictConfig, 
                 split: str, 
                 dataset_name: str, 
                 tokenizer: Tokenizer, 
                 optical_flow_encoder: OpticalFlowEncoder, 
                 accelerator: Accelerator):
        """
        Initializes the dataset.

        Args:
        - cfg (DictConfig): Configuration object.
        - split (str): Split of the dataset (e.g., 'train', 'val', 'test').
        - dataset_name (str): Name of the dataset (e.g., 'LIBERO', 'Bridge V2', 'SimplerEnv').
        - tokenizer (Tokenizer): Tokenizer instance.
        - optical_flow_encoder (OpticalFlowEncoder): Optical flow encoder instance.
        - accelerator (Accelerator): Accelerator instance.
        """
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.optical_flow_encoder = optical_flow_encoder
        self.accelerator = accelerator
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_paths = self.dataset['image_paths']
        self.flow_paths = self.dataset['flow_paths']
        self.tokenized_sequences = []
        self.preloaded_optical_flow = {}
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)
    
    def load_optical_flow(self, flow_path: str) -> np.ndarray:
        """
        Loads optical flow from a file.

        Args:
        - flow_path (str): Path to the optical flow file.

        Returns:
        - np.ndarray: Loaded optical flow.
        """
        try:
            flow = np.load(flow_path)
            return flow
        except Exception as e:
            logger.error(f"Failed to load optical flow from {flow_path}: {e}")
            return None
    
    def tokenize_sequence(self, sequence: List[str]) -> List[str]:
        """
        Tokenizes a sequence of strings.

        Args:
        - sequence (List[str]): Sequence of strings.

        Returns:
        - List[str]: Tokenized sequence.
        """
        try:
            return self.tokenizer.encode_plus(
                sequence, 
                add_special_tokens=True, 
                max_length=self.cfg.max_length, 
                padding='max_length', 
                truncation=True, 
                return_attention_mask=True, 
                return_tensors='pt'
            )['input_ids'].flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to tokenize sequence: {e}")
            return []
    
    def create_interleaved_sequence(self, image: np.ndarray, flow: np.ndarray) -> List[str]:
        """
        Creates an interleaved sequence of image and flow tokens.

        Args:
        - image (np.ndarray): Image array.
        - flow (np.ndarray): Optical flow array.

        Returns:
        - List[str]: Interleaved sequence.
        """
        try:
            image_tokens = self.tokenize_sequence([self.dataset_name, 'image'])
            flow_tokens = self.tokenize_sequence([self.dataset_name, 'flow'])
            interleaved_sequence = image_tokens + flow_tokens
            return interleaved_sequence
        except Exception as e:
            logger.error(f"Failed to create interleaved sequence: {e}")
            return []
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single item from the dataset.

        Args:
        - idx (int): Index of the item.

        Returns:
        - Dict[str, torch.Tensor]: Item from the dataset.
        """
        try:
            image_path = self.image_paths[idx]
            flow_path = self.flow_paths[idx]
            image = Image.open(image_path)
            flow = self.load_optical_flow(flow_path)
            if flow is not None:
                flow = self.optical_flow_encoder.encode(flow)
                interleaved_sequence = self.create_interleaved_sequence(image, flow)
                self.tokenized_sequences.append(interleaved_sequence)
            return {
                'image': torch.tensor(np.array(image)),
                'flow': torch.tensor(flow) if flow is not None else torch.tensor([]),
                'sequence': torch.tensor(self.tokenize_sequence([self.dataset_name, 'image', 'flow']))
            }
        except Exception as e:
            logger.error(f"Failed to get item {idx}: {e}")
            return {}
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of items.

        Args:
        - batch (List[Dict[str, torch.Tensor]]): Batch of items.

        Returns:
        - Dict[str, torch.Tensor]: Collated batch.
        """
        try:
            images = [item['image'] for item in batch]
            flows = [item['flow'] for item in batch]
            sequences = [item['sequence'] for item in batch]
            return {
                'image': torch.stack(images),
                'flow': torch.stack(flows),
                'sequence': torch.stack(sequences)
            }
        except Exception as e:
            logger.error(f"Failed to collate batch: {e}")
            return {}
    
    def create_dataset(self) -> data.Dataset:
        """
        Creates the dataset.

        Returns:
        - data.Dataset: Created dataset.
        """
        try:
            dataset = FlowVLADataset(
                self.cfg, 
                self.split, 
                self.dataset_name, 
                self.tokenizer, 
                self.optical_flow_encoder, 
                self.accelerator
            )
            return dataset
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
    
    def get_tokenized_sequences(self) -> List[List[str]]:
        """
        Returns the tokenized sequences.

        Returns:
        - List[List[str]]: Tokenized sequences.
        """
        try:
            return self.tokenized_sequences
        except Exception as e:
            logger.error(f"Failed to get tokenized sequences: {e}")
            return []

@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    """
    Main function.

    Args:
    - cfg (DictConfig): Configuration object.
    """
    try:
        accelerator = Accelerator()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        optical_flow_encoder = OpticalFlowEncoder(cfg.optical_flow_encoder)
        dataset = FlowVLADataset(
            cfg, 
            cfg.split, 
            cfg.dataset_name, 
            tokenizer, 
            optical_flow_encoder, 
            accelerator
        )
        dataset.create_dataset()
        tokenized_sequences = dataset.get_tokenized_sequences()
        logger.info(f"Tokenized sequences: {tokenized_sequences}")
    except Exception as e:
        logger.error(f"Failed to run main function: {e}")

if __name__ == "__main__":
    main()