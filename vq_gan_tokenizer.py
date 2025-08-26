import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from taming_transformers import VQGAN
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import logging
import os
import json
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VQGANTokenizerException(Exception):
    """Base exception class for VQGANTokenizer"""
    pass

class VQGANTokenizer:
    """
    Implements the shared VQ-GAN tokenizer for both RGB frames and optical flow RGB images.
    Loads pre-trained tokenizer weights and handles encoding/decoding.

    Attributes:
        config (Config): Configuration object
        vqgan (VQGAN): VQ-GAN model
        tokenizer (AutoTokenizer): Tokenizer model
        device (str): Device to use (e.g. "cuda" or "cpu")
    """

    def __init__(self, config: Config):
        """
        Initializes the VQGANTokenizer.

        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.vqgan = VQGAN(config.vqgan_model_name, config.vqgan_model_size)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
        self.device = config.device
        self.vqgan.to(self.device)

    def load_pretrained(self, weights_path: str):
        """
        Loads pre-trained tokenizer weights.

        Args:
            weights_path (str): Path to pre-trained weights file
        """
        try:
            self.vqgan.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"Loaded pre-trained weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load pre-trained weights: {e}")

    def encode(self, image: torch.Tensor) -> List[int]:
        """
        Encodes an image into a list of tokens.

        Args:
            image (torch.Tensor): Image tensor

        Returns:
            List[int]: List of tokens
        """
        try:
            # Preprocess image
            image = image.to(self.device)
            image = F.interpolate(image, size=(self.config.vqgan_model_size, self.config.vqgan_model_size), mode="bilinear")

            # Encode image
            z = self.vqgan.encode(image)
            z_q = self.vqgan.quantize(z)
            tokens = z_q.argmax(dim=1).flatten().tolist()

            return tokens
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise VQGANTokenizerException(f"Failed to encode image: {e}")

    def decode(self, tokens: List[int]) -> torch.Tensor:
        """
        Decodes a list of tokens into an image.

        Args:
            tokens (List[int]): List of tokens

        Returns:
            torch.Tensor: Decoded image tensor
        """
        try:
            # Convert tokens to tensor
            tokens = torch.tensor(tokens, device=self.device)

            # Decode tokens
            z_q = self.vqgan.quantize.embedding(tokens)
            z = self.vqgan.decode(z_q)
            image = F.interpolate(z, size=(self.config.image_size, self.config.image_size), mode="bilinear")

            return image
        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            raise VQGANTokenizerException(f"Failed to decode tokens: {e}")

    def get_vocab_size(self) -> int:
        """
        Returns the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size
        """
        return self.vqgan.quantize.num_embeddings

class VQGANDataset(Dataset):
    """
    Dataset class for VQ-GAN tokenizer.

    Attributes:
        images (List[torch.Tensor]): List of image tensors
        tokenizer (VQGANTokenizer): Tokenizer instance
    """

    def __init__(self, images: List[torch.Tensor], tokenizer: VQGANTokenizer):
        self.images = images
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[int]]:
        image = self.images[index]
        tokens = self.tokenizer.encode(image)

        return image, tokens

def main():
    # Load configuration
    config = Config()

    # Create tokenizer instance
    tokenizer = VQGANTokenizer(config)

    # Load pre-trained weights
    tokenizer.load_pretrained(config.pretrained_weights_path)

    # Test encoding and decoding
    image = torch.randn(1, 3, 256, 256)
    tokens = tokenizer.encode(image)
    decoded_image = tokenizer.decode(tokens)

    logger.info(f"Encoded image into {len(tokens)} tokens")
    logger.info(f"Decoded tokens into image with shape {decoded_image.shape}")

if __name__ == "__main__":
    main()