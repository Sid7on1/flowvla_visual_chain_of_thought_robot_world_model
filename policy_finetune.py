import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate import Accelerator
from wandb import WandbCallback
from model_architecture import FlowVLA
from action_tokenizer import ActionTokenizer
from data_loader import RoboticsTaskDataLoader
from typing import Dict, List, Tuple
import logging
import os
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyFinetuneException(Exception):
    """Base exception class for policy finetune module."""
    pass

class PolicyFinetune:
    def __init__(self, 
                 model_name: str, 
                 pre_trained_weights_path: str, 
                 action_tokenizer_path: str, 
                 data_loader: RoboticsTaskDataLoader, 
                 batch_size: int, 
                 num_epochs: int, 
                 learning_rate: float, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the policy finetune module.

        Args:
        - model_name (str): Name of the pre-trained model.
        - pre_trained_weights_path (str): Path to the pre-trained model weights.
        - action_tokenizer_path (str): Path to the action tokenizer.
        - data_loader (RoboticsTaskDataLoader): Data loader for the robotics task.
        - batch_size (int): Batch size for training.
        - num_epochs (int): Number of epochs for training.
        - learning_rate (float): Learning rate for the optimizer.
        - device (str): Device to use for training (default: "cuda" if available, otherwise "cpu").
        """
        self.model_name = model_name
        self.pre_trained_weights_path = pre_trained_weights_path
        self.action_tokenizer_path = action_tokenizer_path
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Load pre-trained model and tokenizer
        self.model = FlowVLA.from_pretrained(model_name)
        self.action_tokenizer = ActionTokenizer.from_pretrained(action_tokenizer_path)

        # Set up optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

        # Set up accelerator
        self.accelerator = Accelerator(device_placement=True, fp16=True)

    def load_pretrained_weights(self) -> None:
        """
        Load pre-trained weights into the model.
        """
        try:
            self.model.load_state_dict(torch.load(self.pre_trained_weights_path, map_location=self.device))
            logger.info("Loaded pre-trained weights from {}".format(self.pre_trained_weights_path))
        except Exception as e:
            logger.error("Failed to load pre-trained weights: {}".format(str(e)))
            raise PolicyFinetuneException("Failed to load pre-trained weights")

    def compute_policy_loss(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            action_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the policy loss.

        Args:
        - input_ids (torch.Tensor): Input IDs for the model.
        - attention_mask (torch.Tensor): Attention mask for the model.
        - action_labels (torch.Tensor): Action labels for the model.

        Returns:
        - torch.Tensor: Policy loss.
        """
        try:
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=action_labels)
            loss = outputs.loss

            return loss
        except Exception as e:
            logger.error("Failed to compute policy loss: {}".format(str(e)))
            raise PolicyFinetuneException("Failed to compute policy loss")

    def validate_policy(self, 
                        input_ids: torch.Tensor, 
                        attention_mask: torch.Tensor, 
                        action_labels: torch.Tensor) -> Dict[str, float]:
        """
        Validate the policy.

        Args:
        - input_ids (torch.Tensor): Input IDs for the model.
        - attention_mask (torch.Tensor): Attention mask for the model.
        - action_labels (torch.Tensor): Action labels for the model.

        Returns:
        - Dict[str, float]: Validation metrics.
        """
        try:
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=action_labels)
            loss = outputs.loss
            accuracy = torch.sum(torch.argmax(outputs.logits, dim=-1) == action_labels) / len(action_labels)

            return {"loss": loss.item(), "accuracy": accuracy.item()}
        except Exception as e:
            logger.error("Failed to validate policy: {}".format(str(e)))
            raise PolicyFinetuneException("Failed to validate policy")

    def save_policy_checkpoint(self, 
                               checkpoint_path: str) -> None:
        """
        Save the policy checkpoint.

        Args:
        - checkpoint_path (str): Path to save the checkpoint.
        """
        try:
            torch.save(self.model.state_dict(), checkpoint_path)
            logger.info("Saved policy checkpoint to {}".format(checkpoint_path))
        except Exception as e:
            logger.error("Failed to save policy checkpoint: {}".format(str(e)))
            raise PolicyFinetuneException("Failed to save policy checkpoint")

    def finetune_policy(self) -> None:
        """
        Finetune the policy.
        """
        try:
            # Load pre-trained weights
            self.load_pretrained_weights()

            # Create data loaders
            train_dataloader = self.data_loader.get_train_dataloader(self.batch_size)
            val_dataloader = self.data_loader.get_val_dataloader(self.batch_size)

            # Train the model
            for epoch in range(self.num_epochs):
                self.model.train()
                total_loss = 0
                for batch in tqdm(train_dataloader, desc="Training"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    action_labels = batch["action_labels"].to(self.device)

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    loss = self.compute_policy_loss(input_ids, attention_mask, action_labels)

                    # Backward pass
                    loss.backward()

                    # Update the model parameters
                    self.optimizer.step()

                    # Update the total loss
                    total_loss += loss.item()

                # Validate the model
                self.model.eval()
                val_loss = 0
                val_accuracy = 0
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validating"):
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        action_labels = batch["action_labels"].to(self.device)

                        # Forward pass
                        metrics = self.validate_policy(input_ids, attention_mask, action_labels)

                        # Update the validation loss and accuracy
                        val_loss += metrics["loss"]
                        val_accuracy += metrics["accuracy"]

                # Save the checkpoint
                self.save_policy_checkpoint("policy_checkpoint_{}.pth".format(epoch))

                # Log the training and validation metrics
                logger.info("Epoch {}: Training Loss = {:.4f}, Validation Loss = {:.4f}, Validation Accuracy = {:.4f}".format(
                    epoch, total_loss / len(train_dataloader), val_loss / len(val_dataloader), val_accuracy / len(val_dataloader)))

        except Exception as e:
            logger.error("Failed to finetune policy: {}".format(str(e)))
            raise PolicyFinetuneException("Failed to finetune policy")

if __name__ == "__main__":
    # Create a policy finetune instance
    policy_finetune = PolicyFinetune(
        model_name="flowvla-base",
        pre_trained_weights_path="pre_trained_weights.pth",
        action_tokenizer_path="action_tokenizer.json",
        data_loader=RoboticsTaskDataLoader(),
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-5
    )

    # Finetune the policy
    policy_finetune.finetune_policy()