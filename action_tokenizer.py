import torch
import numpy as np
from scipy import spatial
from typing import List, Tuple, Dict
import logging
from config import Config
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionTokenizerException(Exception):
    """Base class for exceptions in this module."""
    pass

class FASTTokenizer:
    """
    Implements FAST (Flexible Action Sequence Tokenization) for discretizing continuous robot actions into tokens during policy fine-tuning stage.

    Attributes:
        config (Config): Configuration object containing settings for the tokenizer.
        vocab (Dict): Vocabulary of action tokens.
        action_bounds (List): List of bounds for each action dimension.
    """

    def __init__(self, config: Config):
        """
        Initializes the FASTTokenizer.

        Args:
            config (Config): Configuration object containing settings for the tokenizer.
        """
        self.config = config
        self.vocab = {}
        self.action_bounds = []

        # Load vocabulary from file if it exists
        if self.config.vocab_file:
            try:
                self.load_vocab(self.config.vocab_file)
            except FileNotFoundError:
                logger.warning(f"Vocabulary file {self.config.vocab_file} not found. Creating a new vocabulary.")

        # Create vocabulary if it doesn't exist
        if not self.vocab:
            self.create_vocab()

    def tokenize(self, actions: np.ndarray) -> List[int]:
        """
        Tokenizes a sequence of continuous robot actions into discrete tokens.

        Args:
            actions (np.ndarray): Sequence of continuous robot actions.

        Returns:
            List[int]: List of discrete action tokens.
        """
        tokens = []
        for action in actions:
            # Find the closest action token in the vocabulary
            token = self.find_closest_token(action)
            tokens.append(token)
        return tokens

    def detokenize(self, tokens: List[int]) -> np.ndarray:
        """
        Detokenizes a sequence of discrete action tokens into continuous robot actions.

        Args:
            tokens (List[int]): List of discrete action tokens.

        Returns:
            np.ndarray: Sequence of continuous robot actions.
        """
        actions = []
        for token in tokens:
            # Get the action corresponding to the token from the vocabulary
            action = self.get_action_from_token(token)
            actions.append(action)
        return np.array(actions)

    def get_action_bounds(self) -> List[Tuple[float, float]]:
        """
        Gets the bounds for each action dimension.

        Returns:
            List[Tuple[float, float]]: List of bounds for each action dimension.
        """
        return self.action_bounds

    def create_vocab(self) -> None:
        """
        Creates a vocabulary of action tokens by discretizing the action space.
        """
        # Calculate the number of tokens for each action dimension
        num_tokens = self.config.num_tokens
        action_dim = self.config.action_dim

        # Calculate the bounds for each action dimension
        self.action_bounds = []
        for i in range(action_dim):
            min_bound = self.config.action_min[i]
            max_bound = self.config.action_max[i]
            self.action_bounds.append((min_bound, max_bound))

        # Create the vocabulary
        self.vocab = {}
        token_idx = 0
        for i in range(num_tokens ** action_dim):
            # Calculate the action corresponding to the current token
            action = self.get_action_from_token_idx(i, num_tokens, action_dim)

            # Add the action to the vocabulary
            self.vocab[token_idx] = action
            token_idx += 1

        # Save the vocabulary to file
        self.save_vocab(self.config.vocab_file)

    def find_closest_token(self, action: np.ndarray) -> int:
        """
        Finds the closest action token in the vocabulary to the given action.

        Args:
            action (np.ndarray): Continuous robot action.

        Returns:
            int: Index of the closest action token in the vocabulary.
        """
        # Calculate the distance between the action and each token in the vocabulary
        distances = []
        for token, token_action in self.vocab.items():
            distance = spatial.distance.euclidean(action, token_action)
            distances.append((token, distance))

        # Find the token with the minimum distance
        closest_token = min(distances, key=lambda x: x[1])[0]
        return closest_token

    def get_action_from_token(self, token: int) -> np.ndarray:
        """
        Gets the action corresponding to the given token from the vocabulary.

        Args:
            token (int): Index of the action token in the vocabulary.

        Returns:
            np.ndarray: Continuous robot action corresponding to the token.
        """
        return self.vocab[token]

    def get_action_from_token_idx(self, token_idx: int, num_tokens: int, action_dim: int) -> np.ndarray:
        """
        Calculates the action corresponding to the given token index.

        Args:
            token_idx (int): Index of the action token.
            num_tokens (int): Number of tokens for each action dimension.
            action_dim (int): Number of action dimensions.

        Returns:
            np.ndarray: Continuous robot action corresponding to the token index.
        """
        action = np.zeros(action_dim)
        for i in range(action_dim):
            # Calculate the value for the current action dimension
            value = (token_idx // (num_tokens ** i)) % num_tokens
            value = value / (num_tokens - 1) * (self.action_bounds[i][1] - self.action_bounds[i][0]) + self.action_bounds[i][0]
            action[i] = value
        return action

    def load_vocab(self, vocab_file: str) -> None:
        """
        Loads the vocabulary from a file.

        Args:
            vocab_file (str): Path to the vocabulary file.
        """
        try:
            with open(vocab_file, 'rb') as f:
                self.vocab = np.load(f, allow_pickle=True).item()
        except Exception as e:
            logger.error(f"Failed to load vocabulary from file {vocab_file}: {e}")

    def save_vocab(self, vocab_file: str) -> None:
        """
        Saves the vocabulary to a file.

        Args:
            vocab_file (str): Path to the vocabulary file.
        """
        try:
            with open(vocab_file, 'wb') as f:
                np.save(f, self.vocab)
        except Exception as e:
            logger.error(f"Failed to save vocabulary to file {vocab_file}: {e}")

class Config:
    def __init__(self, num_tokens: int, action_dim: int, action_min: List[float], action_max: List[float], vocab_file: str):
        self.num_tokens = num_tokens
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.vocab_file = vocab_file

def main():
    # Create a configuration object
    config = Config(num_tokens=10, action_dim=3, action_min=[-1.0, -1.0, -1.0], action_max=[1.0, 1.0, 1.0], vocab_file='vocab.npy')

    # Create a FASTTokenizer
    tokenizer = FASTTokenizer(config)

    # Tokenize a sequence of actions
    actions = np.array([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]])
    tokens = tokenizer.tokenize(actions)
    print(tokens)

    # Detokenize a sequence of tokens
    tokens = [5, 7]
    actions = tokenizer.detokenize(tokens)
    print(actions)

if __name__ == "__main__":
    main()