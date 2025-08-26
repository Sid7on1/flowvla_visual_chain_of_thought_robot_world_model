import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_architecture import FlowVLA
from data_loader import LIBERODataset
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from wandb import init, log
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationException(Exception):
    """Base class for evaluation exceptions."""
    pass

class LIBEROEvaluator:
    """Evaluator for LIBERO benchmark."""
    
    def __init__(self, config: Dict):
        """
        Initialize the evaluator.

        Args:
        config (Dict): Configuration dictionary.
        """
        self.config = config
        self.model = FlowVLA(config['model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.dataset = LIBERODataset(config['dataset'])
        self.data_loader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=False)
        self.suites = ['spatial', 'object', 'goal', 'long-horizon']
        self.success_rates = {suite: 0 for suite in self.suites}
        self.results = []

    def evaluate_libero(self):
        """
        Evaluate the model on the LIBERO benchmark.

        Returns:
        Dict: Dictionary containing the success rates for each suite.
        """
        try:
            for suite in self.suites:
                logger.info(f'Evaluating {suite} suite...')
                self.success_rates[suite] = self.compute_success_rate(suite)
                logger.info(f'{suite} suite success rate: {self.success_rates[suite]}')
            return self.success_rates
        except EvaluationException as e:
            logger.error(f'Evaluation failed: {e}')
            return None

    def compute_success_rate(self, suite: str) -> float:
        """
        Compute the success rate for a given suite.

        Args:
        suite (str): Suite name.

        Returns:
        float: Success rate.
        """
        try:
            success_count = 0
            total_count = 0
            for batch in self.data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                success_count += self.run_episode(outputs, labels, suite)
                total_count += len(labels)
            return success_count / total_count
        except EvaluationException as e:
            logger.error(f'Computing success rate failed: {e}')
            return 0

    def run_episode(self, outputs: torch.Tensor, labels: torch.Tensor, suite: str) -> int:
        """
        Run an episode for a given suite.

        Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.
        suite (str): Suite name.

        Returns:
        int: Number of successful episodes.
        """
        try:
            success_count = 0
            for i in range(len(outputs)):
                output = outputs[i]
                label = labels[i]
                if self.check_success(output, label, suite):
                    success_count += 1
            return success_count
        except EvaluationException as e:
            logger.error(f'Running episode failed: {e}')
            return 0

    def check_success(self, output: torch.Tensor, label: torch.Tensor, suite: str) -> bool:
        """
        Check if an episode is successful.

        Args:
        output (torch.Tensor): Model output.
        label (torch.Tensor): Ground truth label.
        suite (str): Suite name.

        Returns:
        bool: Whether the episode is successful.
        """
        try:
            # Implement the success check logic here
            # For example:
            if suite == 'spatial':
                return torch.allclose(output, label, atol=0.1)
            elif suite == 'object':
                return torch.allclose(output, label, atol=0.1)
            elif suite == 'goal':
                return torch.allclose(output, label, atol=0.1)
            elif suite == 'long-horizon':
                return torch.allclose(output, label, atol=0.1)
        except EvaluationException as e:
            logger.error(f'Checking success failed: {e}')
            return False

    def generate_report(self, success_rates: Dict) -> str:
        """
        Generate an evaluation report.

        Args:
        success_rates (Dict): Dictionary containing the success rates for each suite.

        Returns:
        str: Evaluation report.
        """
        try:
            report = ''
            for suite, success_rate in success_rates.items():
                report += f'{suite} suite success rate: {success_rate}\n'
            return report
        except EvaluationException as e:
            logger.error(f'Generating report failed: {e}')
            return ''

    def save_results(self, success_rates: Dict, report: str):
        """
        Save the evaluation results.

        Args:
        success_rates (Dict): Dictionary containing the success rates for each suite.
        report (str): Evaluation report.
        """
        try:
            results_dir = self.config['results_dir']
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            results_file = os.path.join(results_dir, 'evaluation_results.json')
            with open(results_file, 'w') as f:
                json.dump(success_rates, f)
            report_file = os.path.join(results_dir, 'evaluation_report.txt')
            with open(report_file, 'w') as f:
                f.write(report)
        except EvaluationException as e:
            logger.error(f'Saving results failed: {e}')

def main():
    # Initialize the evaluator
    config = {
        'model': {
            'type': 'FlowVLA',
            'params': {
                'num_layers': 6,
                'num_heads': 8,
                'hidden_size': 512,
                'dropout': 0.1
            }
        },
        'dataset': {
            'type': 'LIBERODataset',
            'params': {
                'data_dir': './data',
                'batch_size': 32
            }
        },
        'batch_size': 32,
        'results_dir': './results'
    }
    evaluator = LIBEROEvaluator(config)

    # Evaluate the model
    success_rates = evaluator.evaluate_libero()
    if success_rates is not None:
        report = evaluator.generate_report(success_rates)
        evaluator.save_results(success_rates, report)
        logger.info('Evaluation completed successfully.')
    else:
        logger.error('Evaluation failed.')

if __name__ == '__main__':
    main()