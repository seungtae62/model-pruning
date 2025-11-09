"""
Lottery Ticket Hypothesis Implementation

This module implements the iterative magnitude pruning algorithm from:
"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
Frankle & Carbin, ICLR 2019
"""

from .masks import create_mask, apply_mask, get_sparsity
from .pruner import MagnitudePruner
from .lottery_trainer import LotteryTicketTrainer
from .utils import copy_model_weights, reinitialize_model

__all__ = [
    'create_mask',
    'apply_mask',
    'get_sparsity',
    'MagnitudePruner',
    'LotteryTicketTrainer',
    'copy_model_weights',
    'reinitialize_model',
]
