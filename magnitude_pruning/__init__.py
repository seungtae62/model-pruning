"""
Han Magnitude Pruning Implementation.

Based on "Learning both Weights and Connections for Efficient Neural Networks"
Song Han et al., NIPS 2015

This module implements magnitude-based pruning with fine-tuning.
"""

from .magnitude_pruner import MagnitudePruner
from .magnitude_trainer import MagnitudePruningTrainer
from .utils import calculate_adjusted_dropout, count_parameters, get_sparsity_stats

__all__ = [
    'MagnitudePruner',
    'MagnitudePruningTrainer',
    'calculate_adjusted_dropout',
    'count_parameters',
    'get_sparsity_stats'
]
