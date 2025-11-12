"""
SynFlow Pruning at Initialization

Based on "Pruning neural networks without any data by iteratively conserving synaptic flow"
Tanaka et al., NeurIPS 2020
"""

from .synflow_pruner import SynFlowPruner, get_input_shape
from .synflow_trainer import SynFlowTrainer
from .utils import (
    apply_mask,
    count_parameters,
    count_nonzero_parameters,
    get_sparsity_stats,
    print_sparsity_stats,
    save_mask,
    load_mask,
    get_compression_ratio
)

__all__ = [
    'SynFlowPruner',
    'SynFlowTrainer',
    'get_input_shape',
    'apply_mask',
    'count_parameters',
    'count_nonzero_parameters',
    'get_sparsity_stats',
    'print_sparsity_stats',
    'save_mask',
    'load_mask',
    'get_compression_ratio'
]
