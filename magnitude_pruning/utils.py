"""
Utility functions for magnitude pruning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


def calculate_adjusted_dropout(original_dropout: float,
                               original_connections: int,
                               remaining_connections: int) -> float:
    """
    Calculate adjusted dropout rate for sparse network.

    Based on Han et al. equation:
    Dr = Do × sqrt(Cir / Cio)
    """
    if original_connections == 0:
        return original_dropout

    ratio = remaining_connections / original_connections
    adjusted_dropout = original_dropout * math.sqrt(ratio)

    # Clamp to reasonable range
    adjusted_dropout = max(0.0, min(0.9, adjusted_dropout))

    return adjusted_dropout


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count total number of parameters in model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module,
                             masks: Optional[Dict[str, torch.Tensor]] = None) -> int:
    if masks is not None:
        return int(sum(mask.sum().item() for mask in masks.values()))
    else:
        total = 0
        for param in model.parameters():
            if param.requires_grad:
                total += (param != 0).sum().item()
        return total


def get_sparsity_stats(masks: Dict[str, torch.Tensor]) -> Dict:
    """
    Calculate sparsity statistics from masks.
    """
    total_params = 0
    remaining_params = 0
    layer_stats = {}

    for name, mask in masks.items():
        layer_total = mask.numel()
        layer_remaining = mask.sum().item()

        total_params += layer_total
        remaining_params += layer_remaining

        layer_stats[name] = {
            'total': layer_total,
            'remaining': int(layer_remaining),
            'sparsity': 100.0 * layer_remaining / layer_total if layer_total > 0 else 0.0
        }

    overall_sparsity = 100.0 * remaining_params / total_params if total_params > 0 else 0.0

    return {
        'overall': overall_sparsity,
        'total_params': total_params,
        'remaining_params': int(remaining_params),
        'layers': layer_stats
    }


def get_layer_names_by_type(model: nn.Module, layer_type: str = 'conv') -> list:
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if layer_type == 'conv' and len(param.shape) == 4:
            names.append(name)
        elif layer_type == 'fc' and len(param.shape) == 2:
            names.append(name)

    return names


def copy_model_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.clone().detach()
    return weights


def load_model_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])
