"""
Utility functions for SynFlow pruning.
"""

import torch
import torch.nn as nn
from typing import Dict
import logging


def apply_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name])

                # Also zero out gradients for pruned weights if they exist
                if param.grad is not None:
                    param.grad.data.mul_(masks[name])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_parameters(model: nn.Module, masks: Dict[str, torch.Tensor]) -> int:
    """
    Count number of non-zero (unpruned) parameters.
    """
    total = 0
    for name, param in model.named_parameters():
        if name in masks:
            total += masks[name].sum().item()
    return int(total)


def get_sparsity_stats(masks: Dict[str, torch.Tensor]) -> Dict:

    total_params = 0
    remaining_params = 0
    by_layer = {}

    for name, mask in masks.items():
        layer_total = mask.numel()
        layer_remaining = mask.sum().item()

        total_params += layer_total
        remaining_params += layer_remaining

        by_layer[name] = {
            'total': layer_total,
            'remaining': int(layer_remaining),
            'remaining_pct': 100.0 * layer_remaining / layer_total
        }

    overall_pct = 100.0 * remaining_params / total_params if total_params > 0 else 0.0

    return {
        'overall': overall_pct,
        'total_params': total_params,
        'remaining_params': int(remaining_params),
        'by_layer': by_layer
    }


def print_sparsity_stats(masks: Dict[str, torch.Tensor], prefix: str = "") -> None:

    stats = get_sparsity_stats(masks)

    logging.info(f"{prefix}Sparsity Statistics:")
    logging.info(f"{prefix}  Overall: {stats['overall']:.2f}% remaining")
    logging.info(f"{prefix}  Total params: {stats['total_params']:,}")
    logging.info(f"{prefix}  Remaining: {stats['remaining_params']:,}")
    logging.info(f"{prefix}  Pruned: {stats['total_params'] - stats['remaining_params']:,}")

    logging.debug(f"{prefix}Per-layer sparsity:")
    for layer_name, layer_stats in stats['by_layer'].items():
        logging.debug(f"{prefix}  {layer_name}: {layer_stats['remaining_pct']:.2f}% "
                     f"({layer_stats['remaining']}/{layer_stats['total']})")


def save_mask(masks: Dict[str, torch.Tensor], path: str) -> None:
    torch.save(masks, path)
    logging.info(f"Masks saved to: {path}")


def load_mask(path: str, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
    masks = torch.load(path, map_location=device)
    logging.info(f"Masks loaded from: {path}")
    return masks


def get_compression_ratio(masks: Dict[str, torch.Tensor]) -> float:
    stats = get_sparsity_stats(masks)
    if stats['remaining_params'] == 0:
        return float('inf')
    return stats['total_params'] / stats['remaining_params']
