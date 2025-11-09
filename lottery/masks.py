"""
Mask management for Lottery Ticket Hypothesis.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import json
import logging


def create_mask(model: nn.Module, prune_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Create a random mask (for initialization).
    """
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Only mask weights, not biases
            masks[name] = torch.ones_like(param, dtype=torch.float32)
    return masks


def apply_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Apply masks to model parameters (zero out pruned weights).
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])


def get_sparsity(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
            'remaining': layer_remaining,
            'sparsity': 100.0 * layer_remaining / layer_total
        }

    overall_sparsity = 100.0 * remaining_params / total_params if total_params > 0 else 0.0

    return {
        'overall': overall_sparsity,
        'total_params': total_params,
        'remaining_params': remaining_params,
        'layers': layer_stats
    }


def save_mask(masks: Dict[str, torch.Tensor], filepath: str) -> None:
    """
    Save masks to file.
    """
    # Convert masks to CPU and save
    cpu_masks = {name: mask.cpu() for name, mask in masks.items()}
    torch.save(cpu_masks, filepath)

    # Save sparsity info
    stats = get_sparsity(masks)
    json_path = filepath.replace('.pth', '_stats.json')
    with open(json_path, 'w') as f:
        serializable_stats = {
            'overall': stats['overall'],
            'total_params': stats['total_params'],
            'remaining_params': stats['remaining_params'],
        }
        json.dump(serializable_stats, f, indent=2)


def load_mask(filepath: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Load masks from file.
    """
    if device is None:
        device = torch.device('cpu')

    masks = torch.load(filepath, map_location=device)
    return masks


def combine_masks(mask1: Dict[str, torch.Tensor],
                  mask2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Combine two masks
    Useful for iterative pruning where we want to keep only weights that survived both rounds.
    """
    combined = {}
    for name in mask1.keys():
        if name in mask2:
            combined[name] = mask1[name] * mask2[name]
        else:
            combined[name] = mask1[name]
    return combined


def print_sparsity_stats(masks: Dict[str, torch.Tensor], prefix: str = "") -> None:

    stats = get_sparsity(masks)

    logging.info(f"{prefix}Sparsity Statistics:")
    logging.info(f"Overall: {stats['overall']:.2f}% of weights remaining")
    logging.info(f"Total parameters: {stats['total_params']:,}")
    logging.info(f"Remaining parameters: {stats['remaining_params']:,}")
    logging.info(f"Pruned parameters: {stats['total_params'] - stats['remaining_params']:,}")

    logging.info(f"Per-layer breakdown:")
    for name, layer_stat in stats['layers'].items():
        logging.info(f"{name}: {layer_stat['sparsity']:.2f}% "
              f"({layer_stat['remaining']:,}/{layer_stat['total']:,})")
