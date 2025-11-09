"""
Utility functions for Lottery Ticket Hypothesis experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def copy_model_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.clone().detach()
    return weights


def load_model_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """
    Load weights into model.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])


def reinitialize_model(model: nn.Module, initialization_method: str = 'glorot') -> None:
    """
    Randomly reinitialize model weights (for comparison with winning tickets).
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if initialization_method == 'glorot':
                nn.init.xavier_normal_(module.weight)
            elif initialization_method == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif initialization_method == 'normal':
                nn.init.normal_(module.weight, mean=0, std=0.01)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            if initialization_method == 'glorot':
                nn.init.xavier_normal_(module.weight)
            elif initialization_method == 'kaiming':
                nn.init.kaiming_normal_(module.weight)
            elif initialization_method == 'normal':
                nn.init.normal_(module.weight, mean=0, std=0.01)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count the number of parameters in a model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module, masks: Optional[Dict[str, torch.Tensor]] = None) -> int:
    if masks is not None:
        return int(sum(mask.sum().item() for mask in masks.values()))
    else:
        return sum((param != 0).sum().item() for param in model.parameters() if param.requires_grad)


def get_layer_names(model: nn.Module, include_biases: bool = False) -> list:
    names = []
    for name, param in model.named_parameters():
        if not include_biases and 'bias' in name:
            continue
        if param.requires_grad:
            names.append(name)
    return names


def get_conv_layer_names(model: nn.Module) -> list:
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) == 4:  # Conv weights are 4D
            names.append(name)
    return names


def get_fc_layer_names(model: nn.Module) -> list:
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) == 2:  # FC weights are 2D
            names.append(name)
    return names


def compare_weights(weights1: Dict[str, torch.Tensor],
                   weights2: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compare two sets of weights
    """
    l2_distances = []
    cosine_sims = []

    for name in weights1.keys():
        if name in weights2:
            w1 = weights1[name].flatten()
            w2 = weights2[name].flatten()

            # L2 distance
            l2_dist = torch.norm(w1 - w2).item()
            l2_distances.append(l2_dist)

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                w1.unsqueeze(0), w2.unsqueeze(0)
            ).item()
            cosine_sims.append(cos_sim)

    return {
        'mean_l2_distance': sum(l2_distances) / len(l2_distances) if l2_distances else 0,
        'mean_cosine_similarity': sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0,
    }
