"""
Magnitude-based pruning following Han et al. NIPS 2015.

Prunes connections based on magnitude threshold:
threshold = quality_parameter × std_dev(layer_weights)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import logging


class MagnitudePruner:
    """
    Magnitude-based pruner using threshold based on standard deviation.
    """

    def __init__(self,
                 quality_parameter: Union[float, Dict[str, float]] = 1.0,
                 global_pruning: bool = False):
        self.quality_parameter = quality_parameter
        self.global_pruning = global_pruning

    def calculate_threshold(self,
                           weights: torch.Tensor,
                           quality_param: float) -> float:
        std_dev = torch.std(weights).item()
        threshold = quality_param * std_dev
        return threshold

    def get_quality_param_for_layer(self, layer_name: str) -> float:
        if isinstance(self.quality_parameter, dict):
            return self.quality_parameter.get(layer_name, 1.0)
        else:
            return self.quality_parameter

    def prune(self,
              model: nn.Module,
              current_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Prune model using magnitude-based thresholding.
        """
        if self.global_pruning:
            return self._prune_global(model, current_masks)
        else:
            return self._prune_layerwise(model, current_masks)

    def _prune_layerwise(self,
                        model: nn.Module,
                        current_masks: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Prune each layer independently.
        """
        new_masks = {}

        for name, param in model.named_parameters():
            # Only prune weights (not biases)
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            # Get quality parameter for this layer
            quality_param = self.get_quality_param_for_layer(name)

            # Get current weights
            weights = param.data.clone()

            # Apply existing mask if available
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            # Calculate threshold
            threshold = self.calculate_threshold(weights, quality_param)

            # Create new mask based on magnitude
            new_mask = (torch.abs(weights) >= threshold).float()

            # Combine with existing mask if doing iterative pruning
            if current_masks is not None and name in current_masks:
                new_mask = new_mask * current_masks[name]

            new_masks[name] = new_mask

            # Log pruning info
            total = new_mask.numel()
            remaining = new_mask.sum().item()
            logging.info(f"Layer {name}: threshold={threshold:.6f}, "
                        f"sparsity={100*remaining/total:.2f}% "
                        f"({int(remaining)}/{total})")

        return new_masks

    def _prune_global(self,
                     model: nn.Module,
                     current_masks: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Collect all weights
        all_weights = []
        param_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            weights = param.data.clone()
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            all_weights.append(torch.abs(weights).flatten())
            param_names.append(name)

        # Concatenate all weights
        all_weights_tensor = torch.cat(all_weights)

        # Calculate global threshold
        quality_param = self.quality_parameter if isinstance(self.quality_parameter, float) else 1.0
        threshold = self.calculate_threshold(all_weights_tensor, quality_param)

        logging.info(f"Global threshold: {threshold:.6f}")

        # Create masks based on global threshold
        new_masks = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            weights = param.data.clone()
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            new_mask = (torch.abs(weights) >= threshold).float()

            # Combine with existing mask
            if current_masks is not None and name in current_masks:
                new_mask = new_mask * current_masks[name]

            new_masks[name] = new_mask

            # Log pruning info
            total = new_mask.numel()
            remaining = new_mask.sum().item()
            logging.info(f"Layer {name}: sparsity={100*remaining/total:.2f}% "
                        f"({int(remaining)}/{total})")

        return new_masks


def create_sensitivity_based_quality_params(model: nn.Module,
                                           base_param: float = 1.0) -> Dict[str, float]:
    """
    Create layer-specific quality parameters based on sensitivity.

    Following Han et al., first conv layer is most sensitive.
    """
    quality_params = {}
    conv_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad or len(param.shape) <= 1:
            continue

        if len(param.shape) == 4:  # Conv layer
            conv_count += 1
            if conv_count == 1:
                # First conv is most sensitive - use smaller threshold
                quality_params[name] = base_param * 0.5
            else:
                quality_params[name] = base_param
        else:  # FC layer
            # FC layers less sensitive - can use larger threshold
            quality_params[name] = base_param * 1.5

    return quality_params
