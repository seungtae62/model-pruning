"""
Magnitude-based pruning for Lottery Ticket Hypothesis.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class MagnitudePruner:
    """
    Magnitude-based pruner that removes weights with smallest absolute values.
        - Layer-wise pruning: prune p% within each layer independently
        - Global pruning: prune p% across all layers collectively
        - Different pruning rates for different layer types (conv vs FC)
    """

    def __init__(self,
                 pruning_rate: float = 0.2,
                 pruning_rate_conv: Optional[float] = None,
                 pruning_rate_fc: Optional[float] = None,
                 global_pruning: bool = False):

        self.pruning_rate = pruning_rate
        self.pruning_rate_conv = pruning_rate_conv if pruning_rate_conv is not None else pruning_rate
        self.pruning_rate_fc = pruning_rate_fc if pruning_rate_fc is not None else pruning_rate
        self.global_pruning = global_pruning

    def prune(self,
              model: nn.Module,
              current_masks: Optional[Dict[str, torch.Tensor]] = None,
              layer_names_to_prune: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Prune the model based on weight magnitudes.
        """
        if self.global_pruning:
            return self._prune_global(model, current_masks, layer_names_to_prune)
        else:
            return self._prune_layerwise(model, current_masks, layer_names_to_prune)

    def _prune_layerwise(self,
                         model: nn.Module,
                         current_masks: Optional[Dict[str, torch.Tensor]],
                         layer_names_to_prune: Optional[List[str]]) -> Dict[str, torch.Tensor]:
        new_masks = {}

        for name, param in model.named_parameters():
            # Prune weights w/o biases
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            if layer_names_to_prune is not None and name not in layer_names_to_prune:
                if current_masks is not None and name in current_masks:
                    new_masks[name] = current_masks[name]
                else:
                    new_masks[name] = torch.ones_like(param, dtype=torch.float32)
                continue

            # Determine pruning rate for this layer
            if 'conv' in name.lower() or len(param.shape) == 4:  # Conv layer
                rate = self.pruning_rate_conv
            else:  # FC layer
                rate = self.pruning_rate_fc

            # Get weights
            weights = param.data.clone()
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            # Calculate threshold for this layer
            weights_abs = torch.abs(weights)

            # Consider non-zero weights (from previous pruning)
            if current_masks is not None and name in current_masks:
                non_zero_weights = weights_abs[current_masks[name] > 0]
            else:
                non_zero_weights = weights_abs.flatten()

            if len(non_zero_weights) == 0:
                new_masks[name] = torch.zeros_like(param, dtype=torch.float32)
                continue

            # Find threshold: prune lowest `rate` fraction of remaining weights
            k = int(len(non_zero_weights) * rate)
            if k == 0:  # Nothing to prune
                if current_masks is not None and name in current_masks:
                    new_masks[name] = current_masks[name]
                else:
                    new_masks[name] = torch.ones_like(param, dtype=torch.float32)
                continue

            threshold = torch.kthvalue(non_zero_weights, k)[0]

            # Create new mask
            new_mask = (weights_abs > threshold).float()

            # Combine with previous mask if doing iterative pruning
            if current_masks is not None and name in current_masks:
                new_mask = new_mask * current_masks[name]

            new_masks[name] = new_mask

        return new_masks

    def _prune_global(self,
                      model: nn.Module,
                      current_masks: Optional[Dict[str, torch.Tensor]],
                      layer_names_to_prune: Optional[List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prune globally across all layers (or specified layers).
        """
        all_weights = []
        param_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            if layer_names_to_prune is not None and name not in layer_names_to_prune:
                continue

            weights = param.data.clone()
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            weights_abs = torch.abs(weights)

            # Only consider non-zero weights
            if current_masks is not None and name in current_masks:
                non_zero_weights = weights_abs[current_masks[name] > 0]
            else:
                non_zero_weights = weights_abs.flatten()

            all_weights.append(non_zero_weights)
            param_names.append(name)

        # Concatenate all weights
        all_weights_tensor = torch.cat(all_weights)

        # Calculate global threshold
        k = int(len(all_weights_tensor) * self.pruning_rate)
        if k == 0:
            return current_masks if current_masks is not None else {
                name: torch.ones_like(param, dtype=torch.float32)
                for name, param in model.named_parameters()
                if param.requires_grad and len(param.shape) > 1
            }

        threshold = torch.kthvalue(all_weights_tensor, k)[0]

        # Create new masks based on global threshold
        new_masks = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or len(param.shape) <= 1:
                continue

            if layer_names_to_prune is not None and name not in layer_names_to_prune:
                if current_masks is not None and name in current_masks:
                    new_masks[name] = current_masks[name]
                else:
                    new_masks[name] = torch.ones_like(param, dtype=torch.float32)
                continue

            weights = param.data.clone()
            if current_masks is not None and name in current_masks:
                weights = weights * current_masks[name]

            weights_abs = torch.abs(weights)
            new_mask = (weights_abs > threshold).float()

            # Combine with previous mask
            if current_masks is not None and name in current_masks:
                new_mask = new_mask * current_masks[name]

            new_masks[name] = new_mask

        return new_masks


def prune_model_by_percentile(model: nn.Module,
                               percentile: float,
                               current_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    pruner = MagnitudePruner(pruning_rate=(percentile / 100.0), global_pruning=True)
    return pruner.prune(model, current_masks)
