"""
SynFlow Pruning at Initialization.

SynFlow uses iterative pruning based on synaptic flow scores calculated from
gradients on all-ones input, making it data-independent.
"""

import torch
import torch.nn as nn
from typing import Dict
import logging


class SynFlowPruner:
    """
    SynFlow pruner - data-independent pruning at initialization.

    Scores weights using: |dR/dw ⊙ w| where R = sum(output) on all-ones input
    """

    def __init__(self,
                 pruning_rate_per_iteration: float = 0.2,
                 num_iterations: int = 100):
        self.pruning_rate_per_iteration = pruning_rate_per_iteration
        self.num_iterations = num_iterations

    def calculate_synflow_scores(self,
                                  model: nn.Module,
                                  input_shape: tuple,
                                  device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Calculate SynFlow scores for all weights.

        Score = |dR/dw ⊙ w| where R = sum(logits) on all-ones input
        """
        model.eval()

        # Store original weights
        original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                original_weights[name] = param.data.clone()

        # Replace all weights with their absolute values
        # This is key to SynFlow - ensures positive flow
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:
                param.data = torch.abs(param.data)

        # Create all-ones input
        input_tensor = torch.ones(input_shape).to(device)
        input_tensor.requires_grad = True

        # Zero gradients
        model.zero_grad()

        # Forward pass
        output = model(input_tensor)

        # Calculate R = sum of all output logits
        R = torch.sum(output)

        # Backward pass to get gradients
        R.backward()

        # Calculate scores: |gradient ⊙ weight|
        scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1 and param.grad is not None:
                # Score = |dR/dw ⊙ w|
                score = torch.abs(param.grad * param.data)
                scores[name] = score.detach()

        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name]

        # Clear gradients
        model.zero_grad()

        return scores

    def prune_by_scores(self,
                        model: nn.Module,
                        scores: Dict[str, torch.Tensor],
                        current_masks: Dict[str, torch.Tensor],
                        pruning_rate: float) -> Dict[str, torch.Tensor]:
        # Collect all scores from unpruned weights
        all_scores = []

        for name, score_tensor in scores.items():
            if name in current_masks:
                # Only consider currently unpruned weights
                mask = current_masks[name]
                unpruned_scores = score_tensor[mask > 0]
                all_scores.append(unpruned_scores.flatten())

        # Concatenate all scores
        all_scores_tensor = torch.cat(all_scores)

        # Calculate threshold: prune lowest `pruning_rate` fraction
        num_to_prune = int(len(all_scores_tensor) * pruning_rate)

        if num_to_prune == 0:
            # Nothing to prune
            return current_masks

        # Find threshold using kthvalue
        threshold = torch.kthvalue(all_scores_tensor, num_to_prune)[0]

        logging.debug(f"SynFlow threshold: {threshold:.6e}, pruning {num_to_prune} weights")

        # Create new masks
        new_masks = {}
        for name, score_tensor in scores.items():
            if name in current_masks:
                # Weights with scores above threshold survive
                new_mask = (score_tensor > threshold).float()
                # Combine with existing mask
                new_mask = new_mask * current_masks[name]
                new_masks[name] = new_mask
            else:
                new_masks[name] = current_masks[name]

        return new_masks

    def iterative_prune(self,
                        model: nn.Module,
                        input_shape: tuple,
                        device: torch.device,
                        target_sparsity: float = 0.8) -> Dict[str, torch.Tensor]:
        """
        Perform iterative SynFlow pruning to reach target sparsity.
        """
        logging.info(f"Starting SynFlow iterative pruning")
        logging.info(f"Target sparsity: {target_sparsity*100:.1f}%, Iterations: {self.num_iterations}")

        # Initialize masks to all ones (no pruning)
        masks = {
            name: torch.ones_like(param, dtype=torch.float32)
            for name, param in model.named_parameters()
            if param.requires_grad and len(param.shape) > 1
        }

        # Calculate total parameters
        total_params = sum(mask.numel() for mask in masks.values())

        for iteration in range(self.num_iterations):
            # Calculate SynFlow scores
            scores = self.calculate_synflow_scores(model, input_shape, device)

            # Prune based on scores
            masks = self.prune_by_scores(
                model, scores, masks, self.pruning_rate_per_iteration
            )

            # Apply masks to model
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.data *= masks[name]

            # Calculate current sparsity
            remaining = sum(mask.sum().item() for mask in masks.values())
            current_sparsity = 1.0 - (remaining / total_params)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logging.info(f"Iteration {iteration+1}/{self.num_iterations}: "
                           f"Sparsity = {current_sparsity*100:.2f}% "
                           f"({remaining:,}/{total_params:,} remaining)")

            # Check if we've reached target sparsity
            if current_sparsity >= target_sparsity:
                logging.info(f"Reached target sparsity at iteration {iteration+1}")
                break

        # Final sparsity report
        remaining = sum(mask.sum().item() for mask in masks.values())
        final_sparsity = 1.0 - (remaining / total_params)
        remaining_pct = (1.0 - final_sparsity) * 100

        logging.info(f"SynFlow pruning complete:")
        logging.info(f"Final sparsity: {final_sparsity*100:.2f}% pruned")
        logging.info(f"Remaining: {remaining_pct:.2f}% ({remaining:,}/{total_params:,})")

        return masks


def get_input_shape(dataset: str, batch_size: int = 64) -> tuple:
    """
    Get input shape for different datasets.
    """
    if dataset.lower() in ['cifar10', 'cifar100']:
        return (batch_size, 3, 32, 32)
    elif dataset.lower() == 'mnist':
        return (batch_size, 1, 28, 28)
    elif dataset.lower() == 'imagenet':
        return (batch_size, 3, 224, 224)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
