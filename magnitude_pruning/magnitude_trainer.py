"""
Magntitude Pruning Trainer - Fine-tuning sparse networks.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List
import os
import json
import logging
from tqdm import tqdm

from .magnitude_pruner import MagnitudePruner
from .utils import calculate_adjusted_dropout, count_parameters, count_nonzero_parameters, get_sparsity_stats


def apply_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])


class MagnitudePruningTrainer:

    def __init__(self,
                 model: nn.Module,
                 pruner: MagnitudePruner,
                 device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.pruner = pruner
        self.device = device

        # Initialize masks to all ones (no pruning initially)
        self.masks = {
            name: torch.ones_like(param, dtype=torch.float32)
            for name, param in self.model.named_parameters()
            if param.requires_grad and len(param.shape) > 1
        }

        self.pruning_history = []
        self.current_iteration = 0

        # Track original parameter count for dropout adjustment
        self.original_param_count = count_parameters(self.model)

    def fine_tune(self,
                  train_loader,
                  val_loader,
                  optimizer,
                  criterion,
                  scheduler=None,
                  num_epochs: int = 100,
                  original_dropout: float = 0.0,
                  save_dir: Optional[str] = None) -> Dict[str, List]:

        logging.info(f"Fine-tuning Iteration {self.current_iteration}")
        stats = get_sparsity_stats(self.masks)
        logging.info(f"Sparsity: {stats['overall']:.2f}% weights remaining")

        # Calculate adjusted dropout if model uses dropout
        if original_dropout > 0:
            current_param_count = count_nonzero_parameters(self.model, self.masks)
            adjusted_dropout = calculate_adjusted_dropout(
                original_dropout,
                self.original_param_count,
                current_param_count
            )
            # TODO: Actual dropout implementation
            logging.info(f"Dropout adjusted: {original_dropout:.3f} → {adjusted_dropout:.3f}")

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

            logging.info(f"Epoch {epoch}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.2f}%")

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'iteration_{self.current_iteration}_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'iteration': self.current_iteration,
                        'model_state_dict': self.model.state_dict(),
                        'masks': self.masks,
                        'val_acc': best_val_acc,
                        'val_loss': val_metrics['loss'],
                        'sparsity': stats['overall']
                    }, save_path)
                    logging.info(f"Best model saved (Val Acc: {best_val_acc:.2f}%)")

            if scheduler is not None:
                scheduler.step()

        logging.info(f"Iteration {self.current_iteration} complete. Best Val Acc: {best_val_acc:.2f}%")

        return history

    def train_epoch(self,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with mask enforcement.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply mask to gradients
            apply_mask(self.model, self.masks)

            optimizer.step()

            # Apply mask to weights
            apply_mask(self.model, self.masks)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        return {
            'loss': running_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    def validate(self, val_loader, criterion) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return {
            'loss': running_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }

    def prune_model(self) -> None:
        """
        Prune the model using magnitude-based pruning.
        """
        logging.info(f"Pruning model (iteration {self.current_iteration})...")

        # Prune to get new masks
        new_masks = self.pruner.prune(self.model, current_masks=self.masks)

        self.masks = new_masks

        # Apply masks immediately
        apply_mask(self.model, self.masks)

        # Log sparsity
        stats = get_sparsity_stats(self.masks)
        logging.info(f"After pruning - Sparsity: {stats['overall']:.2f}% weights remaining")
        logging.info(f"Total params: {stats['total_params']:,}, "
                    f"Remaining: {stats['remaining_params']:,}")

    def run_iterative_pruning(self,
                             train_loader,
                             val_loader,
                             optimizer_fn: Callable,
                             criterion,
                             scheduler_fn: Optional[Callable] = None,
                             num_iterations: int = 5,
                             num_epochs_per_iteration: int = 100,
                             original_dropout: float = 0.0,
                             save_dir: Optional[str] = None,
                             test_loader=None) -> List[Dict]:
        """
        Run iterative pruning and fine-tuning.
        """
        results = []

        for iteration in range(num_iterations):
            self.current_iteration = iteration

            # Create optimizer and scheduler for this iteration
            optimizer = optimizer_fn(self.model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            # Fine-tune
            history = self.fine_tune(
                train_loader, val_loader, optimizer, criterion,
                scheduler, num_epochs_per_iteration, original_dropout, save_dir
            )

            # Test if test_loader provided
            test_acc = None
            if test_loader:
                test_metrics = self.validate(test_loader, criterion)
                test_acc = test_metrics['accuracy']
                logging.info(f"Test Accuracy: {test_acc:.2f}%")

            # Save iteration results
            stats = get_sparsity_stats(self.masks)
            iteration_result = {
                'iteration': iteration,
                'sparsity': stats['overall'],
                'train_history': history,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1],
                'test_acc': test_acc
            }
            results.append(iteration_result)

            # Save masks and results
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Save masks
                torch.save(self.masks, os.path.join(save_dir, f'masks_iteration_{iteration}.pth'))
                # Save iteration results
                with open(os.path.join(save_dir, f'iteration_{iteration}_results.json'), 'w') as f:
                    json.dump({k: v for k, v in iteration_result.items() if k != 'train_history'},
                             f, indent=2)

            # Prune for next iteration (except after last iteration)
            if iteration < num_iterations - 1:
                self.prune_model()

        # Save final summary
        if save_dir:
            with open(os.path.join(save_dir, 'all_iterations_summary.json'), 'w') as f:
                summary = [{k: v for k, v in r.items() if k != 'train_history'} for r in results]
                json.dump(summary, f, indent=2)

        return results
