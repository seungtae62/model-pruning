"""
SynFlow Training - Train pruned network after SynFlow initialization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import os
import json
import logging
from tqdm import tqdm

from .synflow_pruner import SynFlowPruner
from .utils import apply_mask, get_sparsity_stats, count_parameters, count_nonzero_parameters


class SynFlowTrainer:
    """
    Trainer for SynFlow pruned networks.

    SynFlow prunes at initialization, then trains the sparse network.
    """

    def __init__(self,
                 model: nn.Module,
                 pruner: SynFlowPruner,
                 device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.pruner = pruner
        self.device = device

        # Masks will be set after pruning
        self.masks = None

        # Track original parameter count
        self.original_param_count = count_parameters(self.model)

    def prune_at_init(self,
                      input_shape: tuple,
                      target_sparsity: float = 0.8) -> None:
        """
        Perform SynFlow pruning at initialization.
        """
        logging.info("Performing SynFlow pruning at initialization...")

        # Run iterative SynFlow pruning
        self.masks = self.pruner.iterative_prune(
            self.model,
            input_shape,
            self.device,
            target_sparsity
        )

        # Apply masks to zero out pruned weights
        apply_mask(self.model, self.masks)

        # Log final sparsity
        stats = get_sparsity_stats(self.masks)
        logging.info(f"Pruning complete:")
        logging.info(f"  Overall sparsity: {stats['overall']:.2f}% remaining")
        logging.info(f"  Total params: {stats['total_params']:,}")
        logging.info(f"  Remaining params: {stats['remaining_params']:,}")

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

            # Apply mask to gradients before optimizer step
            apply_mask(self.model, self.masks)

            optimizer.step()

            # Apply mask to weights after optimizer step
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

    def train_full(self,
                   train_loader,
                   val_loader,
                   optimizer,
                   criterion,
                   scheduler=None,
                   num_epochs: int = 200,
                   save_dir: Optional[str] = None) -> Dict[str, List]:
        """
        Train the pruned model to completion.
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"Training SynFlow Pruned Network")
        stats = get_sparsity_stats(self.masks)
        logging.info(f"Sparsity: {stats['overall']:.2f}% weights remaining")
        logging.info(f"Epochs: {num_epochs}")
        logging.info(f"{'='*60}")

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

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])

            # Log metrics
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
                    save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'masks': self.masks,
                        'val_acc': best_val_acc,
                        'val_loss': val_metrics['loss'],
                        'sparsity': stats['overall']
                    }, save_path)
                    logging.info(f"Best model saved (Val Acc: {best_val_acc:.2f}%)")

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        logging.info(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")

        return history

    def run_synflow_experiment(self,
                               train_loader,
                               val_loader,
                               test_loader,
                               optimizer,
                               criterion,
                               scheduler=None,
                               input_shape: tuple = (64, 3, 32, 32),
                               target_sparsity: float = 0.8,
                               num_epochs: int = 200,
                               save_dir: Optional[str] = None) -> Dict:
        """
        Run complete SynFlow experiment: prune at init, then train.
        """
        # Step 1: Prune at initialization
        self.prune_at_init(input_shape, target_sparsity)

        # Step 2: Train the pruned network
        history = self.train_full(
            train_loader, val_loader, optimizer, criterion,
            scheduler, num_epochs, save_dir
        )

        # Step 3: Test
        test_metrics = self.validate(test_loader, criterion)
        test_acc = test_metrics['accuracy']
        logging.info(f"Final Test Accuracy: {test_acc:.2f}%")

        # Prepare results
        stats = get_sparsity_stats(self.masks)
        results = {
            'target_sparsity': target_sparsity,
            'actual_sparsity': 100.0 - stats['overall'],
            'remaining_weights_pct': stats['overall'],
            'total_params': stats['total_params'],
            'remaining_params': stats['remaining_params'],
            'train_history': history,
            'best_val_acc': max(history['val_acc']),
            'final_val_acc': history['val_acc'][-1],
            'test_acc': test_acc
        }

        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save masks
            torch.save(self.masks, os.path.join(save_dir, 'synflow_masks.pth'))

            # Save results JSON
            with open(os.path.join(save_dir, 'results.json'), 'w') as f:
                json.dump({k: v for k, v in results.items() if k != 'train_history'},
                         f, indent=2)

            # Save full history
            with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            logging.info(f"Results saved to: {save_dir}")

        return results
