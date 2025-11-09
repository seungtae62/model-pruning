"""
Lottery Ticket Hypothesis Training Loop.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List
import os
import json
import logging
from tqdm import tqdm

from .pruner import MagnitudePruner
from .masks import apply_mask, get_sparsity, save_mask, print_sparsity_stats
from .utils import copy_model_weights, load_model_weights


class LotteryTicketTrainer:
    """
    Trainer for finding winning lottery tickets via iterative pruning.
    """

    def __init__(self,
                 model: nn.Module,
                 pruner: MagnitudePruner,
                 device: torch.device = torch.device('cpu')):
        self.model = model.to(device)
        self.pruner = pruner
        self.device = device

        # Store initial weights θ₀
        self.initial_weights = copy_model_weights(self.model)

        self.masks = {
            name: torch.ones_like(param, dtype=torch.float32)
            for name, param in self.model.named_parameters()
            if param.requires_grad and len(param.shape) > 1
        }

        self.pruning_history = []
        self.current_round = 0

    def reset_to_initial_weights(self) -> None:
        """
        Reset model weights to initial values θ₀.
        """
        load_model_weights(self.model, self.initial_weights)
        logging.info("Weights reset to initial θ₀")

    def train_epoch(self,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
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

            apply_mask(self.model, self.masks)

            optimizer.step()

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
        """
        Validate the model.
        """
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
        Train the model to completion.
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"Training Round {self.current_round}")
        stats = get_sparsity(self.masks)
        logging.info(f"Sparsity: {stats['overall']:.2f}% weights remaining")
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
                    save_path = os.path.join(save_dir, f'round_{self.current_round}_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'masks': self.masks,
                        'val_acc': best_val_acc,
                        'round': self.current_round
                    }, save_path)

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        logging.info(f"\nRound {self.current_round} complete. Best Val Acc: {best_val_acc:.2f}%")

        return history

    def prune_model(self, layer_names_to_prune: Optional[List[str]] = None) -> None:
        """
        Prune the model based on current weights.
        """
        logging.info(f"\nPruning model...")

        # New masks from pruner
        new_masks = self.pruner.prune(
            self.model,
            current_masks=self.masks,
            layer_names_to_prune=layer_names_to_prune
        )

        self.masks = new_masks

        apply_mask(self.model, self.masks)

        print_sparsity_stats(self.masks, prefix="After pruning - ")

    def run_iterative_pruning(self,
                              train_loader,
                              val_loader,
                              optimizer_fn: Callable,
                              criterion,
                              scheduler_fn: Optional[Callable] = None,
                              num_rounds: int = 15,
                              num_epochs_per_round: int = 200,
                              save_dir: Optional[str] = None,
                              test_loader=None) -> List[Dict]:
        """
        Run the full iterative lottery ticket algorithm.
        """
        results = []

        for round_num in range(num_rounds):
            self.current_round = round_num

            optimizer = optimizer_fn(self.model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            # Train
            history = self.train_full(
                train_loader, val_loader, optimizer, criterion,
                scheduler, num_epochs_per_round, save_dir
            )

            # Test if test_loader provided
            test_acc = None
            if test_loader:
                test_metrics = self.validate(test_loader, criterion)
                test_acc = test_metrics['accuracy']
                logging.info(f"Test Accuracy: {test_acc:.2f}%")

            # Save round results
            round_result = {
                'round': round_num,
                'sparsity': get_sparsity(self.masks)['overall'],
                'train_history': history,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1],
                'test_acc': test_acc
            }
            results.append(round_result)

            # Save masks and results
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_mask(self.masks, os.path.join(save_dir, f'masks_round_{round_num}.pth'))
                with open(os.path.join(save_dir, f'round_{round_num}_results.json'), 'w') as f:
                    json.dump({k: v for k, v in round_result.items() if k != 'train_history'}, f, indent=2)

            # Prune for next round
            if round_num < num_rounds - 1:
                self.prune_model()

                self.reset_to_initial_weights()

        # Save final summary
        if save_dir:
            with open(os.path.join(save_dir, 'all_rounds_summary.json'), 'w') as f:
                summary = [{k: v for k, v in r.items() if k != 'train_history'} for r in results]
                json.dump(summary, f, indent=2)

        return results
