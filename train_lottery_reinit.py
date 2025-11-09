"""
Train with Random Reinitialization (for comparison with Lottery Tickets).

This script uses the same masks as the winning ticket but with random initialization.
This demonstrates that it's not just the architecture but the INITIALIZATION that matters.

Usage:
    python train_lottery_reinit.py --masks_path checkpoints/lottery/experiment/masks_round_10.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import logging

from models.resnet import ResNet18
from utils.data_loader import get_cifar10_loaders
from utils.train_utils import train_model, test_model
from lottery.masks import load_mask, apply_mask, print_sparsity_stats
from lottery.utils import reinitialize_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Random Reinitialization Comparison')

    # Required
    parser.add_argument('--masks_path', type=str, required=True,
                        help='Path to saved masks from lottery ticket experiment')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                        help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')

    # Experiment
    parser.add_argument('--num_trials', type=int, default=3,
                        help='Number of random reinit trials')
    parser.add_argument('--save_dir', type=str, default='checkpoints/lottery_reinit',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')

    return parser.parse_args()


def train_with_mask(model, masks, train_loader, val_loader, test_loader, args, trial_num):
    """Train a model with given masks."""

    # Reinitialize model randomly
    logging.info(f"Randomly reinitializing model (Trial {trial_num})...")
    reinitialize_model(model, initialization_method='glorot')

    # Apply masks immediately
    apply_mask(model, masks)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    # Custom training loop with mask application
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply mask before step (keep pruned weights at 0)
            apply_mask(model, masks)

            optimizer.step()

            # Apply mask after step too
            apply_mask(model, masks)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 20 == 0:
            logging.info(f"Epoch {epoch}/{args.epochs} - "
                  f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Best: {best_val_acc:.2f}%")

        scheduler.step()

    # Test
    test_loss = 0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total

    return {
        'best_val_acc': best_val_acc,
        'final_val_acc': history['val_acc'][-1],
        'test_acc': test_acc,
        'history': history
    }


def main():
    args = parse_args()

    # Load masks
    logging.info(f"Loading masks from: {args.masks_path}")
    masks = load_mask(args.masks_path, device=torch.device('cuda'))

    # Print mask statistics
    print_sparsity_stats(masks)

    # Data loaders
    logging.info("Loading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=4
    )

    # Run multiple trials with random reinitialization
    all_results = []

    for trial in range(1, args.num_trials + 1):
        logging.info(f"Trial {trial}/{args.num_trials}")

        # Set seed for this trial
        torch.manual_seed(args.seed + trial)
        torch.cuda.manual_seed(args.seed + trial)

        # Create fresh model
        model = ResNet18().cuda()

        # Train
        results = train_with_mask(model, masks, train_loader, val_loader, test_loader, args, trial)

        logging.info(f"Trial {trial} Results:")
        logging.info(f"Best Val Acc: {results['best_val_acc']:.2f}%")
        logging.info(f"Test Acc: {results['test_acc']:.2f}%")

        all_results.append(results)

    # Summary
    logging.info("RANDOM REINITIALIZATION EXPERIMENT COMPLETE")

    avg_val = sum(r['best_val_acc'] for r in all_results) / len(all_results)
    avg_test = sum(r['test_acc'] for r in all_results) / len(all_results)

    logging.info(f"Results across {args.num_trials} trials:")
    logging.info(f"Average Best Val Acc: {avg_val:.2f}%")
    logging.info(f"Average Test Acc: {avg_test:.2f}%")

    logging.info(f"nPer-trial breakdown:")
    for i, r in enumerate(all_results, 1):
        logging.info(f"Trial {i}: Val={r['best_val_acc']:.2f}%, Test={r['test_acc']:.2f}%")

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        import json
        with open(os.path.join(args.save_dir, 'reinit_results.json'), 'w') as f:
            summary = {
                'masks_path': args.masks_path,
                'num_trials': args.num_trials,
                'avg_val_acc': avg_val,
                'avg_test_acc': avg_test,
                'trials': [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
            }
            json.dump(summary, f, indent=2)
        logging.info(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
