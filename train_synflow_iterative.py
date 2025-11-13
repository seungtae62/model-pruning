"""
Train ResNet18 on CIFAR-10 using Iterative SynFlow Pruning.

Usage:
    python train_synflow_iterative.py --num_rounds 6 --epochs 200 --seed 42
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import logging
from datetime import datetime

from models.resnet import ResNet18
from utils.data_loader import get_cifar10_loaders
from synflow_pruning.synflow_pruner import SynFlowPruner, get_input_shape
from synflow_pruning.synflow_trainer import SynFlowTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Iterative SynFlow Pruning - ResNet18 on CIFAR-10')

    # Model
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs per round')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    # LR Schedule
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                       help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')

    # SynFlow Pruning
    parser.add_argument('--pruning_rate_per_iteration', type=float, default=0.2,
                       help='Fraction to prune each SynFlow iteration')
    parser.add_argument('--num_iterations', type=int, default=20,
                       help='Number of SynFlow pruning iterations (FIXED: 20 works, 100 over-prunes!)')

    # Iterative rounds
    parser.add_argument('--num_rounds', type=int, default=5,
                       help='Number of pruning rounds (sparsity levels)')
    parser.add_argument('--sparsity_levels', nargs='+', type=float, default=None,
                       help='Custom sparsity levels (e.g., 0.0 0.5 0.8 0.9 0.95)')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='synflow_iterative',
                       help='Name of experiment')
    parser.add_argument('--save_dir', type=str, default='checkpoints/synflow',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    return parser.parse_args()


def get_optimizer_fn(args):
    """Create optimizer factory function."""
    def create_optimizer(model):
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    return create_optimizer


def get_scheduler_fn(args):
    """Create LR scheduler factory function."""
    def create_scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.gamma
        )
    return create_scheduler


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save args
    import json
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info(f"Experiment: {args.experiment_name}")
    logging.info(f"Save directory: {save_dir}")
    logging.info(f"Hyperparameters:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Data loaders
    logging.info("\nLoading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")

    # Model (at random initialization)
    logging.info(f"Creating {args.arch} at random initialization...")
    if args.arch == 'resnet18':
        model = ResNet18()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")

    # SynFlow Pruner
    pruner = SynFlowPruner(
        pruning_rate_per_iteration=args.pruning_rate_per_iteration,
        num_iterations=args.num_iterations
    )

    # SynFlow Trainer
    trainer = SynFlowTrainer(model, pruner, device)

    # Get input shape for CIFAR-10
    input_shape = get_input_shape('cifar10', batch_size=64)

    # Optimizer and scheduler factories
    optimizer_fn = get_optimizer_fn(args)
    scheduler_fn = get_scheduler_fn(args)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Determine sparsity levels
    if args.sparsity_levels:
        sparsity_levels = args.sparsity_levels
    else:
        # Default: good coverage with fewer rounds to save time
        if args.num_rounds == 5:
            sparsity_levels = [0.0, 0.5, 0.8, 0.9, 0.95]
        elif args.num_rounds == 6:
            sparsity_levels = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
        elif args.num_rounds == 10:
            sparsity_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        else:
            # Generate evenly spaced levels
            import numpy as np
            sparsity_levels = np.linspace(0.0, 0.95, args.num_rounds).tolist()

    logging.info(f"Iterative SynFlow Configuration:")
    logging.info(f"Number of rounds: {args.num_rounds}")
    logging.info(f"Sparsity levels: {[f'{s*100:.1f}%' for s in sparsity_levels]}")
    logging.info(f"Epochs per round: {args.epochs}")

    # Run iterative SynFlow experiment
    all_results = trainer.run_iterative_synflow_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        criterion=criterion,
        input_shape=input_shape,
        sparsity_levels=sparsity_levels,
        num_epochs=args.epochs,
        save_dir=save_dir
    )

    logging.info(f"ITERATIVE SYNFLOW EXPERIMENT COMPLETE")
    logging.info(f"Results Summary:")
    logging.info(f"{'Round':<8} {'Sparsity':<12} {'Test Acc':<12} {'Best Val Acc':<15}")
    logging.info(f"{'-'*50}")
    for result in all_results:
        logging.info(f"{result['round']:<8} "
                    f"{result['sparsity']:.2f}%{'':<7} "
                    f"{result['test_acc']:.2f}%{'':<7} "
                    f"{result['best_val_acc']:.2f}%")
    logging.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
