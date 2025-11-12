"""
Train ResNet18 on CIFAR-10 using SynFlow Pruning at Initialization.

Based on "Pruning neural networks without any data by iteratively conserving synaptic flow"
Tanaka et al., NeurIPS 2020

SynFlow prunes networks at initialization (before training) in a data-independent manner.

Usage:
    python train_synflow.py --target_sparsity 0.8 --epochs 200 --seed 42
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
    parser = argparse.ArgumentParser(description='SynFlow Pruning - ResNet18 on CIFAR-10')

    # Model
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    # LR Schedule
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                       help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup epochs')

    # SynFlow Pruning
    parser.add_argument('--target_sparsity', type=float, default=0.8,
                       help='Target sparsity (0.8 = prune 80% of weights)')
    parser.add_argument('--pruning_rate_per_iteration', type=float, default=0.2,
                       help='Fraction to prune each SynFlow iteration')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of SynFlow pruning iterations')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='synflow',
                       help='Name of experiment')
    parser.add_argument('--save_dir', type=str, default='checkpoints/synflow',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    return parser.parse_args()


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
        logging.info(f"  {key}: {value}")

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
    logging.info(f"\nCreating {args.arch} at random initialization...")
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

    logging.info(f"SynFlow Pruning Configuration:")
    logging.info(f"Target sparsity: {args.target_sparsity*100:.1f}%")
    logging.info(f"Pruning rate per iteration: {args.pruning_rate_per_iteration}")
    logging.info(f"Number of iterations: {args.num_iterations}")

    # SynFlow Trainer
    trainer = SynFlowTrainer(model, pruner, device)

    # Get input shape for CIFAR-10
    input_shape = get_input_shape('cifar10', batch_size=64)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # LR Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run SynFlow experiment
    logging.info(f"SYNFLOW PRUNING AT INITIALIZATION")

    results = trainer.run_synflow_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        input_shape=input_shape,
        target_sparsity=args.target_sparsity,
        num_epochs=args.epochs,
        save_dir=save_dir
    )

    # Print summary
    logging.info(f"SYNFLOW EXPERIMENT COMPLETE")
    logging.info(f"Results Summary:")
    logging.info(f"Target Sparsity: {results['target_sparsity']*100:.1f}%")
    logging.info(f"Actual Sparsity: {results['actual_sparsity']:.2f}% pruned")
    logging.info(f"Remaining Weights: {results['remaining_weights_pct']:.2f}%")
    logging.info(f"Total Params: {results['total_params']:,}")
    logging.info(f"Remaining Params: {results['remaining_params']:,}")
    logging.info(f"Best Val Acc: {results['best_val_acc']:.2f}%")
    logging.info(f"Final Val Acc: {results['final_val_acc']:.2f}%")
    logging.info(f"Test Acc: {results['test_acc']:.2f}%")
    logging.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
