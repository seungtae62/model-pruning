"""
Train ResNet18 on CIFAR-10 using Lottery Ticket Hypothesis.

This script implements the iterative magnitude pruning algorithm from:
"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
Frankle & Carbin, ICLR 2019

Usage:
    python train_lottery.py --pruning_rate 0.2 --num_rounds 15 --lr 0.1
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
from lottery.pruner import MagnitudePruner
from lottery.lottery_trainer import LotteryTicketTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Lottery Ticket Hypothesis - ResNet18 on CIFAR-10')

    # Model
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Epochs per pruning round')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    # LR Schedule
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                        help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup iterations (0 = no warmup)')

    # Pruning
    parser.add_argument('--pruning_rate', type=float, default=0.2,
                        help='Fraction of weights to prune per round')
    parser.add_argument('--pruning_rate_conv', type=float, default=None,
                        help='Pruning rate for conv layers (overrides default)')
    parser.add_argument('--pruning_rate_fc', type=float, default=None,
                        help='Pruning rate for FC layers (overrides default)')
    parser.add_argument('--num_rounds', type=int, default=15,
                        help='Number of iterative pruning rounds')
    parser.add_argument('--global_pruning', action='store_true',
                        help='Use global pruning instead of layer-wise')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='lottery_ticket',
                        help='Name of experiment')
    parser.add_argument('--save_dir', type=str, default='checkpoints/lottery',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def get_optimizer_fn(args):
    def create_optimizer(model):
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    return create_optimizer


def get_scheduler_fn(args):
    def create_scheduler(optimizer):
        if args.warmup > 0:
            # Warmup scheduler
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=args.warmup
            )
            main_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[m + args.warmup for m in args.milestones],
                gamma=args.gamma
            )
            # Sequential scheduler
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup]
            )
        else:
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
        logging.debug(f"{key}: {value}")

    # Data loaders
    logging.info("Loading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")

    # Model
    logging.info(f"Creating {args.arch}...")
    if args.arch == 'resnet18':
        model = ResNet18()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = model.to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Pruner
    pruner = MagnitudePruner(
        pruning_rate=args.pruning_rate,
        pruning_rate_conv=args.pruning_rate_conv,
        pruning_rate_fc=args.pruning_rate_fc,
        global_pruning=args.global_pruning
    )

    logging.info(f"Pruning configuration:")
    logging.info(f"Strategy: {'Global' if args.global_pruning else 'Layer-wise'}")
    logging.info(f"Pruning rate: {args.pruning_rate * 100:.1f}%")
    if args.pruning_rate_conv:
        logging.info(f"Conv rate: {args.pruning_rate_conv * 100:.1f}%")
    if args.pruning_rate_fc:
        logging.info(f"FC rate: {args.pruning_rate_fc * 100:.1f}%")

    # Lottery Ticket Trainer
    trainer = LotteryTicketTrainer(model, pruner, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run iterative pruning
    logging.info(f"Starting Lottery Ticket Experiment")
    logging.info(f"Pruning rounds: {args.num_rounds}")
    logging.info(f"Epochs per round: {args.epochs}")

    results = trainer.run_iterative_pruning(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_fn=get_optimizer_fn(args),
        criterion=criterion,
        scheduler_fn=get_scheduler_fn(args),
        num_rounds=args.num_rounds,
        num_epochs_per_round=args.epochs,
        save_dir=save_dir,
        test_loader=test_loader
    )

    # Print summary
    logging.info("LOTTERY TICKET EXPERIMENT COMPLETE")
    logging.info(f"Results Summary:")
    logging.info(f"{'Round':<8} {'Sparsity':<12} {'Val Acc':<12} {'Test Acc':<12}")

    for r in results:
        sparsity_str = f"{r['sparsity']:.2f}%"
        val_acc_str = f"{r['best_val_acc']:.2f}%"
        test_acc_str = f"{r['test_acc']:.2f}%" if r['test_acc'] else "N/A"
        logging.info(f"{r['round']:<8} {sparsity_str:<12} {val_acc_str:<12} {test_acc_str:<12}")

    logging.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
