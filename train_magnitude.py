"""
Train ResNet18 on CIFAR-10 using Han Magnitude Pruning.

Based on "Learning both Weights and Connections for Efficient Neural Networks"
Song Han et al., NIPS 2015

Usage:
    python train_han.py --pretrained_path checkpoints/resnet18/resnet18_baseline_best.pth \
                        --quality_param 1.0 --lr_multiplier 0.1 --num_iterations 7
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
from magnitude_pruning.magnitude_pruner import MagnitudePruner, create_sensitivity_based_quality_params
from magnitude_pruning.magnitude_trainer import MagnitudePruningTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Han Magnitude Pruning - ResNet18 on CIFAR-10')

    # Model
    parser.add_argument('--arch', type=str, default='resnet18', help='Model architecture')
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained dense model checkpoint')

    # Training
    parser.add_argument('--epochs_per_iteration', type=int, default=100,
                       help='Fine-tuning epochs per pruning iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--lr_multiplier', type=float, default=0.1,
                       help='LR multiplier for fine-tuning (0.1 = 1/10 LR, 0.01 = 1/100 LR)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    # LR Schedule
    parser.add_argument('--milestones', nargs='+', type=int, default=[50, 75],
                       help='LR decay milestones for fine-tuning')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR decay factor')

    # Pruning
    parser.add_argument('--quality_param', type=float, default=1.0,
                       help='Quality parameter for threshold (multiplies std_dev)')
    parser.add_argument('--num_iterations', type=int, default=7,
                       help='Number of iterative prune-retrain cycles')
    parser.add_argument('--global_pruning', action='store_true',
                       help='Use global pruning instead of layer-wise')
    parser.add_argument('--sensitivity_based', action='store_true',
                       help='Use layer sensitivity-based quality parameters')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='han_pruning',
                       help='Name of experiment')
    parser.add_argument('--save_dir', type=str, default='checkpoints/han',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    return parser.parse_args()


def get_optimizer_fn(args):
    """Create optimizer factory function."""
    def create_optimizer(model):
        # Use reduced learning rate for fine-tuning
        fine_tune_lr = args.lr * args.lr_multiplier
        logging.info(f"Fine-tuning LR: {fine_tune_lr:.6f} (base LR {args.lr} × {args.lr_multiplier})")

        return optim.SGD(
            model.parameters(),
            lr=fine_tune_lr,
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


def load_pretrained_model(model_path: str, model: nn.Module, device: torch.device):
    logging.info(f"Loading pretrained model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            logging.info(f"Pretrained model Val Acc: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    logging.info("Pretrained model loaded successfully")

    return model


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

    # Load pretrained weights
    model = load_pretrained_model(args.pretrained_path, model, device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")

    # Create quality parameters
    if args.sensitivity_based:
        logging.info("Using sensitivity-based quality parameters")
        quality_params = create_sensitivity_based_quality_params(model, args.quality_param)
        logging.info(f"Quality parameters per layer:")
        for name, param in quality_params.items():
            logging.info(f"  {name}: {param:.2f}")
    else:
        quality_params = args.quality_param

    # Pruner
    pruner = MagnitudePruner(
        quality_parameter=quality_params,
        global_pruning=args.global_pruning
    )

    logging.info(f"Pruning configuration:")
    logging.info(f"Strategy: {'Global' if args.global_pruning else 'Layer-wise'}")
    logging.info(f"Quality parameter: {args.quality_param}")
    logging.info(f"Sensitivity-based: {args.sensitivity_based}")

    # Magnitude Pruning Trainer
    trainer = MagnitudePruningTrainer(model, pruner, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run iterative pruning
    logging.info(f"Starting Han Magnitude Pruning")
    logging.info(f"Pruning iterations: {args.num_iterations}")
    logging.info(f"Epochs per iteration: {args.epochs_per_iteration}")
    logging.info(f"Fine-tuning LR: {args.lr * args.lr_multiplier:.6f}")

    results = trainer.run_iterative_pruning(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_fn=get_optimizer_fn(args),
        criterion=criterion,
        scheduler_fn=get_scheduler_fn(args),
        num_iterations=args.num_iterations,
        num_epochs_per_iteration=args.epochs_per_iteration,
        original_dropout=0.0,  # ResNet18 doesn't use dropout
        save_dir=save_dir,
        test_loader=test_loader
    )

    # Print summary
    logging.info("HAN MAGNITUDE PRUNING COMPLETE")
    logging.info(f"Results Summary:")
    logging.info(f"{'Iter':<8} {'Sparsity':<12} {'Val Acc':<12} {'Test Acc':<12}")

    for r in results:
        sparsity_str = f"{r['sparsity']:.2f}%"
        val_acc_str = f"{r['best_val_acc']:.2f}%"
        test_acc_str = f"{r['test_acc']:.2f}%" if r['test_acc'] else "N/A"
        logging.info(f"{r['iteration']:<8} {sparsity_str:<12} {val_acc_str:<12} {test_acc_str:<12}")

    logging.info(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
