import torch
import argparse
import os
import json
import logging

from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import get_cifar10_loaders, train_model, test_model
from utils import get_model_summary, print_model_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Model mapping
MODEL_DICT = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='Model architecture to train')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--milestones', type=int, nargs='+', default=[60, 120, 160])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info(f'Device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    logging.info('Loading CIFAR-10...')
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Create model based on selection
    model_class = MODEL_DICT[args.model]
    model = model_class()
    summary = get_model_summary(model, device=device)
    print_model_summary(summary, model_name=f'{args.model.upper()}-Baseline')

    save_path = os.path.join(args.save_dir, f'{args.model}_baseline_best.pth')

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        milestones=args.milestones,
        gamma=args.gamma,
        save_path=save_path,
        verbose=True
    )

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    test_loss, test_acc = test_model(model, test_loader, device)

    history_path = os.path.join(args.save_dir, f'{args.model}_baseline_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    final_summary = get_model_summary(model, device=device)

    info = {
        'model': args.model,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': checkpoint['val_acc'],
        'best_val_loss': checkpoint['val_loss'],
        'best_epoch': checkpoint['epoch'],
        'total_parameters': final_summary['total_parameters'],
        'flops': final_summary['flops'],
        'inference_time_ms': final_summary['inference_time_ms']
    }

    info_path = os.path.join(args.save_dir, f'{args.model}_baseline_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    logging.info(f'Results saved to {args.save_dir}')


if __name__ == '__main__':
    main()
