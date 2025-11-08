import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=128, val_split=0.1, num_workers=4, data_dir='./data'):
    """
    RGB channel mean/std
    R channel: mean=0.4914, std=0.2470
    G channel: mean=0.4822, std=0.2435
    B channel: mean=0.4465, std=0.2616
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2470, 0.2435, 0.2616])
    ])

    # Load full training dataset to determine split indices
    full_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )

    # Calculate split sizes and generate indices
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Fix seed to ensure same train/val split across runs
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    # Create separate datasets with appropriate transforms
    train_dataset_with_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    val_dataset_no_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )

    # Apply indices to create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_with_aug, train_indices.indices)
    val_dataset = Subset(val_dataset_no_aug, val_indices.indices)

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_cifar10_classes():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
