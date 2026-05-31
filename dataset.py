"""
dataset.py — Data Loading & Augmentation

Provides ImageFolder-based data loaders with HD-compatible transforms:
  HD image → Resize(256) → CenterCrop(224) → Normalize
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Build transforms pipeline.

    Training: RandomResizedCrop + RandomHorizontalFlip for augmentation.
    Validation/Inference: Resize(256) + CenterCrop(224) for deterministic eval.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train and validation DataLoaders from an ImageFolder dataset.

    Expected directory structure:
        data_dir/
        ├── train/
        │   ├── real/
        │   └── ai/
        └── val/
            ├── real/
            └── ai/

    Args:
        data_dir: Root directory containing train/ and val/ subdirs.
        batch_size: Batch size for DataLoader.
        num_workers: Number of parallel data-loading workers.

    Returns:
        (train_loader, val_loader, class_names)
    """
    data_path = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        root=data_path / "train",
        transform=get_transforms(is_training=True),
    )
    val_dataset = datasets.ImageFolder(
        root=data_path / "val",
        transform=get_transforms(is_training=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    print(f"[Data] Classes: {class_names}")
    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_loader, val_loader, class_names
