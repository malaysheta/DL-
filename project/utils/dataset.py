"""
utils/dataset.py
Dataset loading and preprocessing utilities.

The dataset is expected to be organised as:
    data/
        train/
            class_a/  img1.jpg ...
            class_b/  img2.jpg ...
            ...
        val/
            class_a/  ...
            class_b/  ...
        test/
            class_a/  ...
            class_b/  ...

If the dataset only has a single root folder (no pre-existing split), the
build_dataloaders() function will perform a 70/15/15 random split using
ImageFolder + random_split.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str) -> transforms.Compose:
    """
    Return appropriate transforms for each data split.

    Args:
        split: one of 'train', 'val', 'test'
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet statistics (works well for transfer learning)
    std  = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test — no augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders.

    Supports two layouts:
      1. Pre-split: data_dir contains 'train/', 'val/', 'test/' sub-folders.
      2. Single root: data_dir contains class sub-folders directly.
         In this case a 70/15/15 random split is performed.

    Args:
        data_dir:    Path to the data directory.
        batch_size:  Batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_path = Path(data_dir)

    train_path = data_path / 'train'
    val_path   = data_path / 'val'
    test_path  = data_path / 'test'

    # ---- Case 1: pre-split folders exist ----
    if train_path.is_dir() and val_path.is_dir() and test_path.is_dir():
        print("[Dataset] Found pre-split train/val/test folders.")

        train_ds = datasets.ImageFolder(str(train_path), transform=get_transforms('train'))
        val_ds   = datasets.ImageFolder(str(val_path),   transform=get_transforms('val'))
        test_ds  = datasets.ImageFolder(str(test_path),  transform=get_transforms('test'))

        class_names = train_ds.classes

    # ---- Case 2: single root, perform random 70/15/15 split ----
    else:
        print("[Dataset] Pre-split folders not found. Performing 70/15/15 random split.")

        full_ds = datasets.ImageFolder(str(data_path), transform=get_transforms('train'))
        class_names = full_ds.classes

        total  = len(full_ds)
        n_train = int(0.70 * total)
        n_val   = int(0.15 * total)
        n_test  = total - n_train - n_val

        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Override transforms for val/test subsets
        val_ds.dataset.transform  = get_transforms('val')
        test_ds.dataset.transform = get_transforms('test')

    print(f"[Dataset] Classes ({len(class_names)}): {class_names}")
    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Build loaders
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_names
