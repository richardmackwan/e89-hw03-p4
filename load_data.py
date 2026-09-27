"""Load FashionMNIST and create training, validation, and test loaders."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

# __file__ is undefined when this code runs as a notebook cell.
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def load_data(root=PROJECT_DIR / "data"):
    """Return batch-size-32 loaders with a reproducible 55,000/5,000 split."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    training_data = FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_data = FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    train_data, valid_data = random_split(
        training_data,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = load_data()
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(valid_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
