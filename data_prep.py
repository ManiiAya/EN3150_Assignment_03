from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def create_dataloaders(
    dataset_dir="./data/EuroSAT_RGB",
    batch_size=64,
    seed=26,
    image_size=64,
    num_workers=2,
):
    """Create reproducible train, validation, and test loaders."""
    dataset_dir = Path(dataset_dir)
    resize_and_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    train_base = datasets.ImageFolder(
        root=dataset_dir,
        transform=resize_and_tensor,
    )

    train_size = int(0.70 * len(train_base))
    val_size = int(0.15 * len(train_base))
    test_size = len(train_base) - train_size - val_size

    split_generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset, test_subset = random_split(
        train_base,
        [train_size, val_size, test_size],
        generator=split_generator,
    )
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    stats_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in stats_loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / num_pixels
    variance = (channel_squared_sum / num_pixels - mean ** 2).clamp_min(0)
    std = variance.sqrt()
    
    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    normalized_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_base.transform = normalized_transform
    evaluation_base = datasets.ImageFolder(
        root=dataset_dir,
        transform=normalized_transform,
    )
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(evaluation_base, val_indices)
    test_dataset = Subset(evaluation_base, test_indices)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader, test_loader, train_base.classes, mean, std