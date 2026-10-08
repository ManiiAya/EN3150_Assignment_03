import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64, seed=42, root="data"):
    """Download EuroSAT (RGB, 64x64, 10 classes) and split 70/15/15.
    The fixed seed makes the split identical on every machine."""
    transform = transforms.Compose([
        transforms.ToTensor(),                                   # 0-255 -> 0-1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> -1 to 1
    ])
    dataset = datasets.EuroSAT(root=root, download=True, transform=transform)

    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader, dataset.classes


if __name__ == "__main__":
    tr, va, te, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train/Val/Test sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
    images, labels = next(iter(tr))
    print("Batch shape:", images.shape)