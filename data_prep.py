import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class CachedEuroSAT(Dataset):
    """Whole dataset in memory as uint8, normalized to [-1, 1] on the fly."""
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i].float() / 127.5 - 1.0, self.y[i]


def _load_cached(root):
    cache = os.path.join(root, "eurosat_cache.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"], d["classes"]
    ds = datasets.EuroSAT(root=root, download=True, transform=transforms.PILToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))])   # one-time, ~1-2 min
    y = torch.tensor(ds.targets)
    torch.save({"x": x, "y": y, "classes": ds.classes}, cache)
    return x, y, ds.classes


def get_dataloaders(batch_size=64, seed=42, root="data"):
    """EuroSAT (RGB, 64x64, 10 classes), 70/15/15 split with a fixed seed."""
    x, y, classes = _load_cached(root)
    dataset = CachedEuroSAT(x, y)

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
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    tr, va, te, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train/Val/Test sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
    images, labels = next(iter(tr))
    print("Batch shape:", images.shape)