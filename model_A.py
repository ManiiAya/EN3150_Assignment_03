import torch.nn as nn


class ModelA(nn.Module):
    """Standard CNN: 3 x (Conv -> ReLU -> MaxPool), then 2 fully connected layers.
    Input: 3 x 64 x 64 RGB image (EuroSAT). Output: 10 class scores."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 64 * 8 * 8 = 4096
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))