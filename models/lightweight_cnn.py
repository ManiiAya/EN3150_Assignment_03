import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # Depthwise convolution:
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution:
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 3 × 64 × 64
            DepthwiseSeparableConv(3, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 16 × 32 × 32
            DepthwiseSeparableConv(16, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 32 × 16 × 16
            DepthwiseSeparableConv(32, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 64 × 8 × 8
            DepthwiseSeparableConv(64, 96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Output: 96 × 4 × 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x