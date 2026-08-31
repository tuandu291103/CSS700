"""A compact CNN whose final convolutional layer can be targeted by Grad-CAM."""

import torch
from torch import nn


class SmallTabularImageCNN(nn.Module):
    """Dimension-tolerant baseline for small artificial feature grids."""

    def __init__(self, *, input_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.grad_cam_layer = self.features[2]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(pooled)
