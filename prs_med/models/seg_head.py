from __future__ import annotations

import torch.nn as nn


class SegmentationClassifier(nn.Module):
    """Simple classifier head predicting modality labels from SAM embeddings."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, features):
        return self.classifier(features)


