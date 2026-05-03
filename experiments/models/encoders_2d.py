from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.common import ConvBlock2d


class ObservationEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.stage1 = ConvBlock2d(in_channels, base_channels)
        self.stage2 = ConvBlock2d(base_channels, base_channels * 2)
        self.stage3 = ConvBlock2d(base_channels * 2, base_channels * 4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.projection = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return self.projection(x)
