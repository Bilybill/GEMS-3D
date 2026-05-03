from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.common import ConvBlock2d, VolumeToPlaneStem, resize_to_output


class UpBlock2d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock2d(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetObservationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, output_shape: tuple[int, int], base_channels: int = 32):
        super().__init__()
        self.output_shape = output_shape
        self.stem = VolumeToPlaneStem(
            in_channels=in_channels,
            channels=(base_channels, base_channels * 2, base_channels * 4),
            out_channels=base_channels * 4,
        )
        self.enc1 = ConvBlock2d(base_channels * 4, base_channels * 4)
        self.pool1 = nn.AvgPool2d(2)
        self.enc2 = ConvBlock2d(base_channels * 4, base_channels * 8)
        self.pool2 = nn.AvgPool2d(2)
        self.bottleneck = ConvBlock2d(base_channels * 8, base_channels * 8)
        self.up2 = UpBlock2d(base_channels * 8, base_channels * 8, base_channels * 4)
        self.up1 = UpBlock2d(base_channels * 4, base_channels * 4, base_channels * 2)
        self.head = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        plane, _ = self.stem(x)
        skip1 = self.enc1(plane)
        skip2 = self.enc2(self.pool1(skip1))
        bottleneck = self.bottleneck(self.pool2(skip2))
        up2 = self.up2(bottleneck, skip2)
        up1 = self.up1(up2, skip1)
        return resize_to_output(self.head(up1), self.output_shape)
