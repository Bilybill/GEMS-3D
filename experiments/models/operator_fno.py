from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.common import VolumeToPlaneStem, resize_to_output


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat)
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes_h, : self.modes_w] = torch.einsum(
            "bihw,iohw->bohw",
            x_ft[:, :, : self.modes_h, : self.modes_w],
            self.weight,
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_h=modes_h, modes_w=modes_w)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(max(1, min(8, width)), width)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        return F.gelu(self.norm(x))


class FNOObservationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: tuple[int, int],
        width: int = 128,
        depth: int = 4,
        modes_h: int = 16,
        modes_w: int = 16,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.stem = VolumeToPlaneStem(in_channels=in_channels, channels=(32, 64, 128), out_channels=width)
        self.blocks = nn.ModuleList([FNOBlock2d(width=width, modes_h=modes_h, modes_w=modes_w) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x):
        plane, _ = self.stem(x)
        for block in self.blocks:
            plane = block(plane)
        return resize_to_output(self.head(plane), self.output_shape)
