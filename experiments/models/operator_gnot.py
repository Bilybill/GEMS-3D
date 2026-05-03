from __future__ import annotations

import torch
import torch.nn as nn

from experiments.models.common import ResidualTransformerBlock, VolumeToPlaneStem, resize_to_output, sinusoidal_position_encoding


class GNOTObservationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: tuple[int, int],
        width: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 4,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.patch_size = patch_size
        self.stem = VolumeToPlaneStem(in_channels=in_channels, channels=(32, 64, 128), out_channels=width)
        self.patch_embed = nn.Conv2d(width, width, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [ResidualTransformerBlock(dim=width, num_heads=num_heads) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x):
        plane, _ = self.stem(x)
        tokens = self.patch_embed(plane)
        batch, channels, height, width = tokens.shape
        sequence = tokens.flatten(2).transpose(1, 2)
        position = sinusoidal_position_encoding(sequence.shape[1], sequence.shape[2], device=x.device, dtype=x.dtype)
        sequence = sequence + position.unsqueeze(0)
        for block in self.blocks:
            sequence = block(sequence)
        tokens = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        return resize_to_output(self.head(tokens), self.output_shape)
