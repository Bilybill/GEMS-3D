from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels))
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class VolumeEncoder3D(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        blocks = []
        current = in_channels
        for idx, out_channels in enumerate(channels):
            blocks.append(ConvBlock3d(current, out_channels))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward_features(self, x):
        features = []
        current = x
        for idx, block in enumerate(self.blocks):
            current = block(current)
            features.append(current)
            if idx < len(self.blocks) - 1:
                current = self.pool(current)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        last = features[-1]
        pooled = F.adaptive_avg_pool3d(last, output_size=1).flatten(1)
        return last, pooled


class VolumeToPlaneStem(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...] = (32, 64, 128), out_channels: int = 128):
        super().__init__()
        self.encoder = VolumeEncoder3D(in_channels=in_channels, channels=channels)
        self.plane_projection = nn.Conv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        feature_volume, latent = self.encoder(x)
        plane = feature_volume.mean(dim=2)
        plane = self.plane_projection(plane)
        return plane, latent


def build_coordinate_grid(height: int, width: int, device=None, dtype=None):
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1).reshape(height * width, 2)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 4):
        super().__init__()
        layers = []
        current = in_dim
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.GELU())
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def resize_to_output(x, output_shape: tuple[int, int]):
    if x.shape[-2:] == output_shape:
        return x
    return F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)


class ResidualTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_position_encoding(length: int, dim: int, device=None, dtype=None):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 2))
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding
