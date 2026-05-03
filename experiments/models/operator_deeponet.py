from __future__ import annotations

import torch
import torch.nn as nn

from experiments.models.common import MLP, VolumeEncoder3D, build_coordinate_grid


class DeepONetObservationModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: tuple[int, int],
        branch_width: int = 256,
        basis_dim: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.encoder = VolumeEncoder3D(in_channels=in_channels, channels=(32, 64, 128))
        self.branch = nn.Sequential(
            nn.Linear(128, branch_width),
            nn.GELU(),
            nn.Linear(branch_width, out_channels * basis_dim),
        )
        self.trunk = MLP(in_dim=2, hidden_dim=branch_width, out_dim=basis_dim, depth=4)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        _, latent = self.encoder(x)
        branch_weights = self.branch(latent).view(x.shape[0], self.out_channels, -1)
        coords = build_coordinate_grid(
            self.output_shape[0],
            self.output_shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        trunk_basis = self.trunk(coords)
        output = torch.einsum("bck,nk->bcn", branch_weights, trunk_basis)
        output = output.view(x.shape[0], self.out_channels, *self.output_shape)
        return output + self.bias.view(1, -1, 1, 1)
