from __future__ import annotations

import torch
import torch.nn as nn

from experiments.models.common import MLP, VolumeEncoder3D, build_coordinate_grid


class PINNObservationModel(nn.Module):
    """
    Coordinate-conditioned operator surrogate prepared for future physics residuals.

    The current implementation supports supervised learning and exposes a
    coordinate-decoder structure that can later be extended with modality-specific
    PDE residual losses.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: tuple[int, int],
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.output_shape = output_shape
        self.encoder = VolumeEncoder3D(in_channels=in_channels, channels=(32, 64, 128))
        self.latent_projection = nn.Linear(128, latent_dim)
        self.decoder = MLP(in_dim=latent_dim + 2, hidden_dim=hidden_dim, out_dim=out_channels, depth=5)

    def forward(self, x):
        _, latent = self.encoder(x)
        latent = self.latent_projection(latent)
        coords = build_coordinate_grid(
            self.output_shape[0],
            self.output_shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        coords = coords.unsqueeze(0).expand(x.shape[0], -1, -1)
        latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1)
        decoded = self.decoder(torch.cat([latent, coords], dim=-1))
        return decoded.transpose(1, 2).reshape(x.shape[0], self.out_channels, *self.output_shape)
