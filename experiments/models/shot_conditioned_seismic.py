from __future__ import annotations

import torch
import torch.nn as nn

from experiments.models.common import ConvBlock2d, MLP, VolumeToPlaneStem, resize_to_output


class ShotConditionedSeismicModel(nn.Module):
    """
    First-pass shot-conditioned seismic surrogate.

    The model encodes the 3D velocity volume, modulates the 2D latent plane with
    source/frequency conditioning through FiLM, and decodes one gather for each
    requested shot condition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_shape: tuple[int, int],
        condition_dim: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.reference_num_shots = int(out_channels)
        self.condition_dim = int(condition_dim)
        latent_channels = base_channels * 4
        self.stem = VolumeToPlaneStem(
            in_channels=in_channels,
            channels=(base_channels, base_channels * 2, latent_channels),
            out_channels=latent_channels,
        )
        self.condition_mlp = MLP(
            in_dim=self.condition_dim,
            hidden_dim=latent_channels,
            out_dim=latent_channels * 2,
            depth=3,
        )
        self.decoder = nn.Sequential(
            ConvBlock2d(latent_channels, latent_channels),
            ConvBlock2d(latent_channels, base_channels * 2),
            nn.Conv2d(base_channels * 2, 1, kernel_size=1),
        )

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("ShotConditionedSeismicModel expects inputs={'volume': ..., 'condition': ...}.")
        volume = inputs["volume"]
        if volume.ndim == 4:
            volume = volume.unsqueeze(1)
        if volume.ndim != 5:
            raise ValueError(f"Expected velocity volume with shape (B,D,H,W) or (B,C,D,H,W), got {tuple(volume.shape)}.")

        condition = inputs["condition"].to(dtype=volume.dtype, device=volume.device)
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.ndim != 3:
            raise ValueError(f"Expected shot condition with shape (B,S,C), got {tuple(condition.shape)}.")

        batch_size, num_shots, condition_dim = condition.shape
        if condition_dim != self.condition_dim:
            raise ValueError(
                f"Expected shot condition dimension {self.condition_dim}, got {condition_dim}."
            )

        plane, _ = self.stem(volume)
        _, channels, height, width = plane.shape

        flat_condition = condition.reshape(batch_size * num_shots, condition_dim)
        gamma_beta = self.condition_mlp(flat_condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        plane = plane[:, None, :, :, :].expand(batch_size, num_shots, channels, height, width)
        plane = plane.reshape(batch_size * num_shots, channels, height, width)
        plane = plane * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        gathers = resize_to_output(self.decoder(plane), self.output_shape)
        gathers = gathers.reshape(batch_size, num_shots, 1, self.output_shape[0], self.output_shape[1])
        return gathers[:, :, 0]
