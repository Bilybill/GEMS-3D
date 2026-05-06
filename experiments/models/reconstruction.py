from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.datasets.gmes_inverse_dataset import AVAILABLE_MODALITIES, normalize_modalities
from experiments.datasets.gmes_reconstruction_dataset import normalize_properties
from experiments.models.common import ConvBlock2d, ConvBlock3d, MLP, ResidualTransformerBlock
from experiments.models.encoders_2d import ObservationEncoder2D
from experiments.models.encoders_mt import MTObservationEncoder
from experiments.models.encoders_seismic import SeismicObservationEncoder


AVAILABLE_RECONSTRUCTION_MODELS = ("unet", "pinn", "deeponet", "fno", "gnot")


class ObservationFusionEncoder(nn.Module):
    def __init__(
        self,
        modalities: tuple[str, ...],
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.modalities = normalize_modalities(modalities)
        self.embedding_dim = int(embedding_dim)
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            if modality in ("gravity", "magnetic"):
                self.encoders[modality] = ObservationEncoder2D(in_channels=1, embedding_dim=embedding_dim)
            elif modality == "mt":
                self.encoders[modality] = MTObservationEncoder(in_channels=76, embedding_dim=embedding_dim)
            elif modality == "seismic":
                self.encoders[modality] = SeismicObservationEncoder(in_channels=25, embedding_dim=embedding_dim)
            else:
                raise ValueError(f"Unsupported modality '{modality}'. Available: {AVAILABLE_MODALITIES}")

        fused_dim = len(self.modalities) * embedding_dim + len(self.modalities)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs: dict[str, torch.Tensor]):
        available = [tensor for tensor in inputs.values() if tensor is not None]
        if not available:
            raise ValueError("At least one modality tensor must be provided.")
        batch_size = int(available[0].shape[0])
        device = available[0].device
        dtype = available[0].dtype

        embeddings = []
        modality_mask = []
        for modality in self.modalities:
            tensor = inputs.get(modality)
            if tensor is None:
                embeddings.append(torch.zeros(batch_size, self.embedding_dim, device=device, dtype=dtype))
                modality_mask.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
                continue
            embeddings.append(self.encoders[modality](tensor))
            modality_mask.append(torch.ones(batch_size, 1, device=device, dtype=dtype))
        return self.fusion(torch.cat([*embeddings, *modality_mask], dim=1))


class GridVolumeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        output_shape: tuple[int, int, int],
        model_name: str,
        width: int = 64,
    ):
        super().__init__()
        self.output_shape = tuple(int(x) for x in output_shape)
        self.model_name = str(model_name).lower()
        start_shape = tuple(max(2, min(8, dim // 8)) for dim in self.output_shape)
        self.start_shape = start_shape
        self.width = int(width)
        self.project = nn.Linear(latent_dim, self.width * start_shape[0] * start_shape[1] * start_shape[2])
        if self.model_name == "gnot":
            self.refiner = nn.ModuleList([ResidualTransformerBlock(dim=self.width, num_heads=4) for _ in range(2)])
            self.conv = ConvBlock3d(self.width, self.width)
        elif self.model_name == "fno":
            self.refiner = nn.Sequential(
                nn.Conv3d(self.width, self.width, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(self.width, self.width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.conv = ConvBlock3d(self.width, self.width)
        else:
            self.refiner = nn.Sequential(
                ConvBlock3d(self.width, self.width),
                ConvBlock3d(self.width, self.width),
            )
            self.conv = ConvBlock3d(self.width, self.width)
        self.head = nn.Conv3d(self.width, out_channels, kernel_size=1)

    def forward(self, latent):
        batch_size = int(latent.shape[0])
        x = self.project(latent).reshape(batch_size, self.width, *self.start_shape)
        if self.model_name == "gnot":
            tokens = x.flatten(2).transpose(1, 2)
            for block in self.refiner:
                tokens = block(tokens)
            x = tokens.transpose(1, 2).reshape(batch_size, self.width, *self.start_shape)
            x = self.conv(x)
        else:
            x = self.conv(self.refiner(x))
        x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
        return self.head(x)


class CoordinateVolumeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        output_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        depth: int = 5,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.output_shape = tuple(int(x) for x in output_shape)
        self.out_channels = int(out_channels)
        self.chunk_size = int(chunk_size)
        self.decoder = MLP(in_dim=latent_dim + 3, hidden_dim=hidden_dim, out_dim=out_channels, depth=depth)

    def _coords(self, device, dtype):
        axes = [torch.linspace(-1.0, 1.0, dim, device=device, dtype=dtype) for dim in self.output_shape]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)

    def forward(self, latent):
        coords = self._coords(latent.device, latent.dtype)
        outputs = []
        for start in range(0, coords.shape[0], self.chunk_size):
            chunk = coords[start : start + self.chunk_size]
            expanded_latent = latent[:, None, :].expand(-1, chunk.shape[0], -1)
            expanded_coords = chunk[None, :, :].expand(latent.shape[0], -1, -1)
            decoded = self.decoder(torch.cat([expanded_latent, expanded_coords], dim=-1))
            outputs.append(decoded)
        values = torch.cat(outputs, dim=1).transpose(1, 2)
        return values.reshape(latent.shape[0], self.out_channels, *self.output_shape)


class PlaneDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        output_shape: tuple[int, int],
        model_name: str,
        width: int = 64,
    ):
        super().__init__()
        self.output_shape = tuple(int(x) for x in output_shape)
        self.model_name = str(model_name).lower()
        self.width = int(width)
        self.project = nn.Linear(latent_dim, self.width * 8 * 8)
        self.refiner = nn.Sequential(
            ConvBlock2d(self.width, self.width),
            ConvBlock2d(self.width, self.width),
        )
        self.head = nn.Conv2d(self.width, out_channels, kernel_size=1)

    def forward(self, latent):
        x = self.project(latent).reshape(latent.shape[0], self.width, 8, 8)
        x = self.refiner(x)
        x = F.interpolate(x, size=self.output_shape, mode="bilinear", align_corners=False)
        return self.head(x)


class PropertyReconstructionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        modalities: tuple[str, ...],
        target_properties: tuple[str, ...],
        target_shape: tuple[int, int, int],
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        model_name = str(model_name).lower()
        if model_name not in AVAILABLE_RECONSTRUCTION_MODELS:
            raise ValueError(f"Unsupported reconstruction model '{model_name}'.")
        self.model_name = model_name
        self.modalities = normalize_modalities(modalities)
        self.target_properties = normalize_properties(target_properties)
        self.encoder = ObservationFusionEncoder(
            modalities=self.modalities,
            embedding_dim=embedding_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
        if model_name in ("pinn", "deeponet"):
            self.decoder = CoordinateVolumeDecoder(
                latent_dim=fusion_hidden_dim,
                out_channels=len(self.target_properties),
                output_shape=target_shape,
            )
        else:
            self.decoder = GridVolumeDecoder(
                latent_dim=fusion_hidden_dim,
                out_channels=len(self.target_properties),
                output_shape=target_shape,
                model_name=model_name,
            )

    def forward(self, inputs: dict[str, torch.Tensor]):
        return self.decoder(self.encoder(inputs))


class ModalityCompletionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_modalities: tuple[str, ...],
        target_modality: str,
        out_channels: int,
        output_shape: tuple[int, int],
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        model_name = str(model_name).lower()
        if model_name not in AVAILABLE_RECONSTRUCTION_MODELS:
            raise ValueError(f"Unsupported completion model '{model_name}'.")
        self.model_name = model_name
        self.input_modalities = normalize_modalities(input_modalities)
        self.target_modality = normalize_modalities((target_modality,))[0]
        self.encoder = ObservationFusionEncoder(
            modalities=self.input_modalities,
            embedding_dim=embedding_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
        self.decoder = PlaneDecoder(
            latent_dim=fusion_hidden_dim,
            out_channels=out_channels,
            output_shape=output_shape,
            model_name=model_name,
        )

    def forward(self, inputs: dict[str, torch.Tensor]):
        return self.decoder(self.encoder(inputs))


def build_property_reconstruction_model(
    model_name: str,
    modalities: tuple[str, ...],
    target_properties: tuple[str, ...],
    target_shape: tuple[int, int, int],
    embedding_dim: int = 128,
    fusion_hidden_dim: int = 256,
):
    return PropertyReconstructionModel(
        model_name=model_name,
        modalities=modalities,
        target_properties=target_properties,
        target_shape=target_shape,
        embedding_dim=embedding_dim,
        fusion_hidden_dim=fusion_hidden_dim,
    )


def build_modality_completion_model(
    model_name: str,
    input_modalities: tuple[str, ...],
    target_modality: str,
    out_channels: int,
    output_shape: tuple[int, int],
    embedding_dim: int = 128,
    fusion_hidden_dim: int = 256,
):
    return ModalityCompletionModel(
        model_name=model_name,
        input_modalities=input_modalities,
        target_modality=target_modality,
        out_channels=out_channels,
        output_shape=output_shape,
        embedding_dim=embedding_dim,
        fusion_hidden_dim=fusion_hidden_dim,
    )
