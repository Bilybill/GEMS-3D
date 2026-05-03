from __future__ import annotations

import torch
import torch.nn as nn

from experiments.datasets.gmes_inverse_dataset import AVAILABLE_MODALITIES
from experiments.models.encoders_2d import ObservationEncoder2D
from experiments.models.encoders_mt import MTObservationEncoder
from experiments.models.encoders_seismic import SeismicObservationEncoder
from experiments.models.heads_classification import ClassificationHead


class LateFusionClassifier(nn.Module):
    def __init__(
        self,
        modalities: tuple[str, ...],
        num_classes: int,
        embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
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
                raise ValueError(f"Unsupported modality '{modality}'.")

        fused_dim = len(self.modalities) * embedding_dim + len(self.modalities)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )
        self.head = ClassificationHead(in_dim=fusion_hidden_dim, num_classes=num_classes, hidden_dim=fusion_hidden_dim)

    def forward(self, inputs: dict[str, torch.Tensor]):
        embeddings = []
        modality_mask = []
        batch_size = None
        device = None
        dtype = None

        for modality in self.modalities:
            tensor = inputs.get(modality)
            if tensor is not None:
                if batch_size is None:
                    batch_size = int(tensor.shape[0])
                    device = tensor.device
                    dtype = tensor.dtype
                embeddings.append(self.encoders[modality](tensor))
                modality_mask.append(torch.ones(tensor.shape[0], 1, device=tensor.device, dtype=tensor.dtype))
            else:
                if batch_size is None:
                    raise ValueError("At least one modality tensor must be provided.")
                embeddings.append(torch.zeros(batch_size, self.embedding_dim, device=device, dtype=dtype))
                modality_mask.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))

        fused = torch.cat([*embeddings, *modality_mask], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


def canonical_modality_subsets() -> list[tuple[str, ...]]:
    return [
        ("gravity",),
        ("magnetic",),
        ("mt",),
        ("seismic",),
        ("gravity", "magnetic"),
        ("gravity", "mt"),
        ("mt", "seismic"),
        tuple(AVAILABLE_MODALITIES),
    ]
