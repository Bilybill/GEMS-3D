from __future__ import annotations

import torch.nn as nn

from experiments.models.operator_deeponet import DeepONetObservationModel
from experiments.models.operator_fno import FNOObservationModel
from experiments.models.operator_gnot import GNOTObservationModel
from experiments.models.operator_pinn import PINNObservationModel
from experiments.models.operator_unet import UNetObservationModel


MODEL_FAMILY_REGISTRY = {
    "unet": UNetObservationModel,
    "pinn": PINNObservationModel,
    "deeponet": DeepONetObservationModel,
    "fno": FNOObservationModel,
    "gnot": GNOTObservationModel,
}


class JointForwardModel(nn.Module):
    """
    Composite joint surrogate with one modality-specific head per physics.

    The first benchmark version keeps the implementation simple and lets each
    head use the same model family while sharing the aligned four-property input.
    This makes the joint benchmark trainable immediately and leaves room for a
    later shared-trunk variant.
    """

    def __init__(
        self,
        model_name: str,
        in_channels: int,
        output_specs: dict[str, dict[str, tuple[int, int] | int]],
    ):
        super().__init__()
        model_name = str(model_name).lower()
        if model_name not in MODEL_FAMILY_REGISTRY:
            raise ValueError(f"Unsupported joint model family '{model_name}'.")

        constructor = MODEL_FAMILY_REGISTRY[model_name]
        self.heads = nn.ModuleDict(
            {
                modality: constructor(
                    in_channels=in_channels,
                    out_channels=int(spec["out_channels"]),
                    output_shape=tuple(spec["output_shape"]),
                )
                for modality, spec in output_specs.items()
            }
        )

    def forward(self, x):
        return {modality: head(x) for modality, head in self.heads.items()}
