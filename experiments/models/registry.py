from __future__ import annotations

from experiments.models.joint_forward import JointForwardModel, MODEL_FAMILY_REGISTRY
from experiments.models.operator_deeponet import DeepONetObservationModel
from experiments.models.operator_fno import FNOObservationModel
from experiments.models.operator_gnot import GNOTObservationModel
from experiments.models.operator_pinn import PINNObservationModel
from experiments.models.operator_unet import UNetObservationModel
from experiments.models.shot_conditioned_seismic import ShotConditionedSeismicModel


AVAILABLE_FORWARD_MODELS = ("unet", "pinn", "deeponet", "fno", "gnot", "shot_film")


def build_forward_model(
    model_name: str,
    in_channels: int,
    out_channels: int | None = None,
    output_shape: tuple[int, int] | None = None,
    output_specs: dict[str, dict[str, tuple[int, int] | int]] | None = None,
    condition_dim: int | None = None,
):
    model_name = str(model_name).lower()
    if output_specs is not None:
        if model_name == "shot_film":
            raise ValueError("shot_film is a single-shot seismic baseline and is not supported for joint outputs.")
        return JointForwardModel(model_name=model_name, in_channels=in_channels, output_specs=output_specs)
    if out_channels is None or output_shape is None:
        raise ValueError("out_channels and output_shape are required for single-task forward models.")
    if model_name == "shot_film":
        return ShotConditionedSeismicModel(
            in_channels=in_channels,
            out_channels=out_channels,
            output_shape=output_shape,
            condition_dim=condition_dim or 4,
        )
    if model_name == "unet":
        return UNetObservationModel(in_channels=in_channels, out_channels=out_channels, output_shape=output_shape)
    if model_name == "pinn":
        return PINNObservationModel(in_channels=in_channels, out_channels=out_channels, output_shape=output_shape)
    if model_name == "deeponet":
        return DeepONetObservationModel(in_channels=in_channels, out_channels=out_channels, output_shape=output_shape)
    if model_name == "fno":
        return FNOObservationModel(in_channels=in_channels, out_channels=out_channels, output_shape=output_shape)
    if model_name == "gnot":
        return GNOTObservationModel(in_channels=in_channels, out_channels=out_channels, output_shape=output_shape)
    raise ValueError(
        f"Unsupported model '{model_name}'. Available forward models: {', '.join(AVAILABLE_FORWARD_MODELS)}."
    )
