from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from experiments.datasets.modality_transforms import (
    downsample_binary_mask,
    format_mt_target,
    format_planar_input,
    format_seismic_target,
)


AVAILABLE_MODALITIES = ("gravity", "magnetic", "mt", "seismic")
MODALITY_STATUS_KEYS = {
    "gravity": "gravity_status",
    "magnetic": "magnetic_status",
    "mt": "mt_status",
    "seismic": "seismic_status",
}


def normalize_modalities(modalities: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(modality).lower() for modality in modalities)
    unsupported = sorted(set(normalized) - set(AVAILABLE_MODALITIES))
    if unsupported:
        raise ValueError(f"Unsupported modalities: {unsupported}. Available: {AVAILABLE_MODALITIES}")
    if not normalized:
        raise ValueError("At least one modality must be selected.")
    return normalized


def filter_records_for_modalities(records: list[dict], modalities: Iterable[str]) -> list[dict]:
    selected = normalize_modalities(modalities)
    filtered = []
    for record in records:
        if all(record.get(MODALITY_STATUS_KEYS[modality]) == "ok" for modality in selected):
            filtered.append(record)
    return filtered


def build_anomaly_label_mapping(records: list[dict]) -> dict[str, int]:
    anomaly_types = sorted({str(record["anomaly_type"]) for record in records})
    return {anomaly_type: index for index, anomaly_type in enumerate(anomaly_types)}


def _to_tensor(array: np.ndarray):
    if torch is None:
        return array
    return torch.from_numpy(array)


def _build_modality_array(bundle: dict, modality: str) -> np.ndarray:
    if modality == "gravity":
        return format_planar_input(bundle["gravity_data"]).astype(np.float32, copy=False)
    if modality == "magnetic":
        return format_planar_input(bundle["magnetic_data"]).astype(np.float32, copy=False)
    if modality == "mt":
        return format_mt_target(bundle["mt_app_res"], bundle["mt_phase"]).astype(np.float32, copy=False)
    if modality == "seismic":
        return format_seismic_target(bundle["seismic_data"]).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported modality '{modality}'.")


class GMESInverseDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        modalities: Iterable[str],
        anomaly_to_index: dict[str, int] | None = None,
        include_segmentation_target: bool = False,
        segmentation_shape: tuple[int, int, int] = (64, 64, 64),
    ):
        self.records = list(records)
        self.modalities = normalize_modalities(modalities)
        self.anomaly_to_index = anomaly_to_index or build_anomaly_label_mapping(self.records)
        self.include_segmentation_target = bool(include_segmentation_target)
        self.segmentation_shape = tuple(int(x) for x in segmentation_shape)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["bundle_path"], allow_pickle=True) as bundle:
            bundle_dict = {key: bundle[key] for key in bundle.files}

        modality_tensors = {
            modality: _to_tensor(_build_modality_array(bundle_dict, modality))
            for modality in self.modalities
        }
        anomaly_type = str(record["anomaly_type"])
        label = np.int64(self.anomaly_to_index[anomaly_type])

        item = {
            "modalities": modality_tensors,
            "label": torch.tensor(label, dtype=torch.long) if torch is not None else label,
            "metadata": {
                "bundle_path": record["bundle_path"],
                "partition": record["partition"],
                "background_id": record["background_id"],
                "anomaly_type": anomaly_type,
                "anomaly_name_en": record.get("anomaly_name_en", ""),
            },
        }

        if self.include_segmentation_target:
            mask = downsample_binary_mask(bundle_dict["anomaly_label"], output_shape=self.segmentation_shape).astype(np.float32, copy=False)
            item["anomaly_mask"] = _to_tensor(mask[None, ...])

        return item
