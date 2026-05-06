from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from experiments.datasets.gmes_inverse_dataset import (
    MODALITY_STATUS_KEYS,
    _build_modality_array,
    normalize_modalities,
)
from experiments.datasets.modality_transforms import downsample_volume_mean, log_standardize_volume, standardize_volume


PROPERTY_ALIASES = {
    "vp": "vp",
    "v_p": "vp",
    "rho": "rho",
    "density": "rho",
    "phi": "phi",
    "res": "phi",
    "resistivity": "phi",
    "chi": "chi",
    "susceptibility": "chi",
}

PROPERTY_BUNDLE_KEYS = {
    "vp": "vp_model",
    "rho": "rho_model",
    "phi": "res_model",
    "chi": "chi_model",
}


def normalize_properties(properties: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for property_name in properties:
        key = str(property_name).lower()
        if key not in PROPERTY_ALIASES:
            raise ValueError(f"Unsupported property '{property_name}'. Available: {sorted(set(PROPERTY_ALIASES.values()))}")
        normalized.append(PROPERTY_ALIASES[key])
    if not normalized:
        raise ValueError("At least one target property must be selected.")
    return tuple(normalized)


def filter_records_for_reconstruction(records: list[dict], modalities: Iterable[str]) -> list[dict]:
    selected = normalize_modalities(modalities)
    return [
        record
        for record in records
        if all(record.get(MODALITY_STATUS_KEYS[modality]) == "ok" for modality in selected)
    ]


def _to_tensor(array: np.ndarray):
    if torch is None:
        return array
    return torch.from_numpy(array)


def _build_property_array(bundle: dict, property_name: str, target_shape: tuple[int, int, int]) -> np.ndarray:
    key = PROPERTY_BUNDLE_KEYS[property_name]
    volume = downsample_volume_mean(bundle[key], output_shape=target_shape)
    if property_name == "phi":
        return log_standardize_volume(volume).astype(np.float32, copy=False)
    return standardize_volume(volume).astype(np.float32, copy=False)


class GMESPropertyReconstructionDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        modalities: Iterable[str],
        target_properties: Iterable[str],
        target_shape: tuple[int, int, int] = (64, 64, 64),
    ):
        self.records = list(records)
        self.modalities = normalize_modalities(modalities)
        self.target_properties = normalize_properties(target_properties)
        self.target_shape = tuple(int(x) for x in target_shape)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["bundle_path"], allow_pickle=True) as bundle:
            bundle_dict = {key: bundle[key] for key in bundle.files}

        inputs = {
            modality: _to_tensor(_build_modality_array(bundle_dict, modality).astype(np.float32, copy=False))
            for modality in self.modalities
        }
        targets = np.stack(
            [
                _build_property_array(bundle_dict, property_name, self.target_shape)
                for property_name in self.target_properties
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        return {
            "inputs": inputs,
            "targets": _to_tensor(targets),
            "raw_targets": _to_tensor(targets.copy()),
            "target_masks": _to_tensor(np.ones_like(targets, dtype=np.float32)),
            "metadata": {
                "bundle_path": record["bundle_path"],
                "partition": record["partition"],
                "background_id": record["background_id"],
                "anomaly_type": record["anomaly_type"],
                "modalities": self.modalities,
                "target_properties": self.target_properties,
            },
        }


class GMESModalityCompletionDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        input_modalities: Iterable[str],
        target_modality: str,
    ):
        self.records = list(records)
        self.input_modalities = normalize_modalities(input_modalities)
        self.target_modality = normalize_modalities((target_modality,))[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["bundle_path"], allow_pickle=True) as bundle:
            bundle_dict = {key: bundle[key] for key in bundle.files}

        inputs = {
            modality: _to_tensor(_build_modality_array(bundle_dict, modality).astype(np.float32, copy=False))
            for modality in self.input_modalities
        }
        target = _build_modality_array(bundle_dict, self.target_modality).astype(np.float32, copy=False)
        return {
            "inputs": inputs,
            "targets": _to_tensor(target),
            "raw_targets": _to_tensor(target.copy()),
            "target_masks": _to_tensor(np.ones_like(target, dtype=np.float32)),
            "metadata": {
                "bundle_path": record["bundle_path"],
                "partition": record["partition"],
                "background_id": record["background_id"],
                "anomaly_type": record["anomaly_type"],
                "input_modalities": self.input_modalities,
                "target_modality": self.target_modality,
            },
        }
