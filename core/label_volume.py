import os
import numpy as np


def build_label_volume_from_trends(trends, contour_num=12):
    """
    Build a discrete layer-label volume from a continuous trends volume.

    This follows the original sample conversion logic:
    generate evenly spaced contour levels across the full 3D ``gtime`` range,
    then classify the full volume with ``np.digitize``.
    """
    contour_num = int(contour_num)
    if contour_num < 1:
        raise ValueError("contour_num must be >= 1")

    trends = np.asarray(trends, dtype=np.float32)
    if trends.ndim != 3:
        raise ValueError(f"Expected a 3D trends volume, got shape {trends.shape}")

    tmin = float(np.min(trends))
    tmax = float(np.max(trends))

    if tmin == tmax:
        levels = np.empty((0,), dtype=np.float32)
        labels = np.zeros_like(trends, dtype=np.int16)
        return labels, levels

    levels = np.linspace(tmin, tmax, contour_num + 2, dtype=np.float32)[1:-1]
    labels = np.digitize(trends, levels).astype(np.int16, copy=False)
    return labels, levels


def load_label_volume_from_sample_npz(npz_path, contour_num=12, trends_key="gtime"):
    """
    Load a mirrored sample ``.npz`` file and convert its ``gtime`` trends into layer labels.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Sample npz not found: {npz_path}")

    with np.load(npz_path) as data:
        if trends_key not in data:
            raise KeyError(f"Key '{trends_key}' not found in sample npz: {npz_path}")
        trends = np.asarray(data[trends_key], dtype=np.float32)

    labels, levels = build_label_volume_from_trends(trends, contour_num=contour_num)
    return labels, levels
