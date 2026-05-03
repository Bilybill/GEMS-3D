from __future__ import annotations

import numpy as np


MT_CANONICAL_FREQS_HZ = (
    10000.0,
    8000.0,
    6000.0,
    4000.0,
    3000.0,
    2000.0,
    1500.0,
    1000.0,
    800.0,
    600.0,
    400.0,
    300.0,
    200.0,
    150.0,
    100.0,
    80.0,
    60.0,
    40.0,
    30.0,
)


def standardize_volume(volume: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    mean = float(volume.mean())
    std = float(volume.std())
    if std < eps:
        std = 1.0
    return (volume - mean) / std


def log_standardize_volume(volume: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    safe = np.clip(volume, eps, None)
    return standardize_volume(np.log10(safe), eps=eps)


def standardize_map(array: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    return standardize_volume(np.asarray(array, dtype=np.float32), eps=eps)


def format_planar_target(array_2d: np.ndarray) -> np.ndarray:
    array_2d = np.asarray(array_2d, dtype=np.float32)
    if array_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array_2d.shape}.")
    return array_2d[None, ...]


def format_planar_input(array_2d: np.ndarray) -> np.ndarray:
    return format_planar_target(standardize_map(array_2d))


def _align_mt_to_canonical_grid(
    mt_app_res: np.ndarray,
    mt_phase: np.ndarray,
    mt_freqs_hz: np.ndarray,
    canonical_freqs_hz: tuple[float, ...] = MT_CANONICAL_FREQS_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    app_res = np.asarray(mt_app_res, dtype=np.float32)
    phase = np.asarray(mt_phase, dtype=np.float32)
    freqs = np.asarray(mt_freqs_hz, dtype=np.float32).reshape(-1)

    if app_res.shape != phase.shape:
        raise ValueError(f"MT apparent-resistivity shape {app_res.shape} does not match phase shape {phase.shape}.")
    if app_res.ndim != 4:
        raise ValueError(f"Expected MT tensors with shape (freq, x, y, pol), got {app_res.shape}.")
    if app_res.shape[0] != freqs.shape[0]:
        raise ValueError(
            f"MT frequency count {freqs.shape[0]} does not match MT tensor shape {app_res.shape}."
        )

    canonical = np.asarray(canonical_freqs_hz, dtype=np.float32).reshape(-1)
    index_by_freq = {round(float(freq), 6): idx for idx, freq in enumerate(canonical.tolist())}
    observed_mask = np.zeros((canonical.shape[0], *phase.shape[1:]), dtype=np.float32)

    for src_idx, freq in enumerate(freqs.tolist()):
        key = round(float(freq), 6)
        if key in index_by_freq:
            observed_mask[index_by_freq[key]] = 1.0

    sort_idx = np.argsort(freqs)
    sorted_freqs = freqs[sort_idx].astype(np.float64, copy=False)
    sorted_app_res = app_res[sort_idx].astype(np.float32, copy=False)
    sorted_phase = phase[sort_idx].astype(np.float32, copy=False)

    log_src = np.log10(np.clip(sorted_freqs, 1.0e-12, None))
    log_dst = np.log10(np.clip(canonical.astype(np.float64, copy=False), 1.0e-12, None))

    insert_idx = np.searchsorted(log_src, log_dst, side="left")
    right_idx = np.clip(insert_idx, 0, len(log_src) - 1)
    left_idx = np.clip(insert_idx - 1, 0, len(log_src) - 1)

    left_log = log_src[left_idx]
    right_log = log_src[right_idx]
    denom = right_log - left_log
    weights = np.zeros_like(log_dst, dtype=np.float32)
    nonzero = np.abs(denom) > 1.0e-12
    weights[nonzero] = ((log_dst[nonzero] - left_log[nonzero]) / denom[nonzero]).astype(np.float32, copy=False)

    flat_app_res = sorted_app_res.reshape(sorted_app_res.shape[0], -1)
    flat_phase = sorted_phase.reshape(sorted_phase.shape[0], -1)

    aligned_app_res = (
        (1.0 - weights)[:, None] * flat_app_res[left_idx] + weights[:, None] * flat_app_res[right_idx]
    ).reshape(canonical.shape[0], *app_res.shape[1:]).astype(np.float32, copy=False)
    aligned_phase = (
        (1.0 - weights)[:, None] * flat_phase[left_idx] + weights[:, None] * flat_phase[right_idx]
    ).reshape(canonical.shape[0], *phase.shape[1:]).astype(np.float32, copy=False)

    return aligned_app_res, aligned_phase, observed_mask


def format_mt_target(
    mt_app_res: np.ndarray,
    mt_phase: np.ndarray,
    mt_freqs_hz: np.ndarray | None = None,
    app_res_eps: float = 1.0e-6,
    canonical_freqs_hz: tuple[float, ...] = MT_CANONICAL_FREQS_HZ,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    app_res = np.asarray(mt_app_res, dtype=np.float32)
    phase = np.asarray(mt_phase, dtype=np.float32)

    if mt_freqs_hz is not None:
        app_res, phase, mask = _align_mt_to_canonical_grid(
            app_res,
            phase,
            mt_freqs_hz,
            canonical_freqs_hz=canonical_freqs_hz,
        )
    else:
        if app_res.shape != phase.shape:
            raise ValueError(f"MT apparent-resistivity shape {app_res.shape} does not match phase shape {phase.shape}.")
        if app_res.ndim != 4:
            raise ValueError(f"Expected MT tensors with shape (freq, x, y, pol), got {app_res.shape}.")
        mask = np.ones_like(app_res, dtype=np.float32)

    app_res = np.log10(np.clip(app_res, app_res_eps, None))
    phase = phase / 180.0
    # (freq, x, y, pol) -> (channels, x, y)
    app_res = np.transpose(app_res, (0, 3, 1, 2)).reshape(-1, app_res.shape[1], app_res.shape[2])
    phase = np.transpose(phase, (0, 3, 1, 2)).reshape(-1, phase.shape[1], phase.shape[2])
    mask = np.transpose(mask, (0, 3, 1, 2)).reshape(-1, mask.shape[1], mask.shape[2])

    target = np.concatenate([app_res, phase], axis=0)
    target_mask = np.concatenate([mask, mask], axis=0)
    if return_mask:
        return target, target_mask
    return target


def format_seismic_target(
    seismic_data: np.ndarray,
    shot_stride: int = 1,
    receiver_stride: int = 4,
    time_stride: int = 2,
    amplitude_clip_quantile: float | None = 0.995,
    rms_eps: float = 1.0e-8,
) -> np.ndarray:
    seismic = np.asarray(seismic_data, dtype=np.float32)
    if seismic.ndim != 3:
        raise ValueError(f"Expected seismic data with shape (shot, receiver, time), got {seismic.shape}.")

    seismic = seismic[:: max(int(shot_stride), 1), :: max(int(receiver_stride), 1), :: max(int(time_stride), 1)]
    rms = float(np.sqrt(np.mean(np.square(seismic, dtype=np.float64))))
    if rms > float(rms_eps):
        seismic = seismic / rms
    if amplitude_clip_quantile is not None:
        quantile = float(np.quantile(np.abs(seismic), amplitude_clip_quantile))
        if quantile > 0.0:
            seismic = np.clip(seismic, -quantile, quantile) / quantile
    return seismic


def format_seismic_shot_target(
    seismic_data: np.ndarray,
    shot_index: int,
    receiver_stride: int = 4,
    time_stride: int = 2,
    amplitude_clip_quantile: float | None = 0.995,
    rms_eps: float = 1.0e-8,
) -> np.ndarray:
    seismic = np.asarray(seismic_data, dtype=np.float32)
    if seismic.ndim != 3:
        raise ValueError(f"Expected seismic data with shape (shot, receiver, time), got {seismic.shape}.")
    shot_index = int(shot_index)
    if shot_index < 0 or shot_index >= seismic.shape[0]:
        raise IndexError(f"Shot index {shot_index} is out of range for seismic data with {seismic.shape[0]} shots.")

    shot = seismic[shot_index : shot_index + 1, :: max(int(receiver_stride), 1), :: max(int(time_stride), 1)]
    rms = float(np.sqrt(np.mean(np.square(shot, dtype=np.float64))))
    if rms > float(rms_eps):
        shot = shot / rms
    if amplitude_clip_quantile is not None:
        quantile = float(np.quantile(np.abs(shot), amplitude_clip_quantile))
        if quantile > 0.0:
            shot = np.clip(shot, -quantile, quantile) / quantile
    return shot.astype(np.float32, copy=False)


def downsample_binary_mask(mask: np.ndarray, output_shape: tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    mask = np.asarray(mask > 0, dtype=np.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")
    target_z, target_y, target_x = output_shape

    if (
        mask.shape[0] % target_z == 0
        and mask.shape[1] % target_y == 0
        and mask.shape[2] % target_x == 0
    ):
        factor_z = mask.shape[0] // target_z
        factor_y = mask.shape[1] // target_y
        factor_x = mask.shape[2] // target_x
        reduced = mask.reshape(target_z, factor_z, target_y, factor_y, target_x, factor_x).mean(axis=(1, 3, 5))
        return (reduced >= 0.5).astype(np.float32, copy=False)

    z_bins = np.array_split(np.arange(mask.shape[0]), target_z)
    y_bins = np.array_split(np.arange(mask.shape[1]), target_y)
    x_bins = np.array_split(np.arange(mask.shape[2]), target_x)

    reduced = np.zeros(output_shape, dtype=np.float32)
    for iz, z_idx in enumerate(z_bins):
        for iy, y_idx in enumerate(y_bins):
            for ix, x_idx in enumerate(x_bins):
                block = mask[np.ix_(z_idx, y_idx, x_idx)]
                reduced[iz, iy, ix] = float(block.mean() >= 0.5)
    return reduced
