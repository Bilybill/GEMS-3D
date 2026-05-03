import os
import yaml
import numpy as np
import torch
from loguru import logger


CANONICAL_DENSITY_UNIT = "g/cm^3"


def _normalize_density_unit(density_unit: str) -> str:
    unit = (density_unit or CANONICAL_DENSITY_UNIT).lower().replace(" ", "")
    if unit in ["g/cm3", "g/cm^3", "g/cc", "gcc"]:
        return CANONICAL_DENSITY_UNIT
    if unit in ["kg/m3", "kg/m^3", "kgm3"]:
        return "kg/m^3"
    raise ValueError(f"Unknown density_unit: {density_unit}. Use 'g/cm^3' or 'kg/m^3'.")


def _convert_density_to_gcc(arr: np.ndarray, density_unit: str) -> np.ndarray:
    unit = _normalize_density_unit(density_unit)
    out = arr.astype(np.float32, copy=False)
    if unit == "kg/m^3":
        return out / 1000.0
    return out


def load_config(config_path: str) -> dict:
    """Load YAML config (same style as Seismic/forward_modeling/utils.py)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str, config_path: str) -> str:
    """
    Resolve relative paths:
    - If `path` is relative, interpret it relative to the directory containing the config file.
    - If empty/None, return as-is.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(cfg_dir, path))


def load_density_model(config: dict, config_path: str) -> torch.Tensor:
    """
    Load density model as torch.Tensor with shape (nx, ny, nz), float32.

    Returned density is normalized to g/cm^3.
    The gravity solver converts it internally to kg/m^3.

    Supports:
    - .npy: raw numpy array
    - .npz: specify key with model.npz_key (default tries 'density' then first key)
    - .bin: raw float32 binary, requires model.shape
    """
    mconf = config.get("model", {})
    fpath = _resolve_path(mconf.get("file_path", ""), config_path)
    shape = mconf.get("shape", None)
    npz_key = mconf.get("npz_key", None)
    density_unit = _normalize_density_unit(mconf.get("density_unit", CANONICAL_DENSITY_UNIT))

    if fpath and os.path.exists(fpath):
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".npy":
            arr = np.load(fpath)
        elif ext == ".npz":
            z = np.load(fpath)
            if npz_key and npz_key in z:
                arr = z[npz_key]
            elif "density" in z:
                arr = z["density"]
            else:
                # fallback: first key
                arr = z[z.files[0]]
        elif ext == ".bin":
            if shape is None:
                raise ValueError("model.shape is required for .bin density model.")
            arr = np.fromfile(fpath, dtype=np.float32).reshape(tuple(shape))
        else:
            raise ValueError(f"Unsupported density model format: {ext}")
        logger.info(
            f"Loaded density model: {fpath}, shape={arr.shape}, dtype={arr.dtype}, "
            f"input_unit={density_unit}"
        )
    else:
        if shape is None:
            raise FileNotFoundError(f"Density model file not found: {fpath} and model.shape not provided.")
        default_rho = float(mconf.get("default_density", 2.67))
        arr = np.full(tuple(shape), default_rho, dtype=np.float32)
        logger.warning(
            f"Density model not found, using constant density={default_rho} {density_unit}, shape={shape}"
        )

    if arr.ndim != 3:
        raise ValueError(f"Density model must be 3D, got shape={arr.shape}")
    # Ensure (nx, ny, nz)
    if shape is not None and tuple(arr.shape) != tuple(shape):
        logger.warning(f"Density shape mismatch: file={arr.shape}, config.shape={shape}. Using file shape.")

    if density_unit == CANONICAL_DENSITY_UNIT and float(np.nanmax(arr)) > 50.0:
        logger.warning(
            "density_unit is set to g/cm^3 but values look like kg/m^3; auto-converting by /1000 for compatibility."
        )
        arr = arr / 1000.0

    arr = _convert_density_to_gcc(arr, density_unit)
    logger.info(f"Density model normalized to {CANONICAL_DENSITY_UNIT} for project use.")
    return torch.from_numpy(arr.astype(np.float32))
