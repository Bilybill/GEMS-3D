import os
import yaml
import numpy as np
import torch
from loguru import logger

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_path(path: str, config_path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(cfg_dir, path))

def load_susceptibility_model(config: dict, config_path: str) -> torch.Tensor:
    """
    Load susceptibility model as torch.Tensor with shape (nx, ny, nz), float32.
    """
    mconf = config.get("model", {})
    fpath = mconf.get("file_path", "")
    full_path = _resolve_path(fpath, config_path)
    shape = mconf.get("shape", None)
    npz_key = mconf.get("npz_key", None)

    if full_path and os.path.exists(full_path):
        ext = os.path.splitext(full_path)[1].lower()
        if ext == ".npy":
            arr = np.load(full_path)
        elif ext == ".npz":
            z = np.load(full_path)
            if npz_key and npz_key in z:
                arr = z[npz_key]
            elif "susceptibility" in z:
                arr = z["susceptibility"]
            elif "model" in z:
                arr = z["model"]
            else:
                # heuristic: first key
                k = next(iter(z))
                arr = z[k]
        elif ext == ".bin" or ext == ".dat":
            if not shape:
                logger.error("Loading .bin requires model.shape in config.")
                raise ValueError("Shape needed for binary load.")
            arr = np.fromfile(full_path, dtype=np.float32).reshape(shape)
        else:
            logger.warning(f"Unknown model extension {ext}, trying np.loadtxt...")
            arr = np.loadtxt(full_path)
    else:
        # Generate synthetic or fail
        logger.warning(f"Model file {fpath} not found or not specified. Generating synthetic model.")
        if not shape:
            shape = [100, 100, 50]
        nx, ny, nz = shape
        arr = np.zeros((nx, ny, nz), dtype=np.float32)
        # Simple prism in center
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        r = min(nx, ny, nz) // 4
        arr[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 0.05 # SI units
        
    return torch.from_numpy(arr).float()
