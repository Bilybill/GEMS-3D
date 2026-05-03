import argparse
import os
import time

import torch
import numpy as np
from loguru import logger

from utils import load_config, load_density_model
from gra_forward import forward_gravity_gz


def main():
    parser = argparse.ArgumentParser(description="Gravity Forward Modeling (Unified Config Style)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        return

    config = load_config(args.config)

    # Setup logger file output to save_dir (same pattern as Seismic/forward_modeling/main.py)
    save_path_conf = config.get("output", {}).get("save_path", "output/gravity_data.npz")
    save_dir = os.path.dirname(save_path_conf)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    log_dir = save_dir if save_dir else "."
    log_file = os.path.join(log_dir, "forward_modeling.log")
    logger.add(log_file, rotation="10 MB")

    # Device setup
    dev = config.get("simulation", {}).get("device", "cuda")
    device = torch.device("cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Running on {device}")

    # Load density model
    rho = load_density_model(config, args.config).to(device)
    logger.info("Gravity density input normalized to g/cm^3; solver converts internally to kg/m^3.")

    # Grid spacing
    mconf = config.get("model", {})
    dx = float(mconf.get("dx", 10.0))
    dy = float(mconf.get("dy", 10.0))
    dz = float(mconf.get("dz", 10.0))

    # Gravity config
    gconf = config.get("gravity", {})
    heights_m = gconf.get("heights_m", [0.0])
    output_unit = gconf.get("output_unit", "mgal")
    pad_factor = int(gconf.get("pad_factor", 2))
    G = float(gconf.get("G", 6.67430e-11))
    algorithm = gconf.get("algorithm", "prism_exact")
    obs_conf = gconf.get("observation", {"layout": "grid"})

    t0 = time.time()
    data, meta = forward_gravity_gz(
        density=rho,
        dx=dx,
        dy=dy,
        dz=dz,
        heights_m=heights_m,
        obs_conf=obs_conf,
        G=G,
        output_unit=output_unit,
        pad_factor=pad_factor,
        density_unit="g/cm^3",
        algorithm=algorithm,
    )
    t1 = time.time()
    logger.info(f"Forward done. Output shape={tuple(data.shape)}. Time={(t1 - t0):.2f}s")

    # Save output (align with seismic: ensure .npz)
    save_path = save_path_conf
    if save_path.endswith(".npy"):
        save_path = save_path[:-4] + ".npz"
    elif not save_path.endswith(".npz"):
        save_path += ".npz"

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # pack meta (tensors -> numpy)
    meta_np = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            meta_np[k] = v.detach().cpu().numpy()
        elif isinstance(v, (list, tuple)):
            meta_np[k] = np.array(v)
        else:
            meta_np[k] = v

    np.savez(
        save_path,
        data=data.detach().cpu().numpy(),
        **meta_np,
    )
    logger.info(f"Saved gravity data to {save_path}")


if __name__ == "__main__":
    main()
