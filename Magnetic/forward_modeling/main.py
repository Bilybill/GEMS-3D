import argparse
import os
import time

import torch
import numpy as np
from loguru import logger

from utils import load_config, load_susceptibility_model
from mag_forward import forward_mag_tmi

def main():
    parser = argparse.ArgumentParser(description="Magnetic Forward Modeling (Unified Config Style)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["standard_B", "prism_matched"],
        help="Forward mode override. If not set, uses magnetic.mode from config (default: standard_B).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        # If running without args, maybe generate defaults?
        pass

    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Check output path
    out_conf = config.get("output", {})
    save_path = out_conf.get("save_path", "output/magnetic_data.npz")
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Logger
    log_file = os.path.join(save_dir if save_dir else ".", "forward_modeling.log")
    logger.add(log_file, rotation="10 MB")

    # Device
    sim_conf = config.get("simulation", {})
    dev_str = sim_conf.get("device", "cuda")
    device = torch.device("cuda" if (dev_str == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Running on {device}")

    # Load Model (Susceptibility)
    # Shape is (nx, ny, nz)
    model = load_susceptibility_model(config, args.config).to(device)
    nx, ny, nz = model.shape
    logger.info(f"Model shape: {model.shape}")

    # Geometry
    mconf = config.get("model", {})
    dx = float(mconf.get("dx", 10.0))
    dy = float(mconf.get("dy", 10.0))
    dz = float(mconf.get("dz", 10.0))

    # Magnetic Parameters
    # Default Earth field ~ 50,000 nT
    mag_conf = config.get("magnetic", {})
    B0 = float(mag_conf.get("B0", 50000.0)) 
    I_deg = float(mag_conf.get("inclination", 90.0))
    A_deg = float(mag_conf.get("declination", 0.0))
    
    # Magnetization direction (if different from induced)
    M_I = mag_conf.get("mag_inclination", None)
    M_A = mag_conf.get("mag_declination", None)
    if M_I is not None: M_I = float(M_I)
    if M_A is not None: M_A = float(M_A)

    heights_m = mag_conf.get("heights_m", [0.0])
    output_unit = str(mag_conf.get("output_unit", "nt"))
    pad_factor = int(mag_conf.get("pad_factor", 2))
    input_type = str(mag_conf.get("input_type", "susceptibility"))
    mode = str(mag_conf.get("mode", "standard_B"))
    if args.mode is not None:
        mode = args.mode

    # Observation
    obs_conf = config.get("observation", {})
    # default full grid
    if "n_x" not in obs_conf: obs_conf["n_x"] = nx
    if "n_y" not in obs_conf: obs_conf["n_y"] = ny

    # Run Forward
    t0 = time.time()
    logger.info(f"Starting forward modeling (mode={mode})...")
    
    data, meta = forward_mag_tmi(
        susceptibility=model,
        dx=dx, dy=dy, dz=dz,
        heights_m=heights_m,
        obs_conf=obs_conf,
        input_type=input_type,
        B0=B0,
        I_deg=I_deg,
        A_deg=A_deg,
        M_I_deg=M_I,
        M_A_deg=M_A,
        output_unit=output_unit,
        pad_factor=pad_factor,
        mode=mode,
    )

    elapsed = time.time() - t0
    logger.info(f"Forward modeling finished in {elapsed:.4f} s")

    # Save
    result = data.cpu().numpy()
    meta_np = {k: v.numpy() if torch.is_tensor(v) else v for k, v in meta.items()}
    
    np.savez(save_path, data=result, **meta_np)
    logger.info(f"Saved results to {save_path}")

if __name__ == "__main__":
    main()
