import torch
from deepwave import scalar
import argparse
import os
import numpy as np
import random
import time
from utils import load_config, load_velocity_model, get_wavelet, setup_acquisition, plot_acquisition_geometry, resolve_y, check_and_resample_model, calculate_auto_time_duration
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Seismic Forward Modeling using Deepwave (Unified Acquisition)")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        return
        
    config = load_config(args.config)
    
    # Setup logger file output to save_dir
    save_path_conf = config.get('output', {}).get('save_path', 'output/data.npy')
    save_dir = os.path.dirname(save_path_conf)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Use current directory if save_dir is empty string
    log_dir = save_dir if save_dir else "."
    log_file = os.path.join(log_dir, "forward_modeling.log")
    logger.add(log_file, rotation="10 MB")
    
    # Device setup
    dev = config.get("simulation", {}).get("device", "cuda")
    device = torch.device("cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Running on {device}")
    
    # --- Feature 1: Random Frequency Selection ---
    src_conf = config.get('source', {})
    if 'freq' in src_conf:
        f_val = src_conf['freq']
        if isinstance(f_val, list):
            # Pick a random frequency from range [min, max]
            if len(f_val) >= 2:
                # Use integer random selection (inclusive)
                f_min = int(f_val[0])
                f_max = int(f_val[1])
                picked_freq = float(random.randint(f_min, f_max))
                logger.info(f"Random frequency range {f_val} -> picked: {picked_freq:.2f} Hz")
            else:
                picked_freq = float(f_val[0])
            config['source']['freq'] = picked_freq
        else:
            picked_freq = float(f_val)
    else:
        picked_freq = 25.0 # default
        
    # Load Model
    v = load_velocity_model(config).to(device)
    
    # --- Feature 2: Anti-dispersion Resampling ---
    # Must do this BEFORE setting up coordinates and acquisition
    v = check_and_resample_model(v, config, picked_freq)
    
    # Coordinates (Parameters might have been updated by resampling)
    model_conf = config['model']
    dx_val = float(model_conf.get('dx', 10.0))
    dz_val = float(model_conf.get('dz', 10.0))
    
    mode = config['simulation']['mode']
    if mode == "2D":
        # Handle 3D model -> 2D simulation
        if v.ndim == 3:
            ny = v.shape[1]
            y_idx = resolve_y(config['acquisition'].get('default_y', 'center'), ny)
            logger.info(f"Slicing 3D velocity model at y={y_idx} for 2D simulation.")
            v = v[:, y_idx, :] # [nx, ny, nz] -> [nx, nz]

        # Convention: dim0=x, dim1=z -> [dx, dz]
        dx = [dx_val, dz_val]
        logger.info(f"Grid spacing: dx={dx_val}, dz={dz_val}")
    else:
        dy_val = float(model_conf.get('dy', 10.0))
        # Convention: dim0=x, dim1=y, dim2=z -> [dx, dy, dz]
        dx = [dx_val, dy_val, dz_val]
        logger.info(f"Grid spacing: dx={dx_val}, dy={dy_val}, dz={dz_val}")

    dt = float(config['time']['dt'])
    
    # --- Feature 3: Auto-calculate Simulation Duration (Suggestion Only) ---
    if config['time'].get('suggest_t_calculate', False):
        t_auto, nt_auto = calculate_auto_time_duration(v, dx, dt)
        
        if 'nt' in config['time']:
            logger.info(f"Configured nt={config['time']['nt']}. Suggested nt={nt_auto} (based on 2*L/v_bar).")
        else:
            logger.warning(f"No 'nt' found in config. Using auto-calculated nt={nt_auto}")
            config['time']['nt'] = nt_auto

    
    # Wavelet
    wavelet = get_wavelet(config, device)
    
    # Acquisition Geometry
    source_locations, receiver_locations = setup_acquisition(config, device)
    
    n_shots = int(source_locations.shape[0])
    n_src = int(source_locations.shape[1])
    n_rec = int(receiver_locations.shape[1])

    # Plot geometry if requested
    if config['acquisition'].get('show_geometry', False):
        logger.info("Saving acquisition geometry...")
        # Determine save path for geometry plot
        save_path_data = config.get('output', {}).get('save_path', 'output.npy')
        save_dir = os.path.dirname(save_path_data)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(save_path_data))[0]
        geometry_save_path = os.path.join(save_dir, f"{base_name}_geometry.png")
        
        plot_acquisition_geometry(source_locations, receiver_locations, config['model']['shape'], mode, save_path=geometry_save_path)
    
    # Source Amplitudes
    # shape: [n_shots, n_sources_per_shot, nt]
    source_amplitudes = wavelet.unsqueeze(0).unsqueeze(0).repeat(n_shots, n_src, 1)
    
    logger.info("Starting Propagator...")
    logger.info(f"Model shape: {tuple(v.shape)}")
    logger.info(f"Shots: {n_shots}")
    logger.info(f"Receivers/shot: {n_rec}")
    
    # Run Propagator
    # Note: pml_freq acts as a hint for PML width optimization
    pml_freq = float(config['source'].get('freq', 25))
    accuracy = int(config.get("simulation", {}).get("accuracy", 4))
    
    measure_time = config.get("simulation", {}).get("measure_time", False)
    start_time = 0.0
    
    if measure_time:
        start_time = time.time()
        logger.info("Timing started...")

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=pml_freq,
        accuracy=accuracy 
    )
    
    if measure_time:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Forward modeling execution time: {elapsed_time:.4f} seconds")

    receiver_data = out[-1] # Shape: [n_shots, n_receivers, nt]
    
    logger.info("Simulation Complete.")
    logger.info(f"Output shape: {tuple(receiver_data.shape)}")
    
    # Save Results
    save_path = config['output']['save_path']
    # If path ends with .npy, switch to .npz for comprehensive saving
    if save_path.endswith('.npy'):
        save_path = save_path[:-4] + '.npz'
    elif not save_path.endswith('.npz'):
        save_path += '.npz'

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    np.savez(
        save_path, 
        data=receiver_data.cpu().detach().numpy(),
        src=source_locations.cpu().detach().numpy(),
        rec=receiver_locations.cpu().detach().numpy(),
        dt=np.array(dt),
        dx=np.array(dx)
    )
    logger.info(f"Saved data and geometry to {save_path}")

if __name__ == "__main__":
    main()
