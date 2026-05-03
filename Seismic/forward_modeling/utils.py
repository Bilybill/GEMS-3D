import yaml
import torch
import numpy as np
import deepwave
import os
import segyio
import math
from loguru import logger

import matplotlib.pyplot as plt

def plot_acquisition_geometry(source_locations, receiver_locations, model_shape, mode="2D", save_path=None):
    """
    Plots the acquisition geometry.
    source_locations: [n_shots, n_src_per_shot, dim]
    receiver_locations: [n_shots, n_rec_per_shot, dim]
    """
    src = source_locations.cpu().numpy()
    rec = receiver_locations.cpu().numpy()
    
    n_shots = src.shape[0]
    
    fig = plt.figure(figsize=(10, 8))
    
    if mode == "2D":
        ax = fig.add_subplot(111)
        # Plot model boundary
        nx, nz = model_shape
        ax.set_xlim(0, nx)
        ax.set_ylim(nz, 0) # Depth is positive down
        
        # Plot all receivers (flattened)
        # Note: For rolling/patch, receivers move. We can plot all distinct locations or just for first/last shot.
        # Plotting ALL points might be heavy. Let's plot first shot in distinct color, others in gray?
        
        # Plot receivers for all shots
        # Flatten [n_shots, n_rec, 2] -> [N, 2]
        all_recs = rec.reshape(-1, 2)
        ax.scatter(all_recs[:, 0], all_recs[:, 1], s=1, c='gray', alpha=0.1, label='All Receivers')
        
        # Plot sources
        all_srcs = src.reshape(-1, 2)
        ax.scatter(all_srcs[:, 0], all_srcs[:, 1], s=20, c='red', marker='*', label='Sources')
        
        # Highlight first shot geometry
        first_shot_recs = rec[0]
        ax.scatter(first_shot_recs[:, 0], first_shot_recs[:, 1], s=5, c='blue', label='Shot 0 Receivers')
        
        ax.set_xlabel("X (grid index)")
        ax.set_ylabel("Z (grid index)")
        ax.set_title("2D Acquisition Geometry")
        ax.legend()
        
    elif mode == "3D":
        ax = fig.add_subplot(111, projection='3d')
        nx, ny, nz = model_shape
        
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(nz, 0) # Depth positive down
        
        # Subsmaple for plotting if too many
        subsample = 1
        if rec.size > 10000:
             subsample = 10
             
        # Plot sources
        all_srcs = src.reshape(-1, 3)
        ax.scatter(all_srcs[:, 0], all_srcs[:, 1], all_srcs[:, 2], c='red', marker='*', s=50, label='Sources')
        
        # Plot receivers (first shot only to avoid clutter in 3D?)
        # Or maybe just a subset
        first_shot_recs = rec[0]
        ax.scatter(first_shot_recs[:, 0], first_shot_recs[:, 1], first_shot_recs[:, 2], c='blue', s=2, alpha=0.5, label='Shot 0 Receivers')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Acquisition Geometry (Shot 0)")
        ax.legend()
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        logger.info(f"Acquisition geometry plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_velocity_volume(path, shape, model_conf=None):
    """
    Load a velocity volume from path using the same rules as ``load_velocity_model``.

    Parameters
    ----------
    path:
        Input file path. Supports ``.npy``, ``.npz``, ``.bin``, ``.sgy``, ``.segy``.
    shape:
        Target volume shape, typically ``[nx, ny, nz]`` for 3D.
    model_conf:
        Optional model config dict used by composite ``.npz`` scaling logic.
    """
    model_conf = model_conf or {}
    if os.path.exists(path):
        if path.endswith('.npy'):
            v = np.load(path)
        elif path.endswith('.npz'):
            data = np.load(path)
            # Check for keys used in read_npz_2_segy.py logic
            if 'formatS' in data and 'formatD' in data and 'gtime' in data:
                logger.info("Detected composite model components (formatS, formatD, gtime)...")
                Details = data['formatS']
                MDetails = data['formatD']
                Trends = data['gtime']
                
                # Retrieve ranges from config or use defaults from read_npz_2_segy.py
                trends_range = model_conf.get('trends_range', (2000, 6000))
                details_range = model_conf.get('details_range', (-500, 1000))
                mdetails_range = model_conf.get('mdetails_range', (-200, 200))
                
                # Normalize to [0,1] then scale to specified ranges
                def _scale(arr, vmin, vmax):
                    arr = arr.astype(np.float64)
                    if arr.max() != arr.min():
                        arr = (arr - arr.min()) / (arr.max() - arr.min())
                    return vmin + arr * (vmax - vmin)

                Trends = _scale(Trends, trends_range[0], trends_range[1])
                Details = _scale(Details, details_range[0], details_range[1])
                MDetails = _scale(MDetails, mdetails_range[0], mdetails_range[1])

                v = Trends + Details + MDetails
            else:
                # Try to guess key or use first one
                key = list(data.keys())[0] 
                v = data[key]
        elif path.endswith('.bin'):
            # Standard C-order reading. Ensure file size matches expected shape.
            # Note: If the file was generated by read_npz_2_segy.py, it is a raw dump of a 3D volume (nx, ny, nz) in C-order.
            # Users must ensure config['model']['shape'] matches the file's dimensions.
            expected_count = np.prod(shape)
            file_bytes = os.path.getsize(path)
            if file_bytes != expected_count * 4:
                 logger.warning(f"Warning: File size {file_bytes} bytes does not match shape {shape} ({expected_count} floats). Reading first {expected_count} elements if possible.")

            try:
                v = np.fromfile(path, dtype=np.float32)
                if v.size == expected_count:
                    v = v.reshape(shape)
                elif v.size > expected_count:
                     # Maybe the file is larger (e.g. 3D file for 2D sim). Warning already printed.
                     # We take the first chunk which corresponds to the first slice(s) if X is slow dim.
                     # But this is risky. 
                     v = v[:expected_count].reshape(shape)
                else:
                    raise ValueError(f"Loaded {v.size} elements, expected {expected_count} based on shape {shape}.")
            except Exception as e:
                raise ValueError(f"Failed to reshape .bin file: {e}")
        elif path.endswith('.sgy') or path.endswith('.segy'):
            try:
                # Try reading with geometry first (common for 3D)
                with segyio.open(path, "r", strict=False) as f:
                    try:
                        v = segyio.tools.cube(f) # Returns (Inline, Crossline, Depth)
                    except Exception:
                        # Fallback: treat as collection of traces (2D or flattened 3D)
                        v = f.trace.raw[:] # Returns (N_traces, N_samples)
                
                # If doing 3D simulation but loaded 2D (Nx*Ny, Nz), try to reshape if shape matches
                # shape from config is usually [nx, ny, nz] for 3D or [nx, nz] for 2D
                if len(shape) == 3 and v.ndim == 2:
                    expected_traces = shape[0] * shape[1]
                    if v.shape[0] == expected_traces and v.shape[1] == shape[2]:
                        v = v.reshape(shape)
                        logger.info(f"Reshaped SEGY data to {shape}")
                        
            except Exception as e:
                raise ValueError(f"Failed to load SEGY file: {e}")
        else:
            raise ValueError(f"Unsupported file format: {path}")
            
        logger.info(f"Loaded velocity model from {path} with shape {v.shape}")
    else:
        logger.warning(f"Warning: Velocity file {path} not found. Using constant velocity model for testing.")
        v = np.ones(shape) * 2000.0  # Default 2000 m/s

    return np.asarray(v, dtype=np.float32)


def load_velocity_model(config):
    """
    Loads velocity model from file or creates a dummy one if file doesn't exist.
    """
    path = config['model']['file_path']
    shape = config['model']['shape']
    v = load_velocity_volume(path, shape, model_conf=config.get('model', {}))
    return torch.tensor(v, dtype=torch.float32)

def get_wavelet(config, device):
    """
    Generates source wavelet.
    """
    nt = config['time']['nt']
    dt = config['time']['dt']
    src_conf = config['source']
    
    if src_conf.get('type', 'ricker') == 'ricker':
        freq = float(src_conf.get('freq', 25.0))
        peak_time = 1.5 / freq
        wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    else:
        # Placeholder for other wavelets or external file loading
        logger.warning(f"Warning: Wavelet type {src_conf.get('type')} not implemented. Using Ricker.")
        freq = float(src_conf.get('freq', 25.0))
        peak_time = 1.5 / freq
        wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    amp = float(src_conf.get('amplitude', 1.0))
    return (amp * wavelet).to(device)

def resolve_y(y_value, ny: int) -> int:
    if y_value is None:
        return ny // 2
    if isinstance(y_value, str):
        if y_value.lower() == "center":
            return ny // 2
        # allow numeric strings
        if y_value.isdigit():
            return int(y_value)
        raise ValueError(f"Invalid y spec '{y_value}'. Use 'center' or an integer index.")
    return int(y_value)

def check_and_resample_model(v, config, freq):
    """
    Checks for numerical dispersion conditions and refines the grid if necessary.
    Updates config in-place with new grid spacing and acquisition geometry.
    Returns resampled velocity model (torch.Tensor) or original v.
    """
    vmin = v.min().item()
    if vmin <= 0:
        logger.warning(f"Warning: Minimum velocity is {vmin}, skipping dispersion check.")
        return v
    
    # Get parameters
    sim_conf = config.get('simulation', {})
    mode = sim_conf.get('mode', '2D')
    pml_threshold = float(sim_conf.get('pml_threshold', 6.0)) # Default usually 3-6
    
    model_conf = config['model']
    dx_val = float(model_conf.get('dx', 10.0))
    dz_val = float(model_conf.get('dz', 10.0))
    dy_val = float(model_conf.get('dy', 10.0)) if mode == "3D" else 1.0

    # Calculate points per wavelength
    # wavelength = v / f
    # cond = wavelength / d = v / (d * f)
    
    # 2D: [dx, dz] -> check
    # 3D: [dx, dy, dz] -> check
    
    # Determine scaling factors (powers of 2)
    def calc_scale(d_curr):
        cond = vmin / (d_curr * freq)
        if cond < pml_threshold:
            # We need d_new s.t. vmin/(d_new*freq) >= pml_threshold
            # d_new <= vmin / (pml_threshold * freq)
            t_d = vmin / (pml_threshold * freq)
            # Find closest power of 2 refinement: d_curr / 2**a <= t_d
            # 2**a >= d_curr / t_d
            # ratio = d_curr / t_d
            # if ratio > 1:
            #     a = math.ceil(math.log2(ratio))
            #     return int(2**a)
            return math.ceil(d_curr / t_d)
        return 1
        
    sx = calc_scale(dx_val)
    sz = calc_scale(dz_val)
    sy = calc_scale(dy_val) if mode == "3D" else 1
    
    if sx == 1 and sy == 1 and sz == 1:
        return v
        
    logger.info(f"Anti-dispersion check triggered (vmin={vmin:.1f}, freq={freq:.1f}, thresh={pml_threshold}).")
    logger.info(f"Resampling grid: scale_x={sx}, scale_y={sy}, scale_z={sz}")
    
    # 1. Resample Velocity Model
    # v shape: 2D [nx, nz] or 3D [nx, ny, nz]
    # Torch interpolate expects:
    # 3D (volumetric): input (N, C, D, H, W). deepwave 3D is (nx, ny, nz).
    # Let's map nx->D, ny->H, nz->W to match index order?
    # No, usually dim0 is D, dim1 is H, dim2 is W.
    
    v_in = v.unsqueeze(0).unsqueeze(0) # [1, 1, ...]
    
    if mode == "3D":
        # [nx, ny, nz] -> D=nx, H=ny, W=nz
        # scale_factor = (sx, sy, sz)
        # Note: interpolate scale_factor can be tuple.
        # But 'trilinear' or 'area' etc.
        # If sx, sy, sz are integers, 'nearest' or 'trilinear' works.
        # For velocity models, 'trilinear' (linear) is safer than nearest to avoid artifacts, 
        # but 'mode' kwarg for interpolate.
        v_new = torch.nn.functional.interpolate(
            v_in, 
            scale_factor=(sx, sy, sz), 
            mode='trilinear', 
            align_corners=False, 
            recompute_scale_factor=False
        )
    else:
        # [nx, nz] -> H=nx, W=nz (using 2D interpolation)
        # scale_factor = (sx, sz)
        v_new = torch.nn.functional.interpolate(
            v_in, 
            scale_factor=(sx, sz), 
            mode='bilinear', 
            align_corners=False, 
            recompute_scale_factor=False
        )
        
    v_out = v_new.squeeze(0).squeeze(0)
    
    # 2. Update Grid Spacing in Config
    model_conf['dx'] = dx_val / sx
    model_conf['dz'] = dz_val / sz
    if mode == "3D":
        model_conf['dy'] = dy_val / sy
    
    # Update shape
    if mode == "3D":
        model_conf['shape'] = list(v_out.shape)
    else:
        model_conf['shape'] = list(v_out.shape)
        
    logger.info(f"New grid spacing: dx={model_conf['dx']:.2f}, dy={model_conf.get('dy',0):.2f}, dz={model_conf['dz']:.2f}")
    logger.info(f"New model shape: {model_conf['shape']}")
        
    # 3. Update Acquisition Geometry Indices in Config
    # We need to scale all integer indices by the corresponding scale factor.
    # Note: Physical coordinates (if used in 'points') might need handling if stored as indices?
    # utils.setup_acquisition assumes parameters like 'first_source' are INDICES.
    # If layout='points' and users provided physical coordinates? 
    # Current setup_acquisition 'points' mode: "pts = torch.tensor(pts...)" 
    # If they are indices, we scale them. If they are floats (meters), we must check.
    # The doc/code assumes indices (dtype=torch.long in setup_acquisition).
    
    def scale_conf_key(section, key, factor):
        val = config.get(section, {}).get(key)
        if val is not None:
            config[section][key] = int(val * factor)
            
    # Source
    scale_conf_key('source', 'first_source', sx)      # Line
    scale_conf_key('source', 'd_source', sx)          # Line
    scale_conf_key('source', 'source_depth', sz)      # All
    
    scale_conf_key('source', 'first_source_x', sx)    # Grid
    scale_conf_key('source', 'd_source_x', sx)        # Grid
    scale_conf_key('source', 'first_source_y', sy)    # Grid (3D)
    scale_conf_key('source', 'd_source_y', sy)        # Grid (3D)
    
    # If source.locations (points) exists, scale it
    src_locs = config['source'].get('locations')
    if src_locs:
        # List of lists [[x,z], ...] or [[x,y,z], ...]
        new_locs = []
        for pt in src_locs:
            if len(pt) == 2: # x, z
                new_locs.append([int(pt[0]*sx), int(pt[1]*sz)])
            elif len(pt) == 3: # x, y, z
                new_locs.append([int(pt[0]*sx), int(pt[1]*sy), int(pt[2]*sz)])
        config['source']['locations'] = new_locs

    # Receiver
    scale_conf_key('receiver', 'first_receiver', sx)
    scale_conf_key('receiver', 'd_receiver', sx)
    scale_conf_key('receiver', 'receiver_depth', sz)
    
    scale_conf_key('receiver', 'first_receiver_x', sx)
    scale_conf_key('receiver', 'd_receiver_x', sx)
    scale_conf_key('receiver', 'first_receiver_y', sy)
    scale_conf_key('receiver', 'd_receiver_y', sy)
    
    # Rolling / Patch specific
    scale_conf_key('receiver', 'offset_first', sx) # offsets usually along line (X)
    scale_conf_key('receiver', 'd_offset', sx)
    scale_conf_key('receiver', 'receiver_y_offset', sy)
    
    scale_conf_key('receiver', 'patch_half_span', sx) # usually X
    scale_conf_key('receiver', 'patch_d', sx)
    
    # Also default_y in acquisition if numeric
    def_y = config.get('acquisition', {}).get('default_y')
    if isinstance(def_y, int) or (isinstance(def_y, str) and def_y.isdigit()):
        config['acquisition']['default_y'] = int(int(def_y) * sy)

    return v_out


def _oob_apply(locs: torch.Tensor, shape, policy: str) -> torch.Tensor:
    """
    shape: [nx, nz] for 2D, [nx, ny, nz] for 3D
    """
    policy = (policy or "error").lower()
    mins = locs.amin(dim=(0, 1))
    maxs = locs.amax(dim=(0, 1))

    ok = True
    for d, n in enumerate(shape):
        if mins[d].item() < 0 or maxs[d].item() >= n:
            ok = False
            break

    if ok:
        return locs

    if policy == "clamp":
        for d, n in enumerate(shape):
            locs[..., d] = locs[..., d].clamp(0, n - 1)
        return locs

    # error
    raise ValueError(
        f"Acquisition geometry out-of-bounds for shape={shape}. "
        f"mins={mins.tolist()}, maxs={maxs.tolist()}. "
        f"Set acquisition.oob_policy='clamp' to auto-clamp (not recommended for serious experiments)."
    )


def _get_shape_by_mode(config: dict):
    mode = config["simulation"]["mode"]
    shape = config["model"]["shape"]
    if mode == "2D":
        if len(shape) != 2:
            raise ValueError(f"2D mode expects model.shape length 2, got {shape}")
        nx, nz = int(shape[0]), int(shape[1])
        return (nx, nz)
    if mode == "3D":
        if len(shape) != 3:
            raise ValueError(f"3D mode expects model.shape length 3, got {shape}")
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        return (nx, ny, nz)
    raise ValueError(f"Unknown simulation.mode={mode}")


def setup_acquisition(config: dict, device: torch.device):
    """
    Backward compatible acquisition builder.

    Supported (unified):
      - Source layout: line (default/back-compat), grid, points
      - Receiver type: fixed (default/back-compat), rolling, patch
      - Receiver layout: line (default/back-compat), grid

    Returns:
      source_locations: [n_shots, n_src_per_shot, dim]
      receiver_locations: [n_shots, n_receivers_per_shot, dim]

    Side-effects:
      - If source.layout == grid or points, config['source']['n_shots'] is auto-filled (n_shots_x*n_shots_y or len(points))
      - If receiver.layout == grid (fixed/rolling/patch), config['receiver']['n_receivers_per_shot'] is auto-filled (flattened count)
      - For patch/rolling line, config['receiver']['n_receivers_per_shot'] is auto-filled if not provided
    """
    mode = config["simulation"]["mode"]
    dim = 2 if mode == "2D" else 3
    shape = _get_shape_by_mode(config)

    acq_conf = config.get("acquisition", {}) or {}
    oob_policy = acq_conf.get("oob_policy", "error")
    default_y = acq_conf.get("default_y", "center")

    s_conf = config.get("source", {}) or {}
    r_conf = config.get("receiver", {}) or {}

    # --------------
    # Build sources
    # --------------
    n_src_per_shot = int(s_conf.get("n_sources_per_shot", 1))
    s_layout = (s_conf.get("layout") or "line").lower()

    if s_layout == "line":
        n_shots = int(s_conf["n_shots"])
        first = int(s_conf["first_source"])
        ds = int(s_conf["d_source"])
        z_s = int(s_conf.get("source_depth", 2))

        sx = first + torch.arange(n_shots, device=device, dtype=torch.long) * ds

        if dim == 2:
            source_locations = torch.zeros((n_shots, n_src_per_shot, 2), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = z_s
        else:
            nx, ny, nz = shape
            y_s = resolve_y(s_conf.get("source_y", default_y), ny)
            source_locations = torch.zeros((n_shots, n_src_per_shot, 3), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = y_s
            source_locations[:, :, 2] = z_s

    elif s_layout == "grid":
        # 3D areal shots; in 2D we interpret as x-only grid with n_shots_x
        z_s = int(s_conf.get("source_depth", 2))

        nsx = int(s_conf.get("n_shots_x", s_conf.get("n_shots", 1)))
        dsx = int(s_conf.get("d_source_x", s_conf.get("d_source", 1)))
        fsx = int(s_conf.get("first_source_x", s_conf.get("first_source", 0)))
        sx = fsx + torch.arange(nsx, device=device, dtype=torch.long) * dsx

        if dim == 2:
            n_shots = nsx
            config["source"]["n_shots"] = int(n_shots)
            source_locations = torch.zeros((n_shots, n_src_per_shot, 2), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = z_s
        else:
            nx, ny, nz = shape
            nsy = int(s_conf.get("n_shots_y", 1))
            dsy = int(s_conf.get("d_source_y", dsx))
            fsy = int(s_conf.get("first_source_y", ny // 2))
            sy = fsy + torch.arange(nsy, device=device, dtype=torch.long) * dsy

            XS, YS = torch.meshgrid(sx, sy, indexing="ij")
            shot_xy = torch.stack([XS.reshape(-1), YS.reshape(-1)], dim=-1)  # [n_shots, 2]
            n_shots = int(shot_xy.shape[0])

            config["source"]["n_shots"] = int(n_shots)

            source_locations = torch.zeros((n_shots, n_src_per_shot, 3), device=device, dtype=torch.long)
            source_locations[:, :, 0] = shot_xy[:, 0].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = shot_xy[:, 1].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 2] = z_s

    elif s_layout == "grid_random":
        # Randomly select one source per block in a grid
        z_s = int(s_conf.get("source_depth", 2))
        
        if dim == 2:
            nx, nz = shape
            nsx = int(s_conf.get("n_shots_x", s_conf.get("n_shots", 1)))
            
            # Define block boundaries
            x_edges = torch.linspace(0, nx, steps=nsx+1).long()
            
            sx_list = []
            for i in range(nsx):
                xs = x_edges[i].item()
                xe = x_edges[i+1].item()
                if xe <= xs: xe = xs + 1 # protection
                
                rx = torch.randint(xs, xe, (1,)).item()
                sx_list.append(rx)
                
            sx = torch.tensor(sx_list, device=device, dtype=torch.long)
            n_shots = nsx
            config["source"]["n_shots"] = int(n_shots)
            
            source_locations = torch.zeros((n_shots, n_src_per_shot, 2), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = z_s
            
        else: # 3D
            nx, ny, nz = shape
            nsx = int(s_conf.get("n_shots_x", 1))
            nsy = int(s_conf.get("n_shots_y", 1))
            
            x_edges = torch.linspace(0, nx, steps=nsx+1).long()
            y_edges = torch.linspace(0, ny, steps=nsy+1).long()
            
            shot_xy_list = []
            for i in range(nsx):
                xs = x_edges[i].item()
                xe = x_edges[i+1].item()
                if xe <= xs: xe = xs + 1
                
                for j in range(nsy):
                    ys = y_edges[j].item()
                    ye = y_edges[j+1].item()
                    if ye <= ys: ye = ys + 1
                    
                    rx = torch.randint(xs, xe, (1,)).item()
                    ry = torch.randint(ys, ye, (1,)).item()
                    shot_xy_list.append([rx, ry])
            
            shot_xy = torch.tensor(shot_xy_list, device=device, dtype=torch.long)
            n_shots = shot_xy.shape[0]
            config["source"]["n_shots"] = int(n_shots)
            
            source_locations = torch.zeros((n_shots, n_src_per_shot, 3), device=device, dtype=torch.long)
            source_locations[:, :, 0] = shot_xy[:, 0].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = shot_xy[:, 1].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 2] = z_s

    elif s_layout == "points":
        pts = s_conf.get("locations", None)
        if pts is None:
            raise ValueError("source.layout='points' requires source.locations list.")
        pts = torch.tensor(pts, device=device, dtype=torch.long)
        if dim == 2:
            # accept [x,z] or [x,y,z] (drop y)
            if pts.shape[1] == 3:
                pts2 = torch.stack([pts[:, 0], pts[:, 2]], dim=-1)
            elif pts.shape[1] == 2:
                pts2 = pts
            else:
                raise ValueError("2D source points must be [x,z] (2 cols) or [x,y,z] (3 cols).")
            n_shots = int(pts2.shape[0])
            config["source"]["n_shots"] = int(n_shots)
            source_locations = pts2.view(n_shots, 1, 2).repeat(1, n_src_per_shot, 1)
        else:
            nx, ny, nz = shape
            if pts.shape[1] == 2:
                # [x,z] + default y
                y_s = resolve_y(s_conf.get("source_y", default_y), ny)
                pts3 = torch.stack([pts[:, 0], torch.full_like(pts[:, 0], y_s), pts[:, 1]], dim=-1)
            elif pts.shape[1] == 3:
                pts3 = pts
            else:
                raise ValueError("3D source points must be [x,y,z] (3 cols) or [x,z] (2 cols).")
            n_shots = int(pts3.shape[0])
            config["source"]["n_shots"] = int(n_shots)
            source_locations = pts3.view(n_shots, 1, 3).repeat(1, n_src_per_shot, 1)

    else:
        raise ValueError(f"Unknown source.layout={s_layout}")

    # Ensure in-bounds
    source_locations = _oob_apply(source_locations, shape, oob_policy)

    # ----------------
    # Build receivers
    # ----------------
    r_type = (r_conf.get("type") or "fixed").lower()
    r_layout = (r_conf.get("layout") or "line").lower()

    z_r = int(r_conf.get("receiver_depth", s_conf.get("source_depth", 2)))

    # Convenience: shot origins (x,y)
    shot_x = source_locations[:, 0, 0]  # [n_shots]
    shot_y = None
    if dim == 3:
        shot_y = source_locations[:, 0, 1]

    if r_type == "fixed":
        if r_layout == "line":
            n_rec = int(r_conf["n_receivers_per_shot"])
            first = int(r_conf["first_receiver"])
            dr = int(r_conf["d_receiver"])
            rx = first + torch.arange(n_rec, device=device, dtype=torch.long) * dr

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = z_r
            else:
                nx, ny, nz = shape
                y_r = resolve_y(r_conf.get("receiver_y", default_y), ny)
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = y_r
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            # fixed receiver grid, flattened
            if dim == 2:
                # interpret as line along x
                n_rec = int(r_conf.get("n_receivers_x", r_conf.get("n_receivers_per_shot")))
                first = int(r_conf.get("first_receiver_x", r_conf.get("first_receiver", 0)))
                dr = int(r_conf.get("d_receiver_x", r_conf.get("d_receiver", 1)))
                rx = first + torch.arange(n_rec, device=device, dtype=torch.long) * dr
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = z_r
            else:
                nrx = int(r_conf["n_receivers_x"])
                nry = int(r_conf["n_receivers_y"])
                frx = int(r_conf["first_receiver_x"])
                fry = int(r_conf["first_receiver_y"])
                drx = int(r_conf["d_receiver_x"])
                dry = int(r_conf["d_receiver_y"])

                rx = frx + torch.arange(nrx, device=device, dtype=torch.long) * drx
                ry = fry + torch.arange(nry, device=device, dtype=torch.long) * dry
                RX, RY = torch.meshgrid(rx, ry, indexing="ij")
                rec_xy = torch.stack([RX.reshape(-1), RY.reshape(-1)], dim=-1)
                n_rec = int(rec_xy.shape[0])

                config["receiver"]["n_receivers_per_shot"] = int(n_rec)

                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rec_xy[:, 0].unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = rec_xy[:, 1].unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 2] = z_r
        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=fixed")

    elif r_type == "rolling":
        # Receivers move with source
        if r_layout == "line":
            n_rec = int(r_conf["n_receivers_per_shot"])
            offset_first = int(r_conf.get("offset_first", r_conf.get("first_receiver", 0)))
            d_offset = int(r_conf.get("d_offset", r_conf.get("d_receiver", 1)))
            offx = offset_first + torch.arange(n_rec, device=device, dtype=torch.long) * d_offset  # [n_rec]

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + offx.unsqueeze(0)
                receiver_locations[:, :, 1] = z_r
            else:
                y_off = int(r_conf.get("receiver_y_offset", 0))
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + offx.unsqueeze(0)
                receiver_locations[:, :, 1] = (shot_y + y_off).unsqueeze(1).repeat(1, n_rec)
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            if dim != 3:
                raise ValueError("receiver.type='rolling' with layout='grid' is only meaningful in 3D.")
            nrx = int(r_conf["n_receivers_x"])
            nry = int(r_conf["n_receivers_y"])
            offx0 = int(r_conf.get("offset_first_x", r_conf.get("first_receiver_x", 0)))
            offy0 = int(r_conf.get("offset_first_y", r_conf.get("first_receiver_y", 0)))
            doffx = int(r_conf.get("d_offset_x", r_conf.get("d_receiver_x", 1)))
            doffy = int(r_conf.get("d_offset_y", r_conf.get("d_receiver_y", 1)))

            offx = offx0 + torch.arange(nrx, device=device, dtype=torch.long) * doffx
            offy = offy0 + torch.arange(nry, device=device, dtype=torch.long) * doffy
            OFX, OFY = torch.meshgrid(offx, offy, indexing="ij")
            rel = torch.stack([OFX.reshape(-1), OFY.reshape(-1)], dim=-1)
            n_rec = int(rel.shape[0])
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
            receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + rel[:, 0].unsqueeze(0)
            receiver_locations[:, :, 1] = shot_y.unsqueeze(1) + rel[:, 1].unsqueeze(0)
            receiver_locations[:, :, 2] = z_r
        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=rolling")

    elif r_type == "patch":
        # Local receiver window around each shot.
        min_offset = int(r_conf.get("min_offset", 0))

        if r_layout == "line":
            half = int(r_conf.get("patch_half_span", 0))
            if half <= 0:
                raise ValueError("receiver.type='patch' layout='line' requires receiver.patch_half_span > 0")
            step = int(r_conf.get("patch_d", r_conf.get("d_receiver", 1)))
            relx = torch.arange(-half, half + 1, step, device=device, dtype=torch.long)

            if min_offset > 0:
                relx = relx[relx.abs() >= min_offset]

            n_rec = int(relx.numel())
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + relx.unsqueeze(0)
                receiver_locations[:, :, 1] = z_r
            else:
                y_off = int(r_conf.get("receiver_y_offset", 0))
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + relx.unsqueeze(0)
                receiver_locations[:, :, 1] = (shot_y + y_off).unsqueeze(1).repeat(1, n_rec)
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            if dim != 3:
                raise ValueError("receiver.type='patch' with layout='grid' is only supported in 3D.")
            halfx = int(r_conf.get("patch_half_span_x", r_conf.get("patch_half_span", 0)))
            halfy = int(r_conf.get("patch_half_span_y", r_conf.get("patch_half_span", 0)))
            if halfx <= 0 or halfy <= 0:
                raise ValueError("patch grid requires patch_half_span_x > 0 and patch_half_span_y > 0")
            stepx = int(r_conf.get("patch_d_x", r_conf.get("patch_d", 1)))
            stepy = int(r_conf.get("patch_d_y", r_conf.get("patch_d", 1)))

            relx = torch.arange(-halfx, halfx + 1, stepx, device=device, dtype=torch.long)
            rely = torch.arange(-halfy, halfy + 1, stepy, device=device, dtype=torch.long)
            RX, RY = torch.meshgrid(relx, rely, indexing="ij")

            rel = torch.stack([RX.reshape(-1), RY.reshape(-1)], dim=-1)  # [N,2]

            # Optional circle mask in metric units
            mask = torch.ones((rel.shape[0],), device=device, dtype=torch.bool)
            patch_mask = (r_conf.get("patch_mask") or "box").lower()
            if patch_mask == "circle":
                dx = float(config["model"].get("dx", 1.0))
                dy = float(config["model"].get("dy", 1.0))
                radius_m = float(r_conf.get("patch_radius_m", 0.0))
                if radius_m <= 0:
                    raise ValueError("patch_mask='circle' requires receiver.patch_radius_m > 0")
                dist2 = (rel[:, 0].float() * dx) ** 2 + (rel[:, 1].float() * dy) ** 2
                mask = mask & (dist2 <= radius_m ** 2)

            if min_offset > 0:
                # min_offset in grid points (radial)
                r = torch.sqrt(rel[:, 0].float() ** 2 + rel[:, 1].float() ** 2)
                mask = mask & (r >= float(min_offset))

            rel = rel[mask]
            n_rec = int(rel.shape[0])
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
            receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + rel[:, 0].unsqueeze(0)
            receiver_locations[:, :, 1] = shot_y.unsqueeze(1) + rel[:, 1].unsqueeze(0)
            receiver_locations[:, :, 2] = z_r

        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=patch")

    else:
        raise ValueError(f"Unknown receiver.type={r_type}")

    receiver_locations = _oob_apply(receiver_locations, shape, oob_policy)
    return source_locations, receiver_locations

def calculate_auto_time_duration(v, dx, dt):
    """
    Automatically calculates simulation duration based on model diagonal and average velocity.
    Rule: t = 2 * Length_diagonal / v_mean
    
    Args:
        v (Tensor): Velocity model
        dx (list): Grid spacing [dx, dy, dz] or [dx, dz]
        dt (float): Time step interval
        
    Returns:
        tuple: (t_auto, nt_auto)
    """
    v_bar = float(v.mean().cpu().item())
    
    # Calculate physical dimensions
    # v.shape matches dx order
    dimensions = [size * spacing for size, spacing in zip(v.shape, dx)]
    L_diag = np.sqrt(sum(d**2 for d in dimensions))
    
    t_auto = 2.0 * L_diag / v_bar
    nt_auto = int(math.ceil(t_auto / dt))
    
    logger.info(f"Auto-calculated duration (2*L/v_bar): L={L_diag:.1f}m, v_bar={v_bar:.1f}m/s -> t={t_auto:.3f}s")
    
    return t_auto, nt_auto
