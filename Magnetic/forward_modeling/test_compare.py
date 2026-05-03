import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

# Import the GPU function
from mag_forward import forward_mag_tmi

def mat_spatial_prism_magnetic(nx, ny, nz, dx, dy, dz, h, model_slice):
    """
    Python port of `fun_forward_mag.m` to generate ground truth data.
    Uses exact prism integration formula in spatial domain.
    """
    x0 = np.arange(-(nx-1)*dx, (nx-1)*dx + 0.1, dx)
    y0 = np.arange(-(ny-1)*dy, (ny-1)*dy + 0.1, dy)
    z0_vals = np.arange(-(nz*dz+h), -(dz+h) + 0.1, dz)
    
    # Kernel T
    T_exact = np.zeros((2*nx-1, 2*ny-1, nz))
    Ai = [-dx/2, dx/2]
    Bj = [-dy/2, dy/2]
    Ck = [0, dz]
    
    print("Computing Spatial Reference (this might take a few seconds)...")
    for q in range(nz):
        zk_base = z0_vals[q]
        for i_idx, val_i in enumerate(Ai):
            for j_idx, val_j in enumerate(Bj):
                for k_idx, val_k in enumerate(Ck):
                    uijk = (-1)**((i_idx+1) + (j_idx+1) + (k_idx+1))
                    
                    # Compute meshgrid for current shift
                    xi_mesh, yj_mesh = np.meshgrid(x0 + val_i, y0 + val_j, indexing='ij')
                    # Note: indexing='ij' means row=x, col=y. Match output shape (2nx-1, 2ny-1).
                    
                    zk = zk_base + val_k # This is z (negative)
                    rijk = np.sqrt(xi_mesh**2 + yj_mesh**2 + zk**2)
                    
                    # MATLAB: atan(xi*yj/(zk*rijk))
                    # Use np.arctan which matches MATLAB behavior for real numbers
                    with np.errstate(divide="ignore", invalid="ignore"):
                        term = np.arctan(xi_mesh * yj_mesh / (zk * rijk))
                        term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    T_exact[:, :, q] += -uijk * term

    D = np.zeros((nx, ny))
    
    # Convolve
    for q in range(nz):
        # MATLAB logic: T layer q corresponds to depth z0[q].
        # z0 array goes from deep to shallow (-1000 to -100).
        # q=0 is deep. q=nz-1 is shallow.
        
        # Model slicing: Standard is k=0 (top) to k=nz-1 (bottom).
        # We need to match the depth.
        # If z0[0] is -1000 (deepest), we should use model slice corresponding to bottom?
        # If model is created with model[... , 2:6] = 1, we assume index 0 is top.
        # So index (nz-1) is bottom.
        
        # We need to map q (deep->shallow) to model (shallow->deep).
        # q=0 (deepest) -> model index nz-1.
        # q=nz-1 (shallowest) -> model index 0.
        
        layer_m = model_slice[:, :, (nz - 1) - q]
        layer_k = T_exact[:, :, q] 
        
        # Valid convolution
        conv_res = convolve2d(layer_k, layer_m, mode='valid')
        D += conv_res
        
    # Apply global scaling from MATLAB "G_T = 10^2 * Ta"
    return D * 100.0

def _normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(arr)) + 1e-12
    return arr / denom


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def compare_results():
    # 1. Setup Model (Same as MATLAB sample)
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 100., 100., 100.
    
    model = np.zeros((nx, ny, nz))
    # Fill some blocks (near top)
    model[8:12, 8:12, 2:6] = 1.0 
    
    # DEBUG: Print non-zero model stats
    print(f"Model non-zero count: {np.count_nonzero(model)}")
    print(f"Model min/max indices: {np.where(model>0)}")

    obs_conf = {"layout": "grid", "n_x": nx, "n_y": ny}

    
    # 2. Run Spatial Reference (MATLAB Port)
    t0 = time.time()
    d_spatial = mat_spatial_prism_magnetic(nx, ny, nz, dx, dy, dz, h=0, model_slice=model)
    print(f"Spatial Calc Time: {time.time()-t0:.2f}s")

    # 3. Run FFT Forward
    # We compare against MATLAB-prism reference using magnetization mode (J in A/m).
    # (This is consistent with MATLAB sample where model entries are set to 1 and then scaled by 100.)

    t0 = time.time()
    tmi_prism_t, meta_prism = forward_mag_tmi(
        torch.from_numpy(model).float(), dx, dy, dz,
        heights_m=[0], obs_conf=obs_conf,
        input_type="magnetization",
        output_unit="nt",
        mode="prism_matched",
    )
    tmi_prism = tmi_prism_t[0, 0].cpu().numpy()
    t_prism = time.time() - t0

    t0 = time.time()
    tmi_std_t, meta_std = forward_mag_tmi(
        torch.from_numpy(model).float(), dx, dy, dz,
        heights_m=[0], obs_conf=obs_conf,
        input_type="magnetization",
        output_unit="nt",
        mode="standard_B",
        pad_factor=2,
    )
    tmi_std = tmi_std_t[0, 0].cpu().numpy()
    t_std = time.time() - t0

    print("\n--- Comparison Stats (vs MATLAB-prism spatial reference) ---")
    print(f"Spatial (MATLAB port)  max: {d_spatial.max():.6f} nT")

    spa_n = _normalize(d_spatial)

    def _report(name: str, arr: np.ndarray, elapsed: float):
        arr_n = _normalize(arr)
        mae_n = _mae(arr_n, spa_n)
        # Also report absolute MAE in nT for intuition
        mae_abs = _mae(arr, d_spatial)
        ratio = d_spatial.max() / (arr.max() + 1e-12)
        print(f"\n[{name}] time: {elapsed:.4f}s")
        print(f"{name} max: {arr.max():.6f} nT")
        print(f"Ratio (Spatial/{name}): {ratio:.6f}")
        print(f"MAE (normalized): {mae_n:.6f}")
        print(f"MAE (nT): {mae_abs:.6f}")

    _report('prism_matched', tmi_prism, t_prism)
    _report('standard_B',    tmi_std,   t_std)

    # 4. Visualizations (Scheme 1)
    # Plot 1: MATLAB prism vs prism_matched
    diff_prism = np.abs(_normalize(tmi_prism) - spa_n)
    plt.figure(figsize=(15, 5))
    plt.subplot(131); im1 = plt.imshow(tmi_prism, cmap='jet', origin='lower'); plt.title('prism_matched (FFT)'); plt.colorbar(im1)
    plt.subplot(132); im2 = plt.imshow(d_spatial, cmap='jet', origin='lower'); plt.title('MATLAB Port (spatial prism)'); plt.colorbar(im2)
    plt.subplot(133); im3 = plt.imshow(diff_prism, cmap='gray', origin='lower'); plt.title('Diff Norm'); plt.colorbar(im3)
    plt.tight_layout()
    plt.savefig('comparison_prism_matched.png', dpi=200)

    # Plot 2: MATLAB prism vs standard_B
    diff_std = np.abs(_normalize(tmi_std) - spa_n)
    plt.figure(figsize=(15, 5))
    plt.subplot(131); im1 = plt.imshow(tmi_std, cmap='jet', origin='lower'); plt.title('standard_B (FFT)'); plt.colorbar(im1)
    plt.subplot(132); im2 = plt.imshow(d_spatial, cmap='jet', origin='lower'); plt.title('MATLAB Port (spatial prism)'); plt.colorbar(im2)
    plt.subplot(133); im3 = plt.imshow(diff_std, cmap='gray', origin='lower'); plt.title('Diff Norm'); plt.colorbar(im3)
    plt.tight_layout()
    plt.savefig('comparison_standard_B.png', dpi=200)

    print("Plots saved: comparison_prism_matched.png, comparison_standard_B.png")

def compare_cpu_gpu_speed():
    """
    Compare execution speed between CPU and GPU for standard_B mode.
    Scales up the model size for a more meaningful benchmark.
    """
    print("\n--- Speed Benchmark: CPU vs GPU (standard_B) ---")
    
    # Use a larger model for benchmarking relevance (e.g., 256^3)
    # Or keep it moderate (e.g. 128x128x64) so it fits easily and runs quickly.
    nx, ny, nz = 128, 128, 64
    dx, dy, dz = 10.0, 10.0, 10.0
    
    print(f"Model size: {nx}x{ny}x{nz}")
    
    model_np = np.zeros((nx, ny, nz), dtype=np.float32)
    # Add a block in the center
    cx, cy, cz = nx//2, ny//2, nz//2
    model_np[cx-10:cx+10, cy-10:cy+10, cz-5:cz+5] = 0.05
    
    obs_conf = {"layout": "grid", "n_x": nx, "n_y": ny}
    
    # 1. CPU Run
    # Warmup
    try:
        t0 = time.time()
        _ = forward_mag_tmi(
            torch.from_numpy(model_np).float().to("cpu"), dx, dy, dz,
            heights_m=[0], obs_conf=obs_conf, input_type="susceptibility",
            mode="standard_B"
        )
        print(f"CPU Warmup finished.")
    except Exception as e:
        print(f"CPU Run failed: {e}")

    # Meas
    start_cpu = time.time()
    _, _ = forward_mag_tmi(
        torch.from_numpy(model_np).float().to("cpu"), dx, dy, dz,
        heights_m=[0], obs_conf=obs_conf, input_type="susceptibility",
        mode="standard_B"
    )
    t_cpu = time.time() - start_cpu
    print(f"CPU Time: {t_cpu:.4f} s")
    
    # 2. GPU Run (if available)
    if torch.cuda.is_available():
        # Warmup
        try:
            _ = forward_mag_tmi(
                torch.from_numpy(model_np).float().to("cuda"), dx, dy, dz,
                heights_m=[0], obs_conf=obs_conf, input_type="susceptibility",
                mode="standard_B"
            )
            torch.cuda.synchronize()
            print(f"GPU Warmup finished.")
            
            # Meas
            start_gpu = time.time()
            _, _ = forward_mag_tmi(
                torch.from_numpy(model_np).float().to("cuda"), dx, dy, dz,
                heights_m=[0], obs_conf=obs_conf, input_type="susceptibility",
                mode="standard_B"
            )
            torch.cuda.synchronize()
            t_gpu = time.time() - start_gpu
            print(f"GPU Time: {t_gpu:.4f} s")
            print(f"Speedup (CPU/GPU): {t_cpu / t_gpu:.2f}x")
            
        except Exception as e:
            print(f"GPU Run failed: {e}")
    else:
        print("CUDA not available, skipping GPU benchmark.")


if __name__ == "__main__":
    compare_results()
    compare_cpu_gpu_speed()
