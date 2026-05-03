import torch
import math
from typing import Dict, List, Tuple, Optional, Any


def _next_pow2(n: int) -> int:
    """Return the next power-of-two >= n (for FFT padding)."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())

def _unit_scale(output_unit: str) -> float:
    """
    Convert SI (Tesla) to desired unit.
    - "si": 1.0 (Tesla)
    - "nt": 1e9 (NanoTesla)
    """
    u = (output_unit or "nt").lower()
    if u == "si":
        return 1.0
    if u in ["nt", "nanotesla"]:
        return 1e9
    raise ValueError(f"Unknown output_unit: {output_unit}")

def _make_obs_indices(nx: int, ny: int, obs_conf: Dict) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Build observation indices in x/y (integer grid indices).
    Supports:
    - layout: "grid" with optional {n_x, n_y, first_x, first_y, d_x, d_y}
    - layout: "points" with {points: [[ix,iy], ...]}
    Returns (x_idx, y_idx, layout)
    """
    layout = (obs_conf.get("layout", "grid") or "grid").lower()
    if layout == "grid":
        n_x = int(obs_conf.get("n_x", nx))
        n_y = int(obs_conf.get("n_y", ny))
        first_x = int(obs_conf.get("first_x", 0))
        first_y = int(obs_conf.get("first_y", 0))
        d_x = int(obs_conf.get("d_x", 1))
        d_y = int(obs_conf.get("d_y", 1))

        # Create ranges
        # Note: In the gravity code, this produces a grid of indices to select from the full field.
        # If we compute the full field at (nx, ny), we select specific points.
        
        # Ranges for indexing
        x_idx = torch.arange(first_x, first_x + n_x * d_x, d_x, dtype=torch.long)
        y_idx = torch.arange(first_y, first_y + n_y * d_y, d_y, dtype=torch.long)
        
        # Clamp to domain in case specifications go outside
        x_idx = x_idx.clamp(0, nx - 1)
        y_idx = y_idx.clamp(0, ny - 1)
        
        return x_idx, y_idx, "grid"

    if layout == "points":
        pts = obs_conf.get("points", [])
        if not pts:
            raise ValueError("observation.layout=points requires observation.points=[[ix,iy],...]")
        pts_t = torch.tensor(pts, dtype=torch.long)
        x_idx = pts_t[:, 0].clamp(0, nx - 1)
        y_idx = pts_t[:, 1].clamp(0, ny - 1)
        return x_idx, y_idx, "points"

    raise ValueError(f"Unknown observation.layout: {layout}")


# -----------------------------
# Prism-matched kernel (for unit-testing against the provided MATLAB prism code)
# -----------------------------

_PRISM_T_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def _build_prism_T_kernel(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    h: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the T kernel identical to MATLAB `fun_forward_mag.m`.

    Returns a tensor T with shape (2*nx-1, 2*ny-1, nz).

    This is used only in mode="prism_matched" to reproduce the MATLAB reference operator.
    """
    key = (
        "T_prism_v1",
        nx, ny, nz,
        float(dx), float(dy), float(dz), float(h),
        str(device), str(dtype),
    )
    if key in _PRISM_T_CACHE:
        return _PRISM_T_CACHE[key]

    # MATLAB definitions:
    # x0=-(nx-1)*dx:dx:(nx-1)*dx;
    # y0=-(ny-1)*dy:dy:(ny-1)*dy;
    # z0=-(nz*dz+h):dz:-(dz+h);
    x0 = torch.arange(-(nx - 1) * dx, (nx - 1) * dx + 0.5 * dx, dx, device=device, dtype=dtype)
    y0 = torch.arange(-(ny - 1) * dy, (ny - 1) * dy + 0.5 * dy, dy, device=device, dtype=dtype)
    z0 = torch.arange(-(nz * dz + h), -(dz + h) + 0.5 * dz, dz, device=device, dtype=dtype)

    Ai = torch.tensor([-dx / 2.0, dx / 2.0], device=device, dtype=dtype)
    Bj = torch.tensor([-dy / 2.0, dy / 2.0], device=device, dtype=dtype)
    Ck = torch.tensor([0.0, dz], device=device, dtype=dtype)

    X0, Y0 = torch.meshgrid(x0, y0, indexing="ij")
    X0 = X0.unsqueeze(-1)  # (2nx-1, 2ny-1, 1)
    Y0 = Y0.unsqueeze(-1)

    T = torch.zeros((2 * nx - 1, 2 * ny - 1, nz), device=device, dtype=dtype)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                # MATLAB: uijk = (-1)^(i+j+k) with 1-based indices
                uijk = (-1.0) ** ((i + 1) + (j + 1) + (k + 1))

                X = X0 + Ai[i]
                Y = Y0 + Bj[j]
                Z = (z0 + Ck[k]).view(1, 1, nz)

                R = torch.sqrt(X * X + Y * Y + Z * Z)
                denom = Z * R
                denom = torch.where(torch.abs(denom) < 1e-30, torch.full_like(denom, 1e-30), denom)
                ratio = (X * Y) / denom
                term = torch.atan(ratio)
                term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                T = T + (-uijk) * term

    _PRISM_T_CACHE[key] = T
    return T


def _prism_matched_forward_fullgrid(
    J: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    height_m: float,
) -> torch.Tensor:
    """Prism-matched forward (full grid), reproducing MATLAB's Ta*M (before unit scaling).

    J: magnetization intensity (A/m), shape (nx,ny,nz).
    Returns: (nx,ny) tensor.
    """
    nx, ny, nz = J.shape
    device = J.device
    dtype = J.dtype

    T = _build_prism_T_kernel(nx, ny, nz, dx, dy, dz, height_m, device=device, dtype=dtype)
    # MATLAB mapping: q (deep->shallow) uses model layer (nz-1-q)
    J_rev = torch.flip(J, dims=(2,))

    kx = 2 * nx - 1
    ky = 2 * ny - 1
    nx_full = kx + nx - 1
    ny_full = ky + ny - 1
    nx_fft = _next_pow2(nx_full)
    ny_fft = _next_pow2(ny_full)

    # Batch FFT over z (keep z as batch dimension)
    T_hat = torch.fft.rfft2(T, s=(nx_fft, ny_fft), dim=(0, 1))
    J_hat = torch.fft.rfft2(J_rev, s=(nx_fft, ny_fft), dim=(0, 1))
    total_hat = torch.sum(T_hat * J_hat, dim=2)
    conv_full = torch.fft.irfft2(total_hat, s=(nx_fft, ny_fft))
    conv_lin = conv_full[:nx_full, :ny_full]

    start_x = nx - 1
    start_y = ny - 1
    out = conv_lin[start_x:start_x + nx, start_y:start_y + ny]
    return out

def calculate_h_kernel_fft(
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    I: float, A: float,
    I0: float, A0: float,
    height: float = 0.0,
    pad_factor: int = 2,
    device: torch.device = torch.device('cpu')
):
    """Legacy placeholder.

    Earlier versions considered explicitly building a spatial-domain prism kernel and FFT'ing it.
    The current code path uses a standard wavenumber-domain earth-filter operator directly in
    :func:`forward_mag_tmi`, so this helper is intentionally not implemented.
    """
    raise NotImplementedError(
        "calculate_h_kernel_fft() is intentionally not implemented. "
        "Use forward_mag_tmi(), which applies a standard earth-filter formulation directly in the "
        "wavenumber domain."
    )


def forward_mag_tmi(
    susceptibility: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    heights_m: List[float],
    obs_conf: Dict,
    input_type: str = "susceptibility",
    B0: float = 50000.0, # Inducing field magnitude in nT
    I_deg: float = 90.0, # Inclination of inducing field (degrees)
    A_deg: float = 0.0,  # Declination of inducing field (degrees)
    M_I_deg: Optional[float] = None, # Inclination of magnetization (if different, e.g. remanent)
    M_A_deg: Optional[float] = None, # Declination of magnetization
    output_unit: str = "nt",
    pad_factor: int = 2,
    mode: str = "standard_B",
) -> Tuple[torch.Tensor, Dict]:
    """Magnetic forward modeling.

    Supported modes
    ---------------
    mode="standard_B":
        Standard spectral-domain (wavenumber-domain) formulation (route-B). This is the one you
        should use for formal usage (papers / inversion).

    mode="prism_matched":
        Prism-kernel matched *reference* operator that reproduces the provided MATLAB
        `fun_forward_mag.m`/`forward_mag.m` behavior (up to numerical precision). This mode is
        intended for unit tests / regression tests, not for scientific reporting.

    Parameters
    ----------
    susceptibility: (nx, ny, nz)
        - if input_type="susceptibility": dimensionless SI susceptibility χ
        - if input_type="magnetization": magnetization intensity J in A/m
    input_type:
        - "susceptibility" (default): treats the input as χ and uses induced magnetization
          J = χ · H0 with H0 = B0/μ0.
        - "magnetization": treats the input as J (A/m).
    output_unit:
        "nt" (nT) or "si" (Tesla).

    Returns
    -------
    data:
        - layout="grid": (1, n_heights, n_x, n_y)
        - layout="points": (1, n_heights, n_points)
    meta:
        dictionary with geometry and run settings.
    """
    
    if susceptibility.ndim != 3:
        raise ValueError(f"susceptibility must be 3D, got {susceptibility.shape}")
        
    device = susceptibility.device
    nx, ny, nz = susceptibility.shape
    
    if M_I_deg is None: M_I_deg = I_deg
    if M_A_deg is None: M_A_deg = A_deg
    
    # Convert angles to radians
    d2r = math.pi / 180.0
    i0 = I_deg * d2r
    a0 = A_deg * d2r
    mi = M_I_deg * d2r
    ma = M_A_deg * d2r
    
    # Direction cosines of inducing field (F) and magnetization (M)
    # Coordinate system: X (North), Y (East), Z (Down)
    # Standard geophysical convention:
    # m_x = cos(I) cos(A)
    # m_y = cos(I) sin(A)
    # m_z = sin(I)
    
    # Inducing field direction hat{F}
    fx = math.cos(i0) * math.cos(a0)
    fy = math.cos(i0) * math.sin(a0)
    fz = math.sin(i0)
    
    # Magnetization direction hat{M}
    mx = math.cos(mi) * math.cos(ma)
    my = math.cos(mi) * math.sin(ma)
    mz = math.sin(mi)
    
    # ---- Physical constants (SI) ----
    mu0 = 4.0 * math.pi * 1e-7  # N/A^2 = T·m/A
    # Convert inducing field magnitude from nT to Tesla
    B0_T = float(B0) * 1e-9
    
    scale = _unit_scale(output_unit)
    
    # Observation indices
    x_idx, y_idx, layout = _make_obs_indices(nx, ny, obs_conf)
    x_idx = x_idx.to(device)
    y_idx = y_idx.to(device)

    heights = [float(h) for h in (heights_m or [0.0])]

    mode_l = str(mode).strip().lower()

    # -----------------------------
    # mode = prism_matched (reference operator matching MATLAB prism code)
    # -----------------------------
    if mode_l in {"prism_matched", "prism", "matlab"}:
        input_type_l = str(input_type).strip().lower()
        sus_t = susceptibility.to(torch.float32)

        if input_type_l in {"susceptibility", "chi", "kappa"}:
            # Convert susceptibility -> induced magnetization intensity J (A/m)
            H0 = B0_T / mu0
            J = sus_t * float(H0)
        elif input_type_l in {"magnetization", "j", "m"}:
            J = sus_t
        else:
            raise ValueError(f"Unknown input_type={input_type!r}. Use 'susceptibility' or 'magnetization'.")

        # MATLAB uses G_T = 1e2 * T, which equals (mu0/4pi)=1e-7 Tesla converted to nT.
        u_out = (output_unit or "nt").lower()
        if u_out in {"nt", "nanotesla"}:
            prism_scale = 100.0
        elif u_out == "si":
            prism_scale = 1e-7
        else:
            raise ValueError(f"Unknown output_unit: {output_unit}")

        if layout == "grid":
            out = torch.zeros((1, len(heights), x_idx.numel(), y_idx.numel()), device=device, dtype=torch.float32)
        else:
            out = torch.zeros((1, len(heights), x_idx.numel()), device=device, dtype=torch.float32)

        for hi, h in enumerate(heights):
            full = _prism_matched_forward_fullgrid(J, dx=dx, dy=dy, dz=dz, height_m=h) * prism_scale
            if layout == "grid":
                out[0, hi] = full.index_select(0, x_idx).index_select(1, y_idx)
            else:
                out[0, hi] = full[x_idx, y_idx]

        meta = {
            "mode": "prism_matched",
            "layout": layout,
            "obs_x_idx": x_idx.detach().cpu(),
            "obs_y_idx": y_idx.detach().cpu(),
            "heights_m": torch.tensor(heights, dtype=torch.float32),
            "dx": torch.tensor([dx, dy, dz], dtype=torch.float32),
            "mu0": mu0,
            "B0": B0,
            "I": I_deg,
            "A": A_deg,
            "input_type": input_type_l,
            "output_unit": output_unit,
            "pad_factor": None,
            "model_shape": (nx, ny, nz),
        }
        return out, meta

    # Padded dimensions for FFT
    nx_pad = int(pad_factor) * nx
    ny_pad = int(pad_factor) * ny
    
    # Frequency coordinates
    # u corresponds to x (North), v corresponds to y (East)
    # fftfreq returns cycles/unit_sample. multiply by 2pi/L?
    # PyTorch FFT:
    # rfft2 returns frequencies [0, 1, ..., N/2]
    
    u = torch.fft.fftfreq(nx_pad, d=dx, device=device) * 2 * math.pi # Standard fftfreq is 1/T, so * 2pi for angular
    v = torch.fft.rfftfreq(ny_pad, d=dy, device=device) * 2 * math.pi
    
    # Meshgrid of wavenumbers
    # dim=0 is x (rows), dim=1 is y (columns)
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    # Wavenumber K = sqrt(u^2 + v^2)
    K_sq = U**2 + V**2
    K_abs = torch.sqrt(K_sq)
    
    # Handle singularity at K=0 (DC component)
    # For TMI, DC component is 0 if no net monopole ?
    # Usually we set K=0 to a small value or handle explicitly. 
    # But let's look at the multiplier.
    K_abs[0, 0] = 1.0 # Avoid div by zero, we will zero out the term later if needed



    # ---- Earth filter (wavenumber-domain operator for TMI) ----
    # Standard (route-B) spectral-domain expression (Blakely-style) for total-field anomaly:
    #
    #   Theta_m = i*(m_x*kx + m_y*ky) + m_z*k
    #   Theta_f = i*(f_x*kx + f_y*ky) + f_z*k
    #   Filter  = (Theta_m * Theta_f) / k^2
    #
    # and each slab contributes (e^{-k(z_top+h)} - e^{-k(z_bot+h)}).
    #
    # Notes:
    # - We use angular wavenumbers kx,ky in rad/m (u,v computed via 2π*fftfreq).
    # - DC (k=0) is set to zero.
    K_sq[0, 0] = 1.0  # avoid division by zero; we will force DC to 0
    term_m = 1j * (mx * U + my * V) + mz * K_abs
    term_f = 1j * (fx * U + fy * V) + fz * K_abs
    earth_filter = (term_m * term_f) / K_sq
    earth_filter[0, 0] = 0.0
    
    # Precompute RFFT of layers?
    # We have a 3D volume of susceptibility.
    # Usually we treat each layer nz independently.
    # chi is (nx, ny, nz)
    
    # Move to frequency domain
    # chi_pad: (nx_pad, ny_pad, nz)
    sus_t = susceptibility.to(torch.float32)
    # Batch FFT over the last dimension (nz)? No, RFFT2 is over dim 0,1.
    # We can permute to put z in batch dim or iteration.
    # Let's iterate z to save memory or batch if memory allows. 
    # (nx=100, ny=100, nz=100) -> 1M floats. Tiny.
    
    chi_hat = torch.fft.rfft2(sus_t, s=(nx_pad, ny_pad), dim=(0, 1))  # (nx_pad, ny_pad//2 + 1, nz)

    input_type_l = str(input_type).strip().lower()
    if input_type_l in {"susceptibility", "chi", "kappa"}:
        # Overall scale (Tesla) for susceptibility input
        X_term = (0.5 * B0_T) * earth_filter
    elif input_type_l in {"magnetization", "j", "m"}:
        # Overall scale (Tesla) for magnetization intensity input J (A/m)
        X_term = (0.5 * mu0) * earth_filter
    else:
        raise ValueError(f"Unknown input_type={input_type!r}. Use 'susceptibility' or 'magnetization'.")
    
    # Depth term
    # We are integrating over prisms of thickness dz.
    # Integration of e^{-kz} from z1 to z2 is (e^{-kz1} - e^{-kz2})/k
    # z1 = depth_top, z2 = depth_bottom.
    # depth is positive downwards.
    
    # Depth boundaries for each voxel layer (z positive downward from surface)
    z_top = torch.arange(nz, device=device, dtype=torch.float32) * float(dz)
    z_bot = (torch.arange(nz, device=device, dtype=torch.float32) + 1.0) * float(dz)
    
    if layout == "grid":
        out = torch.zeros((1, len(heights), x_idx.numel(), y_idx.numel()), device=device, dtype=torch.float32)
    else:
        out = torch.zeros((1, len(heights), x_idx.numel()), device=device, dtype=torch.float32)
    for hi, h in enumerate(heights):
        # Observation height h (meters above surface). 
        # Surface is z=0. Obs z = -h.
        # Distance to source at z (positive) is z - (-h) = z + h.
        
        # Vectorized depth factor for voxel slabs.
        # Each voxel layer contributes a factor (exp(-K(z_top+h)) - exp(-K(z_bot+h))).
        K_ext = K_abs.unsqueeze(-1)  # (nx_pad, ny_pad_half, 1)
        z_tops = z_top + float(h)
        z_bots = z_bot + float(h)
        
        # e_{-k z} terms
        # (nx_pad, ny_pad_half, nz)
        # We need to handle k=0 case separately.
        # limit (e^{-k z1} - e^{-k z2}) as k->0 is -(z1 - z2) * k? No.
        # It is just thickness? No. DC component of mag field.
        # For k=0, TMI is usually 0 unless we have monopole term.
        
        # exp_factors: (nx_pad, ny_pad_half, nz)
        exp_top = torch.exp(-K_ext * z_tops)
        exp_bot = torch.exp(-K_ext * z_bots)
        
        layer_factors = exp_top - exp_bot
        
        # Apply mask for K=0 to avoid issues if any, though exp(0)-exp(0)=0. Good.
        
        # Sum over layers: sum(chi_hat * layer_factor, dim=2)
        # accumulated: (nx_pad, ny_pad_half)
        acc_hat = torch.sum(chi_hat * layer_factors, dim=2)
        
        # Multiply by common factors
        total_hat = acc_hat * X_term
        
        # Inverse FFT
        tmi_full = torch.fft.irfft2(total_hat, s=(nx_pad, ny_pad))
        
        # Crop
        tmi_full = tmi_full[:nx, :ny] * scale
        
        if layout == "grid":
            # sample by indices
            val = tmi_full.index_select(0, x_idx).index_select(1, y_idx)  # (n_x, n_y)
            out[0, hi] = val
        else:
            val = tmi_full[x_idx, y_idx]  # (n_points,)
            out[0, hi] = val

    meta = {
        "mode": "standard_B",
        "layout": layout,
        "obs_x_idx": x_idx.detach().cpu(),
        "obs_y_idx": y_idx.detach().cpu(),
        "heights_m": torch.tensor(heights, dtype=torch.float32),
        "dx": torch.tensor([dx, dy, dz], dtype=torch.float32),
        "mu0": mu0,
        "B0": B0,
        "I": I_deg,
        "A": A_deg,
        "input_type": input_type_l,
        "output_unit": output_unit,
        "pad_factor": int(pad_factor),
        "model_shape": (nx, ny, nz),
    }

    return out, meta

if __name__ == "__main__":
    # Test code
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 100., 100., 100.
    
    # Create model: a prism in the middle
    # Corresponding to MATLAB: M(9:12,9:12,3:6)=1; (1-based indices)
    # Python 0-based: 8:12 (since 12 is exclusive), etc.
    model = torch.zeros((nx, ny, nz))
    model[8:12, 8:12, 2:6] = 0.05  # example SI susceptibility (dimensionless)
    
    # Example run with a typical inducing field.
    
    obs_conf = {"layout": "grid", "n_x": nx, "n_y": ny}
    
    tmi, meta = forward_mag_tmi(
        model, dx, dy, dz, 
        heights_m=[0], 
        obs_conf=obs_conf, 
        B0=50000, 
        I_deg=90, 
        A_deg=0
    )
    
    print("Output shape:", tmi.shape)
    print("Max anomaly:", tmi.max().item(), "nT")
    print("Min anomaly:", tmi.min().item(), "nT")
