from typing import Dict, List, Tuple

import torch


def _unit_scale(output_unit: str) -> float:
    """
    Convert SI (m/s^2) to desired unit.
    - "si": 1
    - "mgal": 1e5  (1 mGal = 1e-5 m/s^2)
    - "ugal": 1e8  (1 uGal = 1e-8 m/s^2)
    """
    u = (output_unit or "mgal").lower()
    if u == "si":
        return 1.0
    if u == "mgal":
        return 1e5
    if u == "ugal":
        return 1e8
    raise ValueError(f"Unknown output_unit: {output_unit}")


def _make_obs_indices(nx: int, ny: int, obs_conf: Dict) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Build observation indices in x/y (integer grid indices).
    Supports:
    - layout: "grid" with {n_x,n_y,first_x,first_y,d_x,d_y}
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

        x_idx = torch.arange(first_x, first_x + n_x * d_x, d_x, dtype=torch.long)
        y_idx = torch.arange(first_y, first_y + n_y * d_y, d_y, dtype=torch.long)
        # clamp to domain
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


def _normalize_algorithm(algorithm: str) -> str:
    algo = (algorithm or "point_mass_fast").lower()
    aliases = {
        "fast": "point_mass_fast",
        "point_mass": "point_mass_fast",
        "pointmass": "point_mass_fast",
        "point_mass_fast": "point_mass_fast",
        "exact": "prism_exact",
        "prism": "prism_exact",
        "prism_exact": "prism_exact",
    }
    if algo not in aliases:
        raise ValueError(
            f"Unknown gravity algorithm: {algorithm}. "
            "Use 'point_mass_fast' or 'prism_exact'."
        )
    return aliases[algo]


def _build_fft_offsets(
    nx_pad: int,
    ny_pad: int,
    dx: float,
    dy: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.arange(nx_pad, device=device)
    iy = torch.arange(ny_pad, device=device)
    ix = torch.where(ix <= nx_pad // 2, ix, ix - nx_pad)
    iy = torch.where(iy <= ny_pad // 2, iy, iy - ny_pad)
    x = ix.to(dtype) * float(dx)
    y = iy.to(dtype) * float(dy)
    return torch.meshgrid(x, y, indexing="ij")


def _prism_gz_antiderivative(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    r = torch.sqrt(x * x + y * y + z * z + eps)
    tx = torch.clamp(r + x, min=eps)
    ty = torch.clamp(r + y, min=eps)
    return x * torch.log(ty) + y * torch.log(tx) - z * torch.atan2(x * y, z * r)


def _prism_gz_kernel(
    x_center: torch.Tensor,
    y_center: torch.Tensor,
    *,
    dx: float,
    dy: float,
    z_top: float,
    z_bottom: float,
) -> torch.Tensor:
    half_dx = 0.5 * float(dx)
    half_dy = 0.5 * float(dy)
    x1 = x_center - half_dx
    x2 = x_center + half_dx
    y1 = y_center - half_dy
    y2 = y_center + half_dy
    z1 = torch.full_like(x_center, float(z_top))
    z2 = torch.full_like(x_center, float(z_bottom))
    eps = torch.finfo(x_center.dtype).eps

    total = torch.zeros_like(x_center)
    for xv, sx in ((x1, -1.0), (x2, 1.0)):
        for yv, sy in ((y1, -1.0), (y2, 1.0)):
            for zv, sz in ((z1, -1.0), (z2, 1.0)):
                total = total + (sx * sy * sz) * _prism_gz_antiderivative(xv, yv, zv, eps=eps)
    # Convert the standard analytic expression to the project convention:
    # gz is positive downward.
    return -total


def forward_gravity_gz(
    density: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    heights_m: List[float],
    obs_conf: Dict,
    G: float = 6.67430e-11,
    output_unit: str = "mgal",
    pad_factor: int = 2,
    density_unit: str = "g/cm^3",
    algorithm: str = "prism_exact",
) -> Tuple[torch.Tensor, Dict]:
    """
    Gravity forward modeling (vertical component gz) for a 3D density model.

    density: (nx, ny, nz), density_unit (default g/cm^3)
    dx,dy,dz: meters
    heights_m: observation heights above surface (z=0), meters. Example: [0.0]
    obs_conf: observation config dict (grid or points) in index space.
    Returns:
      data: shape
        - grid:   (1, n_heights, n_x, n_y)
        - points: (1, n_heights, n_points)
      meta: dict with observation indices and parameters.
    Notes:
      - algorithm="point_mass_fast":
          *point-mass cell-center approximation*
          dm = rho * dx * dy * dz at each cell center
          gz = G * dm * (z) / r^3
          It is widely used for fast synthetic dataset generation.
      - algorithm="prism_exact":
          exact rectangular-prism gz kernel on a regular grid, evaluated via
          FFT-based horizontal convolution for each depth layer.
    """
    if density.ndim != 3:
        raise ValueError(f"density must be 3D (nx,ny,nz), got {tuple(density.shape)}")

    # The physical kernel uses kg/m^3 internally.
    du = (density_unit or "g/cm^3").lower().replace(" ", "")
    if du in ["g/cm3", "g/cm^3", "g/cc", "gcc"]:
        density = density * 1000.0
        input_density_unit = "g/cm^3"
    elif du in ["kg/m3", "kg/m^3", "kgm3"]:
        input_density_unit = "kg/m^3"
    else:
        raise ValueError(f"Unknown density_unit: {density_unit}. Use 'g/cm^3' or 'kg/m^3'.")

    device = density.device
    nx, ny, nz = density.shape
    heights = [float(h) for h in (heights_m or [0.0])]
    scale = _unit_scale(output_unit)
    algorithm = _normalize_algorithm(algorithm)

    # Observation indices
    x_idx, y_idx, layout = _make_obs_indices(nx, ny, obs_conf)
    # Ensure indices are on the same device as density (e.g. CUDA)
    x_idx = x_idx.to(device)
    y_idx = y_idx.to(device)

    meta = {
        "layout": layout,
        "obs_x_idx": x_idx.detach().cpu(),
        "obs_y_idx": y_idx.detach().cpu(),
        "heights_m": torch.tensor(heights, dtype=torch.float32),
        "dx": torch.tensor([dx, dy, dz], dtype=torch.float32),
        "G": float(G),
        "output_unit": output_unit,
        "density_unit": input_density_unit,
        "internal_density_unit": "kg/m^3",
        "pad_factor": int(pad_factor),
        "model_shape": (nx, ny, nz),
        "algorithm": algorithm,
    }

    # We compute full-field gz on (nx, ny) then sample.
    # Use linear convolution via FFT with padding to avoid wrap-around.
    nx_pad = int(pad_factor) * nx
    ny_pad = int(pad_factor) * ny

    compute_dtype = torch.float64 if algorithm == "prism_exact" else torch.float32
    # Precompute FFT of density slices in batch: (nx_pad, ny_pad//2+1, nz)
    rho = density.to(compute_dtype)
    rho_hat = torch.fft.rfft2(rho, s=(nx_pad, ny_pad), dim=(0, 1))

    # Spatial coordinates for kernel centered at (0,0) using "ifftshift" style grid.
    X, Y = _build_fft_offsets(
        nx_pad,
        ny_pad,
        dx,
        dy,
        device=device,
        dtype=compute_dtype,
    )
    R2_xy = X * X + Y * Y  # (nx_pad, ny_pad)

    # Depth centers (z positive downward), observation is above surface at -h,
    # so vertical distance is zc + h.
    z_centers = (torch.arange(nz, device=device, dtype=compute_dtype) + 0.5) * float(dz)  # (nz,)

    # Output container
    if layout == "grid":
        out = torch.zeros((1, len(heights), x_idx.numel(), y_idx.numel()), device=device, dtype=torch.float32)
    else:
        out = torch.zeros((1, len(heights), x_idx.numel()), device=device, dtype=torch.float32)

    # Compute per height
    cell_mass_scale = float(G) * float(dx) * float(dy) * float(dz)  # G * dV
    for hi, h in enumerate(heights):
        # Accumulate in frequency domain to reduce inverse FFT calls
        acc_hat = None
        for k in range(nz):
            if algorithm == "point_mass_fast":
                z0 = z_centers[k] + float(h)
                # Kernel K(Δx,Δy; z0) = z0 / (Δx^2+Δy^2+z0^2)^(3/2)
                denom = (R2_xy + z0 * z0).pow(1.5)
                K = z0 / denom
            else:
                z_top = float(k * dz + h)
                z_bottom = float((k + 1) * dz + h)
                K = _prism_gz_kernel(
                    X,
                    Y,
                    dx=dx,
                    dy=dy,
                    z_top=z_top,
                    z_bottom=z_bottom,
                )
            # FFT of kernel
            K_hat = torch.fft.rfft2(K, s=(nx_pad, ny_pad))
            term_hat = rho_hat[..., k] * K_hat
            acc_hat = term_hat if acc_hat is None else (acc_hat + term_hat)

        gz_full = torch.fft.irfft2(acc_hat, s=(nx_pad, ny_pad))
        # Crop to original size, centered at [0:nx, 0:ny]
        if algorithm == "point_mass_fast":
            gz_full = gz_full[:nx, :ny] * cell_mass_scale * scale
        else:
            gz_full = gz_full[:nx, :ny] * float(G) * scale

        if layout == "grid":
            # sample by indices
            gz_s = gz_full.index_select(0, x_idx).index_select(1, y_idx)  # (n_x, n_y)
            out[0, hi] = gz_s.to(out.dtype)
        else:
            gz_s = gz_full[x_idx, y_idx]  # (n_points,)
            out[0, hi] = gz_s.to(out.dtype)

    return out, meta
