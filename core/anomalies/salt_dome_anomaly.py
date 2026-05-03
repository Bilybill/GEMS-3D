#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SaltDomeAnomaly implementation refactored to match EllipsoidAnomaly architecture.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from .base import Anomaly

import numpy as np

# --------------------------
# Utility: small helpers
# --------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for float32/float64 arrays."""
    out = np.empty_like(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smoothstep 0..1 -> 0..1"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Catmull-Rom spline (centripetal-ish simplified) for 1D interpolation without SciPy.
    """
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    n = len(xk)
    if n < 4:
        return np.interp(x, xk, yk)

    idx = np.searchsorted(xk, x) - 1
    idx = np.clip(idx, 0, n - 2)

    i1 = idx
    i2 = idx + 1
    i0 = np.clip(i1 - 1, 0, n - 1)
    i3 = np.clip(i2 + 1, 0, n - 1)

    x1 = xk[i1]
    x2 = xk[i2]
    denom = np.maximum(x2 - x1, 1e-9)
    t = (x - x1) / denom
    t = np.clip(t, 0.0, 1.0)

    p0 = yk[i0]
    p1 = yk[i1]
    p2 = yk[i2]
    p3 = yk[i3]

    t2 = t * t
    t3 = t2 * t
    y = 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return y.astype(np.float32)

# --------------------------
# Salt dome parameter set
# --------------------------

@dataclass
class SaltDomeParams:
    # core
    vp_salt_mps: float
    edge_width_m: float

    # depths (meters, Z positive downward)
    z_top_m: float
    z_base_m: float

    # radii (meters)
    stem_radius_m: float
    mid_radius_m: float
    canopy_amp_m: float
    canopy_center_m: float
    canopy_sigma_m: float

    # centerline
    cx_base_m: float
    cy_base_m: float
    tilt_dx_m: float
    tilt_dy_m: float
    meander_amp_m: float
    meander_knots: int

    # roughness
    roughness_eps: float
    roughness_decay_frac: float 

    # azimuth modes
    kmin: int
    kmax: int
    azimuth_power: float

    # warp
    warp_enable: bool
    warp_amp_m: float
    warp_sigma_factor: float
    warp_z_peak_m: float
    warp_z_sigma_m: float
    
    # Grid dimensions used during parameter sampling
    # We might need these to clamp things properly
    Lx: float = 2560.0
    Ly: float = 2560.0
    Lz: float = 3200.0

    # Multi-physics core properties
    rho_salt_gcc: float = 2.15      # Salt density in g/cm^3
    resist_salt_ohmm: float = 3000.0 # Highly resistive
    chi_salt_si: float = -0.00001   # Diamagnetic (negative susceptibility)


# --------------------------
# Main anomaly class
# --------------------------

@dataclass
class SaltDomeAnomaly(Anomaly):
    """
    Refactored Salt Dome Anomaly inheriting from Anomaly.
    Calculates salt mask and applied warp to vp.
    """
    params: SaltDomeParams = None
    rng_seed: int = 20260207
    
    _rng: np.random.Generator = None
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        if self.params is None:
            # Should ideally be provided. If not, it will fail later unless we add default construction logic.
            pass

    @staticmethod
    def create_random_params(grid_shape: tuple, grid_spacing: tuple, seed: int = 20260207) -> SaltDomeParams:
        """Helper to create random params similar to original sample_params."""
        nx, ny, nz = grid_shape
        dx, dy, dz = grid_spacing
        Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
        
        rng = np.random.default_rng(seed)
        
        def U(a, b): return float(rng.uniform(a, b))
        
        # Hardcoded ranges for demo simplicity (based on yaml defaults)
        z_top_m = U(Lz*0.1, Lz*0.4)
        z_base_m = U(Lz*0.6, Lz*0.9)
        if z_base_m < z_top_m + 6*dz: z_base_m = z_top_m + 6*dz
        
        stem_radius_m = U(Lx*0.03, Lx*0.1)

        mid_radius_m = U(Lx*0.1, Lx*0.25)
        canopy_amp_m = U(Lx*0.04, Lx*0.3)
        
        canopy_center_m = z_top_m + U(0.15, 0.35) * (z_base_m - z_top_m)
        canopy_sigma_m = max(U(0.08, 0.20) * (z_base_m - z_top_m), 4*dz)
        
        # Center with margin
        rmax = max(stem_radius_m, mid_radius_m) + canopy_amp_m
        margin = rmax + 100.0
        cx_base_m = U(margin, max(margin+1, Lx-margin))
        cy_base_m = U(margin, max(margin+1, Ly-margin))
        
        # Warp params
        warp_z_peak_m = canopy_center_m + U(-0.05, 0.1) * (z_base_m - z_top_m)
        
        return SaltDomeParams(
            vp_salt_mps=U(4500, 5500),
            edge_width_m=U(20, 60),
            z_top_m=z_top_m,
            z_base_m=z_base_m,
            stem_radius_m=stem_radius_m,
            mid_radius_m=mid_radius_m,
            canopy_amp_m=canopy_amp_m,
            canopy_center_m=canopy_center_m,
            canopy_sigma_m=canopy_sigma_m,
            cx_base_m=cx_base_m,
            cy_base_m=cy_base_m,
            tilt_dx_m=U(-500, 500),
            tilt_dy_m=U(-500, 500),
            meander_amp_m=U(0, 150),
            meander_knots=6,
            roughness_eps=U(0.0, 0.15),
            roughness_decay_frac=U(0.2, 0.8),
            kmin=2, kmax=6, azimuth_power=1.4,
            warp_enable=True,
            warp_amp_m=U(80, 400),
            warp_sigma_factor=U(1.5, 3.0),
            warp_z_peak_m=warp_z_peak_m,
            warp_z_sigma_m=U(300, 900),
            Lx=Lx, Ly=Ly, Lz=Lz
        )

    def mask(self, X, Y, Z) -> np.ndarray:
        """Returns hard mask. We use soft_mask > 0.5."""
        m = self.soft_mask(X, Y, Z)
        return (m > 0.5)

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        # We need to compute SDF.
        # This duplicates some work if apply_to_vp is also called, but that's okay for now.
        _, mask_vol, _ = self._compute_full({'temp': np.zeros_like(Z, dtype=np.float32)}, X, Y, Z, only_mask=True)
        return mask_vol

    def apply_to_vp(self, vp_bg: np.ndarray, X, Y, Z) -> np.ndarray:
        # Backward compatibility
        props = self.apply_properties({'vp': vp_bg}, X, Y, Z)
        return props['vp']

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        """
        Applies salt properties and tectonic warp to all given background properties.
        """
        return self._compute_full(props_dict, X, Y, Z, only_mask=False)

    # --- Internal computation logic ---

    def _compute_full(self, props_bg: dict, X, Y, Z, only_mask: bool = False):
        p = self.params
        rng = self._rng
        
        nx, ny, nz = Z.shape
        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float64)
        dz = z_arr[1] - z_arr[0] if len(z_arr) > 1 else 10.0

        cx, cy = self._centerline(p, z_arr, rng)
        R0 = self._radius_profile(p, z_arr)
        eps = self._roughness_profile(p, z_arr)
        a_zk, phi_k, ks = self._azimuth_coeffs(p, z_arr, rng)
        
        A_z = p.warp_amp_m * np.exp(-((z_arr - p.warp_z_peak_m) ** 2) / max(p.warp_z_sigma_m ** 2, 1e-6))
        A_z[(z_arr < p.z_top_m) | (z_arr > p.z_base_m)] = 0.0
        A_z = A_z.astype(np.float32)
        
        mask_vol = np.empty_like(Z, dtype=np.float32)
        props_new = {k: np.empty_like(v, dtype=np.float32) for k, v in props_bg.items()} if not only_mask else None
        
        salt_vals = {
            'vp': p.vp_salt_mps,
            'rho': p.rho_salt_gcc,
            'resist': p.resist_salt_ohmm,
            'chi': p.chi_salt_si
        }
        
        xx, yy = np.indices((nx, ny), dtype=np.int32)
        
        for iz in range(nz):
            if R0[iz] <= 0.0:
                mask_vol[..., iz] = 0.0
                if not only_mask:
                    for k in props_new:
                        props_new[k][..., iz] = props_bg[k][..., iz]
                continue
            
            dxv = (x_arr - cx[iz]).astype(np.float32)
            dyv = (y_arr - cy[iz]).astype(np.float32)
            r = np.sqrt(dxv[:, None] ** 2 + dyv[None, :] ** 2).astype(np.float32) + 1e-6
            theta = np.arctan2(dyv[None, :], dxv[:, None]).astype(np.float32)
            
            N = np.zeros_like(theta, dtype=np.float32)
            norm = 0.0
            for i, k_azim in enumerate(ks):
                a = a_zk[i, iz]
                if a == 0.0: continue
                N += a * np.cos(k_azim * theta + phi_k[i]).astype(np.float32)
                norm += abs(a)
            if norm > 1e-8:
                N = N / norm
            else:
                N[...] = 0.0
                
            R = R0[iz] * (1.0 + eps[iz] * N)
            R = np.clip(R, 0.25 * R0[iz], 10.0 * R0[iz]).astype(np.float32)
            
            sdf = (R - r).astype(np.float32)
            w = max(p.edge_width_m, 1e-3)
            m = _sigmoid_stable(sdf / w)
            mask_vol[..., iz] = m
            
            if only_mask:
                continue
                
            Az_val = A_z[iz]
            if p.warp_enable and Az_val > 0.0:
                sigma = max(p.warp_sigma_factor * R0[iz], 1e-3)
                uz = (Az_val * np.exp(-(r ** 2) / (2.0 * sigma ** 2))).astype(np.float32)
                zt = (iz + uz / dz).astype(np.float32)
                z1 = np.floor(zt).astype(np.int32)
                z2 = z1 + 1
                z1 = np.clip(z1, 0, nz - 1)
                z2 = np.clip(z2, 0, nz - 1)
                wgt = (zt - z1).astype(np.float32)
                
                for k in props_new:
                    vp1 = props_bg[k][xx, yy, z1]
                    vp2 = props_bg[k][xx, yy, z2]
                    prop_warp = (1.0 - wgt) * vp1 + wgt * vp2
                    target_val = salt_vals.get(k, prop_warp) # Fallback to background if unknown property
                    props_new[k][..., iz] = (1.0 - m) * prop_warp + m * target_val
            else:
                for k in props_new:
                    prop_bg = props_bg[k][..., iz]
                    target_val = salt_vals.get(k, prop_bg)
                    props_new[k][..., iz] = (1.0 - m) * prop_bg + m * target_val
                    
        if only_mask:
            return None, mask_vol, None
        return props_new


    def _centerline(self, p: SaltDomeParams, z: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        z0, z1 = p.z_base_m, p.z_top_m
        denom = max(z0 - z1, 1e-6)
        t = (z0 - z) / denom
        t = np.clip(t, 0.0, 1.0)
        cx_tilt = p.cx_base_m + p.tilt_dx_m * t
        cy_tilt = p.cy_base_m + p.tilt_dy_m * t

        nkn = max(int(p.meander_knots), 4)
        zk = np.linspace(p.z_top_m, p.z_base_m, nkn, dtype=np.float64)
        oxk = rng.normal(0.0, 1.0, size=nkn).astype(np.float64)
        oyk = rng.normal(0.0, 1.0, size=nkn).astype(np.float64)
        oxk = oxk / (np.max(np.abs(oxk)) + 1e-9) * p.meander_amp_m
        oyk = oyk / (np.max(np.abs(oyk)) + 1e-9) * p.meander_amp_m

        ox = _catmull_rom_1d(zk, oxk, z)
        oy = _catmull_rom_1d(zk, oyk, z)

        cx = cx_tilt + ox
        cy = cy_tilt + oy
        
        rmax = p.mid_radius_m + p.canopy_amp_m + 3.0 * p.edge_width_m
        if hasattr(p, 'Lx') and hasattr(p, 'Ly'):
            cx = np.clip(cx, rmax, p.Lx - rmax)
            cy = np.clip(cy, rmax, p.Ly - rmax)
        
        return cx.astype(np.float32), cy.astype(np.float32)

    def _radius_profile(self, p: SaltDomeParams, z: np.ndarray) -> np.ndarray:
        t = (z - p.z_top_m) / max(p.z_base_m - p.z_top_m, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        u = 1.0 - t
        u = _smoothstep(u)
        R_stem = p.stem_radius_m + (p.mid_radius_m - p.stem_radius_m) * u
        bump = p.canopy_amp_m * np.exp(-((z - p.canopy_center_m) ** 2) / max(p.canopy_sigma_m ** 2, 1e-6))
        R = R_stem + bump

        # Tapering at top and bottom to avoid flat cuts
        # Apply a smooth decay in the first and last few samples within the range
        taper_len = max(p.edge_width_m * 1.5, 40.0) # taper over ~40m or 1.5x edge width
        
        d_top = z - p.z_top_m
        d_base = p.z_base_m - z
        
        # Use simple clamping linear taper 0..1 then sqrt to make it rounder
        t_top = np.clip(d_top / taper_len, 0.0, 1.0)
        t_base = np.clip(d_base / taper_len, 0.0, 1.0)
        
        # sqrt(t) gives a steeper rise (rounder cap equivalent) than linear t or t^2
        taper_factor = np.sqrt(t_top) * np.sqrt(t_base)
        
        R = R * taper_factor
        
        R[(z < p.z_top_m) | (z > p.z_base_m)] = 0.0
        return R.astype(np.float32)

    def _roughness_profile(self, p: SaltDomeParams, z: np.ndarray) -> np.ndarray:
        t = (z - p.z_top_m) / max(p.z_base_m - p.z_top_m, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        eps_base = p.roughness_eps * (1.0 - p.roughness_decay_frac)
        eps = eps_base + (p.roughness_eps - eps_base) * (1.0 - _smoothstep(t))
        eps[(z < p.z_top_m) | (z > p.z_base_m)] = 0.0
        return eps.astype(np.float32)

    def _azimuth_coeffs(self, p: SaltDomeParams, z: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        ks = list(range(p.kmin, p.kmax + 1))
        K = len(ks)
        base = rng.normal(0.0, 1.0, size=K).astype(np.float32)
        base = base / (np.max(np.abs(base)) + 1e-9)

        a_zk = np.zeros((K, len(z)), dtype=np.float32)
        
        nkn = max(6, min(12, p.meander_knots + 2))
        zk = np.linspace(p.z_top_m, p.z_base_m, nkn, dtype=np.float64)
        for i in range(K):
            vals = rng.normal(0.0, 1.0, size=nkn).astype(np.float64)
            vals = vals / (np.max(np.abs(vals)) + 1e-9)
            mod = _catmull_rom_1d(zk, vals, z)
            a = base[i] * mod / (float(ks[i]) ** p.azimuth_power)
            a_zk[i, :] = a.astype(np.float32)

        phi_k = rng.uniform(0.0, 2.0 * np.pi, size=K).astype(np.float32)
        return a_zk, phi_k, ks
