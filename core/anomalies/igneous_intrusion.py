#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Igneous Intrusion Anomaly implementation.
Supports:
1. Stratigraphic Sill (Horizon Control) - via layer_labels
2. Geometric Sill (Undulating Sheet)
3. Dyke (Planar with Window)
4. Swarm (Multiple Dykes)
5. Stock (Plug/Laccolith-like)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any
from .base import Anomaly

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)

def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline (no SciPy)."""
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(xk)
    if n < 4:
        # Fallback to linear
        return np.interp(x, xk, yk).astype(np.float32)

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

def _extract_longest_segment_top_bot(
    layer_labels: np.ndarray,
    layer_id: int,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (x,y), find the longest contiguous segment where labels==layer_id.
    """
    nx, ny, nz = layer_labels.shape
    if z_arr.ndim > 1:
        z_arr = z_arr.flatten()
    z_arr = z_arr.astype(np.float32, copy=False)

    flat = (layer_labels.reshape(-1, nz) == int(layer_id))
    anyv = flat.any(axis=1)

    z_top = np.full((nx * ny,), np.nan, dtype=np.float32)
    z_bot = np.full((nx * ny,), np.nan, dtype=np.float32)

    cols = np.where(anyv)[0]
    for c in cols:
        m = flat[c].astype(np.int8, copy=False)
        dm = np.diff(m)
        starts = list(np.where(dm == 1)[0] + 1)
        ends = list(np.where(dm == -1)[0])

        if m[0] == 1:
            starts = [0] + starts
        if m[-1] == 1:
            ends = ends + [nz - 1]

        if len(starts) == 0 or len(ends) == 0:
            continue

        best_len = -1
        best_s, best_e = starts[0], ends[0]
        i_s, i_e = 0, 0
        while i_s < len(starts) and i_e < len(ends):
            s = starts[i_s]
            while i_e < len(ends) and ends[i_e] < s:
                i_e += 1
            if i_e >= len(ends):
                break
            e = ends[i_e]
            L = (e - s + 1)
            if L > best_len:
                best_len = L
                best_s, best_e = s, e
            i_s += 1
            i_e += 1

        z_top[c] = z_arr[best_s]
        z_bot[c] = z_arr[best_e]

    z_top = z_top.reshape(nx, ny)
    z_bot = z_bot.reshape(nx, ny)
    thickness = (z_bot - z_top).astype(np.float32)
    valid = np.isfinite(z_top) & np.isfinite(z_bot) & (thickness > 0)
    return z_top, z_bot, thickness, valid

def _pick_sill_layer_id_from_labels(
    layer_labels: np.ndarray,
    dz: float,
    rng: np.random.Generator,
    min_coverage_frac: float = 0.05,
    n_try: int = 24,
) -> int:
    vals, counts = np.unique(layer_labels.astype(np.int32), return_counts=True)
    if vals.size == 0:
        return -1
    vmin, vmax = int(vals.min()), int(vals.max())
    candidates = vals[(vals > vmin) & (vals < vmax)]
    if candidates.size == 0:
        candidates = vals
    if candidates.size > n_try:
        candidates = rng.choice(candidates, size=n_try, replace=False)

    best_k = int(candidates[0])
    best_score = -1.0

    for k in candidates:
        mask_slice = (layer_labels == int(k))
        proj = mask_slice.any(axis=2)
        cover = proj.mean()
        if cover < min_coverage_frac:
            continue
        thick_est = (mask_slice.sum(axis=2).astype(np.float32) * float(dz))
        med = float(np.median(thick_est[proj])) if np.any(proj) else 0.0
        score = float(cover * (med + 1e-6))
        if score > best_score:
            best_score = score
            best_k = int(k)
    return best_k


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

@dataclass
class IgneousIntrusionParams:
    # Common
    kind: str = "sill" # "sill" | "dyke" | "swarm" | "stock"
    
    # Generic Dimensions (for mapping old params if needed)
    # Note: These are fallback; specific fields below preferred
    cx_m: float = 1500.0
    cy_m: float = 1500.0
    cz_m: float = 1000.0
    length_m: float = 1000.0
    width_m: float = 1000.0
    max_thickness_m: float = 50.0
    strike_deg: float = 0.0
    dip_deg: float = 90.0
    
    # ---- STRATIGRAPHIC SILL (Horizon Control) ----
    sill_layer_id: int = -1                 # >=0 uses layer control
    sill_alpha: float = 0.5
    sill_thick_max_frac_of_layer: float = 0.35 
    sill_min_layer_thickness_m: float = 60.0

    # ---- GEOMETRIC SILL (Merged link params + existing) ----
    sill_xc_m: float = 0.0
    sill_yc_m: float = 0.0
    sill_zc_m: float = 0.0
    sill_thickness_m: float = 80.0
    sill_extent_x_m: float = 2000.0
    sill_extent_y_m: float = 2000.0
    sill_undulation_amp_m: float = 10.0
    sill_undulation_kmax: int = 3
    sill_thickness_var_frac: float = 0.2
    sill_window_irregularity: float = 0.2

    # ---- DYKE (Merged) ----
    dyke_x0_m: float = 0.0
    dyke_y0_m: float = 0.0
    dyke_z0_m: float = 0.0
    dyke_thickness_m: float = 30.0
    dyke_length_m: float = 3000.0
    dyke_width_m: float = 1200.0
    dyke_strike_deg: float = 30.0
    dyke_dip_deg: float = 85.0
    
    # ---- SWARM ----
    swarm_count: int = 8
    swarm_spacing_m: float = 200.0
    swarm_fan_deg: float = 10.0  # 0 means parallel
    swarm_spacing_jitter_frac: float = 0.15
    swarm_parallel_jitter_deg: float = 1.5
    swarm_dip_jitter_deg: float = 1.0
    swarm_thickness_min_m: float = 3.0
    swarm_thickness_max_m: float = 8.0
    swarm_length_min_m: float = 1000.0
    swarm_length_max_m: float = 10000.0
    swarm_echelon_zone_frac: float = 0.45
    swarm_echelon_step_m: float = 120.0
    swarm_top_z_m: float = 0.0
    swarm_base_z_m: float = 3500.0

    # ---- STOCK / PLUG ----
    stock_xc_m: float = 0.0
    stock_yc_m: float = 0.0
    stock_z_top_m: float = 600.0
    stock_z_base_m: float = 2400.0
    stock_radius_m: float = 600.0
    stock_drift_m: float = 200.0
    stock_drift_knots: int = 6
    stock_roughness_eps: float = 0.12
    stock_kmin: int = 2
    stock_kmax: int = 7
    stock_azimuth_power: float = 1.4

    # ---- AUREOLE ----
    aureole_enable: bool = True
    aureole_thickness_m: float = 100.0
    aureole_vp_delta_frac: float = 0.03
    aureole_rho_delta_frac: float = 0.02
    aureole_resist_delta_frac: float = -0.1
    aureole_chi_delta_frac: float = 0.05
    
    # Common
    vp_intr_mps: float = 5500.0
    rho_intr_gcc: float = 3.0
    resist_intr_ohmm: float = 5000.0
    chi_intr_si: float = 0.05
    
    roughness_amp_m: float = 5.0 # For backward compat
    roughness_scale: float = 0.02
    edge_width_m: float = 20.0
    taper_edge_m: float = 50.0

# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

@dataclass
class IgneousIntrusion(Anomaly):
    """
    Igneous Intrusion Anomaly.
    Supports: "sill", "dyke", "swarm", "stock".
    "sill" can be layer-controlled if sill_layer_id >= 0 and layer_labels provided.
    """
    # Override base class fields with defaults to make them optional
    type: str = "igneous"
    strength: float = 0.0
    edge_width_m: float = 20.0
    
    params: IgneousIntrusionParams = field(default_factory=IgneousIntrusionParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)
    rng_seed: int = 42
    
    _rng: np.random.Generator = field(default=None, repr=False)
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        # Note: params is initialized by default_factory if not provided
        if self.params is None:  # In case explicit None was passed
            self.params = IgneousIntrusionParams()
            
        # Update inherited edge_width_m from params if available
        if self.params.edge_width_m != 20.0:
            self.edge_width_m = self.params.edge_width_m
        
        # Populate geometric defaults from old generic params if not set
        p = self.params
        if p.sill_xc_m == 0.0 and p.cx_m != 1500.0: p.sill_xc_m = p.cx_m
        if p.sill_yc_m == 0.0 and p.cy_m != 1500.0: p.sill_yc_m = p.cy_m
        if p.sill_zc_m == 0.0 and p.cz_m != 1000.0: p.sill_zc_m = p.cz_m
        
        if p.dyke_x0_m == 0.0 and p.cx_m != 1500.0: p.dyke_x0_m = p.cx_m
        if p.dyke_y0_m == 0.0 and p.cy_m != 1500.0: p.dyke_y0_m = p.cy_m
        if p.dyke_z0_m == 0.0 and p.cz_m != 1000.0: p.dyke_z0_m = p.cz_m
        
        if p.stock_xc_m == 0.0 and p.cx_m != 1500.0: p.stock_xc_m = p.cx_m
        if p.stock_yc_m == 0.0 and p.cy_m != 1500.0: p.stock_yc_m = p.cy_m

    def mask(self, X, Y, Z) -> np.ndarray:
        m = self.soft_mask(X, Y, Z)
        return m > 0.5
        
    def soft_mask(self, X, Y, Z) -> np.ndarray:
        _, m_core, _ = self._compute_full({'temp': np.zeros_like(Z)}, X, Y, Z, only_mask=True)
        return m_core

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        props = self.apply_properties({'vp': vp}, X, Y, Z)
        return props['vp']

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        props, *_ = self._compute_full(props_dict, X, Y, Z, only_mask=False)
        return props

    # ----------------------------------------------------
    # Unified Logic
    # ----------------------------------------------------
    
    def _compute_full(self, props_bg: dict, X, Y, Z, only_mask: bool = False):
        p = self.params
        rng = self._rng
        nx, ny, nz = Z.shape
        # Assuming regular grid, extracting arrays
        if X.ndim == 3:
            x_arr = X[:, 0, 0].astype(np.float32)
            y_arr = Y[0, :, 0].astype(np.float32)
            z_arr = Z[0, 0, :].astype(np.float32)
            X2d = x_arr[:, None]
            Y2d = y_arr[None, :]
        else:
            # Fallback for odd shapes
            z_arr = Z[0,0,:]
            X2d = X
            Y2d = Y
            
        pre = {}
        # Decision: which logic to run?
        is_strat_sill = (p.kind == "sill") and (self.layer_labels is not None) and (p.sill_layer_id >= 0)
        
        if is_strat_sill:
            pre = self._precompute_sill_stratigraphic(p, X2d, Y2d, z_arr, rng)
        elif p.kind == "sill":
             pre = self._precompute_sill_geometric(p, X2d, Y2d, rng)
        elif p.kind in ["dyke", "swarm"]:
             pre = self._precompute_dyke_list(p, rng)
        elif p.kind == "stock":
             pre = self._precompute_stock(p, z_arr, rng)
             
        props_new = {k: np.empty_like(v, dtype=np.float32) for k, v in props_bg.items()} if not only_mask else None
        m_core_vol = np.empty_like(Z, dtype=np.float32)
        # We assume regular per-slice processing
        
        wd = max(p.edge_width_m, 1e-3)
        
        for iz in range(nz):
            z_val = float(z_arr[iz])
            
            if is_strat_sill:
                sdf = self._sdf_sill_strat_slice(p, X2d, Y2d, z_val, iz, pre)
            elif p.kind == "sill":
                sdf = self._sdf_sill_geo_slice(p, X2d, Y2d, z_val, pre)
            elif p.kind == "dyke":
                sdf = self._sdf_dyke_union_slice(p, X2d, Y2d, z_val, pre, union_mode="single")
            elif p.kind == "swarm":
                sdf = self._sdf_dyke_union_slice(p, X2d, Y2d, z_val, pre, union_mode="swarm")
            elif p.kind == "stock":
                sdf = self._sdf_stock_slice(p, X2d, Y2d, iz, pre)
            else:
                 sdf = np.full((nx, ny), -1e6, dtype=np.float32)
            
            # Clip for safety
            sdf = np.clip(sdf, -1e5, 1e5)
            m_core = _sigmoid_stable(sdf / wd)
            m_core_vol[..., iz] = m_core

            if not only_mask:
                m_aur = 0.0
                if p.aureole_enable and p.aureole_thickness_m > 0:
                     t_aur = float(p.aureole_thickness_m)
                     m_out = _sigmoid_stable(-sdf / wd)
                     m_near = _sigmoid_stable((sdf + t_aur) / wd)
                     m_aur = m_out * m_near
                
                for k, prop_bg in props_bg.items():
                    prop_slice = prop_bg[..., iz].astype(np.float32)
                    
                    if k == 'vp':
                        out_slice = (1.0 - m_core) * prop_slice + m_core * float(p.vp_intr_mps)
                        if p.aureole_enable and abs(float(p.aureole_vp_delta_frac)) > 1e-9:
                             out_slice *= (1.0 + float(p.aureole_vp_delta_frac) * m_aur)
                    elif k == 'rho':
                        out_slice = (1.0 - m_core) * prop_slice + m_core * float(p.rho_intr_gcc)
                        if p.aureole_enable and abs(float(p.aureole_rho_delta_frac)) > 1e-9:
                             out_slice *= (1.0 + float(p.aureole_rho_delta_frac) * m_aur)
                    elif k == 'resist':
                        log_bg = np.log10(np.clip(prop_slice, 1e-3, 1e6))
                        log_intr = np.log10(p.resist_intr_ohmm)
                        log_out = (1.0 - m_core) * log_bg + m_core * log_intr
                        out_slice = 10.0 ** log_out
                        if p.aureole_enable and abs(float(p.aureole_resist_delta_frac)) > 1e-9:
                             out_slice *= (1.0 + float(p.aureole_resist_delta_frac) * m_aur)
                    elif k == 'chi':
                        out_slice = (1.0 - m_core) * prop_slice + m_core * float(p.chi_intr_si)
                        if p.aureole_enable and abs(float(p.aureole_chi_delta_frac)) > 1e-9:
                             out_slice *= (1.0 + float(p.aureole_chi_delta_frac) * m_aur)
                    else:
                        out_slice = prop_slice
                        
                    props_new[k][..., iz] = out_slice.astype(np.float32)

        return props_new, m_core_vol, None


    # ----------------------------------------------------
    # Stratigraphic Sill (Old Logic, adapted)
    # ----------------------------------------------------
    def _precompute_sill_stratigraphic(self, p, X2d, Y2d, z_arr, rng):
        nx, ny = X2d.shape[0], Y2d.shape[1]
        k = int(p.sill_layer_id)
        z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, k, z_arr)

        kmax = max(int(p.sill_undulation_kmax), 1)
        noise = np.zeros((nx, ny), dtype=np.float32)
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                a = rng.normal(0.0, 1.0)
                phi = rng.uniform(0.0, 2*np.pi)
                xx = X2d / max(p.length_m*2, 1e-6)
                yy = Y2d / max(p.width_m*2, 1e-6)
                noise += (a * np.cos(2*np.pi*(kx*xx + ky*yy) + phi)).astype(np.float32)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)

        alpha = np.clip(p.sill_alpha, 0.0, 1.0)
        z_h = (z_top + alpha * thick + p.sill_undulation_amp_m * noise).astype(np.float32)

        tv = rng.normal(0.0, 1.0, size=(nx, ny)).astype(np.float32)
        t0 = p.max_thickness_m
        t_map = t0 * (1.0 + p.sill_thickness_var_frac * tv)
        t_map = np.clip(t_map, 0.25*t0, 3.0*t0)
        
        frac = np.clip(p.sill_thick_max_frac_of_layer, 0.05, 0.95)
        t_cap = frac * thick
        t_map = np.minimum(t_map, t_cap)

        dz = z_arr[1] - z_arr[0] if len(z_arr) > 1 else 10.0
        margin = max(dz, 2.0 * p.edge_width_m)
        ok = valid & (thick >= (p.sill_min_layer_thickness_m + 2.0*margin)) & (t_map >= 2.0*dz)

        z_low = z_top + 0.5*t_map + margin
        z_high = z_bot - 0.5*t_map - margin
        z_h = np.clip(z_h, z_low, np.maximum(z_low, z_high))
        
        t_map[~ok] = 0.0
        
        # Window (Ellipse + Irregularity)
        xc, yc = p.cx_m, p.cy_m
        a0, b0 = p.length_m/2, p.width_m/2
        sdf_win = self._sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0)
        
        return {"z_h": z_h, "t_map": t_map, "sdf_win": sdf_win, "layer_id": k, "use_strat": True}

    def _sdf_sill_strat_slice(self, p, X2d, Y2d, z_val, iz, pre):
        z_h = pre["z_h"]
        t_map = pre["t_map"]
        sdf_win = pre["sdf_win"]
        sdf_thick = 0.5 * t_map - np.abs(z_val - z_h)
        sdf = np.minimum(sdf_thick, sdf_win)
        
        k = pre["layer_id"]
        in_layer = (self.layer_labels[:, :, iz] == k)
        sdf = np.where(in_layer, sdf, -1e6) # Hard gating
        return sdf

    # ----------------------------------------------------
    # Geometric Sill (New Logic)
    # ----------------------------------------------------
    def _precompute_sill_geometric(self, p, X2d, Y2d, rng):
        nx, ny = X2d.shape[0], Y2d.shape[1]
        kmax = max(int(p.sill_undulation_kmax), 1)
        noise = np.zeros((nx, ny), dtype=np.float32)
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                a = rng.normal(0.0, 1.0)
                phi = rng.uniform(0.0, 2.0 * np.pi)
                xx = X2d / max(p.sill_extent_x_m, 1e-6)
                yy = Y2d / max(p.sill_extent_y_m, 1e-6)
                noise += a * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phi)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)
        z_h = float(p.sill_zc_m) + float(p.sill_undulation_amp_m) * noise

        tv = rng.normal(0.0, 1.0, size=(nx, ny)).astype(np.float32)
        t0 = float(p.sill_thickness_m)
        t_map = t0 * (1.0 + float(p.sill_thickness_var_frac) * tv)
        t_map = np.clip(t_map, 0.25 * t0, 3.0 * t0)

        xc, yc = float(p.sill_xc_m), float(p.sill_yc_m)
        a0, b0 = 0.5 * float(p.sill_extent_x_m), 0.5 * float(p.sill_extent_y_m)
        sdf_union = self._sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0)

        n_extra = int(round(p.sill_window_irregularity * 4.0))
        for _ in range(n_extra):
            dx = rng.uniform(-0.25, 0.25) * a0
            dy = rng.uniform(-0.25, 0.25) * b0
            a = a0 * rng.uniform(0.6, 1.2)
            b = b0 * rng.uniform(0.6, 1.2)
            sdf_e = self._sdf_ellipse_pos(X2d, Y2d, xc + dx, yc + dy, a, b)
            sdf_union = np.maximum(sdf_union, sdf_e)

        return {"z_h": z_h.astype(np.float32), "t_map": t_map.astype(np.float32), "sdf_win": sdf_union.astype(np.float32)}

    def _sdf_sill_geo_slice(self, p, X2d, Y2d, z_m, pre):
        z_h = pre["z_h"]
        t_map = pre["t_map"]
        sdf_win = pre["sdf_win"]
        sdf_thick = 0.5 * t_map - np.abs(z_m - z_h)
        return np.minimum(sdf_thick, sdf_win)

    def _sdf_ellipse_pos(self, X2d, Y2d, xc, yc, a, b):
        dx = (X2d - xc).astype(np.float32)
        dy = (Y2d - yc).astype(np.float32)
        rr = np.sqrt((dx / max(a, 1e-6)) ** 2 + (dy / max(b, 1e-6)) ** 2)
        q = (1.0 - rr)
        return q * float(min(a, b))

    # ----------------------------------------------------
    # Dyke / Swarm
    # ----------------------------------------------------
    def _precompute_dyke_list(self, p, rng):
        base = {
            "x0": float(p.dyke_x0_m),
            "y0": float(p.dyke_y0_m),
            "z0": float(p.dyke_z0_m),
            "th": float(p.dyke_thickness_m),
            "len": float(p.dyke_length_m),
            "wid": float(p.dyke_width_m),
            "strike": float(p.dyke_strike_deg),
            "dip": float(p.dyke_dip_deg),
        }
        dykes = []
        if p.kind == "dyke":
            dykes = [base]
        else:
            # Swarm: mostly parallel, steeply dipping narrow dykes with
            # optional local en echelon stepping along strike.
            n = int(p.swarm_count)
            spacing = float(p.swarm_spacing_m)
            fan = float(p.swarm_fan_deg)
            strike0 = base["strike"]
            ang = math.radians(strike0)
            u_perp = np.array([-math.sin(ang), math.cos(ang), 0.0], dtype=np.float32)
            u_strike = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float32)
            u_strike = _unit(u_strike)
            
            idxs = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
            echelon_limit = max(0.0, 0.5 * max(n - 1, 1) * float(p.swarm_echelon_zone_frac))
            th_min = max(0.25, float(p.swarm_thickness_min_m))
            th_max = max(th_min, float(p.swarm_thickness_max_m))
            len_min = max(50.0, float(p.swarm_length_min_m))
            len_max = max(len_min, float(p.swarm_length_max_m))
            z_top = float(p.swarm_top_z_m)
            z_base = max(z_top + 50.0, float(p.swarm_base_z_m))
            dip_jitter = abs(float(p.swarm_dip_jitter_deg))
            strike_jitter = abs(float(p.swarm_parallel_jitter_deg))
            spacing_jitter = abs(float(p.swarm_spacing_jitter_frac))

            for jj in idxs:
                dyke = dict(base)
                fan_term = 0.0
                if fan > 0.0:
                    fan_term = (jj / max((n - 1) / 2.0, 1.0)) * (fan / 2.0)
                dyke["strike"] = float(strike0 + fan_term + rng.normal(0.0, strike_jitter))
                dyke["dip"] = float(np.clip(base["dip"] + rng.normal(0.0, dip_jitter), 82.0, 89.5))
                dyke["th"] = float(rng.uniform(th_min, th_max))
                dyke["len"] = float(rng.uniform(len_min, len_max))
                lateral_spacing = float(spacing * (1.0 + rng.normal(0.0, spacing_jitter)))
                dx, dy = (jj * lateral_spacing) * u_perp[0], (jj * lateral_spacing) * u_perp[1]
                echelon_shift = 0.0
                if abs(float(jj)) <= echelon_limit and abs(float(p.swarm_echelon_step_m)) > 1e-6:
                    echelon_shift = float(jj) * float(p.swarm_echelon_step_m)
                dyke["x0"] = float(base["x0"] + dx + echelon_shift * u_strike[0])
                dyke["y0"] = float(base["y0"] + dy + echelon_shift * u_strike[1])
                dyke["z_top"] = z_top
                dyke["z_base"] = z_base
                dykes.append(dyke)
                
        frames = []
        for d in dykes:
             strike = math.radians(float(d["strike"]))
             dip = math.radians(float(d["dip"]))
             u = np.array([math.cos(strike), math.sin(strike), 0.0], dtype=np.float32) # along strike
             u = _unit(u)
             ddir = np.array([-math.sin(strike), math.cos(strike), 0.0], dtype=np.float32) # dip dir horizontal
             ddir = _unit(ddir)
             # normal: 
             vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
             n = (math.sin(dip) * ddir + math.cos(dip) * vertical)
             n = _unit(n)
             v = np.cross(n, u)
             v = _unit(v)
             if "z_top" in d and "z_base" in d:
                 z_top = float(d["z_top"])
                 z_base = float(d["z_base"])
                 dz_span = max(z_base - z_top, 50.0)
                 v_z = abs(float(v[2]))
                 if v_z > 1e-3:
                     d["wid"] = max(float(d["wid"]), dz_span / v_z)
                 else:
                     d["wid"] = max(float(d["wid"]), dz_span)
                 d["z0"] = 0.5 * (z_top + z_base)
             frames.append({"u": u, "v": v, "n": n})
             
        return {"dykes": dykes, "frames": frames}

    def _sdf_dyke_union_slice(self, p, X2d, Y2d, z_m, pre, union_mode):
        dykes = pre["dykes"]
        frames = pre["frames"]
        sdf_union = np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        
        for d, fr in zip(dykes, frames):
            x0, y0, z0 = float(d["x0"]), float(d["y0"]), float(d["z0"])
            th = float(d["th"])
            half_len = 0.5 * float(d["len"])
            half_wid = 0.5 * float(d["wid"])
            
            dx = (X2d - x0).astype(np.float32)
            dy = (Y2d - y0).astype(np.float32)
            dz = (z_m - z0)
            
            n = fr["n"]
            dplane = n[0] * dx + n[1] * dy + n[2] * dz
            sdf_plane = 0.5 * th - np.abs(dplane)
            
            u = fr["u"]
            v = fr["v"]
            s = u[0] * dx + u[1] * dy + u[2] * dz
            t = v[0] * dx + v[1] * dy + v[2] * dz
            
            rr = np.sqrt((s / max(half_len, 1e-6)) ** 2 + (t / max(half_wid, 1e-6)) ** 2)
            sdf_win = (1.0 - rr) * float(min(half_len, half_wid))
            
            sdf_dyke = np.minimum(sdf_plane, sdf_win)
            sdf_union = np.maximum(sdf_union, sdf_dyke)
            
            if union_mode == "single":
                break
        return sdf_union

    # ----------------------------------------------------
    # Stock
    # ----------------------------------------------------
    def _precompute_stock(self, p, z_arr, rng):
        z = z_arr.astype(np.float64)
        top, base = float(p.stock_z_top_m), float(p.stock_z_base_m)
        denom = max(base - top, 1e-6)
        t = (z - top) / denom
        t = np.clip(t, 0.0, 1.0)
        
        Rmax = float(p.stock_radius_m)
        Rmin = 0.25 * Rmax

        shape = _smoothstep(t)
        taper_top = _smoothstep(1.0 - t)
        R0 = (Rmin + (Rmax - Rmin) * shape) * taper_top
        
        R0[(z < top) | (z > base)] = 0.0
        
        # Drift
        nkn = max(int(p.stock_drift_knots), 4)
        zk = np.linspace(top, base, nkn)
        oxk = rng.normal(0.0, 1.0, size=nkn) * float(p.stock_drift_m)
        oyk = rng.normal(0.0, 1.0, size=nkn) * float(p.stock_drift_m)
        ox = _catmull_rom_1d(zk, oxk, z)
        oy = _catmull_rom_1d(zk, oyk, z)
        cx = float(p.stock_xc_m) + ox
        cy = float(p.stock_yc_m) + oy
        
        # Azimuth roughness
        ks = list(range(int(p.stock_kmin), int(p.stock_kmax) + 1))
        K = len(ks)
        base_amp = rng.normal(0.0, 1.0, size=K)
        # normalize
        mx = np.max(np.abs(base_amp)) + 1e-9
        base_amp /= mx
        
        a_zk = np.zeros((K, len(z)), dtype=np.float32)
        nmod = max(6, min(12, nkn + 2))
        zmodk = np.linspace(top, base, nmod)
        for i, k in enumerate(ks):
             vals = rng.normal(0.0, 1.0, size=nmod)
             vals /= (np.max(np.abs(vals)) + 1e-9)
             mod = _catmull_rom_1d(zmodk, vals, z)
             a = base_amp[i] * mod / (float(k) ** float(p.stock_azimuth_power))
             a_zk[i, :] = a.astype(np.float32)
             
        phi_k = rng.uniform(0.0, 2 * np.pi, size=K).astype(np.float32)
        
        return {"R0": R0.astype(np.float32), "cx": cx.astype(np.float32), "cy": cy.astype(np.float32), 
                "ks": ks, "a_zk": a_zk, "phi_k": phi_k}

    def _sdf_stock_slice(self, p, X2d, Y2d, iz, pre):
        R0 = float(pre["R0"][iz])
        if R0 <= 1e-3:
             return np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        
        cx = float(pre["cx"][iz])
        cy = float(pre["cy"][iz])
        dx = (X2d - cx).astype(np.float32)
        dy = (Y2d - cy).astype(np.float32)
        r = np.sqrt(dx*dx + dy*dy)
        theta = np.arctan2(dy, dx)
        
        ks = pre["ks"]
        a_zk = pre["a_zk"]
        phi_k = pre["phi_k"]
        
        N = np.zeros_like(theta)
        norm = 0.0
        for i, k in enumerate(ks):
            a = float(a_zk[i, iz])
            if abs(a) < 1e-9: continue
            N += a * np.cos(k * theta + float(phi_k[i]))
            norm += abs(a)
        if norm > 1e-9:
            N /= norm
        
        eps = float(p.stock_roughness_eps)
        R = R0 * (1.0 + eps * N)
        return R - r
