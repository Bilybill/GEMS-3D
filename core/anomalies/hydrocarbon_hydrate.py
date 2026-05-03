#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrocarbon / Gas Hydrate Anomaly
- kind="gas": strat/trap lens + optional chimney (gas leakage)
- kind="hydrate": BSR-parallel hydrate layer + optional free-gas layer below BSR
Supports optional layer_labels (0..N, with faults). If provided and layer_id>=0:
  - anchor surfaces are extracted per (x,y) column using the longest contiguous segment.

Implements:
  - soft_mask(): continuous mask in [0,1]
  - apply_to_vp(): inject vp anomaly using soft mask
  - subtype_labels(): component labels (lens/chimney/hydrate/freegas/halo)

Design goal:
  keep consistent with your IgneousIntrusion/MassiveSulfide style: SDF + sigmoid soft edges.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from typing import Optional, Dict, Any, Tuple

from .base import Anomaly


# ----------------------------
# utilities
# ----------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
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

def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline, no SciPy. xk must be sorted."""
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(xk)
    if n < 4:
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

def _fourier_noise_2d(
    X2d: np.ndarray,
    Y2d: np.ndarray,
    rng: np.random.Generator,
    kmax: int = 4,
    scale_x_m: float = 2000.0,
    scale_y_m: float = 2000.0,
) -> np.ndarray:
    """Simple smooth noise by truncated Fourier series, normalized to [-1,1]."""
    nx, ny = X2d.shape[0], Y2d.shape[1]
    noise = np.zeros((nx, ny), dtype=np.float32)
    kmax = max(int(kmax), 1)
    sx = max(float(scale_x_m), 1e-6)
    sy = max(float(scale_y_m), 1e-6)
    xx = (X2d / sx).astype(np.float32)
    yy = (Y2d / sy).astype(np.float32)
    for kx in range(1, kmax + 1):
        for ky in range(1, kmax + 1):
            a = float(rng.normal(0.0, 1.0))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            noise += (a * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phi)).astype(np.float32)
    mx = float(np.max(np.abs(noise)) + 1e-9)
    return (noise / mx).astype(np.float32)

def _sdf_ellipse_pos(X2d: np.ndarray, Y2d: np.ndarray, xc: float, yc: float, a: float, b: float) -> np.ndarray:
    """Positive inside ellipse window, ~distance-like scale."""
    dx = (X2d - xc).astype(np.float32)
    dy = (Y2d - yc).astype(np.float32)
    rr = np.sqrt((dx / max(a, 1e-6)) ** 2 + (dy / max(b, 1e-6)) ** 2)
    q = (1.0 - rr)
    return q * float(min(a, b))

def _extract_longest_segment_top_bot(
    layer_labels: np.ndarray,
    layer_id: int,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (x,y), find the longest contiguous segment where labels==layer_id.
    Returns z_top(x,y), z_bot(x,y), thickness(x,y), valid(x,y)
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

def _pick_layer_id_auto(
    layer_labels: np.ndarray,
    rng: np.random.Generator,
    prefer_shallow: bool = True,
    n_try: int = 32,
    min_coverage_frac: float = 0.05,
) -> int:
    """Heuristic: choose a layer with decent coverage; optionally bias to shallow."""
    vals, counts = np.unique(layer_labels.astype(np.int32), return_counts=True)
    if vals.size == 0:
        return -1
    # remove extreme ends if possible
    vmin, vmax = int(vals.min()), int(vals.max())
    candidates = vals[(vals > vmin) & (vals < vmax)]
    if candidates.size == 0:
        candidates = vals
    if candidates.size > n_try:
        candidates = rng.choice(candidates, size=n_try, replace=False)

    # score by coverage + shallow bias
    best_k = int(candidates[0])
    best_score = -1e9
    for k in candidates:
        mask = (layer_labels == int(k))
        proj = mask.any(axis=2)
        cover = float(proj.mean())
        if cover < min_coverage_frac:
            continue
        # shallow bias: smaller median z index
        z_idx = np.argmax(mask, axis=2)  # first occurrence
        med = float(np.median(z_idx[proj])) if np.any(proj) else 1e9
        score = cover * (1.0 / (1.0 + med)) if prefer_shallow else cover
        if score > best_score:
            best_score = score
            best_k = int(k)
    return best_k


# ----------------------------
# Params
# ----------------------------

@dataclass
class HydrocarbonHydrateParams:
    # kind: "gas" or "hydrate"
    kind: str = "gas"

    # optional stratigraphic anchor (layer_labels)
    layer_id: int = -1              # if >=0 and layer_labels provided -> anchor to that layer
    anchor_alpha: float = 0.5       # center surface within layer: z = z_top + alpha * thickness
    hard_gate_to_layer: bool = True # if True, only inside label==layer_id for strat parts
    min_layer_thickness_m: float = 60.0

    # common location / footprint
    center_x_m: float = 1500.0
    center_y_m: float = 1500.0

    # lens / trap footprint
    lens_extent_x_m: float = 900.0
    lens_extent_y_m: float = 600.0
    lens_window_irregularity: float = 0.6  # 0..1.5 (adds extra ellipses union)
    lens_undulation_amp_m: float = 10.0
    lens_undulation_kmax: int = 3
    lens_thickness_m: float = 120.0
    lens_thickness_var_frac: float = 0.25

    # gas-specific: chimney (leakage / gas plume)
    gas_enable_chimney: bool = True
    chimney_height_m: float = 1000.0
    chimney_radius_top_m: float = 60.0
    chimney_radius_base_m: float = 140.0
    chimney_drift_m: float = 250.0
    chimney_drift_knots: int = 6
    chimney_taper_power: float = 1.0  # 1 linear, >1 more pinched

    # hydrate-specific: BSR-style layer (hydrate above, free gas below)
    hydrate_enable_patchy: bool = True
    hydrate_patch_kmax: int = 4
    hydrate_patch_scale_x_m: float = 1200.0
    hydrate_patch_scale_y_m: float = 1200.0
    hydrate_patch_threshold: float = 0.15  # larger -> fewer patches (range ~ -0.2..0.4)
    hydrate_offset_above_m: float = 40.0   # hydrate center above anchor surface
    hydrate_thickness_m: float = 60.0
    hydrate_extent_x_m: float = 1800.0
    hydrate_extent_y_m: float = 1800.0

    hydrate_enable_free_gas_below: bool = True
    free_gas_offset_below_m: float = 60.0
    free_gas_thickness_m: float = 90.0

    # vp targets
    vp_gas_mps: float = 1800.0
    vp_hydrate_mps: float = 3700.0
    vp_free_gas_mps: float = 2000.0
    
    # rho target params in g/cm^3
    rho_gas_gcc: float = 2.0
    rho_hydrate_gcc: float = 2.3
    rho_free_gas_gcc: float = 2.1
    
    # resist targets (Ohm-m)
    resist_gas_ohmm: float = 100.0
    resist_hydrate_ohmm: float = 200.0
    resist_free_gas_ohmm: float = 50.0

    # halo (optional): represent resistivity high ring / altered zone etc.
    halo_enable: bool = True
    halo_thickness_m: float = 120.0
    halo_vp_delta_frac: float = 0.01  # small perturb on background
    halo_rho_delta_frac: float = 0.01
    halo_resist_delta_frac: float = 0.1

    # soft edge
    edge_width_m: float = 20.0

    # randomness
    rng_seed: int = 123


# ----------------------------
# Anomaly class
# ----------------------------

@dataclass
class HydrocarbonHydrate(Anomaly):
    type: str = "hydrocarbon_hydrate"
    strength: float = 0.0
    edge_width_m: float = 20.0

    params: HydrocarbonHydrateParams = field(default_factory=HydrocarbonHydrateParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)

    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        if self.params is None:
            self.params = HydrocarbonHydrateParams()
        self.edge_width_m = float(self.params.edge_width_m)
        self._rng = np.random.default_rng(int(self.params.rng_seed))

    # ---- required interface ----
    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        _, m_all, _ = self._compute_full({'temp': np.zeros_like(Z, dtype=np.float32)}, X, Y, Z, only_mask=True)
        return m_all

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        props = self.apply_properties({'vp': vp}, X, Y, Z)
        return props['vp']

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        """
        Unified property application for hydrocarbon hydrate system.
        """
        props, *_ = self._compute_full(props_dict, X, Y, Z, only_mask=False)
        return props

    # ---- extra utilities ----
    def subtype_labels(self, X, Y, Z) -> np.ndarray:
        """
        Returns int labels:
          0 bg
          1 gas_lens
          2 gas_chimney
          3 hydrate_layer
          4 free_gas_below
          5 halo
        """
        pre = self._precompute(X, Y, Z)
        nx, ny, nz = Z.shape
        lbl = np.zeros((nx, ny, nz), dtype=np.int32)
        wd = max(float(self.params.edge_width_m), 1e-3)

        x_arr, y_arr, z_arr, X2d, Y2d = pre["grid"]
        for iz in range(nz):
            z = float(z_arr[iz])
            sdf_lens, sdf_chim, sdf_hyd, sdf_fg = self._sdf_components_slice(X2d, Y2d, z, iz, pre)
            # hard labels by component, priority: lens/chimney/hydrate/freegas
            if sdf_lens is not None:
                lbl[..., iz] = np.where(sdf_lens > 0, 1, lbl[..., iz])
            if sdf_chim is not None:
                lbl[..., iz] = np.where(sdf_chim > 0, 2, lbl[..., iz])
            if sdf_hyd is not None:
                lbl[..., iz] = np.where(sdf_hyd > 0, 3, lbl[..., iz])
            if sdf_fg is not None:
                lbl[..., iz] = np.where(sdf_fg > 0, 4, lbl[..., iz])

            # halo as separate region around union (for visualization)
            if self.params.halo_enable and self.params.halo_thickness_m > 0:
                sdf_union = self._sdf_union_slice(X2d, Y2d, z, iz, pre)
                t = float(self.params.halo_thickness_m)
                m_out = _sigmoid_stable(-sdf_union / wd)
                m_near = _sigmoid_stable((sdf_union + t) / wd)
                m_h = m_out * m_near
                lbl[..., iz] = np.where(m_h > 0.5, 5, lbl[..., iz])

        return lbl

    def masks_dict(self, X, Y, Z) -> Dict[str, np.ndarray]:
        """Return soft masks for each component (float32 in [0,1])."""
        pre = self._precompute(X, Y, Z)
        nx, ny, nz = Z.shape
        wd = max(float(self.params.edge_width_m), 1e-3)

        x_arr, y_arr, z_arr, X2d, Y2d = pre["grid"]
        out = {
            "all": np.zeros((nx, ny, nz), dtype=np.float32),
            "gas_lens": np.zeros((nx, ny, nz), dtype=np.float32),
            "gas_chimney": np.zeros((nx, ny, nz), dtype=np.float32),
            "hydrate": np.zeros((nx, ny, nz), dtype=np.float32),
            "free_gas": np.zeros((nx, ny, nz), dtype=np.float32),
            "halo": np.zeros((nx, ny, nz), dtype=np.float32),
        }

        for iz in range(nz):
            z = float(z_arr[iz])
            sdf_lens, sdf_chim, sdf_hyd, sdf_fg = self._sdf_components_slice(X2d, Y2d, z, iz, pre)

            m_lens = _sigmoid_stable(sdf_lens / wd) if sdf_lens is not None else 0.0
            m_chim = _sigmoid_stable(sdf_chim / wd) if sdf_chim is not None else 0.0
            m_hyd  = _sigmoid_stable(sdf_hyd  / wd) if sdf_hyd  is not None else 0.0
            m_fg   = _sigmoid_stable(sdf_fg   / wd) if sdf_fg   is not None else 0.0

            # union for "all"
            m_all = np.maximum.reduce([m_lens, m_chim, m_hyd, m_fg]).astype(np.float32)

            out["gas_lens"][..., iz] = m_lens
            out["gas_chimney"][..., iz] = m_chim
            out["hydrate"][..., iz] = m_hyd
            out["free_gas"][..., iz] = m_fg
            out["all"][..., iz] = m_all

            if self.params.halo_enable and self.params.halo_thickness_m > 0:
                sdf_union = self._sdf_union_slice(X2d, Y2d, z, iz, pre)
                t = float(self.params.halo_thickness_m)
                m_out = _sigmoid_stable(-sdf_union / wd)
                m_near = _sigmoid_stable((sdf_union + t) / wd)
                out["halo"][..., iz] = (m_out * m_near).astype(np.float32)

        return out


    # ----------------------------
    # core computation
    # ----------------------------

    def _compute_full(self, props_bg: dict, X, Y, Z, only_mask: bool = False):
        p = self.params
        pre = self._precompute(X, Y, Z)
        nx, ny, nz = Z.shape

        x_arr, y_arr, z_arr, X2d, Y2d = pre["grid"]
        wd = max(float(p.edge_width_m), 1e-3)

        m_all_vol = np.zeros((nx, ny, nz), dtype=np.float32)
        props_new = {k: np.empty_like(v, dtype=np.float32) for k, v in props_bg.items()} if not only_mask else None

        for iz in range(nz):
            z = float(z_arr[iz])
            sdf_union = self._sdf_union_slice(X2d, Y2d, z, iz, pre)
            sdf_union = np.clip(sdf_union, -1e6, 1e6)
            m_all = _sigmoid_stable(sdf_union / wd)
            m_all_vol[..., iz] = m_all

            if not only_mask:
                sdf_lens, sdf_chim, sdf_hyd, sdf_fg = self._sdf_components_slice(X2d, Y2d, z, iz, pre)

                if p.kind == "gas":
                    m_lens = _sigmoid_stable(sdf_lens / wd) if sdf_lens is not None else 0.0
                    m_chim = _sigmoid_stable(sdf_chim / wd) if sdf_chim is not None else 0.0
                    m_gas = np.maximum(m_lens, m_chim).astype(np.float32)
                    m_hyd = 0.0
                    m_fg = 0.0
                else:  # hydrate
                    m_hyd = _sigmoid_stable(sdf_hyd / wd) if sdf_hyd is not None else 0.0
                    m_gas = 0.0
                    if p.hydrate_enable_free_gas_below and (sdf_fg is not None):
                        m_fg = _sigmoid_stable(sdf_fg / wd)
                        m_fg = (m_fg * (1.0 - m_hyd)).astype(np.float32)
                    else:
                        m_fg = 0.0

                m_halo = 0.0
                if p.halo_enable and p.halo_thickness_m > 0:
                    t = float(p.halo_thickness_m)
                    m_out = _sigmoid_stable(-sdf_union / wd)
                    m_near = _sigmoid_stable((sdf_union + t) / wd)
                    m_halo = (m_out * m_near).astype(np.float32)

                for k, prop_bg in props_bg.items():
                    prop_slice = prop_bg[..., iz].astype(np.float32, copy=False)

                    if k == 'vp':
                        if p.kind == "gas":
                            prop_slice = (1.0 - m_gas) * prop_slice + m_gas * float(p.vp_gas_mps)
                        else:
                            prop_slice = (1.0 - m_hyd) * prop_slice + m_hyd * float(p.vp_hydrate_mps)
                            if m_fg is not float and np.any(m_fg > 0):
                                prop_slice = (1.0 - m_fg) * prop_slice + m_fg * float(p.vp_free_gas_mps)
                        if p.halo_enable and abs(float(p.halo_vp_delta_frac)) > 1e-9:
                            prop_slice = prop_slice * (1.0 + float(p.halo_vp_delta_frac) * m_halo)
                    
                    elif k == 'rho':
                        if p.kind == "gas":
                            prop_slice = (1.0 - m_gas) * prop_slice + m_gas * float(p.rho_gas_gcc)
                        else:
                            prop_slice = (1.0 - m_hyd) * prop_slice + m_hyd * float(p.rho_hydrate_gcc)
                            if m_fg is not float and np.any(m_fg > 0):
                                prop_slice = (1.0 - m_fg) * prop_slice + m_fg * float(p.rho_free_gas_gcc)
                        if p.halo_enable and abs(float(getattr(p, 'halo_rho_delta_frac', 0.0))) > 1e-9:
                            prop_slice = prop_slice * (1.0 + float(p.halo_rho_delta_frac) * m_halo)
                            
                    elif k == 'resist':
                        log_bg = np.log10(np.clip(prop_slice, 1e-3, 1e6))
                        if p.kind == "gas":
                            log_gas = np.log10(p.resist_gas_ohmm)
                            log_out = (1.0 - m_gas) * log_bg + m_gas * log_gas
                        else:
                            log_hyd = np.log10(p.resist_hydrate_ohmm)
                            log_out = (1.0 - m_hyd) * log_bg + m_hyd * log_hyd
                            if m_fg is not float and np.any(m_fg > 0):
                                log_fg = np.log10(p.resist_free_gas_ohmm)
                                log_out = (1.0 - m_fg) * log_out + m_fg * log_fg
                        prop_slice = 10.0 ** log_out
                        if p.halo_enable and abs(float(getattr(p, 'halo_resist_delta_frac', 0.0))) > 1e-9:
                            prop_slice = prop_slice * (1.0 + float(p.halo_resist_delta_frac) * m_halo)
                            
                    props_new[k][..., iz] = prop_slice
                    
        return props_new, m_all_vol, None


    def _precompute(self, X, Y, Z) -> Dict[str, Any]:
        p = self.params
        rng = self._rng
        # assume regular grid
        x_arr = X[:, 0, 0].astype(np.float32) if X.ndim == 3 else X[:, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32) if Y.ndim == 3 else Y[0, :].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32) if Z.ndim == 3 else Z[0, 0, :].astype(np.float32)
        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        pre: Dict[str, Any] = {}
        pre["grid"] = (x_arr, y_arr, z_arr, X2d, Y2d)

        # strat anchor surface (optional)
        layer_id = int(p.layer_id)
        z_top = z_bot = thick = valid = None
        if (self.layer_labels is not None) and (layer_id < 0):
            # auto pick
            layer_id = _pick_layer_id_auto(self.layer_labels, rng, prefer_shallow=(p.kind == "hydrate"))
        if (self.layer_labels is not None) and (layer_id >= 0):
            z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, layer_id, z_arr)
        pre["layer_id"] = layer_id
        pre["z_top"] = z_top
        pre["z_bot"] = z_bot
        pre["thick"] = thick
        pre["valid"] = valid

        # build anchor center surface z_c(x,y)
        if z_top is not None:
            alpha = float(np.clip(p.anchor_alpha, 0.0, 1.0))
            z_c = (z_top + alpha * thick).astype(np.float32)
            # allow small undulation
            n = _fourier_noise_2d(X2d, Y2d, rng, kmax=p.lens_undulation_kmax,
                                  scale_x_m=max(p.lens_extent_x_m, 1.0),
                                  scale_y_m=max(p.lens_extent_y_m, 1.0))
            z_c = (z_c + float(p.lens_undulation_amp_m) * n).astype(np.float32)
            
            # Clip to model bounds to be robust
            z_min = float(z_arr[0] + 2.0 * float(p.edge_width_m))
            z_max = float(z_arr[-1] - 2.0 * float(p.edge_width_m))
            z_c = np.clip(z_c, z_min, z_max)

            # validity mask: avoid too-thin layer
            dz = float(z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 10.0
            margin = max(dz, 2.0 * float(p.edge_width_m))
            ok = (valid & (thick >= (float(p.min_layer_thickness_m) + 2.0 * margin))).astype(bool)
            pre["anchor_ok"] = ok
            pre["z_c"] = z_c
        else:
            # geometric fallback: constant depth center (use middle of model)
            z0 = float(z_arr[len(z_arr) // 2])
            pre["z_c"] = (z0 * np.ones((X2d.shape[0], Y2d.shape[1]), dtype=np.float32))
            pre["anchor_ok"] = np.ones((X2d.shape[0], Y2d.shape[1]), dtype=bool)

        # lens thickness map
        tv = rng.normal(0.0, 1.0, size=pre["z_c"].shape).astype(np.float32)
        t0 = float(p.lens_thickness_m)
        t_map = t0 * (1.0 + float(p.lens_thickness_var_frac) * tv)
        t_map = np.clip(t_map, 0.25 * t0, 3.0 * t0).astype(np.float32)
        pre["t_map"] = t_map

        # lens footprint window union
        xc, yc = float(p.center_x_m), float(p.center_y_m)
        a0, b0 = 0.5 * float(p.lens_extent_x_m), 0.5 * float(p.lens_extent_y_m)
        sdf_win = _sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0)

        n_extra = int(round(float(p.lens_window_irregularity) * 4.0))
        for _ in range(max(n_extra, 0)):
            dx = float(rng.uniform(-0.30, 0.30) * a0)
            dy = float(rng.uniform(-0.30, 0.30) * b0)
            a = float(a0 * rng.uniform(0.6, 1.25))
            b = float(b0 * rng.uniform(0.6, 1.25))
            sdf_e = _sdf_ellipse_pos(X2d, Y2d, xc + dx, yc + dy, a, b)
            sdf_win = np.maximum(sdf_win, sdf_e)
        pre["sdf_win"] = sdf_win.astype(np.float32)

        # chimney drift precompute along z
        if p.gas_enable_chimney:
            topz = float(np.nanmin(pre["z_c"])) - 0.5 * float(p.lens_thickness_m)
            botz = topz + float(p.chimney_height_m)
            # knots
            nkn = max(int(p.chimney_drift_knots), 4)
            zk = np.linspace(topz, botz, nkn).astype(np.float64)
            oxk = rng.normal(0.0, 1.0, size=nkn) * float(p.chimney_drift_m)
            oyk = rng.normal(0.0, 1.0, size=nkn) * float(p.chimney_drift_m)
            ox = _catmull_rom_1d(zk, oxk, z_arr.astype(np.float64))
            oy = _catmull_rom_1d(zk, oyk, z_arr.astype(np.float64))
            pre["chim_topz"] = topz
            pre["chim_botz"] = botz
            pre["chim_cx_z"] = (float(p.center_x_m) + ox).astype(np.float32)
            pre["chim_cy_z"] = (float(p.center_y_m) + oy).astype(np.float32)

        # hydrate patch noise
        if p.kind == "hydrate" and p.hydrate_enable_patchy:
            patch = _fourier_noise_2d(X2d, Y2d, rng, kmax=p.hydrate_patch_kmax,
                                      scale_x_m=p.hydrate_patch_scale_x_m,
                                      scale_y_m=p.hydrate_patch_scale_y_m)
            pre["hydrate_patch"] = patch.astype(np.float32)

        return pre


    # -------------- SDF composition --------------

    def _sdf_components_slice(
        self, X2d: np.ndarray, Y2d: np.ndarray, z_m: float, iz: int, pre: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return component SDFs (positive inside)."""
        p = self.params

        sdf_lens = sdf_chim = sdf_hyd = sdf_fg = None

        if p.kind == "gas":
            sdf_lens = self._sdf_trap_lens_slice(X2d, Y2d, z_m, iz, pre)
            if p.gas_enable_chimney:
                sdf_chim = self._sdf_chimney_slice(X2d, Y2d, z_m, iz, pre)
        else:
            sdf_hyd = self._sdf_hydrate_layer_slice(X2d, Y2d, z_m, iz, pre)
            if p.hydrate_enable_free_gas_below:
                sdf_fg = self._sdf_free_gas_below_slice(X2d, Y2d, z_m, iz, pre)

        return sdf_lens, sdf_chim, sdf_hyd, sdf_fg

    def _sdf_union_slice(self, X2d, Y2d, z_m, iz, pre) -> np.ndarray:
        sdf_lens, sdf_chim, sdf_hyd, sdf_fg = self._sdf_components_slice(X2d, Y2d, z_m, iz, pre)
        parts = []
        for s in (sdf_lens, sdf_chim, sdf_hyd, sdf_fg):
            if s is not None:
                parts.append(s)
        if len(parts) == 0:
            return np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        return np.maximum.reduce(parts).astype(np.float32)


    # -------------- gas: trap lens + chimney --------------

    def _sdf_trap_lens_slice(self, X2d, Y2d, z_m, iz, pre) -> np.ndarray:
        p = self.params
        z_c = pre["z_c"]
        t_map = pre["t_map"]
        sdf_win = pre["sdf_win"]

        # thickness band around anchor surface
        sdf_thick = 0.5 * t_map - np.abs(z_m - z_c)
        sdf = np.minimum(sdf_thick, sdf_win)

        # optional gate to layer label
        layer_id = int(pre["layer_id"])
        if (self.layer_labels is not None) and (layer_id >= 0) and bool(p.hard_gate_to_layer):
            in_layer = (self.layer_labels[:, :, iz] == layer_id)
            ok = pre.get("anchor_ok", None)
            if ok is not None:
                in_layer = in_layer & ok
            sdf = np.where(in_layer, sdf, -1e6)

        return sdf.astype(np.float32)

    def _sdf_chimney_slice(self, X2d, Y2d, z_m, iz, pre) -> np.ndarray:
        p = self.params
        topz = float(pre["chim_topz"])
        botz = float(pre["chim_botz"])
        if (z_m < topz) or (z_m > botz):
            return np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)

        # center drift along z
        cx = float(pre["chim_cx_z"][iz])
        cy = float(pre["chim_cy_z"][iz])

        # radius taper
        t = (z_m - topz) / max((botz - topz), 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        taper = (1.0 - t) ** float(p.chimney_taper_power)
        R = float(p.chimney_radius_base_m) * (1.0 - taper) + float(p.chimney_radius_top_m) * taper

        dx = (X2d - cx).astype(np.float32)
        dy = (Y2d - cy).astype(np.float32)
        r = np.sqrt(dx * dx + dy * dy)
        sdf_r = (R - r).astype(np.float32)

        # also softly limit vertical extent (avoid hard cut)
        vz = min((z_m - topz), (botz - z_m))
        sdf_z = float(vz)  # positive inside
        return np.minimum(sdf_r, sdf_z).astype(np.float32)


    # -------------- hydrate: BSR-parallel layer + free gas --------------

    def _sdf_hydrate_layer_slice(self, X2d, Y2d, z_m, iz, pre) -> np.ndarray:
        p = self.params
        z_c = pre["z_c"]  # anchor surface
        # hydrate above anchor by offset
        z_h = (z_c - float(p.hydrate_offset_above_m)).astype(np.float32)

        # footprint (usually broader than gas trap)
        xc, yc = float(p.center_x_m), float(p.center_y_m)
        a0, b0 = 0.5 * float(p.hydrate_extent_x_m), 0.5 * float(p.hydrate_extent_y_m)
        sdf_win = _sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0).astype(np.float32)

        # thickness band
        t0 = float(p.hydrate_thickness_m)
        sdf_thick = 0.5 * t0 - np.abs(z_m - z_h)
        sdf = np.minimum(sdf_thick, sdf_win)

        # strat gate if desired
        layer_id = int(pre["layer_id"])
        if (self.layer_labels is not None) and (layer_id >= 0) and bool(p.hard_gate_to_layer):
            # Secondary free-gas body:
            # For BSR/hydrate, we do NOT want strict layer containment (label==layer_id)
            # because the anomaly is often offset above/below the layer geometry.
            # Instead, we just use the anchor occurrence mask (ok).
            
            ok = pre.get("anchor_ok", None)
            if ok is not None:
                sdf = np.where(ok, sdf, -1e6)
            
            # Legacy code replaced:
            # in_layer = (self.layer_labels[:, :, iz] == layer_id)
            # if ok is not None: in_layer = in_layer & ok
            # sdf = np.where(in_layer, sdf, -1e6)

        # patchy: apply additional spatial mask (still soft)
        if p.hydrate_enable_patchy and ("hydrate_patch" in pre):
            patch = pre["hydrate_patch"]  # [-1,1]
            # inside patches -> keep sdf; outside -> kill sdf
            # use smooth threshold: m_patch in [0,1]
            thr = float(p.hydrate_patch_threshold)
            m_patch = _sigmoid_stable((patch - thr) / max(float(p.edge_width_m), 10.0))
            # convert to sdf-like gating by scaling
            sdf = np.where(m_patch > 0.05, sdf, -1e6)

        return sdf.astype(np.float32)

    def _sdf_free_gas_below_slice(self, X2d, Y2d, z_m, iz, pre) -> np.ndarray:
        p = self.params
        z_c = pre["z_c"]
        z_fg = (z_c + float(p.free_gas_offset_below_m)).astype(np.float32)

        xc, yc = float(p.center_x_m), float(p.center_y_m)
        a0, b0 = 0.5 * float(p.hydrate_extent_x_m), 0.5 * float(p.hydrate_extent_y_m)
        sdf_win = _sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0).astype(np.float32)

        t0 = float(p.free_gas_thickness_m)
        sdf_thick = 0.5 * t0 - np.abs(z_m - z_fg)
        sdf = np.minimum(sdf_thick, sdf_win)

        # strat gate if desired
        layer_id = int(pre["layer_id"])
        if (self.layer_labels is not None) and (layer_id >= 0) and bool(p.hard_gate_to_layer):
            # Same fix as above (Solution B)
            ok = pre.get("anchor_ok", None)
            if ok is not None:
                sdf = np.where(ok, sdf, -1e6)
            
            # Legacy code replaced:
            # in_layer = (self.layer_labels[:, :, iz] == layer_id)
            # ok = pre.get("anchor_ok", None)
            # if ok is not None: in_layer = in_layer & ok
            # sdf = np.where(in_layer, sdf, -1e6)

        return sdf.astype(np.float32)
