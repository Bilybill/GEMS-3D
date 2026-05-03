#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sediment–Basement Interface (SBI) anomaly / background-builder.

Goal
-----
Simulate a large-scale sediment cover over a crystalline basement with an undulating
basement top surface. This interface controls the *background* characteristics of
gravity/magnetics/EM/seismic at regional scale.

Geological semantics
--------------------
- Basement: "four-high" (high Vp, high density, high resistivity, high susceptibility)
- Sediments: compaction gradients with depth (Vp, density, resistivity increase nonlinearly);
             susceptibility ~ 0 (can add tiny background noise)

Two ways to place the interface
-------------------------------
1) Label-anchored (recommended if you have layer_labels with faults):
   Use a basement layer id (or auto-pick the maximum label) and extract its top depth
   z_intf(x,y) as the interface. This preserves fault offsets implicitly.

2) Procedural:
   Generate a smooth 2D surface via low-frequency Fourier harmonics + tilt + smoothing.

Outputs
-------
- apply_to_vp(vp_bg, X, Y, Z): returns vp with basement enforced (mode configurable)
- soft_mask(X,Y,Z): basement soft mask (0..1)
- build_property_models(X,Y,Z, vp_bg=None): returns dict with
    rho, resist_ohmm, chi_SI, facies_label (1 sediment / 2 basement / 3 interface band),
    z_interface_xy (nx,ny)

Integration
-----------
Save as: core/anomalies/sediment_basement_interface.py
Add to:  core/anomalies/__init__.py  (export the class)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math

from .base import Anomaly


# --------------------------
# Utilities
# --------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _box_blur_1d(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Fast box blur via cumulative sum (separable)."""
    if radius <= 0:
        return a
    
    # Pad input to handle boundaries
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (radius, radius)
    ap = np.pad(a, pad_width, mode="edge")
    
    # Cumulative sum
    cs = np.cumsum(ap, axis=axis, dtype=np.float64)
    
    # We need sum[i..i+2r] = cs[i+2r] - cs[i-1]
    # Pad cs with a 0 at the beginning along 'axis'
    cs_pad_width = [(0, 0)] * a.ndim
    cs_pad_width[axis] = (1, 0)
    cs_padded = np.pad(cs, cs_pad_width, mode='constant', constant_values=0)
    
    # Slices for S = Upper - Lower
    # Upper: indices 2r+1 to 2r+1+N
    sl_upper = [slice(None)] * a.ndim
    sl_upper[axis] = slice(2 * radius + 1, 2 * radius + 1 + a.shape[axis])
    
    # Lower: indices 0 to N
    sl_lower = [slice(None)] * a.ndim
    sl_lower[axis] = slice(0, a.shape[axis])
    
    s = cs_padded[tuple(sl_upper)] - cs_padded[tuple(sl_lower)]
    w = 2 * radius + 1
    return (s / float(w)).astype(np.float32)

def _box_blur_2d(a: np.ndarray, radius: int, iters: int) -> np.ndarray:
    out = a.astype(np.float32, copy=True)
    r = int(max(radius, 0))
    iters = int(max(iters, 0))
    for _ in range(iters):
        out = _box_blur_1d(out, r, axis=0)
        out = _box_blur_1d(out, r, axis=1)
    return out

def _lowfreq_harmonic_noise_2d(
    X2d: np.ndarray,
    Y2d: np.ndarray,
    Lx: float,
    Ly: float,
    amp: float,
    kmax: int,
    rng: np.random.Generator,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """
    Smooth 2D noise via sum_{kx,ky} a cos(2π(kx x/Lx + ky y/Ly)+phi).
    Low k only => large scale undulations.
    """
    nx, ny = X2d.shape[0], Y2d.shape[1]
    if amp <= 0 or kmax <= 0:
        return np.zeros((nx, ny), dtype=np.float32)
    xx = (X2d / max(Lx, 1e-6)).astype(np.float32)
    yy = (Y2d / max(Ly * max(anisotropy, 1e-6), 1e-6)).astype(np.float32)

    noise = np.zeros((nx, ny), dtype=np.float32)
    for kx in range(1, kmax + 1):
        for ky in range(1, kmax + 1):
            a = float(rng.normal(0.0, 1.0))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            noise += (a * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phi)).astype(np.float32)

    noise /= (np.max(np.abs(noise)) + 1e-9)
    return (amp * noise).astype(np.float32)

def _extract_layer_top_depth(
    layer_labels: np.ndarray,
    layer_id: int,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each (x,y), find the first depth index where labels == layer_id.
    Returns:
      z_top (nx,ny) with NaN where absent,
      valid (nx,ny)
    """
    nx, ny, nz = layer_labels.shape
    z_arr = z_arr.astype(np.float32, copy=False).reshape(-1)
    lid = int(layer_id)

    mask = (layer_labels == lid)
    valid = mask.any(axis=2)
    # argmax gives first True for bool arrays (since True=1)
    top_idx = mask.argmax(axis=2).astype(np.int32)
    z_top = z_arr[top_idx]
    z_top = z_top.astype(np.float32)
    z_top[~valid] = np.nan
    return z_top, valid


# --------------------------
# Params
# --------------------------

@dataclass
class SedimentBasementParams:
    # Interface placement
    use_layer_labels: bool = True
    basement_layer_id: int = -1  # if <0 and use_layer_labels: auto pick max label
    z0_m: float = 3500.0         # mean interface depth for procedural mode

    tilt_x_m: float = 0.0        # peak-to-peak interface depth change across x (m)
    tilt_y_m: float = 0.0        # peak-to-peak interface depth change across y (m)

    undulation_amp_m: float = 600.0
    undulation_kmax: int = 4
    undulation_anisotropy: float = 1.0

    smooth_radius: int = 3       # box blur radius (grid cells)
    smooth_iters: int = 3

    edge_width_m: float = 40.0   # soft boundary thickness (meters)
    interface_band_m: float = 80.0  # label the interface band around z_intf

    # How to modify vp
    apply_mode: str = "blend"    # "blend" | "overwrite" | "max"
    vp_basement_mps: float = 6000.0
    vp_basement_grad_mps_per_m: float = 0.0  # optional increase with depth below interface

    # Sediment compaction trend (absolute depth z, from top = 0)
    vp_sed0_mps: float = 1700.0
    vp_sed_inf_mps: float = 4000.0
    vp_sed_scale_m: float = 1600.0
    vp_sed_lateral_frac: float = 0.03  # small lateral variations (fraction)

    # Density params in g/cm^3.
    rho_sed0_gcc: float = 1.95
    rho_sed_inf_gcc: float = 2.45
    rho_sed_scale_m: float = 2200.0
    rho_basement_gcc: float = 2.75

    # Resistivity (ohm·m)
    resist_sed0_ohmm: float = 5.0
    resist_sed_max_ohmm: float = 80.0
    resist_sed_scale_m: float = 1400.0
    resist_basement_ohmm: float = 2000.0

    # Susceptibility (SI)
    chi_sed_SI: float = 0.0005
    chi_basement_SI: float = 0.02
    basement_chi_hetero_frac: float = 0.25
    basement_chi_hetero_kmax: int = 3

    rng_seed: int = 20260208


# --------------------------
# Main Class
# --------------------------

@dataclass
class SedimentBasementInterface(Anomaly):
    """
    Sediment–Basement Interface anomaly / background modifier.
    """

    type: str = "sediment_basement"
    strength: float = 0.0  # not used; kept for compatibility
    edge_width_m: float = 40.0

    params: SedimentBasementParams = field(default_factory=SedimentBasementParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)

    _rng: np.random.Generator = field(default=None, repr=False)
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(int(self.params.rng_seed))
        # keep edge width consistent
        self.edge_width_m = float(self.params.edge_width_m)

    # ---- Base API used by DatasetBuilder ----
    def mask(self, X, Y, Z) -> np.ndarray:
        return (self.soft_mask(X, Y, Z) > 0.5)

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        _, m_b, _ = self._compute_full(vp_bg=None, X=X, Y=Y, Z=Z, only_mask=True)
        return m_b

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        out_dict = {k: v.copy() for k, v in props_dict.items()}
        vp_bg = props_dict.get('vp')
        multiprops = self.build_property_models(X, Y, Z, vp_bg=vp_bg)

        if vp_bg is not None:
            out_dict['vp'] = self.apply_to_vp(vp_bg, X, Y, Z)
        if 'rho' in out_dict:
            out_dict['rho'] = multiprops['rho'].astype(np.float32, copy=False)
        if 'resist' in out_dict:
            out_dict['resist'] = multiprops['resist_ohmm'].astype(np.float32, copy=False)
        if 'chi' in out_dict:
            out_dict['chi'] = multiprops['chi_SI'].astype(np.float32, copy=False)
        return out_dict

    def apply_to_vp(self, vp_bg: np.ndarray, X, Y, Z) -> np.ndarray:
        vp_new, _, _ = self._compute_full(vp_bg=vp_bg, X=X, Y=Y, Z=Z, only_mask=False)
        return vp_new

    # ---- Extra helpers for multi-physics dataset labels ----
    def build_property_models(
        self,
        X, Y, Z,
        vp_bg: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns:
                    rho, resist_ohmm, chi_SI, facies_label, z_interface_xy
        If vp_bg is provided, Vp model will be produced with apply_mode and used as reference,
        otherwise properties follow the param trends.
        """
        p = self.params
        rng = self._rng

        nx, ny, nz = Z.shape
        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32)

        # interface depth map
        z_intf_xy = self._get_or_build_interface(x_arr, y_arr, z_arr)

        # basement mask volume
        m_b = np.empty((nx, ny, nz), dtype=np.float32)
        w = max(float(p.edge_width_m), 1e-3)
        for iz in range(nz):
            sdf = (float(z_arr[iz]) - z_intf_xy).astype(np.float32)
            m_b[..., iz] = _sigmoid_stable(sdf / w)

        # facies label: 1 sediment / 2 basement / 3 interface band
        facies = np.ones((nx, ny, nz), dtype=np.int32)
        band = max(float(p.interface_band_m), 0.0)
        if band > 0:
            half = 0.5 * band
            for iz in range(nz):
                dist = np.abs(float(z_arr[iz]) - z_intf_xy)
                facies[..., iz] = np.where(dist <= half, 3, facies[..., iz])
        facies = np.where(m_b > 0.5, 2, facies).astype(np.int32)

        # optional vp reference
        if vp_bg is not None:
            vp_ref = self.apply_to_vp(vp_bg.astype(np.float32, copy=False), X, Y, Z)
        else:
            vp_ref = None

        # property volumes
        rho_gcc = np.empty((nx, ny, nz), dtype=np.float32)
        resist = np.empty((nx, ny, nz), dtype=np.float32)
        chi = np.empty((nx, ny, nz), dtype=np.float32)

        # basement chi heterogeneity (2D map)
        Lx = float(x_arr[-1] - x_arr[0]) if nx > 1 else 1.0
        Ly = float(y_arr[-1] - y_arr[0]) if ny > 1 else 1.0
        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]
        chi_noise2d = _lowfreq_harmonic_noise_2d(
            X2d, Y2d, Lx, Ly,
            amp=1.0,
            kmax=int(max(p.basement_chi_hetero_kmax, 0)),
            rng=rng,
            anisotropy=1.0
        )
        chi_noise2d = chi_noise2d / (np.max(np.abs(chi_noise2d)) + 1e-9)

        for iz in range(nz):
            z = float(z_arr[iz])

            # Sediment trends vs depth z
            vp_sed = float(p.vp_sed0_mps) + (float(p.vp_sed_inf_mps) - float(p.vp_sed0_mps)) * (1.0 - math.exp(-z / max(float(p.vp_sed_scale_m), 1e-6)))
            rho_sed = float(p.rho_sed0_gcc) + (float(p.rho_sed_inf_gcc) - float(p.rho_sed0_gcc)) * (1.0 - math.exp(-z / max(float(p.rho_sed_scale_m), 1e-6)))
            resist_sed = float(p.resist_sed0_ohmm) * math.exp(z / max(float(p.resist_sed_scale_m), 1e-6))
            resist_sed = float(np.clip(resist_sed, float(p.resist_sed0_ohmm), float(p.resist_sed_max_ohmm)))

            # Basement values (optionally slightly increase with depth below interface)
            # depth below interface varies in x,y; use sdf positive below
            sdf_xy = (z - z_intf_xy).astype(np.float32)
            d_below = np.maximum(sdf_xy, 0.0)

            vp_base = float(p.vp_basement_mps) + float(p.vp_basement_grad_mps_per_m) * d_below
            rho_base = float(p.rho_basement_gcc)
            resist_base = float(p.resist_basement_ohmm)

            # If vp_ref is available, nudge rho using vp_ref (lightly) to keep consistency
            if vp_ref is not None:
                vpv = vp_ref[..., iz].astype(np.float32)
                # A very mild, bounded adjustment to rho to correlate with vp (optional)
                rho_sed_map = (rho_sed + 0.00004 * (vpv - vp_sed)).astype(np.float32)
                rho_sed_map = np.clip(rho_sed_map, 1.7, 2.6).astype(np.float32)
            else:
                rho_sed_map = np.full((nx, ny), rho_sed, dtype=np.float32)

            # susceptibility: sediments near zero; basement higher with heterogeneity
            chi_sed = float(p.chi_sed_SI)
            chi_base = float(p.chi_basement_SI) * (1.0 + float(p.basement_chi_hetero_frac) * chi_noise2d)

            # blend with basement mask
            mb = m_b[..., iz]
            rho_gcc[..., iz] = (1.0 - mb) * rho_sed_map + mb * rho_base
            resist[..., iz] = (1.0 - mb) * resist_sed + mb * resist_base
            chi[..., iz] = (1.0 - mb) * chi_sed + mb * chi_base

        return {
            "rho": rho_gcc,
            "rho_gcc": rho_gcc,
            "resist_ohmm": resist,
            "chi_SI": chi,
            "facies_label": facies,
            "z_interface_xy": z_intf_xy.astype(np.float32),
        }

    # --------------------------
    # Core computation
    # --------------------------

    def _compute_full(self, vp_bg: Optional[np.ndarray], X, Y, Z, only_mask: bool = False):
        p = self.params
        nx, ny, nz = Z.shape

        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32)

        z_intf_xy = self._get_or_build_interface(x_arr, y_arr, z_arr)

        # produce basement mask + vp
        m_b = np.empty((nx, ny, nz), dtype=np.float32)
        vp_new = None if only_mask else np.empty((nx, ny, nz), dtype=np.float32)

        w = max(float(p.edge_width_m), 1e-3)

        # lateral variability map for sediments
        Lx = float(x_arr[-1] - x_arr[0]) if nx > 1 else 1.0
        Ly = float(y_arr[-1] - y_arr[0]) if ny > 1 else 1.0
        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]
        lat = _lowfreq_harmonic_noise_2d(
            X2d, Y2d, Lx, Ly,
            amp=1.0, kmax=2, rng=self._rng, anisotropy=1.0
        )
        lat = lat / (np.max(np.abs(lat)) + 1e-9)

        for iz in range(nz):
            z = float(z_arr[iz])
            sdf = (z - z_intf_xy).astype(np.float32)  # + below interface => basement
            mb = _sigmoid_stable(sdf / w)
            m_b[..., iz] = mb

            if only_mask:
                continue

            assert vp_bg is not None, "vp_bg required when only_mask=False"
            vp_bg_sl = vp_bg[..., iz].astype(np.float32, copy=False)

            # sediment trend
            vp_sed = float(p.vp_sed0_mps) + (float(p.vp_sed_inf_mps) - float(p.vp_sed0_mps)) * (1.0 - math.exp(-z / max(float(p.vp_sed_scale_m), 1e-6)))
            vp_sed_map = vp_sed * (1.0 + float(p.vp_sed_lateral_frac) * lat)

            # basement trend (optionally increases with depth below interface)
            d_below = np.maximum(sdf, 0.0)
            vp_base = float(p.vp_basement_mps) + float(p.vp_basement_grad_mps_per_m) * d_below

            mode = str(p.apply_mode).lower().strip()
            if mode == "overwrite":
                vp_sl = (1.0 - mb) * vp_sed_map + mb * vp_base
            elif mode == "max":
                vp_sl = (1.0 - mb) * vp_bg_sl + mb * np.maximum(vp_bg_sl, vp_base)
            else:  # "blend" default: keep bg above interface, enforce basement inside
                vp_enforced = np.maximum(vp_bg_sl, vp_base)
                vp_sl = (1.0 - mb) * vp_bg_sl + mb * vp_enforced

            vp_new[..., iz] = vp_sl.astype(np.float32)

        return vp_new, m_b, {"z_interface_xy": z_intf_xy}

    def _get_or_build_interface(self, x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
        """
        Cached interface depth map z_intf(x,y).
        """
        key = "z_intf_xy"
        if key in self._cache:
            return self._cache[key]

        p = self.params
        rng = self._rng

        nx, ny = len(x_arr), len(y_arr)
        Lx = float(x_arr[-1] - x_arr[0]) if nx > 1 else 1.0
        Ly = float(y_arr[-1] - y_arr[0]) if ny > 1 else 1.0

        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        z_intf = None

        # --- label anchored ---
        if bool(p.use_layer_labels) and (self.layer_labels is not None):
            labels = self.layer_labels.astype(np.int32, copy=False)
            if int(p.basement_layer_id) >= 0:
                lid = int(p.basement_layer_id)
            else:
                lid = int(np.nanmax(labels))
            z_top, valid = _extract_layer_top_depth(labels, lid, z_arr)

            # fill NaNs
            med = float(np.nanmedian(z_top)) if np.isfinite(z_top).any() else float(p.z0_m)
            z_top = np.where(np.isfinite(z_top), z_top, med).astype(np.float32)

            # smooth to avoid voxel jaggedness but preserve fault offsets partially
            z_intf = _box_blur_2d(z_top, radius=int(p.smooth_radius), iters=int(p.smooth_iters))
        else:
            # --- procedural ---
            base = float(p.z0_m) * np.ones((nx, ny), dtype=np.float32)

            # tilt as peak-to-peak across domain
            if abs(float(p.tilt_x_m)) > 0:
                base += (float(p.tilt_x_m) * (X2d - 0.5 * (x_arr[0] + x_arr[-1])) / max(Lx, 1e-6)).astype(np.float32)
            if abs(float(p.tilt_y_m)) > 0:
                base += (float(p.tilt_y_m) * (Y2d - 0.5 * (y_arr[0] + y_arr[-1])) / max(Ly, 1e-6)).astype(np.float32)

            und = _lowfreq_harmonic_noise_2d(
                X2d, Y2d, Lx, Ly,
                amp=float(p.undulation_amp_m),
                kmax=int(max(p.undulation_kmax, 0)),
                rng=rng,
                anisotropy=float(max(p.undulation_anisotropy, 1e-6)),
            )
            z_intf = base + und
            z_intf = _box_blur_2d(z_intf, radius=int(p.smooth_radius), iters=int(p.smooth_iters))

        # clamp to volume
        zmin, zmax = float(z_arr[0]), float(z_arr[-1])
        z_intf = np.clip(z_intf, zmin + 2.0 * float(p.edge_width_m), zmax - 2.0 * float(p.edge_width_m)).astype(np.float32)

        self._cache[key] = z_intf
        return z_intf
