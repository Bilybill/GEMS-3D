#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brine / Water-bearing Fault Zone anomaly (low resistivity corridor)
- Extract fault-likelihood from vp_ref (implicit faults)
- Keep top-1/top-2 largest connected components
- Skeletonize -> distance transform -> core + damage zone masks
- Optional patchiness (beads/segmentation) along the corridor

Ref: https://chatgpt.com/s/t_6988301699188191aa2a68b0f53dbecf
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label, distance_transform_edt
try:
    from skimage.morphology._skeletonize  import skeletonize_3d
except ImportError:
    # Fallback or warning if skimage is not installed
    import warnings
    def skeletonize_3d(img):
        warnings.warn("skimage not found, skipping skeletonize (using full volume)")
        return img

from .base import Anomaly


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _robust_norm(a: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if float(hi - lo) < eps:
        return np.zeros_like(a, dtype=np.float32)
    x = (a - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _grid_spacing_from_XYZ(X, Y, Z) -> Tuple[float, float, float]:
    # X,Y,Z are meshgrids indexing='ij'
    dx = float(X[1, 0, 0] - X[0, 0, 0]) if X.shape[0] > 1 else 1.0
    dy = float(Y[0, 1, 0] - Y[0, 0, 0]) if Y.shape[1] > 1 else 1.0
    dz = float(Z[0, 0, 1] - Z[0, 0, 0]) if Z.shape[2] > 1 else 1.0
    dx, dy, dz = max(dx, 1e-6), max(dy, 1e-6), max(dz, 1e-6)
    return dx, dy, dz


@dataclass
class BrineFaultZoneParams:
    # topology
    top_k: int = 1                      # 1 or 2
    fault_quantile: float = 0.996       # 0.992~0.998 typical
    smooth_sigma_m: float = 25.0        # Gaussian smoothing on likelihood (meters)
    min_component_voxels: int = 4000    # discard tiny CCs
    skeletonize: bool = True

    # corridor geometry
    core_thickness_m: float = 30.0      # "fault core" thickness (meters, full thickness)
    damage_width_m: float = 180.0       # "damage zone" width (meters, full width)
    edge_width_m: float = 20.0          # sigmoid transition width (meters)

    # patchiness (beads / segmentation along corridor)
    patch_enable: bool = True
    patch_sigma_m: float = 180.0        # low-freq random field smoothing (meters)
    patch_strength: float = 0.75        # 0..1, higher -> more segmented
    patch_threshold: float = 0.0        # threshold in normalized noise
    patch_soft: float = 0.25            # sigmoid softness for patch

    # resistivity painting (Ohm·m)  —— core very low, damage moderately low
    rho_core_ohmm: float = 0.5          # brine core resistivity
    rho_damage_factor: float = 0.25     # rho_damage = rho_bg * factor (clipped >= rho_core)

    # optional: tiny Vp perturbation (fracturing/fluids) if you ever want
    vp_delta_frac_in_damage: float = 0.00   # e.g. -0.02 (2% lower) ; default 0
    vp_delta_frac_in_core: float = 0.00     # e.g. -0.04 ; default 0


@dataclass
class BrineFaultZone(Anomaly):
    """
    NOTE:
    - This anomaly needs vp_ref (the background Vp volume) to extract implicit faults.
    - You can still use it to generate masks/labels even if you don't modify Vp.
    """
    type: str = "brine_fault_zone"
    strength: float = 0.0
    edge_width_m: float = 20.0

    params: BrineFaultZoneParams = field(default_factory=BrineFaultZoneParams)

    # Provide vp_ref at init (recommended) OR call set_vp_ref() before soft_mask/apply
    vp_ref: Optional[np.ndarray] = field(default=None, repr=False)

    rng_seed: int = 20260208
    _rng: np.random.Generator = field(default=None, repr=False)

    # cached
    _pre: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        if self.params.edge_width_m != 20.0:
            self.edge_width_m = float(self.params.edge_width_m)

    def set_vp_ref(self, vp: np.ndarray):
        self.vp_ref = vp
        self._pre = {}

    # --------- public interfaces ----------
    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        self._ensure_precomputed(X, Y, Z)
        return self._pre["m_any"].astype(np.float32)

    def subtype_labels(self, X, Y, Z) -> np.ndarray:
        """
        Fine labels:
        0 = background
        1..k = core of fault i
        11..(10+k) = damage zone of fault i
        """
        self._ensure_precomputed(X, Y, Z)
        return self._pre["subtype"].astype(np.int32)

    def apply_to_vp(self, vp_bg: np.ndarray, X, Y, Z) -> np.ndarray:
        props = self.apply_properties({'vp': vp_bg}, X, Y, Z)
        return props['vp']

    def apply_to_resistivity(self, rho_bg: np.ndarray, X, Y, Z) -> np.ndarray:
        props = self.apply_properties({'resist': rho_bg}, X, Y, Z)
        return props['resist']

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        """
        Unified property mapping: can apply to vp and resist simultaneously based on given dictionary keys.
        """
        self._ensure_precomputed(X, Y, Z)
        p = self.params
        
        m_core = self._pre["m_core_any"]
        m_dmg = self._pre["m_dmg_any"]
        
        out_props = {}
        
        for k, v in props_dict.items():
            vp = v.astype(np.float32, copy=True)
            if k == 'vp':
                if p.vp_delta_frac_in_damage != 0.0:
                    vp = vp * (1.0 + float(p.vp_delta_frac_in_damage) * m_dmg)
                if p.vp_delta_frac_in_core != 0.0:
                    vp = vp * (1.0 + float(p.vp_delta_frac_in_core) * m_core)
                out_props[k] = vp
            elif k == 'resist':
                # per-voxel background
                rho_core_val = float(p.rho_core_ohmm)
                rho_dmg = np.maximum(v * float(p.rho_damage_factor), rho_core_val)
                # damage first, then core overwrite
                vp = (1.0 - m_dmg) * v + m_dmg * rho_dmg
                vp = (1.0 - m_core) * vp + m_core * rho_core_val
                out_props[k] = vp
            else:
                out_props[k] = vp
                
        return out_props

    # --------- internal ----------
    def _ensure_precomputed(self, X, Y, Z):
        if "m_any" in self._pre:
            return
        if self.vp_ref is None:
            raise ValueError("BrineFaultZone requires vp_ref. Pass vp_ref=vp_bg when constructing, or call set_vp_ref().")

        vp = self.vp_ref.astype(np.float32, copy=False)
        assert vp.shape == Z.shape, f"vp_ref shape {vp.shape} must match grid shape {Z.shape}"

        dx, dy, dz = _grid_spacing_from_XYZ(X, Y, Z)
        p = self.params
        rng = self._rng

        # 1) gradients
        vx, vy, vz = np.gradient(vp, dx, dy, dz, edge_order=1)
        gxy = np.sqrt(vx * vx + vy * vy).astype(np.float32)
        gz = np.abs(vz).astype(np.float32)
        gmag = np.sqrt(vx * vx + vy * vy + vz * vz).astype(np.float32)

        # free a bit
        del vx, vy, vz

        ratio = gxy / (gz + 1e-3)
        ratio_n = _robust_norm(ratio, 50.0, 99.7)
        gmag_n = _robust_norm(gmag, 50.0, 99.5)
        L = (ratio_n * gmag_n).astype(np.float32)

        # 2) smooth likelihood
        if p.smooth_sigma_m and p.smooth_sigma_m > 0:
            sx, sy, sz = float(p.smooth_sigma_m / dx), float(p.smooth_sigma_m / dy), float(p.smooth_sigma_m / dz)
            L = gaussian_filter(L, sigma=(sx, sy, sz)).astype(np.float32)

        # 3) threshold
        q = float(np.clip(p.fault_quantile, 0.90, 0.9999))
        thr = float(np.quantile(L, q))
        cand = (L >= thr)

        # 4) connected components, keep top-k
        structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connected
        cc, ncc = cc_label(cand, structure=structure)
        if ncc == 0:
            # nothing found
            m_zero = np.zeros_like(vp, dtype=np.float32)
            self._pre = {
                "m_any": m_zero,
                "m_core_any": m_zero,
                "m_dmg_any": m_zero,
                "subtype": np.zeros_like(vp, dtype=np.int32),
            }
            return

        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]

        keep_ids: List[int] = []
        for cid in order:
            if cid == 0:
                continue
            if sizes[cid] < int(p.min_component_voxels):
                continue
            keep_ids.append(int(cid))
            if len(keep_ids) >= int(max(1, p.top_k)):
                break

        if len(keep_ids) == 0:
            # too small
            m_zero = np.zeros_like(vp, dtype=np.float32)
            self._pre = {
                "m_any": m_zero,
                "m_core_any": m_zero,
                "m_dmg_any": m_zero,
                "subtype": np.zeros_like(vp, dtype=np.int32),
            }
            return

        # optional patch field (shared) for bead-like segmentation
        patch = None
        if p.patch_enable:
            noise = rng.standard_normal(size=vp.shape).astype(np.float32)
            sx, sy, sz = float(p.patch_sigma_m / dx), float(p.patch_sigma_m / dy), float(p.patch_sigma_m / dz)
            noise = gaussian_filter(noise, sigma=(sx, sy, sz)).astype(np.float32)
            noise = _robust_norm(noise, 2.0, 98.0) * 2.0 - 1.0  # [-1,1]
            patch = _sigmoid_stable((noise - float(p.patch_threshold)) / max(float(p.patch_soft), 1e-3))

        # 5) per-fault masks
        m_core_list = []
        m_dmg_list = []

        r_core = 0.5 * float(p.core_thickness_m)
        r_dmg = 0.5 * float(p.damage_width_m)
        w = max(float(p.edge_width_m), 1e-3)

        for cid in keep_ids:
            comp = (cc == cid)

            if p.skeletonize:
                skel = skeletonize_3d(comp).astype(bool)
                if skel.sum() < 16:  # too tiny after skeletonize
                    skel = comp
            else:
                skel = comp

            # distance to skeleton (meters)
            dist = distance_transform_edt(~skel, sampling=(dx, dy, dz)).astype(np.float32)

            m_core = _sigmoid_stable((r_core - dist) / w)
            m_dmg = _sigmoid_stable((r_dmg - dist) / w)
            m_dmg = np.clip(m_dmg - m_core, 0.0, 1.0)

            if patch is not None:
                # apply patchiness stronger to core
                s = float(np.clip(p.patch_strength, 0.0, 1.0))
                m_core = m_core * ((1.0 - s) + s * patch)
                m_dmg = m_dmg * ((1.0 - 0.5 * s) + 0.5 * s * patch)

            m_core_list.append(m_core.astype(np.float32))
            m_dmg_list.append(m_dmg.astype(np.float32))

        # merge & labels
        k_found = len(m_core_list)
        m_core_stack = np.stack(m_core_list, axis=0)  # (k, nx, ny, nz)
        m_dmg_stack = np.stack(m_dmg_list, axis=0)

        m_core_any = np.max(m_core_stack, axis=0)
        m_dmg_any = np.max(m_dmg_stack, axis=0)
        m_any = np.clip(m_core_any + m_dmg_any, 0.0, 1.0)

        # subtype label: core first
        subtype = np.zeros_like(vp, dtype=np.int32)

        core_owner = np.argmax(m_core_stack, axis=0) + 1  # 1..k
        dmg_owner = np.argmax(m_dmg_stack, axis=0) + 1

        subtype[m_dmg_any > 0.5] = 10 + dmg_owner[m_dmg_any > 0.5]
        subtype[m_core_any > 0.5] = core_owner[m_core_any > 0.5]

        self._pre = {
            "m_any": m_any,
            "m_core_any": m_core_any,
            "m_dmg_any": m_dmg_any,
            "subtype": subtype,
        }
