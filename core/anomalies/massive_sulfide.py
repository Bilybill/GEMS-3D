#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Massive Sulfide (SMS / VMS-like) anomaly.

Goal:
- Create a more realistic "massive sulfide system" geometry for multi-physics datasets:
  1) Massive lens / mound (ore body) sitting on/near a stratigraphic horizon (layer_id)
  2) Feeder chimney (pipe) below the lens
  3) Stockwork zone (vein network) below the lens (funnel/envelope + sparse conductive network)
  4) Alteration halo (shell around the core)

Key feature:
- "Attached to a layer label" (layer_id) similar to stratigraphic sill control:
  We extract the local top/bottom surfaces (z_top, z_bot) of the chosen layer_id per (x,y),
  then build the lens center surface z_c(x,y) anchored inside that layer (with small undulation),
  ensuring the lens remains inside the layer thickness. Fault offsets in labels are inherited.

Interface:
- Follows the same pattern as your IgneousIntrusion / SaltDomeAnomaly:
  - mask(X,Y,Z) -> bool
  - soft_mask(X,Y,Z) -> float32 [0,1]
  - apply_to_vp(vp_bg, X,Y,Z) -> vp_new

Extras:
- subtype_labels(X,Y,Z) -> int32 volume with sub-part classes:
    0 host
    1 massive lens
    2 chimney
    3 stockwork
    4 halo (alteration shell)
  (Your DatasetBuilder may ignore this, but it's convenient for saving "真实形态+标签".)

Notes:
- Z is assumed positive downward in meters (consistent with your other anomalies).
- Uses only NumPy (no SciPy).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import math

from .base import Anomaly


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

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


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline interpolation (no SciPy)."""
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


def _sdf_ellipse_pos(X2d: np.ndarray, Y2d: np.ndarray, xc: float, yc: float, a: float, b: float) -> np.ndarray:
    """
    Positive-inside "ellipse window" sdf-like score:
      rr = sqrt((dx/a)^2 + (dy/b)^2)
      q = 1 - rr
      returns q * min(a,b)
    """
    dx = (X2d - xc).astype(np.float32)
    dy = (Y2d - yc).astype(np.float32)
    rr = np.sqrt((dx / max(a, 1e-6)) ** 2 + (dy / max(b, 1e-6)) ** 2)
    q = (1.0 - rr)
    return q.astype(np.float32) * float(min(a, b))


def _extract_longest_segment_top_bot(
    layer_labels: np.ndarray,
    layer_id: int,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (x,y), find the longest contiguous segment where labels==layer_id.
    Returns z_top(x,y), z_bot(x,y), thickness(x,y), valid(x,y).
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


def _pick_layer_id_from_labels(
    layer_labels: np.ndarray,
    dz: float,
    rng: np.random.Generator,
    min_coverage_frac: float = 0.05,
    n_try: int = 24,
) -> int:
    """
    Pick a layer_id with decent areal coverage and thickness.
    """
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
        cover = float(proj.mean())
        if cover < min_coverage_frac:
            continue
        thick_est = (mask_slice.sum(axis=2).astype(np.float32) * float(dz))
        med = float(np.median(thick_est[proj])) if np.any(proj) else 0.0
        score = cover * (med + 1e-6)
        if score > best_score:
            best_score = score
            best_k = int(k)
    return best_k


def _cheap_dilate2d(binary01: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Very cheap dilation without SciPy:
    OR with shifted copies within +-radius in x/y.
    """
    if radius <= 0:
        return binary01
    out = binary01.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            out |= np.roll(np.roll(binary01, dx, axis=0), dy, axis=1)
    return out


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

@dataclass
class MassiveSulfideParams:
    # --- strat control ---
    kind: str = "massive_sulfide"  # fixed
    layer_id: int = -1  # >=0 attaches to that layer; -1 auto-pick if layer_labels provided
    min_layer_thickness_m: float = 80.0
    # anchor position inside the layer: z_anchor = z_top + anchor_alpha * thickness
    # anchor_alpha close to 0 => near top boundary of that layer.
    anchor_alpha: float = 0.05
    # small undulation of the attached center surface (meters)
    anchor_undulation_amp_m: float = 8.0
    anchor_undulation_kmax: int = 3

    # --- footprint / lens ---
    center_x_m: float = 0.0  # 0 means auto-pick
    center_y_m: float = 0.0  # 0 means auto-pick
    lens_extent_x_m: float = 500.0
    lens_extent_y_m: float = 350.0
    lens_window_irregularity: float = 0.35   # 0..1, union extra ellipses
    lens_thickness_m: float = 90.0
    lens_thickness_var_frac: float = 0.25
    lens_thick_max_frac_of_layer: float = 0.6   # cap thickness by layer thickness
    mound_bulge_amp_m: float = 60.0           # central bulge (adds thickness)
    mound_sigma_frac: float = 0.45            # bulge sigma relative to min(extent)

    # --- chimney (feeder pipe) ---
    chimney_enable: bool = True
    chimney_height_m: float = 900.0
    chimney_radius_top_m: float = 35.0
    chimney_radius_base_m: float = 70.0
    chimney_drift_m: float = 120.0
    chimney_drift_knots: int = 6

    # --- stockwork (vein network) ---
    stockwork_enable: bool = True
    stockwork_height_m: float = 900.0
    stockwork_radius_top_m: float = 420.0
    stockwork_radius_base_m: float = 180.0
    # Noise-driven sparse network
    stockwork_noise_kmax: int = 4
    stockwork_vein_threshold: float = 0.65     # higher => sparser
    stockwork_vein_softness: float = 0.12      # sigmoid width on noise
    stockwork_dilate_radius: int = 1           # enlarge connectivity a bit
    # Directional anisotropy (stringy veins)
    stockwork_anisotropy_ratio: float = 2.5    # >1 => stretched veins
    stockwork_anisotropy_angle_deg: float = 20.0

    # --- alteration halo (shell around core) ---
    halo_enable: bool = True
    halo_thickness_m: float = 160.0

    # --- property mapping ---
    vp_massive_mps: float = 6200.0
    vp_chimney_mps: float = 5800.0
    vp_stockwork_mps: float = 5200.0
    halo_vp_delta_frac: float = -0.03  # negative can simulate fractured alteration lowering Vp
    
    rho_massive_gcc: float = 4.0
    rho_chimney_gcc: float = 3.5
    rho_stockwork_gcc: float = 3.0
    halo_rho_delta_frac: float = -0.02
    
    resist_massive_ohmm: float = 0.5
    resist_chimney_ohmm: float = 2.0
    resist_stockwork_ohmm: float = 10.0
    halo_resist_delta_frac: float = -0.2
    
    # A joint-display friendly magnetic scheme:
    # keep the massive core strongest, but also let chimney / stockwork / halo
    # carry a broader magnetic signature so gravity-magnetic-EM-seismic overlays
    # do not collapse into a tiny magnetic hotspot only.
    chi_massive_si: float = 0.012
    chi_chimney_si: float = 0.006
    chi_stockwork_si: float = 0.0018
    halo_chi_delta_frac: float = 0.0
    halo_chi_add_si: float = 5.0e-4

    # --- smoothing / edges ---
    edge_width_m: float = 25.0

    # --- misc ---
    include_halo_in_soft_mask: bool = False  # for "mask()", usually False
    rng_seed: int = 20260207


# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

@dataclass
class MassiveSulfide(Anomaly):
    """
    Massive Sulfide system anomaly, attached to a stratigraphic layer_id (optional).
    """
    type: str = "massive_sulfide"
    strength: float = 0.0
    edge_width_m: float = 25.0

    params: MassiveSulfideParams = field(default_factory=MassiveSulfideParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)

    _rng: np.random.Generator = field(default=None, repr=False)
    _pre: Dict[str, Any] = field(default=None, repr=False)

    def __post_init__(self):
        # rng
        self._rng = np.random.default_rng(int(self.params.rng_seed))
        # sync edge width
        self.edge_width_m = float(self.params.edge_width_m)
        # pre cache set later (after we see X,Y,Z)
        self._pre = None

    # ---------------------------
    # Required interface
    # ---------------------------

    def mask(self, X, Y, Z) -> np.ndarray:
        m = self.soft_mask(X, Y, Z)
        return (m > 0.5)

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        _, m_core, _ = self._compute_full({'temp': np.zeros_like(Z, dtype=np.float32)}, X, Y, Z, only_mask=True)
        return m_core

    def apply_to_vp(self, vp_bg: np.ndarray, X, Y, Z) -> np.ndarray:
        props = self.apply_properties({'vp': vp_bg}, X, Y, Z)
        return props['vp']

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        """
        Apply massive sulfide fields (vp, rho, resist, chi) based on parts (massive, chimney, stockwork, halo).
        """
        props, *_ = self._compute_full(props_dict, X, Y, Z, only_mask=False)
        return props

    # ---------------------------
    # Optional extra label output
    # ---------------------------

    def subtype_labels(self, X, Y, Z, thr: float = 0.5) -> np.ndarray:
        """
        Return integer subtype labels:
          0 host
          1 massive lens
          2 chimney
          3 stockwork
          4 halo
        """
        pre = self._get_precompute(X, Y, Z)
        nx, ny, nz = Z.shape
        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32)

        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        wd = max(float(self.edge_width_m), 1e-3)
        out = np.zeros((nx, ny, nz), dtype=np.int32)

        for iz in range(nz):
            z_m = float(z_arr[iz])
            sdf_massive, sdf_ch, sdf_sw, sdf_core, sdf_halo = self._sdfs_slice(z_m, iz, X2d, Y2d, z_arr, pre)

            m_massive = _sigmoid_stable(sdf_massive / wd)
            m_ch = _sigmoid_stable(sdf_ch / wd)
            m_sw = _sigmoid_stable(sdf_sw / wd)

            lbl = np.zeros((nx, ny), dtype=np.int32)
            lbl[m_sw > thr] = 3
            lbl[m_ch > thr] = 2
            lbl[m_massive > thr] = 1  # massive overrides

            if self.params.halo_enable:
                # halo ring from sdf_core
                m_out = _sigmoid_stable(-sdf_core / wd)
                m_near = _sigmoid_stable((sdf_core + float(self.params.halo_thickness_m)) / wd)
                m_halo = m_out * m_near
                lbl[(lbl == 0) & (m_halo > thr)] = 4

            out[..., iz] = lbl

        return out

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_full(self, props_bg: dict, X, Y, Z, only_mask: bool = False):
        p = self.params
        pre = self._get_precompute(X, Y, Z)

        nx, ny, nz = Z.shape
        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32)

        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        wd = max(float(self.edge_width_m), 1e-3)

        m_core_vol = np.empty((nx, ny, nz), dtype=np.float32)
        props_new = {k: np.empty_like(v, dtype=np.float32) for k, v in props_bg.items()} if not only_mask else None

        for iz in range(nz):
            z_m = float(z_arr[iz])

            sdf_massive, sdf_ch, sdf_sw, sdf_core, sdf_halo = self._sdfs_slice(
                z_m, iz, X2d, Y2d, z_arr, pre
            )

            # core mask
            m_core = _sigmoid_stable(sdf_core / wd).astype(np.float32)

            # halo ring (optional)
            if p.halo_enable and p.halo_thickness_m > 0:
                # outside core but within halo_thickness
                m_out = _sigmoid_stable(-sdf_core / wd)
                m_near = _sigmoid_stable((sdf_core + float(p.halo_thickness_m)) / wd)
                m_halo = (m_out * m_near).astype(np.float32)
            else:
                m_halo = np.zeros_like(m_core, dtype=np.float32)

            # what to expose as "soft_mask"
            if p.include_halo_in_soft_mask:
                m_core_vol[..., iz] = np.maximum(m_core, m_halo)
            else:
                m_core_vol[..., iz] = m_core

            if only_mask:
                continue

            # sub-part weights (avoid double counting by precedence)
            m_massive = _sigmoid_stable(sdf_massive / wd)
            m_ch = _sigmoid_stable(sdf_ch / wd)
            m_sw = _sigmoid_stable(sdf_sw / wd)

            w_massive = m_massive
            w_ch = m_ch * (1.0 - w_massive)
            w_sw = m_sw * (1.0 - w_massive - w_ch)
            w_sw = np.clip(w_sw, 0.0, 1.0)

            w_total = np.clip(w_massive + w_ch + w_sw, 0.0, 1.0)

            for k, prop_bg in props_bg.items():
                prop_slice = prop_bg[..., iz].astype(np.float32)
                
                # Default background mixing
                if k == 'vp':
                    prop_core = w_massive * float(p.vp_massive_mps) + w_ch * float(p.vp_chimney_mps) + w_sw * float(p.vp_stockwork_mps)
                    out_slice = (1.0 - w_total) * prop_slice + prop_core
                    if p.halo_enable and abs(float(p.halo_vp_delta_frac)) > 1e-9:
                        out_slice *= (1.0 + float(p.halo_vp_delta_frac) * m_halo)
                elif k == 'rho':
                    prop_core = (
                        w_massive * float(p.rho_massive_gcc)
                        + w_ch * float(p.rho_chimney_gcc)
                        + w_sw * float(p.rho_stockwork_gcc)
                    )
                    out_slice = (1.0 - w_total) * prop_slice + prop_core
                    if p.halo_enable and abs(float(p.halo_rho_delta_frac)) > 1e-9:
                        out_slice *= (1.0 + float(p.halo_rho_delta_frac) * m_halo)
                elif k == 'chi':
                    prop_core = w_massive * float(p.chi_massive_si) + w_ch * float(p.chi_chimney_si) + w_sw * float(p.chi_stockwork_si)
                    out_slice = (1.0 - w_total) * prop_slice + prop_core
                    if p.halo_enable and abs(float(p.halo_chi_delta_frac)) > 1e-9:
                        out_slice *= (1.0 + float(p.halo_chi_delta_frac) * m_halo)
                    if p.halo_enable and abs(float(getattr(p, 'halo_chi_add_si', 0.0))) > 1e-9:
                        out_slice += float(p.halo_chi_add_si) * m_halo
                elif k == 'resist':
                    # Use logarithmic/Archie-like mixing for conductivity (log(resist) blending)
                    log_bg = np.log10(np.clip(prop_slice, 1e-3, 1e6))
                    log_mass = np.log10(p.resist_massive_ohmm)
                    log_ch = np.log10(p.resist_chimney_ohmm)
                    log_sw = np.log10(p.resist_stockwork_ohmm)
                    
                    log_core = w_massive * log_mass + w_ch * log_ch + w_sw * log_sw
                    log_out = (1.0 - w_total) * log_bg + log_core
                    out_slice = 10.0 ** log_out
                    
                    if p.halo_enable and abs(float(p.halo_resist_delta_frac)) > 1e-9:
                        out_slice *= (1.0 + float(p.halo_resist_delta_frac) * m_halo)
                else:
                    out_slice = prop_slice
                
                props_new[k][..., iz] = out_slice.astype(np.float32)

        return props_new, m_core_vol, None

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def _get_precompute(self, X, Y, Z) -> Dict[str, Any]:
        if self._pre is not None:
            return self._pre

        p = self.params
        rng = self._rng

        nx, ny, nz = Z.shape
        x_arr = X[:, 0, 0].astype(np.float32)
        y_arr = Y[0, :, 0].astype(np.float32)
        z_arr = Z[0, 0, :].astype(np.float32)
        dx = float(x_arr[1] - x_arr[0]) if nx > 1 else 1.0
        dy = float(y_arr[1] - y_arr[0]) if ny > 1 else 1.0
        dz = float(z_arr[1] - z_arr[0]) if nz > 1 else 1.0

        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        Lx = float(x_arr.max() - x_arr.min() + dx)
        Ly = float(y_arr.max() - y_arr.min() + dy)
        Lz = float(z_arr.max() - z_arr.min() + dz)

        # --- Strat attach surfaces ---
        use_strat = (self.layer_labels is not None)
        layer_id = int(p.layer_id)

        if use_strat:
            if layer_id < 0:
                layer_id = _pick_layer_id_from_labels(self.layer_labels, dz, rng)
            z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, layer_id, z_arr)

            # "ok" places where layer is present and thick enough
            margin = max(dz, 2.0 * float(p.edge_width_m))
            ok = valid & (thick >= (float(p.min_layer_thickness_m) + 2.0 * margin))
        else:
            # fallback: build a fake flat horizon at mid-depth
            layer_id = -1
            z_top = np.full((nx, ny), float(z_arr[int(nz * 0.4)]), dtype=np.float32)
            z_bot = np.full((nx, ny), float(z_arr[int(nz * 0.6)]), dtype=np.float32)
            thick = (z_bot - z_top).astype(np.float32)
            ok = np.ones((nx, ny), dtype=bool)

        # --- Choose center (xc,yc) ---
        a0 = 0.5 * float(p.lens_extent_x_m)
        b0 = 0.5 * float(p.lens_extent_y_m)
        footprint_r = max(a0, b0)
        margin_xy = footprint_r + 3.0 * float(p.edge_width_m) + 2.0 * max(dx, dy)

        if float(p.center_x_m) > 0 and float(p.center_y_m) > 0:
            xc = float(p.center_x_m)
            yc = float(p.center_y_m)
        else:
            # select a random valid (x,y) within margins
            x_ok = (x_arr >= (x_arr.min() + margin_xy)) & (x_arr <= (x_arr.max() - margin_xy))
            y_ok = (y_arr >= (y_arr.min() + margin_xy)) & (y_arr <= (y_arr.max() - margin_xy))
            ok2 = ok & x_ok[:, None] & y_ok[None, :]

            cand = np.argwhere(ok2)
            if cand.size == 0:
                # fallback to center of model
                xc = float(0.5 * (x_arr.min() + x_arr.max()))
                yc = float(0.5 * (y_arr.min() + y_arr.max()))
            else:
                ii, jj = cand[rng.integers(0, len(cand))]
                xc = float(x_arr[ii])
                yc = float(y_arr[jj])

        # --- Build attached center surface z_c(x,y) for massive lens ---
        # Low-frequency undulation (inherits faults via z_top/z_bot already)
        kmax = max(int(p.anchor_undulation_kmax), 1)
        noise = np.zeros((nx, ny), dtype=np.float32)
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                a = rng.normal(0.0, 1.0)
                phi = rng.uniform(0.0, 2.0 * np.pi)
                xx = X2d / max(Lx, 1e-6)
                yy = Y2d / max(Ly, 1e-6)
                noise += (a * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phi)).astype(np.float32)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)

        anchor_alpha = float(np.clip(p.anchor_alpha, 0.0, 1.0))
        z_anchor = (z_top + anchor_alpha * thick).astype(np.float32)
        z_c = (z_anchor + float(p.anchor_undulation_amp_m) * noise).astype(np.float32)

        # lens thickness map (vary + bulge at center), then cap by layer thickness
        tv = rng.normal(0.0, 1.0, size=(nx, ny)).astype(np.float32)
        t0 = float(p.lens_thickness_m)
        t_map = t0 * (1.0 + float(p.lens_thickness_var_frac) * tv)
        t_map = np.clip(t_map, 0.25 * t0, 3.0 * t0).astype(np.float32)

        # bulge near center (makes mound-like thickening)
        dx0 = (X2d - xc).astype(np.float32)
        dy0 = (Y2d - yc).astype(np.float32)
        rr2 = dx0 * dx0 + dy0 * dy0
        sigma = float(p.mound_sigma_frac) * float(min(p.lens_extent_x_m, p.lens_extent_y_m))
        sigma = max(sigma, 2.0 * max(dx, dy))
        bulge = float(p.mound_bulge_amp_m) * np.exp(-rr2 / (2.0 * sigma * sigma)).astype(np.float32)
        # add bulge as thickness increase (not vertical shift)
        t_map = (t_map + bulge).astype(np.float32)

        # cap thickness by layer thickness
        frac = float(np.clip(p.lens_thick_max_frac_of_layer, 0.05, 0.95))
        t_cap = (frac * thick).astype(np.float32)
        t_map = np.minimum(t_map, t_cap).astype(np.float32)

        # ensure z_c stays inside layer bounds with margin
        margin_z = max(dz, 2.0 * float(p.edge_width_m))
        z_low = (z_top + 0.5 * t_map + margin_z).astype(np.float32)
        z_high = (z_bot - 0.5 * t_map - margin_z).astype(np.float32)
        z_c = np.clip(z_c, z_low, np.maximum(z_low, z_high)).astype(np.float32)

        # If invalid columns, zero thickness (no massive lens there)
        t_map[~ok] = 0.0

        # --- Lens footprint window (irregular union of ellipses) ---
        sdf_win = _sdf_ellipse_pos(X2d, Y2d, xc, yc, a0, b0)
        n_extra = int(round(float(p.lens_window_irregularity) * 6.0))
        for _ in range(max(0, n_extra)):
            ddx = rng.uniform(-0.30, 0.30) * a0
            ddy = rng.uniform(-0.30, 0.30) * b0
            aa = a0 * rng.uniform(0.6, 1.25)
            bb = b0 * rng.uniform(0.6, 1.25)
            sdf_e = _sdf_ellipse_pos(X2d, Y2d, xc + ddx, yc + ddy, aa, bb)
            sdf_win = np.maximum(sdf_win, sdf_e)

        # --- Determine "attach depth" at chosen center for chimney/stockwork ---
        # Use the z_c at the nearest grid index to (xc,yc)
        ic = int(np.clip(np.argmin(np.abs(x_arr - xc)), 0, nx - 1))
        jc = int(np.clip(np.argmin(np.abs(y_arr - yc)), 0, ny - 1))
        z_attach0 = float(z_c[ic, jc])
        # If center landed in invalid region, fallback to mid depth
        if not np.isfinite(z_attach0):
            z_attach0 = float(z_arr[int(nz * 0.45)])

        # --- Chimney centerline x(z),y(z) and radius r(z) (like salt centerline) ---
        # We'll build along depth from z_attach0 downward.
        chimney_active = np.zeros((nz,), dtype=bool)
        cx_ch = np.full((nz,), xc, dtype=np.float32)
        cy_ch = np.full((nz,), yc, dtype=np.float32)
        r_ch = np.zeros((nz,), dtype=np.float32)

        if bool(p.chimney_enable):
            z0 = z_attach0
            z1 = z_attach0 + float(p.chimney_height_m)
            if z1 < z0 + 3.0 * dz:
                z1 = z0 + 3.0 * dz

            chimney_active = (z_arr >= z0) & (z_arr <= z1)

            # drift knots
            nkn = max(int(p.chimney_drift_knots), 4)
            zk = np.linspace(z0, z1, nkn, dtype=np.float64)
            oxk = rng.normal(0.0, 1.0, size=nkn).astype(np.float64)
            oyk = rng.normal(0.0, 1.0, size=nkn).astype(np.float64)
            oxk = oxk / (np.max(np.abs(oxk)) + 1e-9) * float(p.chimney_drift_m)
            oyk = oyk / (np.max(np.abs(oyk)) + 1e-9) * float(p.chimney_drift_m)

            ox = _catmull_rom_1d(zk, oxk, z_arr)
            oy = _catmull_rom_1d(zk, oyk, z_arr)

            cx_ch = (xc + ox).astype(np.float32)
            cy_ch = (yc + oy).astype(np.float32)

            # radius taper: larger at base, smaller near top
            t = (z_arr - z0) / max(z1 - z0, 1e-6)
            t = np.clip(t, 0.0, 1.0).astype(np.float32)
            r_ch = (float(p.chimney_radius_top_m) * (1.0 - t) + float(p.chimney_radius_base_m) * t).astype(np.float32)
            r_ch[~chimney_active] = 0.0

        # --- Stockwork envelope radius R_env(z) and noise modes ---
        stockwork_active = np.zeros((nz,), dtype=bool)
        r_env = np.zeros((nz,), dtype=np.float32)

        if bool(p.stockwork_enable):
            # stockwork starts near z_attach0 and extends down
            z0s = z_attach0 + 0.10 * float(p.stockwork_height_m)
            z1s = z_attach0 + float(p.stockwork_height_m)
            if z1s < z0s + 3.0 * dz:
                z1s = z0s + 3.0 * dz

            stockwork_active = (z_arr >= z0s) & (z_arr <= z1s)

            t = (z_arr - z0s) / max(z1s - z0s, 1e-6)
            t = np.clip(t, 0.0, 1.0).astype(np.float32)
            # funnel radius decays with depth
            r_env = (float(p.stockwork_radius_top_m) * (1.0 - t) + float(p.stockwork_radius_base_m) * t).astype(np.float32)
            r_env[~stockwork_active] = 0.0

            # Predefine 3D noise modes for veins
            modes = []
            K = max(int(p.stockwork_noise_kmax), 1)
            # a small set of random Fourier modes
            n_modes = 8 + 4 * K
            for _ in range(n_modes):
                kx = rng.integers(1, K + 1)
                ky = rng.integers(1, K + 1)
                kz = rng.integers(0, K + 1)
                a = float(rng.normal(0.0, 1.0))
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                modes.append((kx, ky, kz, a, phi))

            # Normalize amplitudes
            a_abs = np.array([abs(m[3]) for m in modes], dtype=np.float32)
            a_norm = float(a_abs.max() + 1e-9)
            modes = [(kx, ky, kz, a / a_norm, phi) for (kx, ky, kz, a, phi) in modes]
        else:
            modes = []

        self._pre = {
            "Lx": Lx, "Ly": Ly, "Lz": Lz,
            "dx": dx, "dy": dy, "dz": dz,
            "layer_id": layer_id,
            "use_strat": use_strat,
            "ok": ok.astype(bool),
            "z_top": z_top.astype(np.float32),
            "z_bot": z_bot.astype(np.float32),
            "thick": thick.astype(np.float32),
            "xc": float(xc), "yc": float(yc),
            "z_c": z_c.astype(np.float32),
            "t_map": t_map.astype(np.float32),
            "sdf_win": sdf_win.astype(np.float32),
            "z_attach0": float(z_attach0),
            # chimney
            "chimney_active": chimney_active,
            "cx_ch": cx_ch.astype(np.float32),
            "cy_ch": cy_ch.astype(np.float32),
            "r_ch": r_ch.astype(np.float32),
            # stockwork
            "stockwork_active": stockwork_active,
            "r_env": r_env.astype(np.float32),
            "modes": modes,
        }
        return self._pre

    # ------------------------------------------------------------------
    # Slice SDFs
    # ------------------------------------------------------------------

    def _sdfs_slice(
        self,
        z_m: float,
        iz: int,
        X2d: np.ndarray,
        Y2d: np.ndarray,
        z_arr: np.ndarray,
        pre: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (sdf_massive, sdf_chimney, sdf_stockwork, sdf_core, sdf_halo_dummy)
        sdf_halo_dummy is not a true sdf; halo is derived from sdf_core (ring).
        """
        p = self.params

        nx, ny = X2d.shape[0], Y2d.shape[1]

        # default negatives
        sdf_massive = np.full((nx, ny), -1e6, dtype=np.float32)
        sdf_ch = np.full((nx, ny), -1e6, dtype=np.float32)
        sdf_sw = np.full((nx, ny), -1e6, dtype=np.float32)

        # --- Massive lens attached to z_c(x,y) with thickness t_map and footprint sdf_win ---
        t_map = pre["t_map"]
        if float(t_map.max()) > 0:
            z_c = pre["z_c"]
            sdf_win = pre["sdf_win"]

            sdf_thick = 0.5 * t_map - np.abs(z_m - z_c)
            sdf_massive = np.minimum(sdf_thick, sdf_win).astype(np.float32)

            # ensure we don't create lens in invalid columns
            ok = pre["ok"]
            sdf_massive = np.where(ok, sdf_massive, -1e6).astype(np.float32)

        # --- Chimney pipe below the lens ---
        if bool(p.chimney_enable) and bool(pre["chimney_active"][iz]):
            xc = float(pre["cx_ch"][iz])
            yc = float(pre["cy_ch"][iz])
            r0 = float(pre["r_ch"][iz])
            if r0 > 1e-3:
                dx = (X2d - xc).astype(np.float32)
                dy = (Y2d - yc).astype(np.float32)
                rr = np.sqrt(dx * dx + dy * dy).astype(np.float32)
                sdf_ch = (r0 - rr).astype(np.float32)

        # --- Stockwork: envelope (funnel) + sparse vein noise threshold ---
        if bool(p.stockwork_enable) and bool(pre["stockwork_active"][iz]):
            # envelope radius
            Renv = float(pre["r_env"][iz])
            if Renv > 1e-3:
                xc = float(pre["cx_ch"][iz])  # follow chimney path
                yc = float(pre["cy_ch"][iz])

                dx = (X2d - xc).astype(np.float32)
                dy = (Y2d - yc).astype(np.float32)

                # anisotropy coordinates (stringy features)
                ang = math.radians(float(p.stockwork_anisotropy_angle_deg))
                ca, sa = math.cos(ang), math.sin(ang)
                s = (ca * dx + sa * dy).astype(np.float32)
                t = (-sa * dx + ca * dy).astype(np.float32)
                ratio = max(float(p.stockwork_anisotropy_ratio), 1.0)
                s = s / ratio  # stretch in s-direction

                rr = np.sqrt(s * s + t * t).astype(np.float32)
                sdf_env = (Renv - rr).astype(np.float32)

                # 3D-ish coherent noise field at this z
                Lx, Ly, Lz = float(pre["Lx"]), float(pre["Ly"]), float(pre["Lz"])
                z0 = float(z_m - float(z_arr.min()))
                zz = z0 / max(Lz, 1e-6)

                noise = np.zeros((nx, ny), dtype=np.float32)
                for (kx, ky, kz, a, phi) in pre["modes"]:
                    # use (s,t) for anisotropy instead of (x,y)
                    # normalize s,t to model size scale
                    px = (kx * (s / max(Lx, 1e-6))).astype(np.float32)
                    py = (ky * (t / max(Ly, 1e-6))).astype(np.float32)
                    pz = float(kz) * float(zz)
                    phase = (2.0 * np.pi * (px + py + pz) + float(phi)).astype(np.float32)
                    noise += float(a) * np.cos(phase).astype(np.float32)

                noise = noise / (np.max(np.abs(noise)) + 1e-9)

                # convert noise threshold to a "pseudo sdf": (noise - thr) scaled to meters
                thr = float(np.clip(p.stockwork_vein_threshold, -0.99, 0.99))
                # softness for the vein boundary
                soft = max(float(p.stockwork_vein_softness), 1e-3)
                # a smooth vein mask
                m_vein = _sigmoid_stable((noise - thr) / soft)

                # optionally dilate to increase connectivity a bit
                if int(p.stockwork_dilate_radius) > 0:
                    bin0 = (m_vein > 0.5)
                    bin1 = _cheap_dilate2d(bin0, radius=int(p.stockwork_dilate_radius))
                    # mix back softly
                    m_vein = np.maximum(m_vein, bin1.astype(np.float32) * 0.75)

                # approximate sdf for veins: map m_vein to "signed distance-like" via logit
                # sdf ~ wd * logit(m), clamp for safety
                m = np.clip(m_vein, 1e-4, 1.0 - 1e-4)
                sdf_vein = float(max(pre["dx"], pre["dy"])) * np.log(m / (1.0 - m)).astype(np.float32)
                sdf_sw = np.minimum(sdf_env, sdf_vein).astype(np.float32)

        # --- Core union ---
        sdf_core = np.maximum(np.maximum(sdf_massive, sdf_ch), sdf_sw).astype(np.float32)

        # halo is derived from sdf_core; return dummy
        sdf_halo_dummy = np.zeros_like(sdf_core, dtype=np.float32)

        return sdf_massive, sdf_ch, sdf_sw, sdf_core, sdf_halo_dummy
