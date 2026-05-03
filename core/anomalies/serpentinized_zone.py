#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serpentinized Zone Anomaly (Serpentinization / Alteration Corridor)

Geologic idea (why it's "non-traditional" coupling):
- Serpentinization hydrates ultramafic rocks (peridotite) via water-rock reactions.
- Density and Vp typically decrease (more porous / altered mineralogy, lower elastic moduli),
  while magnetic susceptibility can increase due to formation of secondary magnetite.
- This breaks the common "higher Vp ↔ higher density ↔ higher chi" intuition, which is
  excellent for testing joint inversion robustness.

This anomaly can be generated in two ways:
1) Layer-anchored corridor (recommended if you have layer_labels): attach the altered band
   to a target layer_id (or auto-pick), inheriting faults implicitly from labels.
2) Geometric corridor: a curvilinear ribbon/sheet controlled by a 2D centerline spline
   and a depth function.

Outputs:
- soft_mask(): continuous [0,1] mask (core)
- mask(): binary mask (>0.5)
- apply_to_vp(): modifies Vp
- build_property_models(): creates density (g/cm^3) and susceptibility (SI) volumes plus subtype labels
  (you can integrate these into your multi-physics pipeline like SBI did).

Conventions:
- Coordinate arrays X,Y,Z are meters on a regular grid shaped (nx,ny,nz).
- layer_labels is optional, shape (nx,ny,nz), int layer ids 0..N (faults cause offsets).

Author: GMESDataset synthetic anomaly module
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math

# If your project has a common Anomaly base, keep this import.
# Otherwise, the class still works as a plain python object for masking/applying.
try:
    from .base import Anomaly
except Exception:  # pragma: no cover
    class Anomaly:  # minimal fallback
        def mask(self, X, Y, Z): raise NotImplementedError
        def soft_mask(self, X, Y, Z): raise NotImplementedError
        def apply_to_vp(self, vp, X, Y, Z): raise NotImplementedError


# -------------------------
# Utilities
# -------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _nearest_fill_2d(values: np.ndarray, valid: np.ndarray, max_iters: int = 512) -> np.ndarray:
    """Iteratively fill invalid pixels with nearest valid neighbors."""
    out = values.astype(np.float32, copy=True)
    mask = valid.astype(bool, copy=True)
    if mask.all():
        return out
    
    # Init invalid with 0 (or nan, but loop overwrites)
    out[~mask] = 0.0
    
    for _ in range(int(max_iters)):
        if mask.all():
            break
        out_new = out.copy()
        mask_new = mask.copy()
        # 4-neighbor prop
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(out, (dx, dy), axis=(0, 1))
            mask_shifted = np.roll(mask, (dx, dy), axis=(0, 1))
            
            # Identify pixels that are currently INVALID but have a VALID neighbor
            take = mask_shifted & ~mask
            out_new[take] = shifted[take]
            mask_new[take] = True
            
        if np.all(mask_new == mask): # No progress
            break
            
        out, mask = out_new, mask_new
    return out

def _diffuse_smooth_2d(arr: np.ndarray, n_iter: int = 8, lam: float = 0.25) -> np.ndarray:
    """Simple Laplacian smoothing (diffusion)."""
    a = arr.astype(np.float32, copy=True)
    for _ in range(int(n_iter)):
        up = np.roll(a, 1, axis=0)
        dn = np.roll(a, -1, axis=0)
        lf = np.roll(a, 1, axis=1)
        rt = np.roll(a, -1, axis=1)
        a = (1.0 - 4.0 * lam) * a + lam * (up + dn + lf + rt)
    return a

def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline without SciPy. xk must be sorted ascending."""
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

def _extract_longest_segment_top_bot(layer_labels: np.ndarray, layer_id: int, z_arr: np.ndarray):
    """
    For each (x,y), find top/bottom of the longest contiguous segment where labels==layer_id.
    Returns z_top, z_bot, thickness, valid (all shape nx,ny).
    """
    nx, ny, nz = layer_labels.shape
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

def _auto_pick_layer_id(
    layer_labels: np.ndarray,
    z_arr: np.ndarray,
    rng: np.random.Generator,
    mode: str = "pick_near_interface",
    depth_percentile: float = 0.70,
    interface_percentile: float = 0.70,
    min_voxel_frac: float = 0.002,
    prefer_mid: bool = True,
) -> int:
    """
    Depth-aware auto-pick of layer_id for label-anchored anomalies.

    Why: clustering-based labels do NOT guarantee "layer_id == depth order".
         Old logic picked by value range (vmin..vmax), which can select shallow labels and
         place serpentinization near the top.

    Two modes:
      - "pick_by_depth_percentile": pick label whose representative depth (median-z) is closest
        to z at the given percentile (default 0.70 => deeper).
      - "pick_near_interface": try to pick label close to a conceptual "sediment-basement interface".
        Since we don't have explicit SBI, we approximate interface depth by:
            1) if z_center_m is meaningful to caller -> pass a target separately (we don't here),
               so we fall back to percentile-based target (interface_percentile).
        (In SerpentinizedZone._precompute we pass a target depth derived from params.)
    """
    lab = layer_labels.astype(np.int32, copy=False)
    # vals = np.unique(lab) # slow?
    vals, counts = np.unique(lab, return_counts=True)
    if vals.size == 0:
        return -1

    # Basic filtering: remove tiny clusters (often noise/artifacts from clustering)
    total = float(lab.size)
    min_cnt = max(1.0, float(min_voxel_frac) * total)
    keep = counts.astype(np.float64) >= min_cnt
    vals = vals[keep]
    counts = counts[keep]
    if vals.size == 0:
        # fallback: use original vals if everything was filtered
        vals, counts = np.unique(lab, return_counts=True)

    # Optionally avoid extreme label values (still useful as a weak prior)
    if prefer_mid and vals.size >= 4:
        vmin, vmax = int(vals.min()), int(vals.max())
        mid_keep = (vals > vmin) & (vals < vmax)
        if np.any(mid_keep):
            vals = vals[mid_keep]
            counts = counts[mid_keep]

    # Compute representative depth per label: median z (weighted by voxel counts along z)
    z_arr = np.asarray(z_arr, dtype=np.float32).reshape(-1)
    nz = int(z_arr.size)
    # counts_z[k] = number of voxels of this label at depth index k
    # We compute with a single pass over z slices to keep memory stable.
    rep_depth = np.zeros((vals.size,), dtype=np.float32)
    for i, v in enumerate(vals):
        # histogram along z
        hz = np.zeros((nz,), dtype=np.int64)
        for iz in range(nz):
            hz[iz] = int(np.sum(lab[:, :, iz] == int(v)))
        s = int(hz.sum())
        if s <= 0:
            rep_depth[i] = np.nan
            continue
        cdf = np.cumsum(hz).astype(np.int64)
        mid = s // 2
        idx = int(np.searchsorted(cdf, mid, side="left"))
        idx = int(np.clip(idx, 0, nz - 1))
        rep_depth[i] = float(z_arr[idx])

    # drop NaNs if any
    ok = np.isfinite(rep_depth)
    if not np.any(ok):
        # ultimate fallback: random pick
        return int(rng.choice(vals))
    vals = vals[ok]
    counts = counts[ok]
    rep_depth = rep_depth[ok]

    # Targets
    zmin = float(z_arr[0])
    zmax = float(z_arr[-1])
    depth_percentile = float(np.clip(depth_percentile, 0.05, 0.95))
    interface_percentile = float(np.clip(interface_percentile, 0.05, 0.95))
    z_target_depth = zmin + depth_percentile * (zmax - zmin)
    z_target_if = zmin + interface_percentile * (zmax - zmin)

    mode = str(mode).strip().lower()
    if mode in ("pick_by_depth_percentile", "depth_percentile", "percentile"):
        target = z_target_depth
    else:
        # pick_near_interface (default): target is "interface-like" depth
        target = z_target_if

    # Choose label whose rep_depth is closest to target, but prefer labels with larger support
    # Score = distance penalty + small count bonus (so we avoid tiny clusters)
    dist = np.abs(rep_depth - float(target)).astype(np.float64)
    # normalize counts
    w = (counts.astype(np.float64) / (counts.max() + 1e-9))
    score = dist - 0.15 * (w * (zmax - zmin))  # 0.15 is mild support bonus
    j = int(np.argmin(score))
    return int(vals[j])

def _auto_pick_layer_id_depth_aware(
    layer_labels: np.ndarray,
    z_arr: np.ndarray,
    rng: np.random.Generator,
    mode: str = "pick_near_interface",
    depth_percentile: float = 0.70,
    interface_percentile: float = 0.70,
    min_voxel_frac: float = 0.002,
    prefer_mid: bool = True,
) -> int:
    return _auto_pick_layer_id(
        layer_labels, z_arr, rng, mode, depth_percentile, interface_percentile, min_voxel_frac, prefer_mid
    )

def _fault_likelihood_2d_from_labels(layer_labels: np.ndarray) -> np.ndarray:
    """
    Estimate fault/damage likelihood in XY from label discontinuities.
    Idea: large lateral label gradients indicate faults or strong facies boundaries.
    Returns float32 [0,1] map.
    """
    lab = layer_labels.astype(np.int32, copy=False)
    # mark lateral discontinuities in x/y, accumulate over z
    dx = (lab[1:, :, :] != lab[:-1, :, :]).astype(np.float32)
    dy = (lab[:, 1:, :] != lab[:, :-1, :]).astype(np.float32)
    # pad back to (nx,ny,nz)
    dx = np.pad(dx, ((0,1),(0,0),(0,0)), mode="edge")
    dy = np.pad(dy, ((0,0),(0,1),(0,0)), mode="edge")
    g = dx + dy
    m = g.mean(axis=2).astype(np.float32)
    if float(m.max()) > 1e-9:
        m /= float(m.max())
    return m


def _fault_likelihood_2d_from_vp(vp_bg: np.ndarray) -> np.ndarray:
    """
    Fallback: estimate fault/damage likelihood from Vp lateral gradients (XY), averaged over z.
    Returns float32 [0,1] map.
    """
    vp = vp_bg.astype(np.float32, copy=False)
    gx = np.abs(vp[1:, :, :] - vp[:-1, :, :])
    gy = np.abs(vp[:, 1:, :] - vp[:, :-1, :])
    gx = np.pad(gx, ((0,1),(0,0),(0,0)), mode="edge")
    gy = np.pad(gy, ((0,0),(0,1),(0,0)), mode="edge")
    g = (gx + gy).mean(axis=2).astype(np.float32)
    mx = float(g.max())
    if mx > 1e-9:
        g /= mx
    return g


def _topk_components_mask(bin2d: np.ndarray, topk: int = 1) -> np.ndarray:
    """
    Keep top-k largest connected components (4-neighborhood) in a 2D binary mask.
    Pure numpy BFS; ok for 256^2.
    """
    topk = max(1, int(topk))
    h, w = bin2d.shape
    vis = np.zeros_like(bin2d, dtype=np.uint8)
    comps = []

    for i in range(h):
        for j in range(w):
            if bin2d[i, j] and not vis[i, j]:
                q = [(i, j)]
                vis[i, j] = 1
                pts = []
                while q:
                    x, y = q.pop()
                    pts.append((x, y))
                    if x > 0 and bin2d[x-1, y] and not vis[x-1, y]:
                        vis[x-1, y] = 1; q.append((x-1, y))
                    if x+1 < h and bin2d[x+1, y] and not vis[x+1, y]:
                        vis[x+1, y] = 1; q.append((x+1, y))
                    if y > 0 and bin2d[x, y-1] and not vis[x, y-1]:
                        vis[x, y-1] = 1; q.append((x, y-1))
                    if y+1 < w and bin2d[x, y+1] and not vis[x, y+1]:
                        vis[x, y+1] = 1; q.append((x, y+1))
                comps.append(pts)

    if len(comps) == 0:
        return np.zeros_like(bin2d, dtype=bool)
    comps.sort(key=len, reverse=True)
    keep = comps[:topk]
    out = np.zeros_like(bin2d, dtype=bool)
    for pts in keep:
        for x, y in pts:
            out[x, y] = True
    return out


def _distance_gate_from_seed(
    seed: np.ndarray,
    radius_m: float,
    dx_m: float,
    dy_m: float,
) -> np.ndarray:
    """
    Approximate distance-to-seed gate within radius_m using iterative expansion (Manhattan metric).
    Returns bool mask where dist <= radius.
    """
    if radius_m <= 0:
        return np.ones_like(seed, dtype=bool)
    step_m = max(1e-6, float(min(dx_m, dy_m)))
    steps = int(math.ceil(float(radius_m) / step_m))
    gate = seed.astype(bool, copy=True)
    frontier = gate.copy()
    for _ in range(steps):
        if not frontier.any():
            break
        # 4-neighbor dilation
        up = np.roll(frontier, 1, axis=0)
        dn = np.roll(frontier, -1, axis=0)
        lf = np.roll(frontier, 1, axis=1)
        rt = np.roll(frontier, -1, axis=1)
        new = (up | dn | lf | rt) & (~gate)
        gate |= new
        frontier = new
    return gate


def _wrapped_angle_delta(theta_target: np.ndarray, theta_base: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta_target - theta_base), np.cos(theta_target - theta_base)).astype(np.float32)


def _periodic_line_distance(coord: np.ndarray, spacing_m: float) -> np.ndarray:
    spacing_m = max(float(spacing_m), 1e-6)
    return np.abs(((coord + 0.5 * spacing_m) % spacing_m) - 0.5 * spacing_m).astype(np.float32)


# -------------------------
# Params
# -------------------------

@dataclass
class SerpentinizedZoneParams:
    # Geometry / mode
    mode: str = "corridor"  # "corridor" (ribbon/sheet) | "patchy" (blobs along corridor)
    use_layer_labels: bool = True
    layer_id: int = -1          # >=0 fixed; -1 auto-pick (if labels); otherwise ignored

    # ---- NEW: depth-aware auto-pick (no need to change demo params) ----
    # clustering labels are NOT ordered by depth; these controls make auto-pick stable & deeper.
    layer_pick_mode: str = "pick_near_interface"   # "pick_by_depth_percentile" | "pick_near_interface"
    layer_pick_depth_percentile: float = 0.70      # deeper => closer to basement-like depths
    layer_pick_interface_percentile: float = 0.70  # if no interface target given, use this
    layer_pick_min_voxel_frac: float = 0.002       # filter tiny noisy clusters (0.2% voxels)
    layer_pick_prefer_mid: bool = True             # avoid extreme label values as weak prior

    # ---- NEW: mechanism coupling to fault/damage corridor (demo无需改参数) ----
    fault_coupling_enable: bool = True
    fault_source: str = "labels"                         # "labels" | "vp"
    fault_threshold_quantile: float = 0.995              # likelihood阈值分位（越大越稀疏，更像主断裂）
    fault_topk: int = 1                                  # top-1 或 top-2 主断裂
    fault_gate_radius_m: float = 220.0                   # 走廊半径（只在此范围内允许蛇纹岩化）

    # Corridor footprint (XY)
    center_x_m: float = 1300.0
    center_y_m: float = 1300.0
    corridor_length_m: float = 1800.0
    corridor_halfwidth_m: float = 250.0  # lateral half-width (meters)
    corridor_irregularity: float = 0.25  # 0..0.6 (extra lobes)

    # Depth control
    z_center_m: float = 3500.0      # used when not label-anchored
    thickness_m: float = 350.0      # vertical thickness of altered band
    undulation_amp_m: float = 60.0
    undulation_kmax: int = 3

    # Patchiness (for mode="patchy")
    patch_count: int = 6
    patch_radius_m: float = 220.0
    patch_radius_var: float = 0.35

    # Soft edges
    edge_width_m: float = 60.0
    halo_thickness_m: float = 220.0  # alteration halo around core (labels and properties)

    # Internal block-in-matrix architecture
    matrix_volume_frac: float = 0.60
    block_target_frac: float = 0.32
    block_diameter_min_m: float = 5.0
    block_diameter_max_m: float = 30.0
    resolved_block_count: int = 24
    block_noise_modes: int = 9
    block_wrap_strength_deg: float = 26.0
    foliation_spacing_m: float = 45.0
    foliation_contrast_frac: float = 0.08

    # Mesh-style vein network (represented as subgrid density on coarse grids)
    vein_spacing_m: float = 55.0
    vein_irregularity_frac: float = 0.35
    vein_width_min_m: float = 0.1
    vein_width_max_m: float = 1.0
    vein_intensity_frac: float = 0.55
    vein_serp_boost_frac: float = 0.12

    # Relative alteration degree by subcomponent
    block_serp_degree: float = 0.18
    matrix_serp_degree: float = 1.0
    vein_serp_degree: float = 1.12

    # Property deltas (core)
    # Typical: Vp drop from ~8 km/s to ~5 km/s in mantle contexts
    vp_delta_frac: float = -0.25      # multiply background Vp by (1 + delta) inside core
    rho_delta_frac: float = -0.12     # density decrease
    chi_add_SI: float = 0.02          # susceptibility increase (absolute add)
    resist_delta_frac: float = -0.3   # resistivity decrease due to secondary minerals/fluids

    # Background density in g/cm^3.
    rho_bg_gcc: float = 2.70
    chi_bg_SI: float = 0.001

    # Subtype codes (match your demo style; pick unused range)
    subtype_matrix: int = 40
    subtype_halo: int = 41
    subtype_blocks: int = 42
    subtype_veins: int = 43

    rng_seed: int = 123


# -------------------------
# Main anomaly
# -------------------------

@dataclass
class SerpentinizedZone(Anomaly):
    type: str = "serpentinized_zone"
    strength: float = 0.0
    edge_width_m: float = 60.0

    params: SerpentinizedZoneParams = field(default_factory=SerpentinizedZoneParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)

    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    _pre: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(int(self.params.rng_seed))

    # ---- public ----
    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        state = self._compute_state(X=X, Y=Y, Z=Z, vp_bg=None, need_internal=False)
        return state["m_core"]

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        return self.apply_properties({"vp": vp}, X, Y, Z)["vp"]

    def apply_properties(self, props_dict: dict, X, Y, Z) -> dict:
        """
        Unified property interface applying serpentinization effects on vp, rho, chi, and resist.
        """
        state = self._compute_state(X=X, Y=Y, Z=Z, vp_bg=props_dict.get("vp"), need_internal=True)
        out_props = {}
        for k, v in props_dict.items():
            if k in {"vp", "rho", "chi", "resist"}:
                out_props[k] = self._blend_property(k, np.asarray(v, dtype=np.float32), state)
            else:
                out_props[k] = v.copy()
        return out_props

    def build_property_models(
        self,
        X, Y, Z,
        vp_bg: Optional[np.ndarray] = None,
        rho_bg_gcc: Optional[np.ndarray] = None,
        chi_bg_SI: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create density (g/cm^3) and susceptibility volumes and a subtype label volume.
        If vp_bg is provided, the same core mask can also be used to output a modified vp (optional).
        """
        p = self.params
        nx, ny, nz = Z.shape

        if rho_bg_gcc is None:
            rho_bg = np.full((nx, ny, nz), float(p.rho_bg_gcc), dtype=np.float32)
        else:
            rho_bg = rho_bg_gcc.astype(np.float32, copy=False)
            if float(np.nanmax(rho_bg)) > 50.0:
                rho_bg = rho_bg / 1000.0

        if chi_bg_SI is None:
            chi_bg = np.full((nx, ny, nz), float(p.chi_bg_SI), dtype=np.float32)
        else:
            chi_bg = chi_bg_SI.astype(np.float32, copy=False)

        state = self._compute_state(X=X, Y=Y, Z=Z, vp_bg=vp_bg, need_internal=True)
        rho = self._blend_property("rho", rho_bg, state)
        chi = self._blend_property("chi", chi_bg, state)
        sub = self._build_subtype_volume(state)

        out = {
            "rho": rho.astype(np.float32),
            "rho_gcc": rho.astype(np.float32),
            "chi_SI": chi.astype(np.float32),
            "subtype": sub,
            "mask_core": state["m_core"].astype(np.float32),
            "mask_halo": state["m_halo"].astype(np.float32),
        }
        if vp_bg is not None:
            out["vp_mps"] = self._blend_property("vp", vp_bg.astype(np.float32, copy=False), state).astype(np.float32)
        return out

    # ---- internal ----
    def _compute_state(self, X, Y, Z, vp_bg: Optional[np.ndarray] = None, need_internal: bool = True):
        p = self.params
        rng = self._rng
        nx, ny, nz = Z.shape

        x_arr = X[:, 0, 0].astype(np.float32, copy=False)
        y_arr = Y[0, :, 0].astype(np.float32, copy=False)
        z_arr = Z[0, 0, :].astype(np.float32, copy=False)
        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        if not self._pre:
            self._pre = self._precompute(p, X2d, Y2d, z_arr, rng, vp_bg=vp_bg, x_arr=x_arr, y_arr=y_arr)

        wd = max(float(p.edge_width_m), 1e-3)
        halo_t = max(float(p.halo_thickness_m), 1e-3)

        m_core_vol = np.empty((nx, ny, nz), dtype=np.float32)
        m_halo_vol = np.zeros((nx, ny, nz), dtype=np.float32)
        halo_weight_vol = np.zeros((nx, ny, nz), dtype=np.float32)

        if need_internal:
            matrix_weight_vol = np.zeros((nx, ny, nz), dtype=np.float32)
            block_weight_vol = np.zeros((nx, ny, nz), dtype=np.float32)
            vein_weight_vol = np.zeros((nx, ny, nz), dtype=np.float32)
            foliation_vol = np.zeros((nx, ny, nz), dtype=np.float32)
        else:
            matrix_weight_vol = None
            block_weight_vol = None
            vein_weight_vol = None
            foliation_vol = None

        for iz in range(nz):
            z_m = float(z_arr[iz])
            sdf_core = self._sdf_core_slice(p, X2d, Y2d, z_m, iz, self._pre)
            sdf_core = np.clip(sdf_core, -1e6, 1e6)

            m_core = _sigmoid_stable(sdf_core / wd)
            m_core_vol[..., iz] = m_core

            m_out = _sigmoid_stable(-sdf_core / wd)
            m_near = _sigmoid_stable((sdf_core + halo_t) / wd)
            m_halo = m_out * m_near
            m_halo_vol[..., iz] = m_halo

            u = np.clip((sdf_core + halo_t) / halo_t, 0.0, 1.0)
            halo_decay = (u * u * (3.0 - 2.0 * u)) * (1.0 - m_core)
            halo_weight_vol[..., iz] = halo_decay.astype(np.float32)

            if need_internal:
                matrix_w, block_w, vein_w, foliation = self._internal_components_slice(
                    X2d, Y2d, z_m, iz, m_core, self._pre, p
                )
                matrix_weight_vol[..., iz] = matrix_w
                block_weight_vol[..., iz] = block_w
                vein_weight_vol[..., iz] = vein_w
                foliation_vol[..., iz] = foliation

        state = {
            "m_core": m_core_vol,
            "m_halo": m_halo_vol,
            "halo_weight": halo_weight_vol,
        }
        if need_internal:
            state.update(
                {
                    "matrix_w": matrix_weight_vol,
                    "block_w": block_weight_vol,
                    "vein_w": vein_weight_vol,
                    "foliation": foliation_vol,
                }
            )
        return state

    def _blend_property(self, key: str, base: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
        p = self.params
        bg = np.asarray(base, dtype=np.float32)
        m_core = state["m_core"]
        halo_w = state["halo_weight"]
        matrix_w = state.get("matrix_w")
        block_w = state.get("block_w")
        vein_w = state.get("vein_w")
        foliation = state.get("foliation")

        if matrix_w is None or block_w is None or vein_w is None:
            matrix_w = m_core
            block_w = np.zeros_like(m_core, dtype=np.float32)
            vein_w = np.zeros_like(m_core, dtype=np.float32)
            foliation = np.full_like(m_core, 0.5, dtype=np.float32)

        matrix_degree = float(p.matrix_serp_degree) * (
            1.0 + float(p.foliation_contrast_frac) * (2.0 * np.asarray(foliation, dtype=np.float32) - 1.0)
        )
        block_degree = np.full_like(matrix_degree, float(p.block_serp_degree), dtype=np.float32)
        vein_degree = np.full_like(matrix_degree, float(p.vein_serp_degree), dtype=np.float32)

        background_w = np.clip(1.0 - m_core - halo_w, 0.0, 1.0)

        if key == "chi":
            delta = float(p.chi_add_SI)
            halo_prop = bg + delta * halo_w
            matrix_prop = bg + delta * matrix_degree
            block_prop = bg + delta * block_degree
            vein_prop = bg + delta * vein_degree
        else:
            if key == "vp":
                delta = float(p.vp_delta_frac)
            elif key == "rho":
                delta = float(p.rho_delta_frac)
            elif key == "resist":
                delta = float(p.resist_delta_frac)
            else:
                return bg.copy()

            halo_prop = bg * (1.0 + delta * halo_w)
            matrix_prop = bg * (1.0 + delta * matrix_degree)
            block_prop = bg * (1.0 + delta * block_degree)
            vein_prop = bg * (1.0 + delta * vein_degree)

        out = (
            bg * background_w
            + halo_prop * halo_w
            + matrix_prop * matrix_w
            + block_prop * block_w
            + vein_prop * vein_w
        )
        return out.astype(np.float32, copy=False)

    def _build_subtype_volume(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        p = self.params
        sub = np.zeros_like(state["m_core"], dtype=np.int32)
        sub[state["m_halo"] > 0.5] = int(p.subtype_halo)

        core_active = state["m_core"] > 0.5
        matrix_w = state["matrix_w"]
        block_w = state["block_w"]
        vein_w = state["vein_w"]
        m_core = np.maximum(state["m_core"], 1e-6)

        matrix_pick = core_active
        block_pick = core_active & (block_w >= 0.28 * m_core)
        vein_pick = core_active & (vein_w >= 0.12 * m_core) & (~block_pick)

        sub[matrix_pick] = int(p.subtype_matrix)
        sub[block_pick] = int(p.subtype_blocks)
        sub[vein_pick] = int(p.subtype_veins)
        return sub

    def _internal_components_slice(self, X2d, Y2d, z_m: float, iz: int, m_core: np.ndarray, pre, p):
        block_frac = self._block_fraction_slice(X2d, Y2d, z_m, pre, p)
        block_frac = np.clip(block_frac, 0.0, 0.92)

        theta_eff = np.asarray(pre["theta_xy"], dtype=np.float32).copy()
        wrap_strength = math.radians(float(p.block_wrap_strength_deg))
        for block in pre.get("resolved_blocks", []):
            dz_rel = (z_m - float(block["cz"])) / max(float(block["rz"]), 1e-3)
            if abs(dz_rel) > 1.4:
                continue
            dx = (X2d - float(block["cx"])).astype(np.float32)
            dy = (Y2d - float(block["cy"])).astype(np.float32)
            r2 = dx * dx + dy * dy
            influence_scale = max(float(block["rx"]), float(block["ry"])) * 2.4
            influence = np.exp(-r2 / max(influence_scale * influence_scale, 1e-3)).astype(np.float32)
            influence *= float(np.exp(-0.7 * dz_rel * dz_rel))
            theta_target = np.arctan2(dy, dx).astype(np.float32) + 0.5 * np.pi
            theta_eff += wrap_strength * influence * _wrapped_angle_delta(theta_target, theta_eff)

        s_coord = np.cos(theta_eff) * X2d + np.sin(theta_eff) * Y2d
        t_coord = -np.sin(theta_eff) * X2d + np.cos(theta_eff) * Y2d

        fol_spacing = max(float(p.foliation_spacing_m), 1e-3)
        foliation = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * (t_coord / fol_spacing + 0.18 * np.sin((s_coord + 0.4 * z_m) / max(1.8 * fol_spacing, 1e-3)))
        )
        foliation = np.clip(foliation.astype(np.float32), 0.0, 1.0)

        vein_density = self._vein_density_slice(X2d, Y2d, z_m, s_coord, t_coord, theta_eff, pre, p)
        matrix_available = np.clip(1.0 - block_frac, 0.0, 1.0)
        vein_weight = m_core * matrix_available * np.clip(float(p.vein_intensity_frac) * vein_density, 0.0, 0.85)
        block_weight = m_core * block_frac
        matrix_weight = np.clip(m_core - block_weight - vein_weight, 0.0, 1.0)
        return (
            matrix_weight.astype(np.float32),
            block_weight.astype(np.float32),
            vein_weight.astype(np.float32),
            foliation.astype(np.float32),
        )

    def _block_fraction_slice(self, X2d, Y2d, z_m: float, pre, p):
        plane_shape = (X2d.shape[0], Y2d.shape[1])
        block_resolved = np.zeros(plane_shape, dtype=np.float32)
        for block in pre.get("resolved_blocks", []):
            dx = (X2d - float(block["cx"])).astype(np.float32)
            dy = (Y2d - float(block["cy"])).astype(np.float32)
            dz = np.float32(z_m - float(block["cz"]))
            ca = float(block["cos"])
            sa = float(block["sin"])
            xr = ca * dx + sa * dy
            yr = -sa * dx + ca * dy
            rr = (
                (xr / max(float(block["rx"]), 1e-3)) ** 2
                + (yr / max(float(block["ry"]), 1e-3)) ** 2
                + (dz / max(float(block["rz"]), 1e-3)) ** 2
            )
            field = np.clip(1.0 - rr, 0.0, 1.0).astype(np.float32)
            block_resolved = np.maximum(block_resolved, field)

        spectral = np.zeros(plane_shape, dtype=np.float32)
        amp_sum = 1e-6
        for mode in pre.get("block_modes", []):
            arg = (
                float(mode["kx"]) * X2d
                + float(mode["ky"]) * Y2d
                + float(mode["kz"]) * z_m
                + float(mode["phase"])
            )
            spectral += float(mode["amp"]) * np.cos(arg).astype(np.float32)
            amp_sum += abs(float(mode["amp"]))
        spectral = 0.5 + 0.5 * spectral / amp_sum
        block_noise = _sigmoid_stable((spectral - float(pre.get("block_threshold", 0.62))) / 0.08)

        block_frac = np.maximum(block_resolved, 0.75 * block_noise)
        return np.clip(block_frac, 0.0, 1.0).astype(np.float32)

    def _vein_density_slice(self, X2d, Y2d, z_m: float, s_coord, t_coord, theta_eff, pre, p):
        dx_m = float(pre.get("dx_m", 1.0))
        dy_m = float(pre.get("dy_m", 1.0))
        min_vis_width = 0.15 * min(dx_m, dy_m)
        width_eff = max(float(np.random.default_rng(int(p.rng_seed) + int(z_m // max(pre.get("dz_m", 1.0), 1.0))).uniform(
            p.vein_width_min_m, p.vein_width_max_m
        )), min_vis_width)

        spacing1 = max(float(p.vein_spacing_m), 1e-3)
        spacing2 = max(spacing1 * 0.78, 1e-3)
        spacing3 = max(spacing1 * 1.18, 1e-3)
        irr = float(p.vein_irregularity_frac)
        warp1 = irr * 0.35 * spacing1 * np.sin((0.85 * X2d + 1.15 * Y2d + 0.3 * z_m) / max(0.9 * spacing1, 1e-3))
        warp2 = irr * 0.28 * spacing2 * np.cos((1.10 * X2d - 0.70 * Y2d + 0.45 * z_m) / max(1.1 * spacing2, 1e-3))
        oblique = 0.7 * s_coord + 1.1 * t_coord + 0.2 * np.sin(theta_eff) * spacing3

        d1 = _periodic_line_distance(s_coord + warp1, spacing1)
        d2 = _periodic_line_distance(t_coord + warp2, spacing2)
        d3 = _periodic_line_distance(oblique, spacing3)
        soft = max(width_eff * 0.35, 0.25)
        vein1 = _sigmoid_stable((width_eff - d1) / soft)
        vein2 = _sigmoid_stable((width_eff - d2) / soft)
        vein3 = _sigmoid_stable((0.75 * width_eff - d3) / max(0.75 * soft, 0.2))
        vein_density = np.maximum.reduce([vein1, vein2, vein3])
        return np.clip(vein_density.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _polyline_tangent_angle_map(X2d, Y2d, xs, ys):
        dx = X2d[..., None] - xs[None, None, :]
        dy = Y2d[..., None] - ys[None, None, :]
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2, axis=2)
        tx = np.gradient(xs.astype(np.float32))
        ty = np.gradient(ys.astype(np.float32))
        theta = np.arctan2(ty, tx).astype(np.float32)
        return theta[idx].astype(np.float32)

    def _precompute(self, p, X2d, Y2d, z_arr, rng, vp_bg=None, x_arr=None, y_arr=None):
        # Decide anchor layer if requested and available
        use_labels = bool(p.use_layer_labels) and (self.layer_labels is not None)
        layer_id = int(p.layer_id)

        if use_labels and layer_id < 0:
            # Depth-aware pick: avoid shallow labels produced by clustering.
            layer_id = _auto_pick_layer_id_depth_aware(
                self.layer_labels,
                z_arr=z_arr,
                rng=rng,
                mode=str(p.layer_pick_mode),
                depth_percentile=float(p.layer_pick_depth_percentile),
                interface_percentile=float(p.layer_pick_interface_percentile),
                min_voxel_frac=float(p.layer_pick_min_voxel_frac),
                prefer_mid=bool(p.layer_pick_prefer_mid),
            )

        pre: Dict[str, Any] = {"use_labels": use_labels, "layer_id": layer_id}

        # --- Mechanism gate: fault/damage corridor in XY (optional) ---
        if bool(p.fault_coupling_enable):
            if str(p.fault_source).lower() == "vp" and vp_bg is not None:
                lk = _fault_likelihood_2d_from_vp(vp_bg)
            elif self.layer_labels is not None:
                 lk = _fault_likelihood_2d_from_labels(self.layer_labels)
            elif vp_bg is not None:
                 lk = _fault_likelihood_2d_from_vp(vp_bg)
            else:
                 lk = np.zeros(X2d.shape, dtype=np.float32)

            q = float(np.clip(p.fault_threshold_quantile, 0.90, 0.9999))
            thr = float(np.quantile(lk, q))
            seeds = (lk >= thr)
            seeds = _topk_components_mask(seeds, topk=int(p.fault_topk))
            
            dx_m = 1.0
            dy_m = 1.0
            if x_arr is not None and x_arr.size > 1:
                dx_m = float(x_arr[1] - x_arr[0])
            if y_arr is not None and y_arr.size > 1:
                dy_m = float(y_arr[1] - y_arr[0])
                
            gate = _distance_gate_from_seed(seeds, float(p.fault_gate_radius_m), dx_m=dx_m, dy_m=dy_m)
            pre["fault_gate_xy"] = gate.astype(bool)
        else:
            pre["fault_gate_xy"] = None

        # 1) Build a curvilinear centerline in XY (polyline + Catmull-Rom)
        # Parameter s in [0,1]
        nkn = 6
        L = float(p.corridor_length_m)
        xc, yc = float(p.center_x_m), float(p.center_y_m)
        # Create control points with gentle meander
        s_kn = np.linspace(0.0, 1.0, nkn)
        # Base direction: random azimuth
        theta = rng.uniform(0.0, 2*np.pi)
        dx_dir, dy_dir = np.cos(theta), np.sin(theta)
        # Perp
        px, py = -dy_dir, dx_dir
        # Control points
        xk = xc + (s_kn - 0.5) * L * dx_dir
        yk = yc + (s_kn - 0.5) * L * dy_dir
        # Add lateral meander
        meander = (0.18 * L) * rng.normal(0.0, 1.0, size=nkn)
        xk = xk + meander * px
        yk = yk + meander * py

        # Sample polyline dense for distance computation
        ns = 160
        s = np.linspace(0.0, 1.0, ns).astype(np.float32)
        xs = _catmull_rom_1d(s_kn, xk, s).astype(np.float32)
        ys = _catmull_rom_1d(s_kn, yk, s).astype(np.float32)

        pre["xs"] = xs
        pre["ys"] = ys
        pre["theta_xy"] = self._polyline_tangent_angle_map(X2d, Y2d, xs, ys)

        # 2) Depth function zc(x,y)
        nx, ny = X2d.shape[0], Y2d.shape[1]
        kmax = max(int(p.undulation_kmax), 1)
        noise = np.zeros((nx, ny), dtype=np.float32)
        # Low-frequency cosine noise
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                a = rng.normal(0.0, 1.0)
                phi = rng.uniform(0.0, 2*np.pi)
                xx = X2d / max(float(p.corridor_length_m), 1e-6)
                yy = Y2d / max(float(p.corridor_length_m), 1e-6)
                noise += (a * np.cos(2*np.pi*(kx*xx + ky*yy) + phi)).astype(np.float32)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)

        if use_labels and layer_id >= 0:
            z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, layer_id, z_arr)
            # put serpentinization near middle of that layer, plus undulation
            zc = (0.5 * (z_top + z_bot) + float(p.undulation_amp_m) * noise).astype(np.float32)
            
            # --- Nearest fill plus diffuse smoothing ---
            # Step 1: Set invalid to nan
            zc = np.where(valid, zc, np.nan)
            
            # Step 2: Nearest fill
            zc = _nearest_fill_2d(zc, valid)
            
            # Step 3: Diffuse smooth (optional but recommended)
            zc = _diffuse_smooth_2d(zc, n_iter=8, lam=0.20)
            
            # Step 4: Physical clip
            z_min_bound = float(z_arr[0]) + 2.0 * float(p.edge_width_m)
            z_max_bound = float(z_arr[-1]) - 2.0 * float(p.edge_width_m)
            zc = np.clip(zc, z_min_bound, z_max_bound)

            pre["zc"] = zc
            pre["valid_layer"] = valid
        else:
            pre["zc"] = (float(p.z_center_m) + float(p.undulation_amp_m) * noise).astype(np.float32)
            pre["valid_layer"] = None

        # --- Debug info (based on user request) ---
        zc = pre["zc"]
        valid = pre["valid_layer"]
        if valid is not None:
            coverage = valid.mean()
            print(f"[SerpentinizedZone Debug] valid_layer coverage: {coverage:.4f}")
            print(f"[SerpentinizedZone Debug] zc (all) min/max: {zc.min():.1f} / {zc.max():.1f}")
            if valid.any():
                zcv = zc[valid]
                print(f"[SerpentinizedZone Debug] zc (valid) min/max/mean: {zcv.min():.1f} / {zcv.max():.1f} / {zcv.mean():.1f}")
            if (~valid).any():
                zci = zc[~valid]
                print(f"[SerpentinizedZone Debug] zc (invalid) min/max/mean: {zci.min():.1f} / {zci.max():.1f} / {zci.mean():.1f}")
        else:
            print(f"[SerpentinizedZone Debug] No layer validation used. zc min/max: {zc.min():.1f} / {zc.max():.1f}")
        # ------------------------------------------

        # 3) Build an XY window SDF around the centerline:
        #    sdf_xy = halfwidth - distance_to_polyline
        dist_xy = self._distance_to_polyline_xy(X2d, Y2d, xs, ys)
        base_hw = float(p.corridor_halfwidth_m)

        # add window irregularity: union with a few ellipses along the line
        sdf_xy = (base_hw - dist_xy).astype(np.float32)
        n_extra = int(round(float(p.corridor_irregularity) * 6.0))
        if n_extra > 0:
            for _ in range(n_extra):
                j = int(rng.integers(0, len(xs)))
                ex, ey = float(xs[j]), float(ys[j])
                a = base_hw * rng.uniform(0.8, 1.6)
                b = base_hw * rng.uniform(0.6, 1.3)
                sdf_e = self._sdf_ellipse_pos(X2d, Y2d, ex, ey, a, b)
                sdf_xy = np.maximum(sdf_xy, sdf_e)

        pre["sdf_xy"] = sdf_xy

        dx_m = float(x_arr[1] - x_arr[0]) if (x_arr is not None and x_arr.size > 1) else 1.0
        dy_m = float(y_arr[1] - y_arr[0]) if (y_arr is not None and y_arr.size > 1) else 1.0
        dz_m = float(z_arr[1] - z_arr[0]) if z_arr.size > 1 else 1.0
        pre["dx_m"] = dx_m
        pre["dy_m"] = dy_m
        pre["dz_m"] = dz_m

        block_modes = []
        n_modes = max(3, int(p.block_noise_modes))
        lam_min = max(1.5 * min(dx_m, dy_m), float(p.block_diameter_min_m) * 1.25)
        lam_max = max(lam_min + 1.0, float(p.block_diameter_max_m) * 2.8)
        for _ in range(n_modes):
            lam_xy = rng.uniform(lam_min, lam_max)
            az = rng.uniform(0.0, 2.0 * np.pi)
            kx = (2.0 * np.pi / max(lam_xy, 1e-6)) * np.cos(az)
            ky = (2.0 * np.pi / max(lam_xy, 1e-6)) * np.sin(az)
            lam_z = rng.uniform(max(1.2 * dz_m, 18.0), max(float(p.block_diameter_max_m) * 2.4, 30.0))
            kz = rng.choice([-1.0, 1.0]) * (2.0 * np.pi / max(lam_z, 1e-6))
            block_modes.append(
                {
                    "amp": float(rng.uniform(0.7, 1.2)),
                    "kx": float(kx),
                    "ky": float(ky),
                    "kz": float(kz),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
        pre["block_modes"] = block_modes
        pre["block_threshold"] = float(np.clip(0.58 + 0.20 * float(p.matrix_volume_frac), 0.55, 0.78))

        resolved_blocks = []
        block_count = max(8, int(p.resolved_block_count))
        for _ in range(block_count):
            j = int(rng.integers(0, len(xs)))
            cx = float(xs[j] + rng.normal(0.0, 0.35 * max(float(p.corridor_halfwidth_m), dx_m)))
            cy = float(ys[j] + rng.normal(0.0, 0.35 * max(float(p.corridor_halfwidth_m), dy_m)))
            ix = int(np.clip(round((cx - float(X2d[0, 0])) / max(dx_m, 1e-6)), 0, X2d.shape[0] - 1))
            iy = int(np.clip(round((cy - float(Y2d[0, 0])) / max(dy_m, 1e-6)), 0, Y2d.shape[1] - 1))
            cz_local = float(pre["zc"][ix, iy]) + rng.normal(0.0, 0.18 * float(p.thickness_m))
            diam = float(rng.uniform(p.block_diameter_min_m, p.block_diameter_max_m))
            rx = max(0.5 * diam * rng.uniform(0.7, 1.25), 0.45 * dx_m)
            ry = max(0.5 * diam * rng.uniform(0.7, 1.55), 0.45 * dy_m)
            rz = max(0.35 * diam * rng.uniform(0.6, 1.10), 0.35 * dz_m)
            az = float(pre["theta_xy"][ix, iy] + rng.normal(0.0, 0.45))
            resolved_blocks.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "cz": float(np.clip(cz_local, float(z_arr[0]), float(z_arr[-1]))),
                    "rx": float(rx),
                    "ry": float(ry),
                    "rz": float(rz),
                    "az": az,
                    "cos": float(np.cos(az)),
                    "sin": float(np.sin(az)),
                }
            )
        pre["resolved_blocks"] = resolved_blocks

        # 4) Patchiness seeds (optional)
        if p.mode == "patchy":
            pc = int(p.patch_count)
            jidx = rng.integers(0, len(xs), size=pc)
            centers = [(float(xs[j]), float(ys[j])) for j in jidx]
            pre["patch_centers"] = centers

        return pre

    def _sdf_core_slice(self, p, X2d, Y2d, z_m, iz, pre):
        # Vertical band sdf_z
        zc = pre["zc"]
        half_t = 0.5 * float(p.thickness_m)
        sdf_z = (half_t - np.abs(z_m - zc)).astype(np.float32)

        # Lateral corridor sdf_xy
        sdf_xy = pre["sdf_xy"]

        sdf = np.minimum(sdf_z, sdf_xy)
        
        # --- Fault Mechanism Gate ---
        gate_xy = pre.get("fault_gate_xy", None)
        if gate_xy is not None:
             sdf = np.where(gate_xy, sdf, -1e6)

        # If label-anchored: apply a hard gate to invalid regions.
        # Note: We rely on the smoothed 'zc' for the shape, not the voxel-level labels, 
        # so we don't check 'in_layer' anymore. We simply gate columns that were totally invalid.
        valid = pre.get("valid_layer", None)
        if valid is not None:
             sdf = np.where(valid, sdf, -1e6)

        # Patchiness: keep only blobs along corridor
        if p.mode == "patchy":
            sdf = np.minimum(sdf, self._sdf_patchiness(X2d, Y2d, pre, p))

        return sdf

    def _sdf_patchiness(self, X2d, Y2d, pre, p):
        centers = pre.get("patch_centers", [])
        if not centers:
            return np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        rng = self._rng
        r0 = float(p.patch_radius_m)
        rv = float(p.patch_radius_var)
        sdf_u = np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        for (cx, cy) in centers:
            rr = r0 * rng.uniform(1.0 - rv, 1.0 + rv)
            dx = (X2d - cx).astype(np.float32)
            dy = (Y2d - cy).astype(np.float32)
            r = np.sqrt(dx*dx + dy*dy)
            sdf = (rr - r).astype(np.float32)
            sdf_u = np.maximum(sdf_u, sdf)
        return sdf_u

    @staticmethod
    def _distance_to_polyline_xy(X2d, Y2d, xs, ys):
        """
        Approx distance from each (x,y) to a polyline defined by points (xs,ys).
        Uses nearest point among dense samples (fast enough for nx~256,ny~256, ns~160).
        """
        # Shape: (nx,ny,ns)
        dx = X2d[..., None] - xs[None, None, :]
        dy = Y2d[..., None] - ys[None, None, :]
        d2 = dx*dx + dy*dy
        return np.sqrt(np.min(d2, axis=2)).astype(np.float32)

    @staticmethod
    def _sdf_ellipse_pos(X2d, Y2d, xc, yc, a, b):
        dx = (X2d - float(xc)).astype(np.float32)
        dy = (Y2d - float(yc)).astype(np.float32)
        rr = np.sqrt((dx / max(float(a), 1e-6)) ** 2 + (dy / max(float(b), 1e-6)) ** 2)
        q = (1.0 - rr)
        return (q * float(min(a, b))).astype(np.float32)
