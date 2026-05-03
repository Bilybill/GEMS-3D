#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Deepwave shot gathers saved as NPZ.

Expected NPZ content (your saving code):
np.savez(
    save_path,
    data=receiver_data.cpu().detach().numpy(),    # [S, R, T]
    src=source_locations.cpu().detach().numpy(),  # [S, nsrc, dim]
    rec=receiver_locations.cpu().detach().numpy(),# [S, R, dim]
    dt=np.array(dt),
    dx=np.array(dx)
)

What you get
- Interactive shot slider
- Shot gather image (time vs offset or receiver index), optional AGC + percentile clipping
- Geometry (shots + receivers for current shot)
- CMP fold/coverage heatmap (midpoint histogram)
- Offset histogram (current shot)
- Optional wiggle view (press "w" or click button)

Run
  python visualize_npz_gathers.py --file outputs/run1.npz
Optional
  python visualize_npz_gathers.py --file outputs/run1.npz --sort_by_offset --agc_ms 300 --save_dir figs
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, CheckButtons, Button


def _as_float(x) -> float:
    arr = np.array(x).reshape(-1)
    return float(arr[0])


def _infer_dim_from_src(src: np.ndarray) -> int:
    if src.ndim != 3:
        raise ValueError(f"Expected src shape [S, nsrc, dim], got {src.shape}")
    return int(src.shape[-1])


def _compute_offset_azimuth(src0: np.ndarray, rec: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    src0: [dim] shot location (grid indices)
    rec : [R, dim] receiver locations (grid indices)
    dx  : [dim] spacing in meters

    Returns:
      offset_m: [R]
      azimuth_deg: [R] (3D only; 2D returns zeros)
    """
    dim = src0.shape[-1]
    d = (rec.astype(np.float64) - src0.astype(np.float64)) * np.asarray(dx, dtype=np.float64)
    if dim == 2:
        offset = np.abs(d[:, 0])
        az = np.zeros_like(offset)
    else:
        offset = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return offset, az


def _running_rms_agc(x: np.ndarray, win: int) -> np.ndarray:
    """
    x: [R, T]
    win: window length in samples
    Returns AGC-scaled x (per trace)
    """
    if win <= 1:
        return x
    R, T = x.shape
    win = min(win, T)

    xx = x.astype(np.float64) ** 2
    c = np.cumsum(xx, axis=1)
    c = np.concatenate([np.zeros((R, 1), dtype=np.float64), c], axis=1)  # [R, T+1]
    s = c[:, win:] - c[:, :-win]  # [R, T+1-win]
    rms = np.sqrt(np.maximum(s / win, 1e-12))

    pad_left = win // 2
    pad_right = T - rms.shape[1] - pad_left
    rms_full = np.pad(rms, ((0, 0), (pad_left, pad_right)), mode="edge")
    return x / rms_full


def _clip_symmetric(x: np.ndarray, pct: float) -> float:
    pct = float(np.clip(pct, 50.0, 100.0))
    vmax = np.percentile(np.abs(x), pct)
    if vmax <= 0:
        vmax = float(np.max(np.abs(x)) + 1e-12)
    return float(vmax)


def _wiggle(ax, t: np.ndarray, x_axis: np.ndarray, gather: np.ndarray, max_traces: int = 200, scale: float | None = None):
    """
    Simple wiggle plot: gather is [R, T]; t is [T]; x_axis is [R] (offset or index)
    """
    R, T = gather.shape
    if R > max_traces:
        idx = np.linspace(0, R - 1, max_traces).astype(int)
        g = gather[idx]
        x = x_axis[idx]
    else:
        g = gather
        x = x_axis

    if scale is None:
        if len(x) > 1:
            dx_med = np.median(np.diff(np.sort(x)))
            dx_med = dx_med if dx_med > 0 else 1.0
        else:
            dx_med = 1.0
        peak = np.percentile(np.abs(g), 99)
        peak = peak if peak > 0 else (np.max(np.abs(g)) + 1e-12)
        scale = 0.5 * dx_med / peak

    for i in range(g.shape[0]):
        tr = g[i] * scale
        ax.plot(x[i] + tr, t, linewidth=0.8)
        ax.fill_betweenx(t, x[i], x[i] + np.maximum(tr, 0), alpha=0.2)

    ax.set_ylim(t[-1], t[0])
    ax.set_ylabel("time (s)")


def _cmp_fold_heatmap(src: np.ndarray, rec: np.ndarray, dx: np.ndarray,
                      bin_x_m: float, bin_y_m: float,
                      max_shots: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CMP (midpoint) fold heatmap in XY.
    Returns (fold, x_edges, y_edges). For 2D returns a 1-row fold map.
    """
    S = src.shape[0]
    dim = src.shape[-1]

    if max_shots is not None and S > max_shots:
        sel = np.linspace(0, S - 1, max_shots).astype(int)
        src = src[sel]
        rec = rec[sel]

    s0 = src[:, 0, :]  # [S, dim]

    if dim == 2:
        mx = (s0[:, None, 0] + rec[:, :, 0]) / 2.0 * dx[0]
        mx = mx.reshape(-1)
        x_min, x_max = float(mx.min()), float(mx.max())
        x_edges = np.arange(x_min, x_max + bin_x_m, bin_x_m)
        h, _ = np.histogram(mx, bins=x_edges)
        fold = h[np.newaxis, :]
        return fold, x_edges, np.array([0.0, 1.0])

    mx = (s0[:, None, 0] + rec[:, :, 0]) / 2.0 * dx[0]
    my = (s0[:, None, 1] + rec[:, :, 1]) / 2.0 * dx[1]
    mx = mx.reshape(-1)
    my = my.reshape(-1)

    x_min, x_max = float(mx.min()), float(mx.max())
    y_min, y_max = float(my.min()), float(my.max())

    x_edges = np.arange(x_min, x_max + bin_x_m, bin_x_m)
    y_edges = np.arange(y_min, y_max + bin_y_m, bin_y_m)

    H, y_edges2, x_edges2 = np.histogram2d(my, mx, bins=[y_edges, x_edges])
    return H, x_edges2, y_edges2


def _detect_fixed_receivers(rec: np.ndarray) -> bool:
    if rec.shape[0] <= 1:
        return True
    return np.all(rec == rec[0:1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to .npz file containing data/src/rec/dt/dx")
    ap.add_argument("--shot", type=int, default=0, help="Initial shot index")
    ap.add_argument("--sort_by_offset", action="store_true", help="Sort receivers by offset for display")
    ap.add_argument("--agc_ms", type=float, default=0.0, help="AGC window in milliseconds (0 disables)")
    ap.add_argument("--clip_pct", type=float, default=99.0, help="Percentile for symmetric clipping (50-100)")
    ap.add_argument("--cmp_bin_x_m", type=float, default=40.0, help="CMP bin size in X (meters)")
    ap.add_argument("--cmp_bin_y_m", type=float, default=40.0, help="CMP bin size in Y (meters)")
    ap.add_argument("--cmp_max_shots", type=int, default=0, help="Subsample shots for CMP (0=use all)")
    ap.add_argument("--wiggle", action="store_true", help="Open an extra wiggle window initially")
    ap.add_argument("--save_dir", default="", help="If set, 'Save Figure' button writes PNGs here")
    args = ap.parse_args()

    npz = np.load(args.file, allow_pickle=True)
    data = np.asarray(npz["data"])  # [S,R,T]
    src = np.asarray(npz["src"])
    rec = np.asarray(npz["rec"])
    dt = _as_float(npz["dt"])
    dx = np.asarray(npz["dx"], dtype=np.float64).reshape(-1)

    if data.ndim != 3:
        raise ValueError(f"Expected data shape [S,R,T], got {data.shape}")
    S, R, T = data.shape
    dim = _infer_dim_from_src(src)
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dim={dim} from src shape {src.shape}")

    t = np.arange(T, dtype=np.float64) * dt

    fixed_receivers = _detect_fixed_receivers(rec)
    rec_mode = "fixed" if fixed_receivers else "variable (rolling/patch)"

    print(f"[Loaded] file={args.file}")
    print(f"  data: {data.shape}  (S={S}, R={R}, T={T})")
    print(f"  src : {src.shape}")
    print(f"  rec : {rec.shape}  -> {rec_mode}")
    print(f"  dt  : {dt}")
    print(f"  dx  : {dx.tolist()}")

    cmp_max = None if args.cmp_max_shots <= 0 else int(args.cmp_max_shots)
    print("[CMP] Computing fold heatmap ...")
    fold, x_edges, y_edges = _cmp_fold_heatmap(src, rec, dx, args.cmp_bin_x_m, args.cmp_bin_y_m, max_shots=cmp_max)
    print(f"[CMP] fold shape: {fold.shape}")

    # UI state
    shot0 = int(np.clip(args.shot, 0, S - 1))
    state = dict(
        shot=shot0,
        sort=bool(args.sort_by_offset),
        agc_ms=float(args.agc_ms),
        clip_pct=float(args.clip_pct),
        wiggle_open=False,
    )

    # Figure layout
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1.2, 1.1], height_ratios=[1.0, 0.25])
    ax_gather = fig.add_subplot(gs[0, 0])
    ax_geo = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_fold = fig.add_subplot(gs[1, 0:2])
    ax_ui = fig.add_subplot(gs[1, 2])
    ax_ui.axis("off")

    # Gather placeholder + colorbar
    im = ax_gather.imshow(np.zeros((T, R)), aspect="auto", origin="upper")
    cbar = fig.colorbar(im, ax=ax_gather, fraction=0.046, pad=0.02)
    cbar.set_label("amplitude (clipped)")

    # Fold heatmap (static)
    if dim == 3:
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        fold_im = ax_fold.imshow(fold, aspect="auto", origin="lower", extent=extent)
        fig.colorbar(fold_im, ax=ax_fold, fraction=0.046, pad=0.02).set_label("fold (counts)")
        ax_fold.set_xlabel("CMP x (m)")
        ax_fold.set_ylabel("CMP y (m)")
        ax_fold.set_title("CMP fold / coverage (midpoint histogram)")
    else:
        extent = [x_edges[0], x_edges[-1], 0, 1]
        fold_im = ax_fold.imshow(fold, aspect="auto", origin="lower", extent=extent)
        fig.colorbar(fold_im, ax=ax_fold, fraction=0.046, pad=0.02).set_label("fold (counts)")
        ax_fold.set_xlabel("CMP x (m)")
        ax_fold.set_yticks([])
        ax_fold.set_title("CMP fold (1D)")

    # Shot slider
    slider_ax = fig.add_axes([0.15, 0.05, 0.55, 0.03])
    shot_slider = Slider(slider_ax, "shot", 0, S - 1, valinit=shot0, valstep=1)

    # Checkboxes
    cb_ax = fig.add_axes([0.75, 0.06, 0.2, 0.12])
    labels = ["sort by offset", "AGC"]
    actives = [state["sort"], state["agc_ms"] > 0]
    checks = CheckButtons(cb_ax, labels, actives)

    # Buttons
    btn_save_ax = fig.add_axes([0.75, 0.20, 0.2, 0.05])
    btn_save = Button(btn_save_ax, "Save Figure")
    btn_wig_ax = fig.add_axes([0.75, 0.27, 0.2, 0.05])
    btn_wig = Button(btn_wig_ax, "Wiggle Window")

    txt = ax_ui.text(0.0, 1.0, "", va="top", fontsize=10)

    # Wiggle window
    wig_fig = None
    wig_ax = None

    def _get_gather_for_shot(s: int):
        g = data[s].astype(np.float64)  # [R,T]
        src0 = src[s, 0, :].astype(np.float64)
        recs = rec[s].astype(np.float64)

        offset_m, az_deg = _compute_offset_azimuth(src0, recs, dx)
        order = np.arange(R)
        if state["sort"]:
            order = np.argsort(offset_m)

        g = g[order]
        offset_m_s = offset_m[order]
        az_s = az_deg[order]

        if state["agc_ms"] > 0:
            win = int(round((state["agc_ms"] / 1000.0) / dt))
            win = max(win, 1)
            g = _running_rms_agc(g, win)

        # normalize for display
        denom = np.max(np.abs(g))
        denom = denom if denom > 0 else 1.0
        g = g / denom

        return g, src0, recs, offset_m_s, az_s, order

    def _update_all(s: int):
        nonlocal wig_fig, wig_ax
        s = int(np.clip(s, 0, S - 1))
        state["shot"] = s

        g, src0, recs, offset_m_s, az_s, order = _get_gather_for_shot(s)
        vmax = _clip_symmetric(g, state["clip_pct"])

        # Gather
        ax_gather.clear()
        
        # Select plotting method: pcolormesh is safer for non-uniform offsets
        if state["sort"]:
            # Plot offset vs time
            im2 = ax_gather.pcolormesh(offset_m_s, t, g.T, shading='auto', rasterized=True)
            ax_gather.set_xlabel("offset (m)")
        else:
            # Plot receiver index vs time
            x_vals = np.arange(R)
            im2 = ax_gather.pcolormesh(x_vals, t, g.T, shading='auto', rasterized=True)
            ax_gather.set_xlabel("receiver index")

        im2.set_clim(-vmax, vmax)
        ax_gather.set_ylim(t[-1], t[0]) # Invert Y axis so 0 is at top

        # Update colorbar (redraw in the existing cbar axis)
        cbar.ax.clear()
        cbar_new = fig.colorbar(im2, cax=cbar.ax)
        cbar_new.set_label("amplitude (clipped)")

        ax_gather.set_ylabel("time (s)")
        ax_gather.set_title(f"Shot gather | shot={s} | AGC={'on' if state['agc_ms']>0 else 'off'} | sort={state['sort']}")

        # Geometry
        ax_geo.clear()
        if dim == 3:
            sx_m = src[:, 0, 0] * dx[0]
            sy_m = src[:, 0, 1] * dx[1]
            ax_geo.scatter(sx_m, sy_m, s=8)
            ax_geo.scatter([src0[0]*dx[0]], [src0[1]*dx[1]], s=60, marker="x")
            ax_geo.scatter(recs[:, 0] * dx[0], recs[:, 1] * dx[1], s=6)
            ax_geo.set_xlabel("x (m)")
            ax_geo.set_ylabel("y (m)")
            ax_geo.set_title("Geometry (shots + receivers of current shot)")
        else:
            sx_m = src[:, 0, 0] * dx[0]
            ax_geo.scatter(sx_m, np.zeros_like(sx_m), s=8)
            ax_geo.scatter([src0[0]*dx[0]], [0], s=60, marker="x")
            ax_geo.scatter(recs[:, 0] * dx[0], np.zeros((recs.shape[0],)), s=6)
            ax_geo.set_xlabel("x (m)")
            ax_geo.set_yticks([])
            ax_geo.set_title("Geometry (2D projection)")

        # Offset hist
        ax_hist.clear()
        off = offset_m_s
        ax_hist.hist(off, bins=30)
        ax_hist.set_xlabel("offset (m)")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Offset histogram (current shot)")

        # Info text
        if dim == 3:
            msg = (
                f"File: {os.path.basename(args.file)}\n"
                f"Mode: 3D\n"
                f"Shots: {S}\n"
                f"Receivers/shot: {R}\n"
                f"dt: {dt}\n"
                f"dx,dy,dz: {dx.tolist()}\n"
                f"Receiver mode: {rec_mode}\n\n"
                f"Current shot: {s}\n"
                f"src (idx): {src0.astype(int).tolist()}\n"
                f"src (m): [{src0[0]*dx[0]:.1f}, {src0[1]*dx[1]:.1f}, {src0[2]*dx[2]:.1f}]\n"
                f"offset range (m): [{off.min():.1f}, {off.max():.1f}]\n"
                f"clip pct: {state['clip_pct']}\n"
            )
        else:
            msg = (
                f"File: {os.path.basename(args.file)}\n"
                f"Mode: 2D\n"
                f"Shots: {S}\n"
                f"Receivers/shot: {R}\n"
                f"dt: {dt}\n"
                f"dx,dz: {dx.tolist()}\n\n"
                f"Current shot: {s}\n"
                f"src (idx): {src0.astype(int).tolist()}\n"
                f"offset range (m): [{off.min():.1f}, {off.max():.1f}]\n"
                f"clip pct: {state['clip_pct']}\n"
            )
        txt.set_text(msg)

        # Wiggle update
        if wig_fig is not None and plt.fignum_exists(wig_fig.number):
            wig_ax.clear()
            xax = off if state["sort"] else np.arange(R)
            wig_ax.set_xlabel("offset (m)" if state["sort"] else "receiver index")
            _wiggle(wig_ax, t, xax, g, max_traces=200)
            wig_ax.set_title(f"Wiggle | shot={s}")
            wig_fig.canvas.draw_idle()

        fig.canvas.draw_idle()

    def on_slider(val):
        _update_all(int(val))

    shot_slider.on_changed(on_slider)

    def on_checks(label):
        if label == "sort by offset":
            state["sort"] = not state["sort"]
        elif label == "AGC":
            if state["agc_ms"] > 0:
                state["agc_ms"] = 0.0
            else:
                state["agc_ms"] = 300.0
        _update_all(state["shot"])

    checks.on_clicked(on_checks)

    def _ensure_save_dir() -> str:
        if not args.save_dir:
            return ""
        os.makedirs(args.save_dir, exist_ok=True)
        return args.save_dir

    def on_save(event):
        out_dir = _ensure_save_dir()
        if not out_dir:
            print("[Save] --save_dir not set; skipping.")
            return
        s = state["shot"]
        out = os.path.join(out_dir, f"viz_shot_{s:04d}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[Save] wrote {out}")

        if wig_fig is not None and plt.fignum_exists(wig_fig.number):
            out2 = os.path.join(out_dir, f"viz_wiggle_shot_{s:04d}.png")
            wig_fig.savefig(out2, dpi=200, bbox_inches="tight")
            print(f"[Save] wrote {out2}")

    btn_save.on_clicked(on_save)

    def open_wiggle(event=None):
        nonlocal wig_fig, wig_ax
        if wig_fig is not None and plt.fignum_exists(wig_fig.number):
            return
        wig_fig = plt.figure(figsize=(8, 6))
        wig_ax = wig_fig.add_subplot(1, 1, 1)
        state["wiggle_open"] = True
        _update_all(state["shot"])
        wig_fig.show()

    btn_wig.on_clicked(open_wiggle)

    def on_key(event):
        if event.key == "w":
            open_wiggle()
        elif event.key == "s":
            on_save(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # init
    _update_all(shot0)
    if args.wiggle:
        open_wiggle()

    if args.save_dir:
        # 如果指定了 save_dir，自动保存当前炮的可视化结果
        on_save(None)
        # 注意：如果要在后台运行不弹窗，可以考虑不调用 plt.show()
        # 但如果用户也想看交互，这里还是保留 plt.show()
        
    plt.show()


if __name__ == "__main__":
    main()
