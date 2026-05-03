#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified visualizer for Deepwave shot gathers.
Only displays the shot gather data (Time vs Offset).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def _as_float(x) -> float:
    arr = np.array(x).reshape(-1)
    return float(arr[0])

def _clip_symmetric(x: np.ndarray, pct: float) -> float:
    pct = float(np.clip(pct, 50.0, 100.0))
    vmax = np.percentile(np.abs(x), pct)
    if vmax <= 0:
        vmax = float(np.max(np.abs(x)) + 1e-12)
    return float(vmax)

def _compute_offset(src0: np.ndarray, rec: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    src0: [dim]
    rec : [R, dim]
    dx  : [dim]
    """
    dim = src0.shape[-1]
    d = (rec.astype(np.float64) - src0.astype(np.float64)) * np.asarray(dx, dtype=np.float64)
    if dim == 2:
        return np.abs(d[:, 0])
    else:
        return np.sqrt(np.sum(d**2, axis=1))

def main():
    parser = argparse.ArgumentParser(description="Deepwave Output Viewer - Shot Gathers Only")
    parser.add_argument("--file", required=True, help="Path to .npz file")
    parser.add_argument("--shot", type=int, default=0, help="Initial shot index")
    parser.add_argument("--clip_pct", type=float, default=99.0, help="Percentile for clipping (default 99)")
    parser.add_argument("--output", help="Path to save the visualization. If directory or ends with /, saves with input filename.")
    args = parser.parse_args()

    # Load data
    try:
        npz = np.load(args.file, allow_pickle=True)
        data = npz["data"]  # Expected shape: [S, R, T]
        dt = _as_float(npz["dt"])
        
        # Geometry for offset calculation
        if "src" in npz and "rec" in npz and "dx" in npz:
            src = npz["src"] # [S, nsrc, dim]
            rec = npz["rec"] # [S, R, dim]
            dx = npz["dx"].flatten()
            has_geom = True
        else:
            has_geom = False
            src, rec, dx = None, None, None
            print("Warning: Geometry (src/rec/dx) not found. Sorting by index.")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if data.ndim != 3:
        print(f"Error: Expected data shape [Shots, Receivers, Time], but got {data.shape}")
        return

    S, R, T = data.shape
    t_max = (T - 1) * dt

    print(f"Loaded {args.file}")
    print(f"Data shape: {data.shape} (Shots={S}, Receivers={R}, TimeSteps={T})")
    print(f"Sampling interval dt: {dt} s")

    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for slider

    init_shot = np.clip(args.shot, 0, S - 1)
    
    # helper to get data for a shot
    def get_shot_data(idx):
        g = data[idx]  # [R, T]
        
        # Sort by offset if geometry exists
        if has_geom:
            s0 = src[idx, 0, :]
            r = rec[idx]
            offsets = _compute_offset(s0, r, dx)
            order = np.argsort(offsets)
            
            g_sorted = g[order].T # [T, R]
            offsets_sorted = offsets[order]
            
            # Return sorted offsets for pcolormesh
            x_vals = offsets_sorted
            xlabel = "Offset (m)"
        else:
            g_sorted = g.T
            # For index-based plotting
            x_vals = np.arange(R)
            xlabel = "Receiver Index"
            
        return g_sorted, x_vals, xlabel

    # Initial Plot
    gather, x_vals, xlbl = get_shot_data(init_shot)
    vmax = _clip_symmetric(gather, args.clip_pct)
    
    # Prepare Time Axis
    times = np.linspace(0, t_max, T)

    # Use pcolormesh for non-uniform grid to avoid distortion
    # shading='auto' handles centering/edges automatically
    im = ax.pcolormesh(x_vals, times, gather, shading='auto', cmap='seismic', 
                       vmin=-vmax, vmax=vmax, rasterized=True)
    
    ax.set_ylim(t_max, 0) # Flip time axis
    ax.set_title(f"Shot Gather #{init_shot}")
    ax.set_xlabel(xlbl)
    ax.set_ylabel("Time (s)")
    
    # Create colorbar once
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")

    # Slider configuration
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Shot Index",
        valmin=0,
        valmax=S - 1,
        valinit=init_shot,
        valstep=1
    )

    def update(val):
        idx = int(slider.val)
        g, x, lbl = get_shot_data(idx)
        
        # Recalculate clipping for dynamic range adaptation per shot
        v = _clip_symmetric(g, args.clip_pct)
        
        # pcolormesh cannot easily update coordinates in-place like imshow.
        # We clear the axes and replot.
        ax.cla()
        
        im_new = ax.pcolormesh(x, times, g, shading='auto', cmap='seismic', 
                               vmin=-v, vmax=v, rasterized=True)
        
        ax.set_ylim(t_max, 0)
        ax.set_xlabel(lbl)
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Shot Gather #{idx}")
        
        # Update colorbar to point to new mappable
        # reusing the existing colorbar axis
        plt.colorbar(im_new, cax=cbar.ax)
        cbar.ax.set_ylabel("Amplitude")
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if args.output:
        out_path = args.output
        # Check if output is a directory (exists as dir) or looks like one (ends with separator)
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)
            # Use input filename as base
            base_name = os.path.splitext(os.path.basename(args.file))[0]
            out_path = os.path.join(out_path, f"{base_name}.png")
        else:
            # Check if parent directory exists, if so create it
            parent_dir = os.path.dirname(out_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
        print(f"Visualization saved to {out_path}")

    plt.show()

if __name__ == "__main__":
    main()