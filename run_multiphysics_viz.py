import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import cigvis

from core.multiphysics import build_multiphysics_model
from core.presets import build_default_viz_presets, read_segy_volume
from core.viz_utils import extract_subtype_labels

def _plot_igneous_swarm_debug_view(anomaly, mask_final, vp_multi, vp_bg, dx, dy, dz, save_dir, name_en, name_zh):
    if getattr(getattr(anomaly, "params", None), "kind", "") != "swarm":
        return

    mask_binary = np.asarray(mask_final > 0, dtype=np.float32)
    if mask_binary.ndim != 3 or not np.any(mask_binary):
        return

    nx, ny, nz = mask_binary.shape
    top_nz = max(1, min(nz, int(round(nz * 0.08))))
    shallow_projection = np.max(mask_binary[:, :, :top_nz], axis=2)
    footprint_projection = np.max(mask_binary, axis=2)
    thickness_map_m = mask_binary.sum(axis=2) * float(dz)
    vp_delta_shallow = np.mean(vp_multi[:, :, :top_nz] - vp_bg[:, :, :top_nz], axis=2)

    x_extent = np.arange(nx) * float(dx)
    y_extent = np.arange(ny) * float(dy)
    extent = [float(x_extent[0]), float(x_extent[-1] if nx > 1 else dx), float(y_extent[0]), float(y_extent[-1] if ny > 1 else dy)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    panels = [
        (shallow_projection.T, "magma", "Shallow Exposure Projection", "Active dyke presence"),
        (footprint_projection.T, "viridis", "Full Footprint Projection", "Active dyke presence"),
        (thickness_map_m.T, "plasma", "Integrated Dyke Thickness", "Accumulated thickness (m)"),
        (vp_delta_shallow.T, "coolwarm", "Shallow Vp Contrast", "Vp anomaly (m/s)"),
    ]

    # Recreate the dyke list with the original seed so the debug overlay matches
    # the swarm geometry used during model construction.
    dyke_pre = anomaly._precompute_dyke_list(anomaly.params, np.random.default_rng(anomaly.rng_seed))
    dykes = dyke_pre.get("dykes", [])

    for ax, (panel, cmap, title, cbar_label) in zip(axes.flat, panels):
        im = ax.imshow(panel, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
        ax.set_title(f"{title} - {name_zh}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if np.any(footprint_projection > 0):
            ax.contour(
                x_extent,
                y_extent,
                footprint_projection.T,
                levels=[0.5],
                colors="white",
                linewidths=0.8,
                alpha=0.8,
            )

        for idx, dyke in enumerate(dykes):
            strike_rad = np.deg2rad(float(dyke["strike"]))
            half_len = 0.5 * float(dyke["len"])
            cx = float(dyke["x0"])
            cy = float(dyke["y0"])
            dx_line = half_len * np.cos(strike_rad)
            dy_line = half_len * np.sin(strike_rad)
            color = plt.cm.tab20(idx % 20)
            ax.plot(
                [cx - dx_line, cx + dx_line],
                [cy - dy_line, cy + dy_line],
                color=color,
                linewidth=1.2,
                alpha=0.95,
            )
            ax.scatter([cx], [cy], color=color, s=10, alpha=0.95)

    out_file = f"IgneousSwarm_Debug_{name_en}-{name_zh}.png"
    out_path = os.path.join(save_dir, out_file)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved igneous swarm debug view: {out_path}")


def generate_multiphysics_and_plot(anomaly, anomaly_type_str, name_en, name_zh, vp_bg, label_vol, dx, dy, dz, run_app=False, show_colorbar=True):
    """
    基于速度模型生成多物理场属性（密度，电阻率，磁化率），叠加异常体，并使用 cigvis 进行可视化。
    """
    print(f"\n-> Generating Multiphysics for {name_zh} ({anomaly_type_str})...")
    
    model = build_multiphysics_model(vp_bg, label_vol, [anomaly], dx, dy, dz)
    vp_multi = model["vp"]
    rho_multi = model["rho"]
    res_multi = model["resist"]
    chi_multi = model["chi"]
    mask_final = model["anomaly_label"]
    X, Y, Z = model["X"], model["Y"], model["Z"]
    
    # 保存与可视化准备
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "outputs", "ModelFig")
    os.makedirs(save_dir, exist_ok=True)
    
    # 磁化率放大数值便于可视化
    chi_multi_scaled = chi_multi * 1e5
    
    # 获取各个场的合理取值范围 (clim)，用于切片色标和表面物理场上色
    def _safe_clim(vmin, vmax, eps=1e-5):
        return [vmin, vmax] if vmax > vmin else [vmin, vmax + eps]

    def _robust_upper_clim(values, upper_pct=99.5):
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return _safe_clim(0.0, 1.0)
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        robust_vmax = float(np.percentile(finite, upper_pct))
        robust_vmax = max(robust_vmax, vmin + 1e-5)
        return _safe_clim(vmin, min(vmax, robust_vmax))

    clim_vp = _safe_clim(vp_multi.min(), vp_multi.max())
    clim_rho = _safe_clim(rho_multi.min(), rho_multi.max())
    
    log_res_multi = np.log10(np.clip(res_multi, a_min=1e-5, a_max=None))
    # 电阻率跨度极大，往往取对数以保证可视化效果
    clim_res = _safe_clim(log_res_multi.min(), log_res_multi.max())
    # Susceptibility often contains a tiny high-chi ore body inside a much larger
    # weakly magnetic background. A robust upper percentile keeps that anomaly
    # visible instead of letting a few extreme cells flatten the whole panel.
    clim_chi = _robust_upper_clim(chi_multi_scaled, upper_pct=99.5)

    # 在四个子窗口分别创建模型基础切片
    nodes_vp = cigvis.create_slices(vp_multi, cmap='jet', clim=clim_vp)
    nodes_rho = cigvis.create_slices(rho_multi, cmap='jet', clim=clim_rho)
    nodes_res = cigvis.create_slices(log_res_multi, cmap='jet', clim=clim_res)
    nodes_chi = cigvis.create_slices(chi_multi_scaled, cmap='jet', clim=clim_chi)
    
    # ---------------- 异常体可视化 ----------------
    sub_labels = extract_subtype_labels(anomaly, X, Y, Z, mask_final, vp_bg=vp_bg)
    
    if sub_labels is not None and len(np.unique(sub_labels)) > 1:
        for code in np.unique(sub_labels):
            if code == 0: continue
            
            sub_mask = (sub_labels == code).astype(np.float32)
            
            for nds, vol, clim in zip(
                [nodes_vp, nodes_rho, nodes_res, nodes_chi],
                [vp_multi, rho_multi, log_res_multi, chi_multi_scaled],
                [clim_vp, clim_rho, clim_res, clim_chi]
            ):
                mask_nodes = cigvis.create_bodys(sub_mask, level=0.5, color='white', alpha=1.0)
                if mask_nodes:
                    import scipy.ndimage as ndimage
                    node = mask_nodes[0]
                    verts = node.mesh_data.get_vertices()
                    if verts is not None and len(verts) > 0:
                        # 采用三线性插值(order=1)以消除浮点坐标强转整数带来的莫尔条纹/锯齿
                        coords = verts.T  # map_coordinates expect shape (3, N)
                        vals = ndimage.map_coordinates(vol, coords, order=1, mode='nearest')
                        
                        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
                        scalar_map = cm.ScalarMappable(norm=norm, cmap='jet')
                        colors = scalar_map.to_rgba(vals, alpha=1.0)
                        
                        node.mesh_data.set_vertex_colors(colors)

                    if isinstance(mask_nodes, list):
                        nds.extend(mask_nodes)
                    else:
                        nds.append(mask_nodes)
    # ---------------- 异常体可视化结束 ----------------

    # 统一增加 colorbar
    if show_colorbar:
        cb_vp = cigvis.create_colorbar_from_nodes(nodes_vp, label_str="Vp (m/s)")
        cb_rho = cigvis.create_colorbar_from_nodes(nodes_rho, label_str="Density (g/cm^3)")
        cb_res = cigvis.create_colorbar_from_nodes(nodes_res, label_str="log10(Res)")
        cb_chi = cigvis.create_colorbar_from_nodes(nodes_chi, label_str="Suscept(1e-5SI)")
        
        for nds, cb in zip((nodes_vp, nodes_rho, nodes_res, nodes_chi), (cb_vp, cb_rho, cb_res, cb_chi)):
            if isinstance(cb, list):
                nds.extend(cb)
            else:
                nds.append(cb)
    
    out_file = f"Multiphysics_{name_en}-{name_zh}.png"
    out_path = os.path.join(save_dir, out_file)
    
    # (2, 2) 网格绘制四个物理场：Vp, Rho, Res(log), Chi
    cigvis.plot3D(
        [nodes_vp, nodes_rho, nodes_res, nodes_chi], 
        grid=(2, 2), 
        savename=out_file, 
        savedir=save_dir, 
        run_app=run_app, 
        title=[f"Vp (m/s) - {name_zh}", f"Density (g/cm^3)", f"log10(Resistivity) (Ohm.m)", f"Susceptibility (x1e-5 SI)"]
    )
    print(f"Saved Multiphysics Visualization: {out_path}")

    if anomaly_type_str == "igneous_swarm":
        _plot_igneous_swarm_debug_view(
            anomaly,
            mask_final,
            vp_multi,
            vp_bg,
            dx,
            dy,
            dz,
            save_dir,
            name_en,
            name_zh,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_app", action="store_true", help="Launch cigvis GUI for checking visuals interactively.")
    parser.add_argument("--hide_colorbar", action="store_true", help="Hide colorbars in the output visualization.")
    parser.add_argument("--vp_segy", type=str, required=True, help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default=None, help="Optional path to background label SEGY.")
    args = parser.parse_args()

    vp_segy_path = args.vp_segy
    label_segy_path = args.label_segy
    
    try:
        vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
        label_vol, _ = read_segy_volume(label_segy_path)
        nx, ny, nz = vp_bg.shape
        print(f'Velocity shape = {nx, ny, nz}, dx={dx}, dy={dy}, dz={dz}')
    except Exception as e:
        raise RuntimeError(f"Failed to read SEGY volumes: {e}")

    for preset in build_default_viz_presets(vp_bg, label_vol, (dx, dy, dz), include_stock=False):
        generate_multiphysics_and_plot(
            preset.anomaly,
            preset.key,
            preset.name_en,
            preset.name_zh,
            vp_bg,
            label_vol,
            dx,
            dy,
            dz,
            run_app=args.run_app,
            show_colorbar=not args.hide_colorbar,
        )

if __name__ == '__main__':
    main()
