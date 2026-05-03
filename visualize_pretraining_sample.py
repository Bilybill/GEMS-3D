import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-gmesdataset"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cigvis
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    import scipy.ndimage as ndimage
except Exception:  # pragma: no cover - fallback for environments without scipy
    ndimage = None

from plot_saved_forward_data import (
    _load_bundle,
    _prepare_output_dir,
    _render_plot3d,
    _robust_clim,
    _safe_clim,
    plot_gravity_magnetic,
    plot_mt_3d,
    plot_multiphysics_observation_systems,
    plot_seismic_shot_3d,
)


MODEL_3D_FILENAME = "01_model_multiphysics_with_anomaly.png"
MODEL_LABEL_FILENAME = "01b_model_label_anomaly_slices.png"


def _merge_nodes(nodes, colorbar):
    try:
        return nodes + [colorbar]
    except TypeError:
        return nodes + colorbar


def _sample_volume_at_vertices(volume, vertices):
    if ndimage is None:
        idx = np.clip(np.round(vertices).astype(np.int32), 0, np.array(volume.shape) - 1)
        return volume[idx[:, 0], idx[:, 1], idx[:, 2]]
    coords = vertices.T
    return ndimage.map_coordinates(volume, coords, order=1, mode="nearest")


def _add_anomaly_overlay(nodes, volume, anomaly_mask, clim, cmap="jet"):
    if anomaly_mask is None:
        return nodes
    mask = np.asarray(anomaly_mask > 0, dtype=np.float32)
    if mask.ndim != 3 or not np.any(mask):
        return nodes

    mask_nodes = cigvis.create_bodys(mask, level=0.5, color="white", alpha=1.0)
    if not mask_nodes:
        return nodes

    node_list = mask_nodes if isinstance(mask_nodes, list) else [mask_nodes]
    norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    for node in node_list:
        verts = node.mesh_data.get_vertices()
        if verts is None or len(verts) == 0:
            continue
        vals = _sample_volume_at_vertices(volume, verts)
        colors = scalar_map.to_rgba(vals, alpha=1.0)
        node.mesh_data.set_vertex_colors(colors)
    nodes.extend(node_list)
    return nodes


def _save_model_label_panel(model_bundle, out_dir):
    label_volume = np.asarray(model_bundle["label_volume"], dtype=np.int16)
    anomaly_label = np.asarray(model_bundle["anomaly_label"], dtype=np.int16)
    if label_volume.size == 0:
        print("[-] Skip label/anomaly slice panel because label_volume is unavailable.")
        return

    nx, ny, nz = label_volume.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    panels = [
        (label_volume[:, :, iz].T, "Label Volume | Z slice"),
        (label_volume[:, iy, :].T, "Label Volume | Y slice"),
        (anomaly_label[:, :, iz].T, "Anomaly Label | Z slice"),
        (anomaly_label[:, iy, :].T, "Anomaly Label | Y slice"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (img, title) in zip(axes.flat, panels):
        im = ax.imshow(img, origin="lower", cmap="tab20")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y/Z")
        plt.colorbar(im, ax=ax, shrink=0.8)

    out_path = Path(out_dir) / MODEL_LABEL_FILENAME
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[+] Saved: {out_path}")


def plot_model_multiphysics(bundle, out_dir, run_app=False):
    vp = np.asarray(bundle["vp_model"], dtype=np.float32)
    rho = np.asarray(bundle["rho_model"], dtype=np.float32)
    res = np.asarray(bundle["res_model"], dtype=np.float32)
    chi = np.asarray(bundle["chi_model"], dtype=np.float32) * 1e5
    anomaly_mask = np.asarray(bundle["anomaly_label"], dtype=np.int16)

    log_res = np.log10(np.clip(res, 1e-6, None))
    pos = [vp.shape[0] // 2, vp.shape[1] // 2, vp.shape[2] // 2]

    vp_clim = _robust_clim(vp)
    rho_clim = _robust_clim(rho)
    res_clim = _robust_clim(log_res)
    chi_clim = _safe_clim(np.min(chi), np.percentile(chi, 99.5))

    panels = []
    volumes = [
        (vp, "Vp (m/s)", vp_clim),
        (rho, "Density (g/cm^3)", rho_clim),
        (log_res, "log10(Resistivity)", res_clim),
        (chi, "Susceptibility (x1e-5 SI)", chi_clim),
    ]

    for volume, label, clim in volumes:
        nodes = cigvis.create_slices(volume, cmap="jet", pos=pos, clim=clim)
        nodes = _add_anomaly_overlay(nodes, volume, anomaly_mask, clim, cmap="jet")
        cbar = cigvis.create_colorbar(cmap="jet", clim=clim, label_str=label)
        panels.append(_merge_nodes(nodes, cbar))

    ok = _render_plot3d(
        panels,
        Path(out_dir) / MODEL_3D_FILENAME,
        run_app=run_app,
        grid=(2, 2),
        share=True,
        size=(1600, 1200),
        title=[v[1] for v in volumes],
    )
    if not ok:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        mid = vp.shape[2] // 2
        fallback = [
            (vp[:, :, mid].T, "Vp (m/s)", vp_clim),
            (rho[:, :, mid].T, "Density (g/cm^3)", rho_clim),
            (log_res[:, :, mid].T, "log10(Resistivity)", res_clim),
            (chi[:, :, mid].T, "Susceptibility (x1e-5 SI)", chi_clim),
        ]
        anomaly_outline = (anomaly_mask[:, :, mid] > 0).T.astype(np.float32)
        for ax, (img, title, clim) in zip(axes.flat, fallback):
            im = ax.imshow(img, origin="lower", cmap="jet", vmin=clim[0], vmax=clim[1])
            if np.any(anomaly_outline > 0):
                ax.contour(anomaly_outline, levels=[0.5], colors="white", linewidths=0.8)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.8)
        out_path = Path(out_dir) / MODEL_3D_FILENAME
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"[+] Saved fallback model panel: {out_path}")

    _save_model_label_panel(bundle, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize one pretraining sample: multiphysics models with anomaly body + forward responses.")
    parser.add_argument("--sample-dir", type=str, default=None, help="Directory containing model_bundle.npz and forward_bundle.npz.")
    parser.add_argument("--model-bundle", type=str, default=None, help="Explicit path to model_bundle.npz.")
    parser.add_argument("--forward-bundle", type=str, default=None, help="Explicit path to forward_bundle.npz.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures. Defaults to <sample-dir>/viz_sample.")
    parser.add_argument("--shot-index", type=int, default=0, help="Seismic shot index to visualize.")
    parser.add_argument("--run-app", action="store_true", help="Launch cigvis interactive windows in addition to saving figures.")
    args = parser.parse_args()

    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        model_bundle_path = sample_dir / "model_bundle.npz"
        forward_bundle_path = sample_dir / "forward_bundle.npz"
    else:
        model_bundle_path = Path(args.model_bundle) if args.model_bundle else None
        forward_bundle_path = Path(args.forward_bundle) if args.forward_bundle else None
        sample_dir = None

    if model_bundle_path is None or not model_bundle_path.exists():
        raise FileNotFoundError("model_bundle.npz not found. Provide --sample-dir or --model-bundle.")
    has_forward_bundle = forward_bundle_path is not None and forward_bundle_path.exists()

    _, model_bundle = _load_bundle(model_bundle_path)
    if has_forward_bundle:
        _, forward_bundle = _load_bundle(forward_bundle_path)
        base_out_dir = args.output_dir or (sample_dir / "viz_sample" if sample_dir is not None else forward_bundle_path.parent / "viz_sample")
        out_dir = _prepare_output_dir(base_out_dir, forward_bundle_path)
    else:
        forward_bundle = None
        fallback_path = model_bundle_path if model_bundle_path is not None else Path(args.output_dir or ".")
        base_out_dir = args.output_dir or (sample_dir / "viz_sample" if sample_dir is not None else model_bundle_path.parent / "viz_sample")
        out_dir = _prepare_output_dir(base_out_dir, fallback_path)

    print(f"Loaded model bundle  : {model_bundle_path}")
    if has_forward_bundle:
        print(f"Loaded forward bundle: {forward_bundle_path}")
    else:
        print("Forward bundle not found, only model-side visualization will be generated.")
    plot_model_multiphysics(model_bundle, out_dir, run_app=args.run_app)
    if has_forward_bundle:
        plot_multiphysics_observation_systems(forward_bundle, out_dir, shot_index=args.shot_index)
        plot_mt_3d(forward_bundle, out_dir, run_app=args.run_app)
        plot_seismic_shot_3d(forward_bundle, out_dir, shot_index=args.shot_index, run_app=args.run_app)
        plot_gravity_magnetic(forward_bundle, out_dir)
    print("Sample visualization completed.")


if __name__ == "__main__":
    main()
