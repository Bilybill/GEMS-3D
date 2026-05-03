import argparse
import os
from pathlib import Path

# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cigvis
import matplotlib.pyplot as plt
import numpy as np


VELOCITY_3D_FILENAME = "01_3d_velocity_model.png"
MT_3D_FILENAME = "02_3d_mt_te_tm_appres_phase.png"
SEISMIC_3D_FILENAME = "03_3d_seismic_shot.png"
SEISMIC_2D_SLICES_FILENAME = "03c_seismic_shot_iline_xline.png"
MULTIPHYSICS_OBS_FILENAME = "00_multiphysics_observation_systems.png"
FIELDS_2D_FILENAME = "04_gravity_magnetic.png"


def _safe_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _safe_clim(vmin, vmax, eps=1e-6):
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + eps
    return [vmin, vmax]


def _robust_clim(values, lower_pct=0.5, upper_pct=99.5):
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0, 1.0]
    return _safe_clim(np.percentile(finite, lower_pct), np.percentile(finite, upper_pct))


def _merge_nodes(nodes, colorbar):
    try:
        return nodes + [colorbar]
    except TypeError:
        return nodes + colorbar


def _render_plot3d(nodes, output_path, run_app=False, **kwargs):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cigvis.plot3D(
            nodes,
            savename=output_path.name,
            savedir=str(output_path.parent.resolve()) + os.sep,
            run_app=run_app,
            **kwargs,
        )
        print(f"[+] Saved: {output_path}")
        return True
    except Exception as exc:
        print(f"[!] cigvis plot3D failed for {output_path.name}: {exc}")
        return False


def _save_volume_slice_panel(volume, output_path, title, cmap="jet", clim=None):
    volume = np.asarray(volume, dtype=np.float32)
    nx, ny, nz = volume.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    slices = [
        (volume[ix, :, :].T, f"{title} | X slice {ix}", "Y", "Z"),
        (volume[:, iy, :].T, f"{title} | Y slice {iy}", "X", "Z"),
        (volume[:, :, iz].T, f"{title} | Z slice {iz}", "X", "Y"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if clim is None:
        clim = _robust_clim(volume)
    for ax, (img, sub_title, xlabel, ylabel) in zip(axes, slices):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1], aspect="auto")
        ax.set_title(sub_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, shrink=0.8)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[+] Saved fallback slices: {output_path}")


def _save_volume_slice_panel_with_scales(volume, output_path, title, cmap="jet", clim=None, axis_scales=(1.0, 1.0, 1.0)):
    volume = np.asarray(volume, dtype=np.float32)
    sx, sy, sz = [max(float(v), 1e-6) for v in axis_scales]
    nx, ny, nz = volume.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    slices = [
        (
            volume[ix, :, :].T,
            f"{title} | X slice {ix}",
            "Y",
            "T",
            [0.0, ny * sy, 0.0, nz * sz],
        ),
        (
            volume[:, iy, :].T,
            f"{title} | Y slice {iy}",
            "X",
            "T",
            [0.0, nx * sx, 0.0, nz * sz],
        ),
        (
            volume[:, :, iz].T,
            f"{title} | T slice {iz}",
            "X",
            "Y",
            [0.0, nx * sx, 0.0, ny * sy],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if clim is None:
        clim = _robust_clim(volume)
    for ax, (img, sub_title, xlabel, ylabel, extent) in zip(axes, slices):
        im = ax.imshow(
            img,
            origin="lower",
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            aspect="equal",
            extent=extent,
        )
        ax.set_title(sub_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, shrink=0.8)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[+] Saved fallback slices: {output_path}")


def _load_bundle(bundle_path):
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"forward bundle not found: {bundle_path}")
    data = np.load(bundle_path, allow_pickle=True)
    return bundle_path, data


def _prepare_output_dir(requested_output_dir, bundle_path):
    candidates = []
    if requested_output_dir:
        candidates.append(Path(requested_output_dir))
    else:
        candidates.append(bundle_path.parent / "custom_plots")
    candidates.append(Path("/tmp/gmes_forward_custom_plots"))

    for out_dir in candidates:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_dir != candidates[0]:
                print(f"[!] Output directory fallback in use: {out_dir}")
            return out_dir
        except OSError:
            continue

    raise OSError("Failed to create any writable output directory for visualization outputs.")


def plot_velocity_model_3d(bundle, out_dir, run_app=False):
    vp = np.asarray(bundle["vp_model"], dtype=np.float32)
    pos = [vp.shape[0] // 2, vp.shape[1] // 2, vp.shape[2] // 2]
    nodes = cigvis.create_slices(vp, cmap="jet", pos=pos)
    cbar = cigvis.create_colorbar(cmap="jet", clim=_robust_clim(vp), label_str="Vp (m/s)")
    ok = _render_plot3d(
        _merge_nodes(nodes, cbar),
        Path(out_dir) / VELOCITY_3D_FILENAME,
        run_app=run_app,
        xyz_axis=True,
        size=(1000, 800),
    )
    if not ok:
        _save_volume_slice_panel(vp, Path(out_dir) / VELOCITY_3D_FILENAME, "Velocity Model", cmap="jet")


def plot_mt_3d(bundle, out_dir, run_app=False):
    mt_status = str(_safe_scalar(bundle["mt_status"]))
    if mt_status != "ok":
        print(f"[-] Skip MT visualization because mt_status={mt_status}")
        return

    app_res = np.asarray(bundle["mt_app_res"], dtype=np.float32)
    phase = np.asarray(bundle["mt_phase"], dtype=np.float32)
    freqs = np.asarray(bundle["mt_freqs_hz"], dtype=np.float32) if "mt_freqs_hz" in bundle else None

    te_app = np.log10(np.clip(app_res[..., 0], 1e-6, None)).transpose(1, 2, 0)
    tm_app = np.log10(np.clip(app_res[..., 1], 1e-6, None)).transpose(1, 2, 0)
    te_phase = np.asarray(phase[..., 0], dtype=np.float32).transpose(1, 2, 0)
    tm_phase = np.asarray(phase[..., 1], dtype=np.float32).transpose(1, 2, 0)

    pos = [te_app.shape[0] // 2, te_app.shape[1] // 2, te_app.shape[2] // 2]

    nodes_te_app = cigvis.create_slices(te_app, cmap="jet", pos=pos)
    nodes_tm_app = cigvis.create_slices(tm_app, cmap="jet", pos=pos)
    nodes_te_phase = cigvis.create_slices(te_phase, cmap="jet", pos=pos)
    nodes_tm_phase = cigvis.create_slices(tm_phase, cmap="jet", pos=pos)

    cb_te_app = cigvis.create_colorbar(cmap="jet", clim=_robust_clim(te_app), label_str="TE log10(AppRes)")
    cb_tm_app = cigvis.create_colorbar(cmap="jet", clim=_robust_clim(tm_app), label_str="TM log10(AppRes)")
    cb_te_phase = cigvis.create_colorbar(cmap="jet", clim=_robust_clim(te_phase), label_str="TE Phase (deg)")
    cb_tm_phase = cigvis.create_colorbar(cmap="jet", clim=_robust_clim(tm_phase), label_str="TM Phase (deg)")

    panels = [
        _merge_nodes(nodes_te_app, cb_te_app),
        _merge_nodes(nodes_tm_app, cb_tm_app),
        _merge_nodes(nodes_te_phase, cb_te_phase),
        _merge_nodes(nodes_tm_phase, cb_tm_phase),
    ]

    if freqs is not None and freqs.size > 0:
        title_suffix = f" ({freqs.size} freqs: {float(freqs.max()):g} Hz -> {float(freqs.min()):g} Hz)"
    else:
        title_suffix = ""

    ok = _render_plot3d(
        panels,
        Path(out_dir) / MT_3D_FILENAME,
        run_app=run_app,
        grid=(2, 2),
        share=True,
        size=(1600, 1200),
        title=[
            f"TE AppRes{title_suffix}",
            f"TM AppRes{title_suffix}",
            "TE Phase",
            "TM Phase",
        ],
    )
    if not ok:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        fallback_panels = [
            (te_app[:, :, te_app.shape[2] // 2].T, "TE log10(AppRes)", _robust_clim(te_app)),
            (tm_app[:, :, tm_app.shape[2] // 2].T, "TM log10(AppRes)", _robust_clim(tm_app)),
            (te_phase[:, :, te_phase.shape[2] // 2].T, "TE Phase", _robust_clim(te_phase)),
            (tm_phase[:, :, tm_phase.shape[2] // 2].T, "TM Phase", _robust_clim(tm_phase)),
        ]
        for ax, (img, title, clim) in zip(axes.flat, fallback_panels):
            im = ax.imshow(img, origin="lower", cmap="jet", vmin=clim[0], vmax=clim[1])
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.8)
        out_path = Path(out_dir) / MT_3D_FILENAME
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"[+] Saved fallback MT panel: {out_path}")


def _shot_to_receiver_volume(seismic_shot, receiver_locations):
    rec = np.asarray(receiver_locations, dtype=np.int32)
    shot = np.asarray(seismic_shot, dtype=np.float32)
    x_unique = np.unique(rec[:, 0])
    y_unique = np.unique(rec[:, 1])

    x_index = {int(v): i for i, v in enumerate(x_unique.tolist())}
    y_index = {int(v): i for i, v in enumerate(y_unique.tolist())}

    volume = np.zeros((len(x_unique), len(y_unique), shot.shape[1]), dtype=np.float32)
    hit = np.zeros((len(x_unique), len(y_unique)), dtype=np.uint8)
    for i in range(rec.shape[0]):
        ix = x_index[int(rec[i, 0])]
        iy = y_index[int(rec[i, 1])]
        volume[ix, iy, :] = shot[i, :]
        hit[ix, iy] = 1

    return volume, x_unique, y_unique, hit


def _seismic_axis_scales(bundle, x_unique, y_unique, nt):
    dx = float(_safe_scalar(bundle["dx"])) if "dx" in bundle else 1.0
    dy = float(_safe_scalar(bundle["dy"])) if "dy" in bundle else 1.0
    dt = float(_safe_scalar(bundle["seismic_dt"])) if "seismic_dt" in bundle else 1.0

    if len(x_unique) > 1:
        x_extent = max(float(x_unique[-1] - x_unique[0]) * dx, dx)
    else:
        x_extent = dx
    if len(y_unique) > 1:
        y_extent = max(float(y_unique[-1] - y_unique[0]) * dy, dy)
    else:
        y_extent = dy

    time_extent_ms = max(float(max(nt - 1, 1)) * dt * 1000.0, 1.0)
    ref_extent = max(x_extent, y_extent, time_extent_ms, 1.0)
    return (
        x_extent / ref_extent,
        y_extent / ref_extent,
        time_extent_ms / ref_extent,
    )


def _locations_to_metric(locations, dx, dy, dz):
    loc = np.asarray(locations, dtype=np.float32)
    scale = np.array([dx, dy, dz], dtype=np.float32)
    return loc * scale


def _subsample_points(points, max_points=3000):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= max_points:
        return pts
    step = int(np.ceil(pts.shape[0] / max_points))
    return pts[::step]


def _make_surface_grid_points(nx, ny, dx, dy, x_offset=0.0, y_offset=0.0, center_offset=False):
    offset = 0.5 if center_offset else 0.0
    x = (np.arange(nx, dtype=np.float32) + offset) * dx + x_offset
    y = (np.arange(ny, dtype=np.float32) + offset) * dy + y_offset
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.zeros_like(xx, dtype=np.float32)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def plot_multiphysics_observation_systems(bundle, out_dir, shot_index=0):
    dx = float(_safe_scalar(bundle["dx"])) if "dx" in bundle else 1.0
    dy = float(_safe_scalar(bundle["dy"])) if "dy" in bundle else 1.0
    dz = float(_safe_scalar(bundle["dz"])) if "dz" in bundle else 1.0
    nx, ny, nz = bundle["vp_model"].shape
    x_extent = max((nx - 1) * dx, dx)
    y_extent = max((ny - 1) * dy, dy)
    z_extent = max((nz - 1) * dz, dz)

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    summary_lines = [f"Model extent: {x_extent:.0f} m x {y_extent:.0f} m x {z_extent:.0f} m"]

    gravity_status = str(_safe_scalar(bundle["gravity_status"])) if "gravity_status" in bundle else "not_run"
    if gravity_status == "ok" and "gravity_data" in bundle:
        gx, gy = np.asarray(bundle["gravity_data"]).shape
        grav_pts = _make_surface_grid_points(gx, gy, dx, dy)
        grav_plot = _subsample_points(grav_pts, max_points=2500)
        ax3d.scatter(grav_plot[:, 0], grav_plot[:, 1], grav_plot[:, 2], c="forestgreen", marker="s", s=4, alpha=0.10, label="Gravity obs")
        axxy.scatter(grav_plot[:, 0], grav_plot[:, 1], c="forestgreen", marker="s", s=7, alpha=0.12, label=f"Gravity ({gx}x{gy})")
        summary_lines.append(f"Gravity: {gx * gy} surface points")

    magnetic_status = str(_safe_scalar(bundle["magnetic_status"])) if "magnetic_status" in bundle else "not_run"
    if magnetic_status == "ok" and "magnetic_data" in bundle:
        mx, my = np.asarray(bundle["magnetic_data"]).shape
        mag_pts = _make_surface_grid_points(mx, my, dx, dy)
        mag_plot = _subsample_points(mag_pts, max_points=2500)
        ax3d.scatter(mag_plot[:, 0], mag_plot[:, 1], mag_plot[:, 2], c="purple", marker="x", s=5, alpha=0.12, label="Magnetic obs")
        axxy.scatter(mag_plot[:, 0], mag_plot[:, 1], c="purple", marker="x", s=8, alpha=0.15, label=f"Magnetic ({mx}x{my})")
        summary_lines.append(f"Magnetic: {mx * my} surface points")

    mt_status = str(_safe_scalar(bundle["mt_status"])) if "mt_status" in bundle else "not_run"
    if mt_status == "ok":
        if "mt_res_model" in bundle:
            mt_nx, mt_ny, _ = np.asarray(bundle["mt_res_model"]).shape
        else:
            _, mt_nx, mt_ny, _ = np.asarray(bundle["mt_app_res"]).shape
        mt_dx = float(_safe_scalar(bundle["mt_dx"])) if "mt_dx" in bundle else dx
        mt_dy = float(_safe_scalar(bundle["mt_dy"])) if "mt_dy" in bundle else dy
        mt_pts = _make_surface_grid_points(mt_nx, mt_ny, mt_dx, mt_dy, center_offset=True)
        mt_plot = _subsample_points(mt_pts, max_points=2500)
        ax3d.scatter(mt_plot[:, 0], mt_plot[:, 1], mt_plot[:, 2], c="darkorange", marker="v", s=10, alpha=0.55, label="MT stations")
        axxy.scatter(mt_plot[:, 0], mt_plot[:, 1], c="darkorange", marker="v", s=12, alpha=0.65, label=f"MT ({mt_nx}x{mt_ny})")
        summary_lines.append(f"MT: {mt_nx * mt_ny} surface stations")

    seismic_status = str(_safe_scalar(bundle["seismic_status"])) if "seismic_status" in bundle else "not_run"
    if seismic_status == "ok" and "seismic_source_locations" in bundle and "seismic_receiver_locations" in bundle:
        source_locations = np.asarray(bundle["seismic_source_locations"], dtype=np.int32)
        receiver_locations = np.asarray(bundle["seismic_receiver_locations"], dtype=np.int32)
        shot_index = int(np.clip(shot_index, 0, source_locations.shape[0] - 1))

        src_all = np.unique(source_locations.reshape(-1, source_locations.shape[-1]), axis=0)
        rec_all = np.unique(receiver_locations.reshape(-1, receiver_locations.shape[-1]), axis=0)
        src_shot = np.unique(source_locations[shot_index], axis=0)
        rec_shot = np.unique(receiver_locations[shot_index], axis=0)

        src_all_m = _locations_to_metric(src_all, dx, dy, dz)
        rec_all_m = _locations_to_metric(rec_all, dx, dy, dz)
        src_shot_m = _locations_to_metric(src_shot, dx, dy, dz)
        rec_shot_m = _locations_to_metric(rec_shot, dx, dy, dz)

        rec_all_plot = _subsample_points(rec_all_m, max_points=3000)
        rec_shot_plot = _subsample_points(rec_shot_m, max_points=3000)

        ax3d.scatter(rec_all_plot[:, 0], rec_all_plot[:, 1], rec_all_plot[:, 2], c="lightgray", s=5, alpha=0.20, label="Seismic receivers")
        ax3d.scatter(rec_shot_plot[:, 0], rec_shot_plot[:, 1], rec_shot_plot[:, 2], c="dodgerblue", s=10, alpha=0.85, label=f"Shot {shot_index} receivers")
        ax3d.scatter(src_all_m[:, 0], src_all_m[:, 1], src_all_m[:, 2], c="crimson", marker="^", s=48, alpha=0.85, label="Seismic sources")
        ax3d.scatter(src_shot_m[:, 0], src_shot_m[:, 1], src_shot_m[:, 2], c="gold", edgecolors="black", marker="*", s=170, label=f"Shot {shot_index} source")

        axxy.scatter(rec_all_plot[:, 0], rec_all_plot[:, 1], c="lightgray", s=7, alpha=0.25, label="Seismic receivers")
        axxy.scatter(rec_shot_plot[:, 0], rec_shot_plot[:, 1], c="dodgerblue", s=12, alpha=0.95, label=f"Shot {shot_index} receivers")
        axxy.scatter(src_all_m[:, 0], src_all_m[:, 1], c="crimson", marker="^", s=52, alpha=0.9, label="Seismic sources")
        axxy.scatter(src_shot_m[:, 0], src_shot_m[:, 1], c="gold", edgecolors="black", marker="*", s=180, label=f"Shot {shot_index} source")

        summary_lines.append(f"Seismic: {src_all.shape[0]} sources, {rec_all.shape[0]} unique receivers")

    ax3d.set_xlim(0.0, x_extent)
    ax3d.set_ylim(0.0, y_extent)
    ax3d.set_zlim(0.0, z_extent)
    ax3d.invert_zaxis()
    if hasattr(ax3d, "set_box_aspect"):
        ax3d.set_box_aspect((x_extent, y_extent, max(z_extent * 0.15, dz * 6.0)))
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Multiphysics Observation Systems (3D)")
    ax3d.legend(loc="upper left", fontsize=8)

    axxy.set_xlim(0.0, x_extent)
    axxy.set_ylim(0.0, y_extent)
    axxy.set_aspect("equal")
    axxy.set_xlabel("X (m)")
    axxy.set_ylabel("Y (m)")
    axxy.set_title("Plan View")
    axxy.legend(loc="upper right", fontsize=8)
    axxy.grid(alpha=0.25, linestyle="--")
    axxy.plot([0.0, x_extent, x_extent, 0.0, 0.0], [0.0, 0.0, y_extent, y_extent, 0.0], color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axxy.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        transform=axxy.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    output_path = Path(out_dir) / MULTIPHYSICS_OBS_FILENAME
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def _plot_seismic_shot_slices_2d(bundle, out_dir, volume, shot_index, amp, x_unique, y_unique):
    dx = float(_safe_scalar(bundle["dx"])) if "dx" in bundle else 1.0
    dy = float(_safe_scalar(bundle["dy"])) if "dy" in bundle else 1.0
    dt = float(_safe_scalar(bundle["seismic_dt"])) if "seismic_dt" in bundle else 1.0

    ix = volume.shape[0] // 2
    iy = volume.shape[1] // 2
    t_max_ms = max(float(max(volume.shape[2] - 1, 1)) * dt * 1000.0, 1.0)
    x_extent = [0.0, max(float(volume.shape[0] - 1), 1.0) * dx]
    y_extent = [0.0, max(float(volume.shape[1] - 1), 1.0) * dy]

    iline = volume[:, iy, :].T
    xline = volume[ix, :, :].T

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panels = [
        (
            axes[0],
            iline,
            [x_extent[0], x_extent[1], 0.0, t_max_ms],
            f"Iline Slice | Y={int(y_unique[iy])} (receiver index)",
            "X (m)",
        ),
        (
            axes[1],
            xline,
            [y_extent[0], y_extent[1], 0.0, t_max_ms],
            f"Xline Slice | X={int(x_unique[ix])} (receiver index)",
            "Y (m)",
        ),
    ]

    for ax, img, extent, title, xlabel in panels:
        im = ax.imshow(
            img,
            cmap="gray",
            vmin=-amp,
            vmax=amp,
            aspect="auto",
            extent=extent,
            origin="lower",
        )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (ms)")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")

    fig.suptitle(f"Seismic Shot {shot_index} 2D Slices", fontsize=12)
    output_path = Path(out_dir) / SEISMIC_2D_SLICES_FILENAME
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[+] Saved: {output_path}")


def plot_seismic_shot_3d(bundle, out_dir, shot_index=0, run_app=False):
    seismic_status = str(_safe_scalar(bundle["seismic_status"]))
    if seismic_status != "ok":
        print(f"[-] Skip seismic visualization because seismic_status={seismic_status}")
        return

    seismic = np.asarray(bundle["seismic_data"], dtype=np.float32)
    receiver_locations = np.asarray(bundle["seismic_receiver_locations"], dtype=np.int32)
    source_locations = np.asarray(bundle["seismic_source_locations"], dtype=np.int32)

    n_shots = seismic.shape[0]
    shot_index = int(np.clip(shot_index, 0, n_shots - 1))
    shot = seismic[shot_index]
    receiver_locs = receiver_locations[shot_index]
    source_loc = source_locations[shot_index, 0]

    volume, x_unique, y_unique, hit = _shot_to_receiver_volume(shot, receiver_locs)
    pos = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
    amp = float(np.percentile(np.abs(volume), 95.0))
    amp = max(amp, 1e-6)
    axis_scales = _seismic_axis_scales(bundle, x_unique, y_unique, volume.shape[2])
    nodes = cigvis.create_slices(volume, cmap="gray", pos=pos)
    cbar = cigvis.create_colorbar(cmap="gray", clim=[-amp, amp], label_str="Amplitude")

    title = (
        f"Seismic Shot {shot_index} | src=({int(source_loc[0])}, {int(source_loc[1])}, {int(source_loc[2])}) "
        f"| rec grid={len(x_unique)}x{len(y_unique)}"
    )
    ok = _render_plot3d(
        _merge_nodes(nodes, cbar),
        Path(out_dir) / SEISMIC_3D_FILENAME,
        run_app=run_app,
        xyz_axis=True,
        axis_scales=axis_scales,
        size=(1100, 850),
        title=title,
    )
    if not ok:
        _save_volume_slice_panel_with_scales(
            volume,
            Path(out_dir) / SEISMIC_3D_FILENAME,
            f"Seismic Shot {shot_index}",
            cmap="gray",
            clim=[-amp, amp],
            axis_scales=axis_scales,
        )

    _plot_seismic_shot_slices_2d(bundle, out_dir, volume, shot_index, amp, x_unique, y_unique)

    coverage_path = Path(out_dir) / "03b_seismic_receiver_coverage.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(hit.T, origin="lower", cmap="viridis")
    plt.scatter(
        [np.searchsorted(x_unique, source_loc[0])],
        [np.searchsorted(y_unique, source_loc[1])],
        c="red",
        s=35,
        label="source",
    )
    plt.title(f"Receiver Coverage (Shot {shot_index})")
    plt.xlabel("Receiver X index")
    plt.ylabel("Receiver Y index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(coverage_path, dpi=180)
    plt.close()
    print(f"[+] Saved: {coverage_path}")


def plot_gravity_magnetic(bundle, out_dir):
    grav_status = str(_safe_scalar(bundle["gravity_status"])) if "gravity_status" in bundle else "not_run"
    mag_status = str(_safe_scalar(bundle["magnetic_status"])) if "magnetic_status" in bundle else "not_run"
    if grav_status != "ok" and mag_status != "ok":
        print("[-] Skip gravity/magnetic visualization because both fields are unavailable")
        return

    dx = float(_safe_scalar(bundle["dx"]))
    dy = float(_safe_scalar(bundle["dy"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    if grav_status == "ok":
        grav = np.asarray(bundle["gravity_data"], dtype=np.float32)
        extent = [0.0, grav.shape[0] * dx, 0.0, grav.shape[1] * dy]
        im = axes[0].imshow(grav.T, origin="lower", extent=extent, cmap="jet")
        plt.colorbar(im, ax=axes[0], shrink=0.85, label="mGal")
        axes[0].set_title("Gravity Anomaly")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
    else:
        axes[0].set_axis_off()
        axes[0].set_title(f"Gravity unavailable ({grav_status})")

    if mag_status == "ok":
        mag = np.asarray(bundle["magnetic_data"], dtype=np.float32)
        extent = [0.0, mag.shape[0] * dx, 0.0, mag.shape[1] * dy]
        im = axes[1].imshow(mag.T, origin="lower", extent=extent, cmap="jet")
        plt.colorbar(im, ax=axes[1], shrink=0.85, label="nT")
        axes[1].set_title("Magnetic Anomaly")
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Y (m)")
    else:
        axes[1].set_axis_off()
        axes[1].set_title(f"Magnetic unavailable ({mag_status})")

    output_path = Path(out_dir) / FIELDS_2D_FILENAME
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"[+] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GMESDataset forward_bundle.npz outputs using cigvis and matplotlib."
    )
    parser.add_argument(
        "--save_dir",
        default="./outputs/ForwardOutput",
        help="Directory containing forward_bundle.npz generated by run_multiphysics_forward.py",
    )
    parser.add_argument(
        "--bundle_path",
        default=None,
        help="Optional explicit path to forward_bundle.npz. Overrides --save_dir.",
    )
    parser.add_argument(
        "--shot_index",
        type=int,
        default=0,
        help="Seismic shot index to visualize in 3D.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for saved visualization figures. Defaults to <bundle_dir>/custom_plots, then falls back to /tmp if needed.",
    )
    parser.add_argument(
        "--run_app",
        action="store_true",
        help="Launch cigvis interactive window in addition to saving figures.",
    )
    args = parser.parse_args()

    bundle_path = args.bundle_path or os.path.join(args.save_dir, "forward_bundle.npz")
    bundle_path, bundle = _load_bundle(bundle_path)
    out_dir = _prepare_output_dir(args.output_dir, bundle_path)

    print(f"Loaded forward bundle: {bundle_path}")
    plot_multiphysics_observation_systems(bundle, out_dir, shot_index=args.shot_index)
    plot_velocity_model_3d(bundle, out_dir, run_app=args.run_app)
    plot_mt_3d(bundle, out_dir, run_app=args.run_app)
    plot_seismic_shot_3d(bundle, out_dir, shot_index=args.shot_index, run_app=args.run_app)
    plot_gravity_magnetic(bundle, out_dir)
    print("All visualizations completed.")


if __name__ == "__main__":
    main()
