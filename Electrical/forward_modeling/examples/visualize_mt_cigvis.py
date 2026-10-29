"""Visualize a saved 3-D MT model and response volume with cigvis."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

MPL_CACHE = Path("/tmp/gems3d-matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


GEMS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    GEMS_ROOT
    / "outputs"
    / "mt_datfolder_demo"
    / "serpentinized_nips19"
    / "mt_demo_results.npz"
)
SKIN_DEPTH_COEFFICIENT_M = 503.0


def _load_results(path: Path) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise FileNotFoundError(f"MT result file not found: {path}")
    with np.load(path) as data:
        required = {
            "frequencies_hz",
            "spacing_m",
            "resistivity_model",
            "apparent_resistivity",
            "phase_deg",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"MT result file is missing: {', '.join(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required}
        if "background_resistivity_ohm_m" in data.files:
            arrays["background_resistivity_ohm_m"] = np.asarray(
                data["background_resistivity_ohm_m"]
            )

    frequencies = arrays["frequencies_hz"].astype(np.float64)
    spacing = arrays["spacing_m"].astype(np.float64)
    model = arrays["resistivity_model"].astype(np.float32)
    apparent = arrays["apparent_resistivity"].astype(np.float32)
    phase = arrays["phase_deg"].astype(np.float32)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array")
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"spacing_m must contain three positive values, got {spacing}")
    if model.ndim != 3 or np.any(~np.isfinite(model)) or np.any(model <= 0):
        raise ValueError("resistivity_model must be a positive finite 3-D array")
    expected_response_shape = (frequencies.size, model.shape[0], model.shape[1], 2)
    if (
        apparent.shape != expected_response_shape
        or phase.shape != expected_response_shape
    ):
        raise ValueError(
            "response arrays must have shape "
            f"{expected_response_shape}, got {apparent.shape} and {phase.shape}"
        )
    if (
        np.any(~np.isfinite(apparent))
        or np.any(apparent <= 0)
        or np.any(~np.isfinite(phase))
    ):
        raise ValueError(
            "MT responses must be finite and apparent resistivity positive"
        )

    order = np.argsort(frequencies)[::-1]
    result = {
        "frequencies_hz": frequencies[order],
        "spacing_m": spacing,
        "resistivity_model": model,
        "apparent_resistivity": apparent[order],
        "phase_deg": phase[order],
    }
    if "background_resistivity_ohm_m" in arrays:
        background = float(arrays["background_resistivity_ohm_m"].item())
        if not np.isfinite(background) or background <= 0:
            raise ValueError("saved background_resistivity_ohm_m must be positive")
        result["background_resistivity_ohm_m"] = background
    return result


def _boundary_mean(model: np.ndarray) -> float:
    nx, ny, nz = model.shape
    total = float(model.astype(np.float64).sum())
    count = nx * ny * nz
    if min(model.shape) > 2:
        total -= float(model[1:-1, 1:-1, 1:-1].astype(np.float64).sum())
        count -= (nx - 2) * (ny - 2) * (nz - 2)
    return total / count


def _skin_depth_m(
    resistivity_ohm_m: np.ndarray | float, frequency_hz: np.ndarray
) -> np.ndarray:
    return SKIN_DEPTH_COEFFICIENT_M * np.sqrt(
        np.asarray(resistivity_ohm_m, dtype=np.float64) / frequency_hz
    )


def _robust_clim(
    values: np.ndarray, lower: float = 2.0, upper: float = 98.0
) -> list[float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot determine color limits from non-finite data")
    limits = np.percentile(finite, [lower, upper]).astype(float)
    if limits[1] <= limits[0]:
        padding = max(abs(limits[0]) * 0.01, 1.0e-6)
        limits += np.array([-padding, padding])
    return limits.tolist()


def _parse_slice_frequencies(text: str, frequencies: np.ndarray) -> list[int]:
    requested = [float(value) for value in text.split(",") if value.strip()]
    if not requested or any(
        not np.isfinite(value) or value <= 0 for value in requested
    ):
        raise ValueError("slice frequencies must be positive comma-separated values")
    indices = {
        int(np.argmin(np.abs(np.log(frequencies) - np.log(value))))
        for value in requested
    }
    return sorted(indices)


def _import_cigvis():
    try:
        import cigvis
    except ImportError as exc:
        raise RuntimeError(
            "cigvis is required; install it in the active environment"
        ) from exc
    return cigvis


def _render_cigvis(
    cigvis,
    nodes,
    path: Path,
    *,
    run_app: bool,
    size: tuple[int, int],
    axis_scales: list[float],
    grid: tuple[int, int] | None = None,
    share: bool = False,
    title=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "cigvis requires an OpenGL display. Run with `xvfb-run -a` "
            "in a headless shell."
        )
    cigvis.plot3D(
        nodes,
        grid=grid,
        share=share,
        savename=path.name,
        savedir=str(path.parent.resolve()) + os.sep,
        size=size,
        run_app=run_app,
        axis_scales=axis_scales,
        title=title,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cigvis did not create the requested image: {path}")


def _create_axis(cigvis, shape, samplings, labels):
    return cigvis.create_axis(
        shape,
        axis_pos="auto",
        tick_nums=6,
        ticks_font_size=11,
        labels_font_size=13,
        samplings=[np.asarray(values) for values in samplings],
        axis_labels=labels,
        line_width=1.5,
        ticks_length=2.0,
    )


def _render_model(
    cigvis,
    model: np.ndarray,
    spacing: np.ndarray,
    output: Path,
    run_app: bool,
) -> None:
    dx, dy, dz = spacing
    log_model = np.log10(model)
    pos = {
        "x": [model.shape[0] // 2],
        "y": [model.shape[1] // 2],
        "z": [model.shape[2] // 2],
    }
    clim = _robust_clim(log_model, 1.0, 99.0)
    nodes = cigvis.create_slices(
        log_model,
        pos=pos,
        clim=clim,
        cmap="turbo",
        interpolation="nearest",
        intersection_lines=True,
        line_width=1.2,
    )
    coordinates_km = [
        (np.arange(size, dtype=float) + 0.5) * delta / 1000.0
        for size, delta in zip(model.shape, spacing)
    ]
    nodes += _create_axis(
        cigvis,
        model.shape,
        coordinates_km,
        ["X (km)", "Y (km)", "Depth (km)"],
    )
    nodes.append(cigvis.create_colorbar("turbo", clim, label_str="log10 rho (Ohm m)"))
    _render_cigvis(
        cigvis,
        nodes,
        output,
        run_app=run_app,
        size=(1200, 900),
        axis_scales=[1.0, float(dy / dx), float(dz / dx)],
        title="GMESDataset 3-D resistivity model",
    )


def _response_panel(
    cigvis,
    volume: np.ndarray,
    frequencies: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_indices: list[int],
    cmap: str,
    label: str,
):
    clim = _robust_clim(volume)
    nodes = cigvis.create_slices(
        volume,
        pos={
            "x": [volume.shape[0] // 2],
            "y": [volume.shape[1] // 2],
            "z": z_indices,
        },
        clim=clim,
        cmap=cmap,
        interpolation="nearest",
        intersection_lines=False,
    )
    nodes += _create_axis(
        cigvis,
        volume.shape,
        [x_km, y_km, frequencies],
        ["X (km)", "Y (km)", "Frequency (Hz)"],
    )
    nodes.append(cigvis.create_colorbar(cmap, clim, label_str=label))
    return nodes


def _render_responses(
    cigvis,
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: np.ndarray,
    spacing: np.ndarray,
    edge_crop: int,
    slice_indices: list[int],
    output: Path,
    run_app: bool,
) -> None:
    nx, ny = apparent.shape[1:3]
    if edge_crop < 0 or 2 * edge_crop >= min(nx, ny):
        raise ValueError(
            f"edge_crop={edge_crop} is invalid for a {nx}x{ny} response grid"
        )
    spatial_slice = slice(edge_crop, -edge_crop or None)
    apparent = apparent[:, spatial_slice, spatial_slice, :]
    phase = phase[:, spatial_slice, spatial_slice, :]
    dx, dy, _ = spacing
    x_km = (np.arange(apparent.shape[1]) + edge_crop + 0.5) * dx / 1000.0
    y_km = (np.arange(apparent.shape[2]) + edge_crop + 0.5) * dy / 1000.0

    log_rho_xy = np.log10(apparent[..., 0]).transpose(1, 2, 0)
    log_rho_yx = np.log10(apparent[..., 1]).transpose(1, 2, 0)
    phase_xy = phase[..., 0].transpose(1, 2, 0)
    phase_yx = phase[..., 1].transpose(1, 2, 0)
    panels = [
        _response_panel(
            cigvis,
            log_rho_xy,
            frequencies,
            x_km,
            y_km,
            slice_indices,
            "viridis",
            "log10 rhoa Zxy",
        ),
        _response_panel(
            cigvis,
            log_rho_yx,
            frequencies,
            x_km,
            y_km,
            slice_indices,
            "viridis",
            "log10 rhoa Zyx",
        ),
        _response_panel(
            cigvis,
            phase_xy,
            frequencies,
            x_km,
            y_km,
            slice_indices,
            "RdBu_r",
            "phase Zxy (deg)",
        ),
        _response_panel(
            cigvis,
            phase_yx,
            frequencies,
            x_km,
            y_km,
            slice_indices,
            "RdBu_r",
            "phase Zyx (deg)",
        ),
    ]
    selected = ", ".join(f"{frequencies[index]:g}" for index in slice_indices)
    crop_note = f"edge crop: {edge_crop} cells" if edge_crop else "full receiver grid"
    _render_cigvis(
        cigvis,
        panels,
        output,
        run_app=run_app,
        size=(1900, 1450),
        axis_scales=[1.0, 1.0, 1.5],
        grid=(2, 2),
        share=True,
        title=[
            f"Zxy apparent rho | {crop_note}",
            f"Zyx apparent rho | slices {selected} Hz",
            "Zxy phase",
            "Zyx phase",
        ],
    )


def _write_depth_products(
    frequencies: np.ndarray,
    apparent: np.ndarray,
    spacing: np.ndarray,
    model_shape: tuple[int, int, int],
    background_rho: float,
    edge_crop: int,
    output_dir: Path,
) -> tuple[Path, Path, list[dict[str, float]]]:
    reference_depth = _skin_depth_m(background_rho, frequencies)
    proxy_xy = _skin_depth_m(
        apparent[..., 0].astype(np.float64), frequencies[:, None, None]
    )
    proxy_yx = _skin_depth_m(
        apparent[..., 1].astype(np.float64), frequencies[:, None, None]
    )
    spatial_slice = slice(edge_crop, -edge_crop or None)
    interior_xy = proxy_xy[:, spatial_slice, spatial_slice]
    interior_yx = proxy_yx[:, spatial_slice, spatial_slice]
    center = (apparent.shape[1] // 2, apparent.shape[2] // 2)
    center_xy = proxy_xy[:, center[0], center[1]]
    center_yx = proxy_yx[:, center[0], center[1]]
    rows = []
    for index, frequency in enumerate(frequencies):
        values_xy = interior_xy[index]
        values_yx = interior_yx[index]
        rows.append(
            {
                "frequency_hz": float(frequency),
                "period_s": float(1.0 / frequency),
                "background_skin_depth_m": float(reference_depth[index]),
                "background_skin_depth_cells": float(
                    reference_depth[index] / spacing[2]
                ),
                "center_rhoa_depth_proxy_zxy_m": float(center_xy[index]),
                "center_rhoa_depth_proxy_zyx_m": float(center_yx[index]),
                "interior_rhoa_depth_proxy_zxy_p05_m": float(
                    np.percentile(values_xy, 5.0)
                ),
                "interior_rhoa_depth_proxy_zxy_median_m": float(
                    np.median(values_xy)
                ),
                "interior_rhoa_depth_proxy_zxy_p95_m": float(
                    np.percentile(values_xy, 95.0)
                ),
                "interior_rhoa_depth_proxy_zyx_p05_m": float(
                    np.percentile(values_yx, 5.0)
                ),
                "interior_rhoa_depth_proxy_zyx_median_m": float(
                    np.median(values_yx)
                ),
                "interior_rhoa_depth_proxy_zyx_p95_m": float(
                    np.percentile(values_yx, 95.0)
                ),
            }
        )

    csv_path = output_dir / "frequency_depth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output_dir / "frequency_depth.png"
    xy_p05 = np.asarray(
        [row["interior_rhoa_depth_proxy_zxy_p05_m"] for row in rows]
    ) / 1000.0
    xy_p50 = np.asarray(
        [row["interior_rhoa_depth_proxy_zxy_median_m"] for row in rows]
    ) / 1000.0
    xy_p95 = np.asarray(
        [row["interior_rhoa_depth_proxy_zxy_p95_m"] for row in rows]
    ) / 1000.0
    yx_p05 = np.asarray(
        [row["interior_rhoa_depth_proxy_zyx_p05_m"] for row in rows]
    ) / 1000.0
    yx_p50 = np.asarray(
        [row["interior_rhoa_depth_proxy_zyx_median_m"] for row in rows]
    ) / 1000.0
    yx_p95 = np.asarray(
        [row["interior_rhoa_depth_proxy_zyx_p95_m"] for row in rows]
    ) / 1000.0
    fig, axis = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    axis.fill_between(
        frequencies,
        xy_p05,
        xy_p95,
        color="#70a5d8",
        alpha=0.18,
        label="rhoa Zxy proxy: P05-P95",
    )
    axis.fill_between(
        frequencies,
        yx_p05,
        yx_p95,
        color="#75b36a",
        alpha=0.18,
        label="rhoa Zyx proxy: P05-P95",
    )
    axis.semilogx(
        frequencies,
        reference_depth / 1000.0,
        "o-",
        color="#c23b22",
        label=f"background rho={background_rho:.1f} Ohm m",
    )
    axis.semilogx(
        frequencies,
        xy_p50,
        "s--",
        color="#236192",
        label="rhoa Zxy proxy: median",
    )
    axis.semilogx(
        frequencies,
        yx_p50,
        "^--",
        color="#3d7c35",
        label="rhoa Zyx proxy: median",
    )
    model_depth_km = model_shape[2] * spacing[2] / 1000.0
    axis.axhline(
        model_depth_km,
        color="#333333",
        linestyle=":",
        label=f"model depth={model_depth_km:.2f} km",
    )
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("half-space skin-depth scale / proxy (km)")
    axis.set_title("MT frequency-depth reference and rhoa-derived proxies (not DOI)")
    axis.grid(True, which="both", alpha=0.25)
    axis.invert_xaxis()
    axis.invert_yaxis()
    axis.legend(loc="best")
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    return csv_path, figure_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--slice-frequencies",
        default="10000,1000,200,30",
        help="frequencies whose horizontal response slices are shown",
    )
    parser.add_argument(
        "--background-rho",
        type=float,
        default=None,
        help="reference half-space resistivity; default is the model boundary mean",
    )
    parser.add_argument(
        "--edge-crop",
        type=int,
        default=4,
        help="receiver cells removed on each side of MT response volumes",
    )
    parser.add_argument(
        "--run-app",
        action="store_true",
        help="keep interactive cigvis windows open after saving",
    )
    args = parser.parse_args()

    results_path = args.results.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_path.parent / "cigvis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_results(results_path)
    frequencies = data["frequencies_hz"]
    model = data["resistivity_model"]
    apparent = data["apparent_resistivity"]
    phase = data["phase_deg"]
    spacing = data["spacing_m"]
    if args.edge_crop < 0 or 2 * args.edge_crop >= min(model.shape[:2]):
        raise ValueError(
            f"edge_crop={args.edge_crop} is invalid for a "
            f"{model.shape[0]}x{model.shape[1]} grid"
        )
    background_rho = (
        float(args.background_rho)
        if args.background_rho is not None
        else (
            float(data["background_resistivity_ohm_m"])
            if "background_resistivity_ohm_m" in data
            else _boundary_mean(model)
        )
    )
    if not np.isfinite(background_rho) or background_rho <= 0:
        raise ValueError("background_rho must be positive and finite")
    slice_indices = _parse_slice_frequencies(args.slice_frequencies, frequencies)

    csv_path, depth_figure, depth_rows = _write_depth_products(
        frequencies,
        apparent,
        spacing,
        model.shape,
        background_rho,
        args.edge_crop,
        output_dir,
    )
    cigvis = _import_cigvis()
    cigvis.set_axis_reversed(False, False, True)
    model_figure = output_dir / "cigvis_model.png"
    response_figure = output_dir / "cigvis_mt_response_volume.png"
    _render_model(cigvis, model, spacing, model_figure, args.run_app)
    _render_responses(
        cigvis,
        apparent,
        phase,
        frequencies,
        spacing,
        args.edge_crop,
        slice_indices,
        response_figure,
        args.run_app,
    )

    summary = {
        "results": str(results_path),
        "output_dir": str(output_dir),
        "shape": list(model.shape),
        "spacing_m": spacing.tolist(),
        "background_rho_ohm_m": background_rho,
        "background_source": (
            "command_line"
            if args.background_rho is not None
            else (
                "saved_solver_metadata"
                if "background_resistivity_ohm_m" in data
                else "model_boundary_mean"
            )
        ),
        "skin_depth_formula": "delta_m = 503 * sqrt(rho_ohm_m / frequency_hz)",
        "depth_interpretation": (
            "background values are uniform-half-space skin-depth scales; "
            "Zxy/Zyx values derived from apparent resistivity are nonlocal "
            "proxies, not DOI"
        ),
        "edge_crop_cells": args.edge_crop,
        "slice_frequencies_hz": [float(frequencies[index]) for index in slice_indices],
        "files": {
            "model_cigvis": str(model_figure),
            "responses_cigvis": str(response_figure),
            "frequency_depth_figure": str(depth_figure),
            "frequency_depth_csv": str(csv_path),
        },
        "frequency_depth": depth_rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "frequency_hz  background_skin_depth_m  "
        "rhoa_proxy_zxy_median_m  rhoa_proxy_zyx_median_m"
    )
    for row in depth_rows:
        print(
            f"{row['frequency_hz']:12g}  "
            f"{row['background_skin_depth_m']:23.1f}  "
            f"{row['interior_rhoa_depth_proxy_zxy_median_m']:24.1f}  "
            f"{row['interior_rhoa_depth_proxy_zyx_median_m']:24.1f}"
        )
    console_summary = {
        key: value for key, value in summary.items() if key != "frequency_depth"
    }
    print("MT_CIGVIS_SUMMARY=" + json.dumps(console_summary, sort_keys=True))


if __name__ == "__main__":
    main()
