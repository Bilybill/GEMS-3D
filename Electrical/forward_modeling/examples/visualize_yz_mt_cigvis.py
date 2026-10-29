"""Visualize one Y-Z model section and the MT responses at the same X line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MPL_CACHE = Path("/tmp/gems3d-matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.examples.visualize_mt_cigvis import (  # noqa: E402
    DEFAULT_RESULTS,
    _create_axis,
    _import_cigvis,
    _load_results,
    _render_cigvis,
    _robust_clim,
)


def _index(value: int | None, size: int, name: str) -> int:
    index = size // 2 if value is None else value
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{name} must be an integer")
    if index < 0 or index >= size:
        raise ValueError(f"{name}={index} is outside [0, {size - 1}]")
    return index


def _frequency_indices(text: str, frequencies: np.ndarray) -> list[int]:
    requested = [float(value) for value in text.split(",") if value.strip()]
    if not requested or any(not np.isfinite(value) or value <= 0 for value in requested):
        raise ValueError("slice frequencies must be positive comma-separated values")
    return sorted(
        {
            int(np.argmin(np.abs(np.log(frequencies) - np.log(value))))
            for value in requested
        }
    )


def _plot_yz_response_section(
    model: np.ndarray,
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: np.ndarray,
    spacing: np.ndarray,
    x_index: int,
    y_index: int,
    output: Path,
) -> None:
    """Save a readable 2-D overview for one fixed-X line."""
    _, dy, dz = spacing
    ny, nz = model.shape[1:]
    y_extent = [0.0, ny * dy / 1000.0]
    depth_extent = [nz * dz / 1000.0, 0.0]
    frequency_order = np.argsort(frequencies)
    frequency_axis = frequencies[frequency_order]
    y_position = (y_index + 0.5) * dy / 1000.0

    log_model = np.log10(model[x_index].T)
    log_rhoa = np.log10(apparent[frequency_order, x_index, :, :])
    log_rhoa = log_rhoa.transpose(0, 2, 1)
    phase_line = phase[frequency_order, x_index, :, :].transpose(0, 2, 1)
    rho_clim = _robust_clim(log_rhoa)
    phase_limit = max(float(np.percentile(np.abs(phase_line), 98.0)), 1.0)

    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.15, 1.0, 1.0))
    model_axis = figure.add_subplot(grid[:, 0])
    rho_xy_axis = figure.add_subplot(grid[0, 1])
    phase_xy_axis = figure.add_subplot(grid[0, 2])
    rho_yx_axis = figure.add_subplot(grid[1, 1])
    phase_yx_axis = figure.add_subplot(grid[1, 2])

    model_image = model_axis.imshow(
        log_model,
        extent=[*y_extent, *depth_extent],
        origin="upper",
        aspect="auto",
        cmap="turbo",
        vmin=_robust_clim(log_model, 1.0, 99.0)[0],
        vmax=_robust_clim(log_model, 1.0, 99.0)[1],
    )
    model_axis.axvline(y_position, color="white", linestyle="--", linewidth=1.2)
    model_axis.set_title(f"Model Y-Z section | x index={x_index}")
    model_axis.set_xlabel("Y (km)")
    model_axis.set_ylabel("Depth (km)")
    figure.colorbar(model_image, ax=model_axis, label="log10 rho (Ohm m)")

    panel_data = (
        (rho_xy_axis, log_rhoa[:, 0, :], "log10 rhoa Zxy", "viridis", rho_clim),
        (phase_xy_axis, phase_line[:, 0, :], "phase Zxy (deg)", "RdBu_r", (-phase_limit, phase_limit)),
        (rho_yx_axis, log_rhoa[:, 1, :], "log10 rhoa Zyx", "viridis", rho_clim),
        (phase_yx_axis, phase_line[:, 1, :], "phase Zyx (deg)", "RdBu_r", (-phase_limit, phase_limit)),
    )
    for axis, values, title, cmap, clim in panel_data:
        image = axis.imshow(
            values,
            extent=[*y_extent, frequency_axis[0], frequency_axis[-1]],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
        )
        axis.set_yscale("log")
        axis.axvline(y_position, color="white", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Y (km)")
        axis.set_ylabel("Frequency (Hz)")
        figure.colorbar(image, ax=axis, fraction=0.046)

    figure.suptitle(
        f"Fixed-X MT section: x={(x_index + 0.5) * spacing[0] / 1000.0:.3f} km; "
        f"marked station y={(y_index + 0.5) * dy / 1000.0:.3f} km",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_station_sounding(
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: np.ndarray,
    x_index: int,
    y_index: int,
    output: Path,
) -> None:
    order = np.argsort(frequencies)
    frequency_axis = frequencies[order]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for component, label in enumerate(("Zxy", "Zyx")):
        axes[0].loglog(
            frequency_axis,
            apparent[order, x_index, y_index, component],
            "o-",
            label=label,
        )
        axes[1].semilogx(
            frequency_axis,
            phase[order, x_index, y_index, component],
            "o-",
            label=label,
        )
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Apparent resistivity (Ohm m)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle(f"MT sounding at x index={x_index}, y index={y_index}")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _response_volume_nodes(
    cigvis,
    values: np.ndarray,
    frequencies: np.ndarray,
    y_km: np.ndarray,
    frequency_indices: list[int],
    cmap: str,
    label: str,
    clim: tuple[float, float] | None = None,
):
    """Create a cigvis Y-frequency plane from shape (frequency, y)."""
    # cigvis requires at least two samples on every axis for tick generation;
    # duplicate the display-only fixed-section axis without changing values.
    volume = np.repeat(values.T[:, np.newaxis, :], 2, axis=1)
    clim = _robust_clim(volume) if clim is None else list(clim)
    nodes = cigvis.create_slices(
        volume,
        pos={"x": [], "y": [0], "z": frequency_indices},
        clim=clim,
        cmap=cmap,
        interpolation="nearest",
        intersection_lines=False,
    )
    nodes += _create_axis(
        cigvis,
        volume.shape,
        [y_km, np.asarray([0.0, 1.0]), frequencies],
        ["Y (km)", "Fixed-X section", "Frequency (Hz)"],
    )
    nodes.append(cigvis.create_colorbar(cmap, clim, label_str=label))
    return nodes


def _render_yz_cigvis(
    cigvis,
    model: np.ndarray,
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: np.ndarray,
    spacing: np.ndarray,
    x_index: int,
    frequency_indices: list[int],
    model_output: Path,
    response_output: Path,
    run_app: bool,
) -> None:
    dx, dy, dz = spacing
    log_model = np.log10(model)
    model_clim = _robust_clim(log_model, 1.0, 99.0)
    model_nodes = cigvis.create_slices(
        log_model,
        pos={"x": [x_index], "y": [], "z": []},
        clim=model_clim,
        cmap="turbo",
        interpolation="nearest",
        intersection_lines=False,
    )
    y_km = (np.arange(model.shape[1], dtype=float) + 0.5) * dy / 1000.0
    z_km = (np.arange(model.shape[2], dtype=float) + 0.5) * dz / 1000.0
    model_nodes += _create_axis(
        cigvis,
        model.shape,
        [(np.arange(model.shape[0]) + 0.5) * dx / 1000.0, y_km, z_km],
        ["X (km)", "Y (km)", "Depth (km)"],
    )
    model_nodes.append(
        cigvis.create_colorbar("turbo", model_clim, label_str="log10 rho (Ohm m)")
    )
    _render_cigvis(
        cigvis,
        model_nodes,
        model_output,
        run_app=run_app,
        size=(1200, 900),
        axis_scales=[1.0, float(dy / dx), float(dz / dx)],
        title=f"Y-Z resistivity section at X index {x_index}",
    )

    rho_xy = np.log10(apparent[:, x_index, :, 0])
    rho_yx = np.log10(apparent[:, x_index, :, 1])
    phase_limit = max(
        float(np.percentile(np.abs(phase[:, x_index, :, :]), 98.0)), 1.0
    )
    response_panels = [
        _response_volume_nodes(
            cigvis,
            rho_xy,
            frequencies,
            y_km,
            frequency_indices,
            "viridis",
            "log10 rhoa Zxy",
        ),
        _response_volume_nodes(
            cigvis,
            rho_yx,
            frequencies,
            y_km,
            frequency_indices,
            "viridis",
            "log10 rhoa Zyx",
        ),
        _response_volume_nodes(
            cigvis,
            phase[:, x_index, :, 0],
            frequencies,
            y_km,
            frequency_indices,
            "RdBu_r",
            "phase Zxy (deg)",
            (-phase_limit, phase_limit),
        ),
        _response_volume_nodes(
            cigvis,
            phase[:, x_index, :, 1],
            frequencies,
            y_km,
            frequency_indices,
            "RdBu_r",
            "phase Zyx (deg)",
            (-phase_limit, phase_limit),
        ),
    ]
    selected = ", ".join(f"{frequencies[index]:g}" for index in frequency_indices)
    _render_cigvis(
        cigvis,
        response_panels,
        response_output,
        run_app=run_app,
        size=(1900, 1450),
        axis_scales=[1.0, 1.0, 1.5],
        grid=(2, 2),
        share=True,
        title=[
            f"Fixed-X Zxy apparent rho | slices {selected} Hz",
            f"Fixed-X Zyx apparent rho | slices {selected} Hz",
            f"Fixed-X Zxy phase | slices {selected} Hz",
            f"Fixed-X Zyx phase | slices {selected} Hz",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--x-index", type=int, default=None)
    parser.add_argument("--y-index", type=int, default=None)
    parser.add_argument(
        "--slice-frequencies",
        default="10000,1000,200,30",
        help="frequencies shown as Y-frequency cigvis slices",
    )
    parser.add_argument("--run-app", action="store_true")
    args = parser.parse_args()

    results_path = args.results.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_path.parent / "cigvis_yz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_results(results_path)
    frequencies = data["frequencies_hz"]
    model = data["resistivity_model"]
    apparent = data["apparent_resistivity"]
    phase = data["phase_deg"]
    spacing = data["spacing_m"]
    x_index = _index(args.x_index, model.shape[0], "x_index")
    y_index = _index(args.y_index, model.shape[1], "y_index")
    frequency_indices = _frequency_indices(args.slice_frequencies, frequencies)

    static_output = output_dir / f"yz_mt_section_x{x_index}.png"
    sounding_output = output_dir / f"yz_mt_sounding_x{x_index}_y{y_index}.png"
    _plot_yz_response_section(
        model,
        apparent,
        phase,
        frequencies,
        spacing,
        x_index,
        y_index,
        static_output,
    )
    _plot_station_sounding(
        apparent, phase, frequencies, x_index, y_index, sounding_output
    )

    cigvis = _import_cigvis()
    cigvis.set_axis_reversed(False, False, True)
    model_cigvis = output_dir / f"cigvis_yz_model_x{x_index}.png"
    response_cigvis = output_dir / f"cigvis_yz_mt_response_x{x_index}.png"
    _render_yz_cigvis(
        cigvis,
        model,
        apparent,
        phase,
        frequencies,
        spacing,
        x_index,
        frequency_indices,
        model_cigvis,
        response_cigvis,
        args.run_app,
    )

    summary = {
        "results": str(results_path),
        "output_dir": str(output_dir),
        "x_index": x_index,
        "x_center_m": float((x_index + 0.5) * spacing[0]),
        "y_index": y_index,
        "y_center_m": float((y_index + 0.5) * spacing[1]),
        "model_shape": list(model.shape),
        "spacing_m": spacing.tolist(),
        "slice_frequencies_hz": [float(frequencies[index]) for index in frequency_indices],
        "component_order": ["Zxy", "Zyx"],
        "interpretation": (
            "The Y-Z model plane is fixed at x_index. The MT panels show the "
            "surface response along the same fixed-X line as Y-frequency planes."
        ),
        "files": {
            "static_yz_section": str(static_output),
            "station_sounding": str(sounding_output),
            "cigvis_yz_model": str(model_cigvis),
            "cigvis_yz_response": str(response_cigvis),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
