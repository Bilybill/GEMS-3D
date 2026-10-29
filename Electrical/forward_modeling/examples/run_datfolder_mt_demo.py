"""Run and visualize the optimized MT solver on a GMESDataset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


GEMS_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = GEMS_ROOT.parent
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.mt_forward import MTForward3D  # noqa: E402


DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "GMESDataset"
    / "DATAFOLDER"
    / "Cache"
    / "ForwardBatch"
    / "serpentinized"
    / "forward_bundle.npz"
)
DEFAULT_OUTPUT = (
    GEMS_ROOT
    / "outputs"
    / "mt_datfolder_demo"
    / "serpentinized_optimized"
)
NIPS_MT_CANONICAL_FREQS_HZ = (
    10000.0,
    8000.0,
    6000.0,
    4000.0,
    3000.0,
    2000.0,
    1500.0,
    1000.0,
    800.0,
    600.0,
    400.0,
    300.0,
    200.0,
    150.0,
    100.0,
    80.0,
    60.0,
    40.0,
    30.0,
)


def _parse_frequencies(text: str) -> list[float]:
    if text.strip().lower() in {"nips19", "paper19"}:
        return list(NIPS_MT_CANONICAL_FREQS_HZ)
    frequencies = [float(value) for value in text.split(",") if value.strip()]
    if not frequencies or any(
        not np.isfinite(value) or value <= 0.0 for value in frequencies
    ):
        raise ValueError(
            "frequencies must be a non-empty comma-separated list of positive values"
        )
    return frequencies


def _representative_frequency_indices(
    frequencies: list[float], maximum: int = 6
) -> np.ndarray:
    if len(frequencies) <= maximum:
        return np.arange(len(frequencies), dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, len(frequencies) - 1, maximum)).astype(np.int64)
    )


def _scalar(bundle: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in bundle.files:
        return default
    value = bundle[key]
    return value.item() if value.shape == () else value


def _load_model(
    input_path: Path,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, object]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"input bundle not found: {input_path}")
    with np.load(input_path, allow_pickle=True) as bundle:
        model_key = "mt_res_model" if "mt_res_model" in bundle.files else "res_model"
        if model_key not in bundle.files:
            raise KeyError("bundle must contain mt_res_model or res_model")
        model = np.asarray(bundle[model_key], dtype=np.float64)
        spacing_keys = (
            ("mt_dx", "mt_dy", "mt_dz")
            if "mt_dx" in bundle.files
            else ("dx", "dy", "dz")
        )
        spacing = tuple(float(_scalar(bundle, key)) for key in spacing_keys)
        metadata = {
            "model_key": model_key,
            "source_anomaly": str(_scalar(bundle, "anomaly_name_en", "unknown")),
            "source_anomaly_type": str(_scalar(bundle, "anomaly_type", "unknown")),
            "source_mt_status": str(_scalar(bundle, "mt_status", "unknown")),
            "source_frequencies_hz": (
                np.asarray(bundle["mt_freqs_hz"], dtype=float).tolist()
                if "mt_freqs_hz" in bundle.files
                else None
            ),
        }
    if model.ndim != 3 or min(model.shape) <= 0:
        raise ValueError(f"MT model must be a non-empty 3-D array, got {model.shape}")
    if not np.isfinite(model).all() or np.any(model <= 0.0):
        raise ValueError("MT model must contain positive finite resistivity values")
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError(f"invalid model spacing: {spacing}")
    return model, spacing, metadata


def _refine_vertical_model(
    model: np.ndarray,
    spacing: tuple[float, float, float],
    factor: int,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Split each vertical cell into equal, piecewise-constant subcells."""
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise TypeError("vertical_refinement must be an integer")
    if factor <= 0:
        raise ValueError("vertical_refinement must be positive")
    if factor == 1:
        return model, spacing
    refined = np.repeat(model, factor, axis=2)
    return refined, (spacing[0], spacing[1], spacing[2] / factor)


def _boundary_mean(model: np.ndarray) -> float:
    boundary_sum = float(model.astype(np.float64).sum())
    boundary_count = int(np.prod(model.shape))
    if min(model.shape) > 2:
        boundary_sum -= float(model[1:-1, 1:-1, 1:-1].sum(dtype=np.float64))
        boundary_count -= int(np.prod(np.asarray(model.shape) - 2))
    return boundary_sum / boundary_count


def _geometric_padding_extent(
    base_cell_width: float, layers: int, growth: float
) -> float:
    return base_cell_width * sum(growth**index for index in range(1, layers + 1))


def _sha256_float32(model: np.ndarray) -> str:
    values = np.ascontiguousarray(model, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_model(
    model: np.ndarray, spacing: tuple[float, float, float], output: Path
) -> None:
    plt = _get_pyplot()
    nx, ny, nz = model.shape
    dx, dy, dz = spacing
    mid_y, mid_z = ny // 2, nz // 2
    log_model = np.log10(model)
    vmin, vmax = np.percentile(log_model, [1.0, 99.0])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x_extent = [0.0, nx * dx]
    y_extent = [0.0, ny * dy]
    z_extent = [nz * dz, 0.0]
    image_xz = axes[0].imshow(
        log_model[:, mid_y, :].T,
        extent=[*x_extent, *z_extent],
        aspect="auto",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"3-D resistivity model, y={(mid_y + 0.5) * dy:.0f} m")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("depth (m)")
    figure.colorbar(image_xz, ax=axes[0], label="log10 resistivity (Ohm m)")
    image_xy = axes[1].imshow(
        log_model[:, :, mid_z].T,
        extent=[*x_extent, *y_extent[::-1]],
        aspect="equal",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Horizontal slice, z={(mid_z + 0.5) * dz:.0f} m")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    figure.colorbar(image_xy, ax=axes[1], label="log10 resistivity (Ohm m)")
    figure.suptitle("GMESDataset 3-D MT input model")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_responses(
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: list[float],
    spacing: tuple[float, float, float],
    output: Path,
) -> None:
    plt = _get_pyplot()
    nx, ny = apparent.shape[1:3]
    dx, dy, _ = spacing
    plot_indices = _representative_frequency_indices(frequencies)
    log_apparent = np.log10(np.maximum(apparent, 1.0e-12))
    rho_min, rho_max = np.percentile(log_apparent, [2.0, 98.0])
    phase_limit = max(float(np.percentile(np.abs(phase), 98.0)), 1.0)
    figure, axes = plt.subplots(
        len(plot_indices),
        4,
        figsize=(16, 4.0 * len(plot_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [0.0, nx * dx, ny * dy, 0.0]
    for row, frequency_index in enumerate(plot_indices):
        frequency = frequencies[frequency_index]
        panels = (
            (
                log_apparent[frequency_index, :, :, 0],
                "Zxy apparent rho",
                "viridis",
                rho_min,
                rho_max,
                "log10 Ohm m",
            ),
            (
                log_apparent[frequency_index, :, :, 1],
                "Zyx apparent rho",
                "viridis",
                rho_min,
                rho_max,
                "log10 Ohm m",
            ),
            (
                phase[frequency_index, :, :, 0],
                "Zxy phase",
                "RdBu_r",
                -phase_limit,
                phase_limit,
                "degrees",
            ),
            (
                phase[frequency_index, :, :, 1],
                "Zyx phase",
                "RdBu_r",
                -phase_limit,
                phase_limit,
                "degrees",
            ),
        )
        for column, (data, label, cmap, lower, upper, colorbar_label) in enumerate(
            panels
        ):
            image = axes[row, column].imshow(
                data.T,
                extent=extent,
                aspect="equal",
                origin="upper",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            axes[row, column].set_title(f"{label}\n{frequency:g} Hz")
            axes[row, column].set_xlabel("x (m)")
            axes[row, column].set_ylabel("y (m)")
            figure.colorbar(
                image,
                ax=axes[row, column],
                label=colorbar_label,
                fraction=0.046,
            )
    figure.suptitle(
        "Optimized CUDA MT response on a GMESDataset 3-D model", fontsize=15
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_station_sounding(
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: list[float],
    output: Path,
) -> None:
    plt = _get_pyplot()
    order = np.argsort(frequencies)
    freq = np.asarray(frequencies)[order]
    center = (apparent.shape[1] // 2, apparent.shape[2] // 2)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for polarization, label in enumerate(("Zxy", "Zyx")):
        axes[0].loglog(
            freq,
            apparent[order, center[0], center[1], polarization],
            "o-",
            label=label,
        )
        axes[1].semilogx(
            freq,
            phase[order, center[0], center[1], polarization],
            "o-",
            label=label,
        )
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("apparent resistivity (Ohm m)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("phase (degrees)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    figure.suptitle("Central-station MT sounding")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_response_variations(
    apparent: np.ndarray,
    phase: np.ndarray,
    frequencies: list[float],
    spacing: tuple[float, float, float],
    output: Path,
) -> None:
    """Plot spatial variations after removing each frequency map median."""
    plt = _get_pyplot()
    nx, ny = apparent.shape[1:3]
    dx, dy, _ = spacing
    margin = min(4, max(1, min(nx, ny) // 5))
    plot_indices = _representative_frequency_indices(frequencies)
    interior = (slice(margin, -margin), slice(margin, -margin))
    delta_rho = np.log10(np.maximum(apparent, 1.0e-12))
    delta_rho -= np.median(delta_rho[:, interior[0], interior[1], :], axis=(1, 2), keepdims=True)
    delta_rho = delta_rho[:, interior[0], interior[1], :]
    delta_phase = phase - np.median(
        phase[:, interior[0], interior[1], :], axis=(1, 2), keepdims=True
    )
    delta_phase = delta_phase[:, interior[0], interior[1], :]
    rho_limit = max(float(np.percentile(np.abs(delta_rho), 98.0)), 0.01)
    phase_limit = max(float(np.percentile(np.abs(delta_phase), 98.0)), 0.1)
    figure, axes = plt.subplots(
        len(plot_indices),
        4,
        figsize=(16, 4.0 * len(plot_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [margin * dx, (nx - margin) * dx, (ny - margin) * dy, margin * dy]
    for row, frequency_index in enumerate(plot_indices):
        frequency = frequencies[frequency_index]
        panels = (
            (
                delta_rho[frequency_index, :, :, 0],
                "Delta log10 rhoa Zxy",
                rho_limit,
                "log10 Ohm m",
            ),
            (
                delta_rho[frequency_index, :, :, 1],
                "Delta log10 rhoa Zyx",
                rho_limit,
                "log10 Ohm m",
            ),
            (
                delta_phase[frequency_index, :, :, 0],
                "Delta phase Zxy",
                phase_limit,
                "degrees",
            ),
            (
                delta_phase[frequency_index, :, :, 1],
                "Delta phase Zyx",
                phase_limit,
                "degrees",
            ),
        )
        for column, (data, title, limit, colorbar_label) in enumerate(panels):
            image = axes[row, column].imshow(
                data.T,
                extent=extent,
                aspect="equal",
                origin="upper",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set_title(f"{title}\n{frequency:g} Hz")
            axes[row, column].set_xlabel("x (m)")
            axes[row, column].set_ylabel("y (m)")
            figure.colorbar(
                image,
                ax=axes[row, column],
                label=colorbar_label,
                fraction=0.046,
            )
    figure.suptitle(
        f"Interior MT variation relative to map median ({margin}-cell edge removed)",
        fontsize=15,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frequencies",
        default="1000,200,20",
        help="comma-separated Hz values, or nips19 for the paper's 10 kHz-30 Hz grid",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--air-layers", type=int, default=8)
    parser.add_argument("--air-resistivity", type=float, default=1.0e8)
    parser.add_argument(
        "--vertical-refinement",
        type=int,
        default=1,
        help="split every vertical cell into this many constant-resistivity cells",
    )
    parser.add_argument("--padding-layers", type=int, default=10)
    parser.add_argument("--padding-growth", type=float, default=1.4)
    parser.add_argument(
        "--background-resistivity",
        type=float,
        default=None,
        help="padding/remote-boundary resistivity; default is the source boundary mean",
    )
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--solver-max-iterations", type=int, default=50000)
    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "--frequency-continuation",
        dest="frequency_continuation",
        action="store_true",
    )
    continuation_group.add_argument(
        "--cold-start",
        dest="frequency_continuation",
        action="store_false",
    )
    parser.set_defaults(frequency_continuation=False)
    args = parser.parse_args()

    frequencies = _parse_frequencies(args.frequencies)
    source_model, source_spacing, source_metadata = _load_model(
        args.input_bundle.resolve()
    )
    model, spacing = _refine_vertical_model(
        source_model, source_spacing, args.vertical_refinement
    )
    background_resistivity = (
        _boundary_mean(source_model)
        if args.background_resistivity is None
        else args.background_resistivity
    )
    source_model_sha256 = _sha256_float32(source_model)
    refined_model_sha256 = _sha256_float32(model)
    padding_extent = {
        "x_each_side": _geometric_padding_extent(
            spacing[0], args.padding_layers, args.padding_growth
        ),
        "y_each_side": _geometric_padding_extent(
            spacing[1], args.padding_layers, args.padding_growth
        ),
        "bottom": _geometric_padding_extent(
            spacing[2], args.padding_layers, args.padding_growth
        ),
    }
    if args.device.startswith("cuda"):
        try:
            torch.cuda.init()
        except Exception as exc:
            raise RuntimeError("CUDA was requested but could not be initialized") from exc
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rho = torch.as_tensor(model, dtype=torch.float64, device=device)
    solver = MTForward3D(
        freqs=frequencies,
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
        boundary_mode="explicit_air",
        air_layers=args.air_layers,
        air_resistivity=args.air_resistivity,
        solver_tolerance=args.solver_tolerance,
        solver_max_iterations=args.solver_max_iterations,
        frequency_continuation=args.frequency_continuation,
        background_resistivity=background_resistivity,
        padding_layers=args.padding_layers,
        padding_growth=args.padding_growth,
    )
    print(
        "Refined MT configuration: "
        f"shape={model.shape}, spacing={spacing} m, "
        f"padding={args.padding_layers}x{args.padding_growth:g}, "
        f"background={background_resistivity:.6g} Ohm m",
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        apparent_tensor, phase_tensor = solver(rho)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    apparent = apparent_tensor.detach().cpu().numpy()
    phase = phase_tensor.detach().cpu().numpy()

    _plot_model(model, spacing, args.output_dir / "model_slices.png")
    _plot_responses(
        apparent,
        phase,
        frequencies,
        spacing,
        args.output_dir / "mt_response_maps.png",
    )
    _plot_response_variations(
        apparent,
        phase,
        frequencies,
        spacing,
        args.output_dir / "mt_response_variations.png",
    )
    _plot_station_sounding(
        apparent,
        phase,
        frequencies,
        args.output_dir / "central_station_sounding.png",
    )
    np.savez_compressed(
        args.output_dir / "mt_demo_results.npz",
        frequencies_hz=np.asarray(frequencies),
        spacing_m=np.asarray(spacing),
        resistivity_model=model.astype(np.float32),
        apparent_resistivity=apparent.astype(np.float32),
        phase_deg=phase.astype(np.float32),
        source_shape=np.asarray(source_model.shape, dtype=np.int32),
        source_spacing_m=np.asarray(source_spacing),
        vertical_refinement=np.int32(args.vertical_refinement),
        padding_layers=np.int32(args.padding_layers),
        padding_growth=np.float64(args.padding_growth),
        background_resistivity_ohm_m=np.float64(background_resistivity),
        air_layers=np.int32(args.air_layers),
        air_resistivity_ohm_m=np.float64(args.air_resistivity),
        solver_tolerance=np.float64(args.solver_tolerance),
        solver_max_iterations=np.int32(args.solver_max_iterations),
        frequency_continuation=np.bool_(args.frequency_continuation),
        axis_order=np.asarray(("x", "y", "z")),
        component_order=np.asarray(("Zxy", "Zyx")),
        resampling_method=np.asarray("piecewise_constant_cell_split_z"),
        phase_convention=np.asarray(
            "atan2(Im(Z),Re(Z)) degrees; Zxy/Zyx signs retained"
        ),
        source_model_sha256=np.asarray(source_model_sha256),
        refined_model_sha256=np.asarray(refined_model_sha256),
        padding_extent_m=np.asarray(tuple(padding_extent.values())),
    )
    summary = {
        "input_bundle": str(args.input_bundle.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "shape": list(model.shape),
        "spacing_m": list(spacing),
        "physical_extent_m": [
            model.shape[index] * spacing[index] for index in range(3)
        ],
        "source_shape": list(source_model.shape),
        "source_spacing_m": list(source_spacing),
        "source_physical_extent_m": [
            source_model.shape[index] * source_spacing[index]
            for index in range(3)
        ],
        "vertical_refinement": args.vertical_refinement,
        "resampling_method": "piecewise_constant_cell_split_z",
        "axis_order": ["x", "y", "z"],
        "component_order": ["Zxy", "Zyx"],
        "phase_convention": "atan2(Im(Z),Re(Z)) degrees; Zxy/Zyx signs retained",
        "source_model_sha256_float32": source_model_sha256,
        "refined_model_sha256_float32": refined_model_sha256,
        "frequencies_hz": frequencies,
        "map_frequencies_hz": [
            frequencies[index]
            for index in _representative_frequency_indices(frequencies)
        ],
        "boundary_mode": "explicit_air",
        "air_layers": args.air_layers,
        "air_thickness_m": args.air_layers * spacing[2],
        "air_resistivity_ohm_m": args.air_resistivity,
        "background_resistivity_ohm_m": background_resistivity,
        "source_boundary_mean_resistivity_ohm_m": _boundary_mean(source_model),
        "refined_boundary_mean_resistivity_ohm_m": _boundary_mean(model),
        "padding_layers": args.padding_layers,
        "padding_growth": args.padding_growth,
        "padding_extent_m": padding_extent,
        "padded_shape": [
            model.shape[0] + 2 * args.padding_layers,
            model.shape[1] + 2 * args.padding_layers,
            model.shape[2] + args.air_layers + args.padding_layers,
        ],
        "solver_tolerance": args.solver_tolerance,
        "solver_max_iterations": args.solver_max_iterations,
        "frequency_continuation": args.frequency_continuation,
        "elapsed_seconds": elapsed,
        "resistivity_range_ohm_m": [float(model.min()), float(model.max())],
        "apparent_resistivity_range_ohm_m": [
            float(apparent.min()),
            float(apparent.max()),
        ],
        "phase_range_deg": [float(phase.min()), float(phase.max())],
        "source_metadata": source_metadata,
        "figures": {
            "model": str((args.output_dir / "model_slices.png").resolve()),
            "responses": str((args.output_dir / "mt_response_maps.png").resolve()),
            "variations": str(
                (args.output_dir / "mt_response_variations.png").resolve()
            ),
            "sounding": str(
                (args.output_dir / "central_station_sounding.png").resolve()
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
