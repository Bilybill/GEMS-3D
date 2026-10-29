"""Cross-validate the CUDA MT solver against MT3D_MFEM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MPL_CACHE = Path("/tmp/gems3d-matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.colors as colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


GEMS_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = GEMS_ROOT.parent
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.mt_forward import MTForward3D  # noqa: E402


MU0 = 4.0e-7 * np.pi
GEMS_PADDING_LAYERS = 10
GEMS_PADDING_GROWTH = 1.4
STATION_DEPTH_FRACTION = 1.0e-4


def _parse_tuple(text: str, value_type, name: str):
    values = tuple(value_type(value) for value in text.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    return values


def _load_mfem_bridge(mfem_root: Path):
    bridge_path = mfem_root / "scripts" / "create_legacy_cuda_mt3d_example.py"
    spec = importlib.util.spec_from_file_location("mt3d_mfem_bridge", bridge_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MFEM bridge generator from {bridge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _centered_block_model(
    shape: tuple[int, int, int], background_rho: float, anomaly_rho: float
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    nx, ny, nz = shape
    model = np.full(shape, background_rho, dtype=np.float64)
    width_x = max(2, nx // 8)
    width_y = max(2, ny // 8)
    x0 = (nx - width_x) // 2
    y0 = (ny - width_y) // 2
    z0 = max(1, nz // 6)
    z1 = min(nz - 1, max(z0 + 2, nz // 2))
    block = (slice(x0, x0 + width_x), slice(y0, y0 + width_y), slice(z0, z1))
    model[block] = anomaly_rho
    return model, block


def _write_mfem_case(
    case_dir: Path,
    case_name: str,
    model: np.ndarray,
    spacing: tuple[float, float, float],
    frequencies: list[float],
    air_layers: int,
    background_rho: float,
    padding_layers: int,
    padding_growth: float,
    mfem_root: Path,
    bridge,
) -> None:
    nx, ny, nz = model.shape
    dx, dy, dz = spacing
    mesh_path = case_dir / "mesh" / f"{case_name}.msh"
    x_levels, y_levels, z_levels = bridge.padded_mesh_levels(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        npad=padding_layers,
        growth=padding_growth,
        air_layers=air_layers,
    )
    markers = bridge.write_structured_tet_mesh_from_levels(
        mesh_path, x_levels, y_levels, z_levels, air_layers
    )
    padded_model = np.full(
        (
            nx + 2 * padding_layers,
            ny + 2 * padding_layers,
            nz + padding_layers,
        ),
        background_rho,
        dtype=np.float64,
    )
    padding = padding_layers
    padded_model[padding : padding + nx, padding : padding + ny, :nz] = model
    bridge.write_cond(
        case_dir / f"{case_name}.cond",
        padded_model,
        markers,
        1.0 / background_rho,
    )
    _write_subcell_comp(
        case_dir / f"{case_name}.comp", frequencies, nx, ny, dx, dy, dz
    )
    bridge.write_config(case_dir / f"{case_name}.config", case_name)

    solver_text = (mfem_root / "examples" / "3D-1" / "3D-1.solver").read_text()
    solver_text = solver_text.replace(
        "fgmres_maxit         100", "fgmres_maxit         300"
    )
    solver_text = solver_text.replace(
        "prec_type            0", "prec_type            1"
    )
    solver_text = solver_text.replace(
        "pcg_maxit            20", "pcg_maxit            50"
    )
    solver_text = solver_text.replace(
        "pcg_tol              1e-2", "pcg_tol              1e-3"
    )
    (case_dir / f"{case_name}.solver").write_text(solver_text)


def _write_subcell_comp(
    path: Path,
    frequencies: list[float],
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    # Cell centers lie on a tetrahedral face diagonal. Four symmetric samples
    # avoid ambiguous one-sided interpolation and are averaged as complex Z.
    offsets = ((-0.2, 0.0), (0.2, 0.0), (0.0, -0.2), (0.0, 0.2))
    station_depth = STATION_DEPTH_FRACTION * dz
    with path.open("w") as stream:
        stream.write(f"{len(frequencies)}\n")
        for period_id, frequency in enumerate(frequencies, start=1):
            stream.write(f"{period_id}\t{1.0 / frequency:.12g}\n")
        stream.write(f"\n{nx * ny * len(offsets)}\n")
        site_id = 1
        for ix in range(nx):
            center_x = (ix + 0.5 - nx / 2.0) * dx
            for iy in range(ny):
                center_y = (iy + 0.5 - ny / 2.0) * dy
                for offset_x, offset_y in offsets:
                    x = center_x + offset_x * dx
                    y = center_y + offset_y * dy
                    stream.write(
                        f"{site_id}\t{x:.12g}\t{y:.12g}\t{station_depth:.12g}\n"
                    )
                    site_id += 1


def _run_mfem(
    case_dir: Path, case_name: str, executable: Path, processes: int
) -> None:
    command = [
        "mpirun",
        "-np",
        str(processes),
        str(executable),
        f"{case_name}.config",
    ]
    result = subprocess.run(
        command,
        cwd=case_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    log = result.stdout + result.stderr
    (case_dir / "mt3d.log").write_text(log)
    if result.returncode != 0:
        raise RuntimeError(
            f"MT3D_MFEM failed for {case_name}; see {case_dir / 'mt3d.log'}"
        )
    if "No convergence" in log:
        raise RuntimeError(
            f"MT3D_MFEM did not converge for {case_name}; see {case_dir / 'mt3d.log'}"
        )


def _read_comp_metadata(
    path: Path,
) -> tuple[list[float], dict[int, tuple[float, float, float]]]:
    tokens = path.read_text().split()
    cursor = 0
    n_periods = int(tokens[cursor])
    cursor += 1
    periods = []
    for _ in range(n_periods):
        cursor += 1  # period identifier
        period = float(tokens[cursor])
        cursor += 1
        if not np.isfinite(period) or period <= 0.0:
            raise RuntimeError(f"invalid MFEM period in {path}")
        periods.append(period)
    n_sites = int(tokens[cursor])
    cursor += 1
    sites = {}
    for _ in range(n_sites):
        site_id = int(tokens[cursor])
        sites[site_id] = tuple(float(tokens[cursor + offset]) for offset in (1, 2, 3))
        cursor += 4
    return [1.0 / period for period in periods], sites


def _validate_reused_mfem_metadata(
    results_dir: Path,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    frequencies: list[float],
    air_layers: int,
    background_rho: float,
    anomaly_rho: float,
    padding_layers: int,
    padding_growth: float,
) -> None:
    """Reject a result directory generated for a different validation case."""
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"reused MFEM results must include metadata at {summary_path}"
        )
    try:
        metadata = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read MFEM validation metadata: {summary_path}") from exc

    if tuple(metadata.get("shape", ())) != tuple(shape):
        raise ValueError(
            f"MFEM result shape {metadata.get('shape')} does not match requested {list(shape)}"
        )
    recorded_spacing = metadata.get("spacing_m")
    if recorded_spacing is None or not np.allclose(
        recorded_spacing, spacing, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"MFEM result spacing {recorded_spacing} does not match requested {list(spacing)}"
        )
    recorded_frequencies = metadata.get("frequencies_hz")
    if recorded_frequencies is None or not np.allclose(
        recorded_frequencies, frequencies, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError(
            f"MFEM result frequencies {recorded_frequencies} do not match requested {frequencies}"
        )
    if int(metadata.get("air_layers", -1)) != int(air_layers):
        raise ValueError("MFEM result air-layer count does not match the requested case")
    if not np.isclose(
        float(metadata.get("background_rho_ohm_m", np.nan)), background_rho
    ):
        raise ValueError("MFEM result background resistivity does not match the requested case")
    if not np.isclose(
        float(metadata.get("anomaly_rho_ohm_m", np.nan)), anomaly_rho
    ):
        raise ValueError("MFEM result anomaly resistivity does not match the requested case")
    padding = metadata.get("matched_side_bottom_padding", {})
    if int(padding.get("layers", -1)) != int(padding_layers) or not np.isclose(
        float(padding.get("growth", np.nan)), padding_growth
    ):
        raise ValueError("MFEM result padding configuration does not match the requested case")


def _read_mfem_responses(
    case_dir: Path,
    case_name: str,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    frequencies: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, _ = shape
    dx, dy, _ = spacing
    comp_frequencies, sites = _read_comp_metadata(case_dir / f"{case_name}.comp")
    if len(comp_frequencies) != len(frequencies) or not np.allclose(
        comp_frequencies, frequencies, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError(
            f"{case_name}.comp frequencies {comp_frequencies} do not match requested {frequencies}"
        )
    impedance = np.zeros((len(frequencies), nx, ny, 2), dtype=np.complex128)
    sample_count = np.zeros((len(frequencies), nx, ny), dtype=np.int64)

    for frequency_index, frequency in enumerate(frequencies):
        solution_path = (
            case_dir
            / "MT_forward_solutions"
            / f"period{frequency_index + 1}_amr1.sol"
        )
        rows = np.loadtxt(solution_path, comments="%", ndmin=2)
        for row in rows:
            site_id = int(row[1])
            x, y, _ = sites[site_id]
            ix = int(round(x / dx + nx / 2.0 - 0.5))
            iy = int(round(y / dy + ny / 2.0 - 0.5))
            if not (0 <= ix < nx and 0 <= iy < ny):
                raise RuntimeError(f"MFEM site {site_id} lies outside the model grid")
            impedance[frequency_index, ix, iy, 0] += complex(row[4], row[5])
            impedance[frequency_index, ix, iy, 1] += complex(row[6], row[7])
            sample_count[frequency_index, ix, iy] += 1

        if not np.all(sample_count[frequency_index] == 4):
            raise RuntimeError("each MFEM cell must contain four subcell samples")
        impedance[frequency_index] /= sample_count[frequency_index, :, :, None]

    apparent = np.empty(impedance.shape, dtype=np.float64)
    for frequency_index, frequency in enumerate(frequencies):
        apparent[frequency_index] = (
            np.abs(impedance[frequency_index]) ** 2 / (2.0 * np.pi * frequency * MU0)
        )
    phase = np.rad2deg(np.angle(impedance))
    if not np.isfinite(apparent).all() or not np.isfinite(phase).all():
        raise RuntimeError("MT3D_MFEM response contains missing or non-finite values")
    return apparent, phase


def _run_gems(
    model: np.ndarray,
    spacing: tuple[float, float, float],
    frequencies: list[float],
    device_index: int,
    boundary_mode: str,
    air_layers: int,
    air_resistivity: float,
    solver_tolerance: float,
    solver_max_iterations: int,
    frequency_continuation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda", device_index)
    rho = torch.as_tensor(model, dtype=torch.float64, device=device)
    solver = MTForward3D(
        freqs=frequencies,
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
        boundary_mode=boundary_mode,
        air_layers=air_layers,
        air_resistivity=air_resistivity,
        solver_tolerance=solver_tolerance,
        solver_max_iterations=solver_max_iterations,
        frequency_continuation=frequency_continuation,
    ).to(device)
    with torch.inference_mode():
        apparent, phase = solver(rho)
    torch.cuda.synchronize(device)
    return apparent.cpu().numpy(), phase.cpu().numpy()


def _wrapped_phase_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.remainder(first - second + 90.0, 180.0) - 90.0


def _comparison_metrics(
    gems_apparent: np.ndarray,
    gems_phase: np.ndarray,
    mfem_apparent: np.ndarray,
    mfem_phase: np.ndarray,
    margin: int,
    frequencies: list[float] | None = None,
) -> dict[str, object]:
    crop = (slice(None), slice(margin, -margin), slice(margin, -margin), slice(None))
    gems_rho = gems_apparent[crop]
    mfem_rho = mfem_apparent[crop]
    log_difference = np.log10(gems_rho) - np.log10(mfem_rho)
    phase_difference = _wrapped_phase_difference(gems_phase[crop], mfem_phase[crop])
    symmetric_relative = 2.0 * np.abs(gems_rho - mfem_rho) / (gems_rho + mfem_rho)

    gems_log_flat = np.log10(gems_rho).ravel()
    mfem_log_flat = np.log10(mfem_rho).ravel()
    if np.std(gems_log_flat) > 0.0 and np.std(mfem_log_flat) > 0.0:
        correlation = float(np.corrcoef(gems_log_flat, mfem_log_flat)[0, 1])
    else:
        correlation = float("nan")

    metrics: dict[str, object] = {
        "median_abs_log10_rho": float(np.median(np.abs(log_difference))),
        "p95_abs_log10_rho": float(np.percentile(np.abs(log_difference), 95.0)),
        "rmse_log10_rho": float(np.sqrt(np.mean(log_difference**2))),
        "median_symmetric_relative_rho_percent": float(
            100.0 * np.median(symmetric_relative)
        ),
        "relative_l2_rho_percent": float(
            100.0 * np.linalg.norm(gems_rho - mfem_rho) / np.linalg.norm(mfem_rho)
        ),
        "log10_rho_spatial_correlation": correlation,
        "phase_bias_deg": float(np.mean(phase_difference)),
        "phase_mae_deg": float(np.mean(np.abs(phase_difference))),
        "phase_rmse_deg": float(np.sqrt(np.mean(phase_difference**2))),
        "phase_p95_deg": float(np.percentile(np.abs(phase_difference), 95.0)),
        "phase_max_deg": float(np.max(np.abs(phase_difference))),
    }
    if frequencies is not None:
        metrics["by_frequency_hz"] = {
            f"{frequency:g}": _comparison_metrics(
                gems_apparent[index : index + 1],
                gems_phase[index : index + 1],
                mfem_apparent[index : index + 1],
                mfem_phase[index : index + 1],
                margin,
            )
            for index, frequency in enumerate(frequencies)
        }
    return metrics


def _anomaly_response_metrics(
    gems_uniform_apparent: np.ndarray,
    gems_uniform_phase: np.ndarray,
    gems_block_apparent: np.ndarray,
    gems_block_phase: np.ndarray,
    mfem_uniform_apparent: np.ndarray,
    mfem_uniform_phase: np.ndarray,
    mfem_block_apparent: np.ndarray,
    mfem_block_phase: np.ndarray,
    margin: int,
    frequencies: list[float] | None = None,
) -> dict[str, object]:
    crop = (slice(None), slice(margin, -margin), slice(margin, -margin), slice(None))
    gems_rho_ratio = gems_block_apparent[crop] / gems_uniform_apparent[crop]
    mfem_rho_ratio = mfem_block_apparent[crop] / mfem_uniform_apparent[crop]
    rho_difference = np.log10(gems_rho_ratio) - np.log10(mfem_rho_ratio)

    gems_phase_delta = _wrapped_phase_difference(
        gems_block_phase[crop], gems_uniform_phase[crop]
    )
    mfem_phase_delta = _wrapped_phase_difference(
        mfem_block_phase[crop], mfem_uniform_phase[crop]
    )
    phase_difference = _wrapped_phase_difference(
        gems_phase_delta, mfem_phase_delta
    )

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        first_flat = first.ravel()
        second_flat = second.ravel()
        if np.std(first_flat) == 0.0 or np.std(second_flat) == 0.0:
            return float("nan")
        return float(np.corrcoef(first_flat, second_flat)[0, 1])

    metrics: dict[str, object] = {
        "median_abs_log10_rho_ratio": float(
            np.median(np.abs(rho_difference))
        ),
        "p95_abs_log10_rho_ratio": float(
            np.percentile(np.abs(rho_difference), 95.0)
        ),
        "rmse_log10_rho_ratio": float(np.sqrt(np.mean(rho_difference**2))),
        "log10_rho_ratio_spatial_correlation": correlation(
            np.log10(gems_rho_ratio), np.log10(mfem_rho_ratio)
        ),
        "phase_delta_spatial_correlation": correlation(
            gems_phase_delta, mfem_phase_delta
        ),
        "phase_delta_bias_deg": float(np.mean(phase_difference)),
        "phase_delta_mae_deg": float(np.mean(np.abs(phase_difference))),
        "phase_delta_rmse_deg": float(
            np.sqrt(np.mean(phase_difference**2))
        ),
        "phase_delta_p95_deg": float(
            np.percentile(np.abs(phase_difference), 95.0)
        ),
        "phase_delta_max_deg": float(np.max(np.abs(phase_difference))),
    }
    if frequencies is not None:
        metrics["by_frequency_hz"] = {
            f"{frequency:g}": _anomaly_response_metrics(
                gems_uniform_apparent[index : index + 1],
                gems_uniform_phase[index : index + 1],
                gems_block_apparent[index : index + 1],
                gems_block_phase[index : index + 1],
                mfem_uniform_apparent[index : index + 1],
                mfem_uniform_phase[index : index + 1],
                mfem_block_apparent[index : index + 1],
                mfem_block_phase[index : index + 1],
                margin,
            )
            for index, frequency in enumerate(frequencies)
        }
    return metrics


def _center_anomaly_summary(
    gems_results: dict[str, tuple[np.ndarray, np.ndarray]],
    mfem_results: dict[str, tuple[np.ndarray, np.ndarray]],
    block_slices: tuple[slice, slice, slice],
    frequencies: list[float],
) -> dict[str, dict[str, dict[str, float]]]:
    summary = {}
    for frequency_index, frequency in enumerate(frequencies):
        components = {}
        for component, label in enumerate(("Zxy", "Zyx")):
            selection = (
                frequency_index,
                block_slices[0],
                block_slices[1],
                component,
            )
            values = {}
            for owner, results in (("gems", gems_results), ("mfem", mfem_results)):
                uniform_apparent, uniform_phase = results["uniform"]
                block_apparent, block_phase = results["block"]
                values[f"{owner}_rho_ratio"] = float(
                    np.median(
                        block_apparent[selection] / uniform_apparent[selection]
                    )
                )
                values[f"{owner}_phase_delta_deg"] = float(
                    np.median(
                        _wrapped_phase_difference(
                            block_phase[selection], uniform_phase[selection]
                        )
                    )
                )
            components[label] = values
        summary[f"{frequency:g}"] = components
    return summary


def _uniform_metrics(
    apparent: np.ndarray,
    phase: np.ndarray,
    rho: float,
    frequencies: list[float] | None = None,
) -> dict[str, object]:
    expected_phase = np.full_like(phase, 45.0)
    phase_error = _wrapped_phase_difference(phase, expected_phase)
    metrics: dict[str, object] = {
        "median_apparent_rho_ohm_m": float(np.median(apparent)),
        "median_apparent_rho_error_percent": float(
            100.0 * np.median(np.abs(apparent - rho)) / rho
        ),
        "max_apparent_rho_error_percent": float(
            100.0 * np.max(np.abs(apparent - rho)) / rho
        ),
        "median_phase_error_deg": float(np.median(np.abs(phase_error))),
        "max_phase_error_deg": float(np.max(np.abs(phase_error))),
    }
    if frequencies is not None:
        metrics["by_frequency_hz"] = {
            f"{frequency:g}": _uniform_metrics(
                apparent[index : index + 1], phase[index : index + 1], rho
            )
            for index, frequency in enumerate(frequencies)
        }
    return metrics


def _plot_comparison(
    output: Path,
    frequency: float,
    spacing: tuple[float, float, float],
    mfem_apparent: np.ndarray,
    mfem_phase: np.ndarray,
    gems_apparent: np.ndarray,
    gems_phase: np.ndarray,
    uniform_summary: dict,
) -> None:
    nx, ny = gems_apparent.shape[1:3]
    extent = (
        -nx * spacing[0] / 2000.0,
        nx * spacing[0] / 2000.0,
        -ny * spacing[1] / 2000.0,
        ny * spacing[1] / 2000.0,
    )
    frequency_index = 0
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 17.0), constrained_layout=True)

    for component in range(2):
        rho_row = 2 * component
        phase_row = rho_row + 1
        label = "Zxy" if component == 0 else "Zyx"
        mfem_rho = mfem_apparent[frequency_index, :, :, component]
        gems_rho = gems_apparent[frequency_index, :, :, component]
        rho_min = float(min(mfem_rho.min(), gems_rho.min()))
        rho_max = float(max(mfem_rho.max(), gems_rho.max()))
        rho_norm = colors.LogNorm(vmin=rho_min, vmax=rho_max)
        rho_images = []
        for column, (title, values) in enumerate(
            (("MT3D_MFEM", mfem_rho), ("GEMS-3D CUDA", gems_rho))
        ):
            rho_images.append(
                axes[rho_row, column].imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    norm=rho_norm,
                    aspect="equal",
                )
            )
            axes[rho_row, column].set_title(f"{title}: apparent rho {label}")
        rho_difference = np.log10(gems_rho) - np.log10(mfem_rho)
        difference_limit = max(float(np.max(np.abs(rho_difference))), 1.0e-6)
        difference_image = axes[rho_row, 2].imshow(
            rho_difference.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-difference_limit,
            vmax=difference_limit,
            aspect="equal",
        )
        axes[rho_row, 2].set_title("log10 rho: GEMS - MFEM")
        figure.colorbar(
            rho_images[0], ax=axes[rho_row, :2], label="Apparent rho (Ohm m)"
        )
        figure.colorbar(difference_image, ax=axes[rho_row, 2], label="dex")

        mfem_phase_map = np.mod(mfem_phase[frequency_index, :, :, component], 180.0)
        gems_phase_map = np.mod(gems_phase[frequency_index, :, :, component], 180.0)
        phase_min = float(min(mfem_phase_map.min(), gems_phase_map.min()))
        phase_max = float(max(mfem_phase_map.max(), gems_phase_map.max()))
        phase_images = []
        for column, (title, values) in enumerate(
            (("MT3D_MFEM", mfem_phase_map), ("GEMS-3D CUDA", gems_phase_map))
        ):
            phase_images.append(
                axes[phase_row, column].imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=phase_min,
                    vmax=phase_max,
                    aspect="equal",
                )
            )
            axes[phase_row, column].set_title(f"{title}: principal phase {label}")
        phase_difference = _wrapped_phase_difference(gems_phase_map, mfem_phase_map)
        phase_limit = max(float(np.max(np.abs(phase_difference))), 1.0e-6)
        phase_difference_image = axes[phase_row, 2].imshow(
            phase_difference.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-phase_limit,
            vmax=phase_limit,
            aspect="equal",
        )
        axes[phase_row, 2].set_title("Wrapped phase: GEMS - MFEM")
        figure.colorbar(
            phase_images[0], ax=axes[phase_row, :2], label="Phase (degrees)"
        )
        figure.colorbar(
            phase_difference_image, ax=axes[phase_row, 2], label="degrees"
        )

    for axis in axes.ravel():
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
    frequency_key = f"{frequency:g}"
    mfem_uniform_metrics = uniform_summary["mfem"].get(
        "by_frequency_hz", {}
    ).get(frequency_key, uniform_summary["mfem"])
    gems_uniform_metrics = uniform_summary["gems"].get(
        "by_frequency_hz", {}
    ).get(frequency_key, uniform_summary["gems"])
    figure.suptitle(
        f"MT solver cross-validation at {frequency:g} Hz\n"
        "Uniform 100 Ohm m median errors: "
        f"MFEM {mfem_uniform_metrics['median_apparent_rho_error_percent']:.2f}% / "
        f"{mfem_uniform_metrics['median_phase_error_deg']:.2f} deg; "
        f"GEMS {gems_uniform_metrics['median_apparent_rho_error_percent']:.2f}% / "
        f"{gems_uniform_metrics['median_phase_error_deg']:.2f} deg",
        fontsize=14,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_anomaly_response(
    output: Path,
    frequency: float,
    spacing: tuple[float, float, float],
    mfem_uniform: tuple[np.ndarray, np.ndarray],
    mfem_block: tuple[np.ndarray, np.ndarray],
    gems_uniform: tuple[np.ndarray, np.ndarray],
    gems_block: tuple[np.ndarray, np.ndarray],
) -> None:
    nx, ny = gems_block[0].shape[1:3]
    extent = (
        -nx * spacing[0] / 2000.0,
        nx * spacing[0] / 2000.0,
        -ny * spacing[1] / 2000.0,
        ny * spacing[1] / 2000.0,
    )
    frequency_index = 0
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 17.0), constrained_layout=True)

    for component in range(2):
        label = "Zxy" if component == 0 else "Zyx"
        rho_row = 2 * component
        phase_row = rho_row + 1
        mfem_rho_ratio = (
            mfem_block[0][frequency_index, :, :, component]
            / mfem_uniform[0][frequency_index, :, :, component]
        )
        gems_rho_ratio = (
            gems_block[0][frequency_index, :, :, component]
            / gems_uniform[0][frequency_index, :, :, component]
        )
        ratio_min = float(min(mfem_rho_ratio.min(), gems_rho_ratio.min()))
        ratio_max = float(max(mfem_rho_ratio.max(), gems_rho_ratio.max()))
        ratio_images = []
        for column, (title, values) in enumerate(
            (("MT3D_MFEM", mfem_rho_ratio), ("GEMS-3D CUDA", gems_rho_ratio))
        ):
            ratio_images.append(
                axes[rho_row, column].imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=ratio_min,
                    vmax=ratio_max,
                    aspect="equal",
                )
            )
            axes[rho_row, column].set_title(f"{title}: block/uniform rho {label}")
        rho_difference = np.log10(gems_rho_ratio) - np.log10(mfem_rho_ratio)
        rho_limit = max(float(np.max(np.abs(rho_difference))), 1.0e-6)
        rho_difference_image = axes[rho_row, 2].imshow(
            rho_difference.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-rho_limit,
            vmax=rho_limit,
            aspect="equal",
        )
        axes[rho_row, 2].set_title("log10 rho ratio: GEMS - MFEM")
        figure.colorbar(
            ratio_images[0], ax=axes[rho_row, :2], label="Block / uniform"
        )
        figure.colorbar(rho_difference_image, ax=axes[rho_row, 2], label="dex")

        mfem_phase_delta = _wrapped_phase_difference(
            mfem_block[1][frequency_index, :, :, component],
            mfem_uniform[1][frequency_index, :, :, component],
        )
        gems_phase_delta = _wrapped_phase_difference(
            gems_block[1][frequency_index, :, :, component],
            gems_uniform[1][frequency_index, :, :, component],
        )
        phase_limit = max(
            float(np.max(np.abs(mfem_phase_delta))),
            float(np.max(np.abs(gems_phase_delta))),
            1.0e-6,
        )
        phase_images = []
        for column, (title, values) in enumerate(
            (("MT3D_MFEM", mfem_phase_delta), ("GEMS-3D CUDA", gems_phase_delta))
        ):
            phase_images.append(
                axes[phase_row, column].imshow(
                    values.T,
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-phase_limit,
                    vmax=phase_limit,
                    aspect="equal",
                )
            )
            axes[phase_row, column].set_title(f"{title}: phase change {label}")
        phase_difference = _wrapped_phase_difference(
            gems_phase_delta, mfem_phase_delta
        )
        difference_limit = max(float(np.max(np.abs(phase_difference))), 1.0e-6)
        phase_difference_image = axes[phase_row, 2].imshow(
            phase_difference.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-difference_limit,
            vmax=difference_limit,
            aspect="equal",
        )
        axes[phase_row, 2].set_title("Phase change: GEMS - MFEM")
        figure.colorbar(
            phase_images[0], ax=axes[phase_row, :2], label="degrees"
        )
        figure.colorbar(
            phase_difference_image, ax=axes[phase_row, 2], label="degrees"
        )

    for axis in axes.ravel():
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
    figure.suptitle(
        f"Anomaly response relative to uniform half-space at {frequency:g} Hz",
        fontsize=14,
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GEMS_ROOT / "outputs" / "mt3d_mfem_cross_validation",
    )
    parser.add_argument(
        "--mfem-results-dir",
        type=Path,
        help="reuse MFEM case directories from another validation output",
    )
    parser.add_argument("--mfem-root", type=Path, default=REPOSITORY_ROOT / "MT3D_MFEM")
    parser.add_argument("--shape", default="16,16,24")
    parser.add_argument("--spacing", default="500,500,125")
    parser.add_argument("--frequencies", default="4")
    parser.add_argument("--air-layers", type=int, default=8)
    parser.add_argument(
        "--mfem-padding-layers", type=int, default=GEMS_PADDING_LAYERS
    )
    parser.add_argument(
        "--mfem-padding-growth", type=float, default=GEMS_PADDING_GROWTH
    )
    parser.add_argument("--background-rho", type=float, default=100.0)
    parser.add_argument("--anomaly-rho", type=float, default=10.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--gems-boundary-mode",
        choices=("legacy", "explicit_air"),
        default="legacy",
    )
    parser.add_argument("--gems-air-layers", type=int, default=8)
    parser.add_argument("--gems-air-resistivity", type=float, default=1.0e8)
    parser.add_argument("--gems-solver-tolerance", type=float)
    parser.add_argument("--gems-solver-max-iterations", type=int, default=50000)
    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "--gems-frequency-continuation",
        dest="gems_frequency_continuation",
        action="store_true",
        help="reuse higher-frequency fields as lower-frequency initial guesses",
    )
    continuation_group.add_argument(
        "--gems-cold-start",
        dest="gems_frequency_continuation",
        action="store_false",
        help="clear the field between frequencies instead of using continuation",
    )
    parser.set_defaults(gems_frequency_continuation=None)
    parser.add_argument("--mfem-processes", type=int, default=4)
    parser.add_argument("--run-mfem", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shape = _parse_tuple(args.shape, int, "shape")
    spacing = _parse_tuple(args.spacing, float, "spacing")
    frequencies = [float(value) for value in args.frequencies.split(",")]
    if any(value <= 0 for value in (*shape, *spacing, *frequencies)):
        raise ValueError("shape, spacing, and frequencies must be positive")
    if args.air_layers <= 0 or args.background_rho <= 0 or args.anomaly_rho <= 0:
        raise ValueError("air layers and resistivities must be positive")
    if args.mfem_padding_layers < 0 or args.mfem_padding_growth <= 1.0:
        raise ValueError("MFEM padding layers must be nonnegative and growth > 1")
    if args.mfem_processes <= 0:
        raise ValueError("MFEM process count must be positive")
    if (
        args.gems_air_layers <= 0
        or not np.isfinite(args.gems_air_resistivity)
        or args.gems_air_resistivity <= 0.0
    ):
        raise ValueError("GEMS air layers and resistivity must be positive")
    if (
        args.gems_solver_tolerance is not None
        and (
            not np.isfinite(args.gems_solver_tolerance)
            or args.gems_solver_tolerance <= 0.0
        )
    ) or args.gems_solver_max_iterations <= 0:
        raise ValueError("GEMS solver tolerance and maximum iterations must be positive")
    if args.gems_solver_tolerance is None:
        gems_solver_tolerance = (
            1.0e-8 if args.gems_boundary_mode == "explicit_air" else 1.0e-7
        )
    else:
        gems_solver_tolerance = args.gems_solver_tolerance
    gems_frequency_continuation = (
        args.gems_frequency_continuation
        if args.gems_frequency_continuation is not None
        else args.gems_boundary_mode != "explicit_air"
    )
    if args.run_mfem and args.mfem_results_dir is not None:
        raise ValueError("--mfem-results-dir cannot be combined with --run-mfem")
    if not torch.cuda.is_available():
        raise RuntimeError("GEMS-3D cross-validation requires CUDA")

    output_dir = args.output_dir.resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.run_mfem and output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; use --overwrite or choose another directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mfem_results_dir = (
        args.mfem_results_dir.resolve()
        if args.mfem_results_dir is not None
        else output_dir
    )

    if args.mfem_results_dir is not None:
        _validate_reused_mfem_metadata(
            mfem_results_dir,
            shape,
            spacing,
            frequencies,
            args.air_layers,
            args.background_rho,
            args.anomaly_rho,
            args.mfem_padding_layers,
            args.mfem_padding_growth,
        )

    mfem_root = args.mfem_root.resolve()
    mfem_executable = mfem_root / "MT3D"
    bridge = _load_mfem_bridge(mfem_root)
    uniform_model = np.full(shape, args.background_rho, dtype=np.float64)
    block_model, block_slices = _centered_block_model(
        shape, args.background_rho, args.anomaly_rho
    )
    models = {"uniform": uniform_model, "block": block_model}

    if args.run_mfem:
        for case_name, model in models.items():
            case_dir = output_dir / case_name
            _write_mfem_case(
                case_dir,
                case_name,
                model,
                spacing,
                frequencies,
                args.air_layers,
                args.background_rho,
                args.mfem_padding_layers,
                args.mfem_padding_growth,
                mfem_root,
                bridge,
            )
            _run_mfem(
                case_dir, case_name, mfem_executable, args.mfem_processes
            )

    mfem_results = {}
    gems_results = {}
    for case_name, model in models.items():
        mfem_results[case_name] = _read_mfem_responses(
            mfem_results_dir / case_name,
            case_name,
            shape,
            spacing,
            frequencies,
        )
        gems_results[case_name] = _run_gems(
            model,
            spacing,
            frequencies,
            args.device,
            args.gems_boundary_mode,
            args.gems_air_layers,
            args.gems_air_resistivity,
            gems_solver_tolerance,
            args.gems_solver_max_iterations,
            gems_frequency_continuation,
        )

    margin = max(1, min(shape[0], shape[1]) // 8)
    uniform_summary = {
        "mfem": _uniform_metrics(
            *mfem_results["uniform"], args.background_rho, frequencies
        ),
        "gems": _uniform_metrics(
            *gems_results["uniform"], args.background_rho, frequencies
        ),
    }
    block_comparison = _comparison_metrics(
        *gems_results["block"], *mfem_results["block"], margin, frequencies
    )
    anomaly_response_comparison = _anomaly_response_metrics(
        *gems_results["uniform"],
        *gems_results["block"],
        *mfem_results["uniform"],
        *mfem_results["block"],
        margin,
        frequencies,
    )
    summary = {
        "shape": list(shape),
        "spacing_m": list(spacing),
        "frequencies_hz": frequencies,
        "gems_boundary": {
            "mode": args.gems_boundary_mode,
            "air_layers": (
                args.gems_air_layers
                if args.gems_boundary_mode == "explicit_air"
                else 0
            ),
            "air_resistivity_ohm_m": args.gems_air_resistivity,
        },
        "gems_solver": {
            "method": "left-Jacobi BiCGStab",
            "tolerance": gems_solver_tolerance,
            "max_iterations": args.gems_solver_max_iterations,
            "frequency_continuation": gems_frequency_continuation,
        },
        "mfem_results_dir": str(mfem_results_dir),
        "comparison_scope": {
            "same_core_resistivity_model": True,
            "same_side_bottom_padding": bool(
                args.mfem_padding_layers == GEMS_PADDING_LAYERS
                and np.isclose(args.mfem_padding_growth, GEMS_PADDING_GROWTH)
            ),
            "same_top_boundary_value_problem": False,
            "top_boundary_note": (
                "GEMS uses "
                + (
                    "explicit air with a remote 1D E Dirichlet boundary"
                    if args.gems_boundary_mode == "explicit_air"
                    else "a fixed tangential E boundary on the earth surface"
                )
                + "; MT3D_MFEM uses explicit air and a remote natural H "
                "boundary condition."
            ),
            "phase_difference_convention": "wrapped modulo 180 degrees",
            "mfem_station_sampling": (
                "mean of four symmetric subcell stations just inside the earth"
            ),
        },
        "air_layers": args.air_layers,
        "station_depth_m": STATION_DEPTH_FRACTION * spacing[2],
        "mfem_processes": args.mfem_processes,
        "matched_side_bottom_padding": {
            "layers": args.mfem_padding_layers,
            "growth": args.mfem_padding_growth,
            "matches_gems": bool(
                args.mfem_padding_layers == GEMS_PADDING_LAYERS
                and np.isclose(args.mfem_padding_growth, GEMS_PADDING_GROWTH)
            ),
            "mfem_earth_shape": [
                shape[0] + 2 * args.mfem_padding_layers,
                shape[1] + 2 * args.mfem_padding_layers,
                shape[2] + args.mfem_padding_layers,
            ],
        },
        "background_rho_ohm_m": args.background_rho,
        "anomaly_rho_ohm_m": args.anomaly_rho,
        "block_index_bounds": [
            [block_slices[axis].start, block_slices[axis].stop] for axis in range(3)
        ],
        "interior_margin_cells": margin,
        "uniform_halfspace": uniform_summary,
        "block_cross_comparison": block_comparison,
        "anomaly_response_comparison": anomaly_response_comparison,
        "center_anomaly_response": _center_anomaly_summary(
            gems_results, mfem_results, block_slices, frequencies
        ),
    }

    figure_path = output_dir / "mt3d_mfem_cross_validation.png"
    _plot_comparison(
        figure_path,
        frequencies[0],
        spacing,
        *mfem_results["block"],
        *gems_results["block"],
        uniform_summary,
    )
    anomaly_figure_path = output_dir / "mt3d_mfem_anomaly_response.png"
    _plot_anomaly_response(
        anomaly_figure_path,
        frequencies[0],
        spacing,
        mfem_results["uniform"],
        mfem_results["block"],
        gems_results["uniform"],
        gems_results["block"],
    )
    data_path = output_dir / "mt3d_mfem_cross_validation.npz"
    np.savez_compressed(
        data_path,
        block_model_rho=block_model,
        frequencies_hz=np.asarray(frequencies),
        spacing_m=np.asarray(spacing),
        mfem_uniform_app=mfem_results["uniform"][0],
        mfem_uniform_phase=mfem_results["uniform"][1],
        gems_uniform_app=gems_results["uniform"][0],
        gems_uniform_phase=gems_results["uniform"][1],
        mfem_block_app=mfem_results["block"][0],
        mfem_block_phase=mfem_results["block"][1],
        gems_block_app=gems_results["block"][0],
        gems_block_phase=gems_results["block"][1],
    )
    summary["figure"] = str(figure_path)
    summary["anomaly_figure"] = str(anomaly_figure_path)
    summary["data"] = str(data_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("MT3D_MFEM_CROSS_VALIDATION=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
