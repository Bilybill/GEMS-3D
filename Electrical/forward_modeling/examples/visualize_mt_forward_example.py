"""Run and visualize a reproducible 3D MT forward-modeling example."""

import argparse
import json
import os
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
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.mt_forward import MTForward3D, mt_forward_cuda  # noqa: E402


def _wrapped_phase_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.remainder(actual - expected + 180.0, 360.0) - 180.0


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("plot data contains no finite values")
    lower = float(finite.min())
    upper = float(finite.max())
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0e-6)
        lower -= padding
        upper += padding
    return lower, upper


def _format_map_axis(axis, title: str, x_extent_km: float, y_extent_km: float) -> None:
    axis.set_title(title)
    axis.set_xlabel("x (km)")
    axis.set_ylabel("y (km)")
    axis.set_xlim(0.0, x_extent_km)
    axis.set_ylim(0.0, y_extent_km)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mt_forward_example.png")
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shape", default="16,16,20")
    parser.add_argument("--spacing", default="250,250,100")
    parser.add_argument("--frequency", type=float, default=4.0)
    parser.add_argument("--background-rho", type=float, default=100.0)
    parser.add_argument("--anomaly-rho", type=float, default=10.0)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.shape.split(","))
    spacing = tuple(float(value) for value in args.spacing.split(","))
    if len(shape) != 3 or any(value < 6 for value in shape):
        raise ValueError("shape must contain three dimensions, each at least 6")
    if len(spacing) != 3 or any(value <= 0.0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if (
        args.frequency <= 0.0
        or args.background_rho <= 0.0
        or args.anomaly_rho <= 0.0
    ):
        raise ValueError("frequency and resistivities must be positive")
    if mt_forward_cuda is None:
        raise RuntimeError("mt_forward_cuda is not built or cannot be imported")
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    nx, ny, nz = shape
    dx, dy, dz = spacing

    background = torch.full(
        shape, args.background_rho, dtype=torch.float64, device=device
    )
    model = background.clone()
    x_start, x_stop = nx // 4, nx - nx // 4
    y_start, y_stop = ny // 4, ny - ny // 4
    z_start = max(1, nz // 5)
    z_stop = min(nz - 1, max(z_start + 1, 3 * nz // 5))
    model[x_start:x_stop, y_start:y_stop, z_start:z_stop] = args.anomaly_rho

    solver = MTForward3D(
        freqs=[args.frequency], dx=dx, dy=dy, dz=dz
    ).to(device)
    with torch.inference_mode():
        background_app, background_phase = solver(background)
        anomaly_app, anomaly_phase = solver(model)
    torch.cuda.synchronize(device)

    background_app_np = background_app.cpu().numpy()[0]
    background_phase_np = background_phase.cpu().numpy()[0]
    anomaly_app_np = anomaly_app.cpu().numpy()[0]
    anomaly_phase_np = anomaly_phase.cpu().numpy()[0]
    model_np = model.cpu().numpy()

    expected_phase = np.array([45.0, -135.0])[None, None, :]
    app_relative_error = (
        np.abs(background_app_np - args.background_rho) / args.background_rho
    )
    phase_error = np.abs(_wrapped_phase_error(background_phase_np, expected_phase))
    interior = (slice(2, -2), slice(2, -2), slice(None))

    all_outputs_finite = bool(
        np.isfinite(background_app_np).all()
        and np.isfinite(background_phase_np).all()
        and np.isfinite(anomaly_app_np).all()
        and np.isfinite(anomaly_phase_np).all()
        and (anomaly_app_np > 0.0).all()
    )
    if not all_outputs_finite:
        raise RuntimeError("MT forward example produced invalid output")

    summary = {
        "shape": list(shape),
        "spacing_m": list(spacing),
        "frequency_hz": args.frequency,
        "background_rho_ohm_m": args.background_rho,
        "anomaly_rho_ohm_m": args.anomaly_rho,
        "halfspace_app_max_relative_error_percent": float(
            100.0 * app_relative_error.max()
        ),
        "halfspace_app_interior_max_relative_error_percent": float(
            100.0 * app_relative_error[interior].max()
        ),
        "halfspace_phase_max_error_deg": float(phase_error.max()),
        "halfspace_phase_interior_max_error_deg": float(
            phase_error[interior].max()
        ),
        "anomaly_app_min_ohm_m": float(anomaly_app_np.min()),
        "anomaly_app_max_ohm_m": float(anomaly_app_np.max()),
        "all_outputs_finite": all_outputs_finite,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_output = args.output.with_suffix(".npz")
    np.savez_compressed(
        data_output,
        model_rho=model_np,
        background_app_res=background_app_np,
        background_phase_deg=background_phase_np,
        anomaly_app_res=anomaly_app_np,
        anomaly_phase_deg=anomaly_phase_np,
        frequency_hz=np.array(args.frequency),
        spacing_m=np.asarray(spacing),
    )

    x_extent_km = nx * dx / 1000.0
    y_extent_km = ny * dy / 1000.0
    z_extent_km = nz * dz / 1000.0
    z_index = (z_start + z_stop - 1) // 2
    y_index = ny // 2

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    model_norm = colors.LogNorm(vmin=model_np.min(), vmax=model_np.max())
    horizontal = axes[0, 0].imshow(
        model_np[:, :, z_index].T,
        origin="lower",
        extent=(0.0, x_extent_km, 0.0, y_extent_km),
        cmap="viridis",
        norm=model_norm,
        aspect="equal",
    )
    _format_map_axis(
        axes[0, 0],
        f"Resistivity at depth {(z_index + 0.5) * dz / 1000.0:.2f} km",
        x_extent_km,
        y_extent_km,
    )
    axes[1, 0].imshow(
        model_np[:, y_index, :].T,
        origin="upper",
        extent=(0.0, x_extent_km, z_extent_km, 0.0),
        cmap="viridis",
        norm=model_norm,
        aspect="auto",
    )
    axes[1, 0].set_title(
        f"Resistivity section at y={(y_index + 0.5) * dy / 1000.0:.2f} km"
    )
    axes[1, 0].set_xlabel("x (km)")
    axes[1, 0].set_ylabel("depth (km)")
    figure.colorbar(
        horizontal,
        ax=[axes[0, 0], axes[1, 0]],
        label="Resistivity (Ohm m)",
    )

    app_min, app_max = _finite_limits(anomaly_app_np)
    app_norm = colors.Normalize(vmin=app_min, vmax=app_max)
    app_images = []
    for polarization, axis in enumerate((axes[0, 1], axes[0, 2])):
        app_images.append(
            axis.imshow(
                anomaly_app_np[:, :, polarization].T,
                origin="lower",
                extent=(0.0, x_extent_km, 0.0, y_extent_km),
                cmap="magma",
                norm=app_norm,
                aspect="equal",
            )
        )
        label = "Zxy" if polarization == 0 else "Zyx"
        _format_map_axis(
            axis,
            f"Apparent resistivity {label}, {args.frequency:g} Hz",
            x_extent_km,
            y_extent_km,
        )
    figure.colorbar(
        app_images[0],
        ax=[axes[0, 1], axes[0, 2]],
        label="Apparent resistivity (Ohm m)",
    )

    principal_phase = np.mod(anomaly_phase_np, 180.0)
    phase_min, phase_max = _finite_limits(principal_phase)
    phase_norm = colors.Normalize(vmin=phase_min, vmax=phase_max)
    phase_images = []
    for polarization, axis in enumerate((axes[1, 1], axes[1, 2])):
        phase_images.append(
            axis.imshow(
                principal_phase[:, :, polarization].T,
                origin="lower",
                extent=(0.0, x_extent_km, 0.0, y_extent_km),
                cmap="coolwarm",
                norm=phase_norm,
                aspect="equal",
            )
        )
        label = "Zxy" if polarization == 0 else "Zyx"
        _format_map_axis(
            axis,
            f"Principal phase {label}, {args.frequency:g} Hz",
            x_extent_km,
            y_extent_km,
        )
    figure.colorbar(
        phase_images[0],
        ax=[axes[1, 1], axes[1, 2]],
        label="Phase (degrees, modulo 180)",
    )

    figure.suptitle(
        "3D MT conductive-block example\n"
        f"Uniform half-space check: max apparent-resistivity error "
        f"{summary['halfspace_app_max_relative_error_percent']:.2f}%, "
        f"max phase error {summary['halfspace_phase_max_error_deg']:.2f} deg",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    summary["figure"] = str(args.output.resolve())
    summary["data"] = str(data_output.resolve())
    print("MT_EXAMPLE_JSON=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
