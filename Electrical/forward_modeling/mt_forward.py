import math

import torch
import torch.nn as nn

try:
    from . import mt_forward_cuda
except ImportError:
    try:
        import mt_forward_cuda  # type: ignore[no-redef]
    except ImportError:
        mt_forward_cuda = None


PHOENIX_COEFFS = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]
BOUNDARY_MODES = {"legacy", "explicit_air"}


def _validate_positive_finite_scalar(value, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number, not bool.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return normalized


def _validate_rho_tensor(rho_tensor: torch.Tensor) -> None:
    if not isinstance(rho_tensor, torch.Tensor):
        raise TypeError("rho_tensor must be a torch.Tensor.")
    if rho_tensor.dim() != 3:
        raise ValueError(
            f"rho_tensor must have shape (NX, NY, NZ), got {tuple(rho_tensor.shape)}."
        )
    if any(size <= 0 for size in rho_tensor.shape):
        raise ValueError("rho_tensor dimensions NX, NY, and NZ must all be positive.")
    if rho_tensor.is_complex() or rho_tensor.dtype == torch.bool:
        raise TypeError("rho_tensor must contain real-valued resistivity data.")
    invalid = torch.logical_or(~torch.isfinite(rho_tensor), rho_tensor <= 0.0)
    if bool(invalid.any().item()):
        raise ValueError(
            "rho_tensor resistivity values must all be positive and finite."
        )


def _normalize_frequency_list(freqs) -> list[float] | None:
    if freqs is None:
        return None
    try:
        values = list(freqs)
    except TypeError as exc:
        raise TypeError("freqs must be an iterable of positive finite frequencies.") from exc
    return [
        _validate_positive_finite_scalar(freq, f"freqs[{index}]")
        for index, freq in enumerate(values)
    ]


def _estimate_boundary_background_rho(rho_tensor: torch.Tensor) -> float:
    nx, ny, nz = rho_tensor.shape
    rho64 = rho_tensor.to(torch.float64)
    boundary_sum = rho64.sum()
    boundary_count = nx * ny * nz
    if nx > 2 and ny > 2 and nz > 2:
        boundary_sum = boundary_sum - rho64[1:-1, 1:-1, 1:-1].sum()
        boundary_count -= (nx - 2) * (ny - 2) * (nz - 2)
    return float((boundary_sum / boundary_count).item())


def _generate_phoenix_frequencies(f_min: float, f_max: float) -> list[float]:
    f_min = _validate_positive_finite_scalar(f_min, "f_min")
    f_max = _validate_positive_finite_scalar(f_max, "f_max")
    if f_min > f_max:
        f_min, f_max = f_max, f_min

    max_power = math.ceil(math.log10(f_max))
    min_power = math.floor(math.log10(f_min))

    freqs = []
    for p in range(max_power, min_power - 1, -1):
        try:
            power_of_10 = 10.0 ** p
        except OverflowError:
            continue
        if not math.isfinite(power_of_10) or power_of_10 == 0.0:
            continue
        for coeff in PHOENIX_COEFFS:
            current_f = coeff * power_of_10
            if current_f <= f_max and current_f >= f_min:
                freqs.append(float(current_f))

    if not freqs:
        freqs = [f_max] if f_min == f_max else [f_max, f_min]
    return freqs


def generate_mt_frequencies(f_min: float, f_max: float) -> list[float]:
    """
    Public helper for building the Phoenix-style MT frequency list used by
    the project from a requested frequency range.
    """
    return _generate_phoenix_frequencies(f_min, f_max)


def resolve_auto_mt_frequencies(rho_tensor: torch.Tensor, dz: float):
    _validate_rho_tensor(rho_tensor)
    dz = _validate_positive_finite_scalar(dz, "dz")
    return _resolve_auto_mt_frequencies(rho_tensor, dz)


def _resolve_auto_mt_frequencies(rho_tensor: torch.Tensor, dz: float):
    bg_rho = _estimate_boundary_background_rho(rho_tensor)
    skin_depth_min = dz * 2.0
    total_depth = rho_tensor.shape[2] * dz
    skin_depth_max = total_depth / 1.5

    f_max = bg_rho * (503.0 / skin_depth_min) ** 2
    f_min = bg_rho * (503.0 / skin_depth_max) ** 2
    freqs = _generate_phoenix_frequencies(f_min, f_max)
    return freqs, bg_rho, f_min, f_max


class MTForward3D(nn.Module):
    def __init__(
        self,
        freqs=None,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        boundary_mode="legacy",
        air_layers=8,
        air_resistivity=1.0e8,
        solver_tolerance=None,
        solver_max_iterations=50000,
        frequency_continuation=None,
        background_resistivity=None,
        padding_layers=10,
        padding_growth=1.4,
    ):
        """
        PyTorch wrapper for CUDA MT 3D Forward Modeling.
        Args:
            freqs: Optional list of frequencies (Hz). If omitted or empty,
                the frequency range is auto-generated from the model extent and
                boundary background resistivity following MTForward3D.
            dx: Cell width in x-direction (m)
            dy: Cell width in y-direction (m)
            dz: Cell width in z-direction (m)
            boundary_mode: ``"legacy"`` fixes tangential E at the earth
                surface. ``"explicit_air"`` moves that boundary to the top of
                an explicit air layer while retaining the existing remote 1D
                Dirichlet excitation on the outer boundary.
            air_layers: Number of air cells for ``"explicit_air"``.
            air_resistivity: Air resistivity in Ohm-m.
            solver_tolerance: Relative tolerance applied to both the true and
                Jacobi-scaled residual norms. If omitted, uses ``1e-7`` for
                ``"legacy"`` and the reproducibility-tested ``1e-8`` for
                ``"explicit_air"``.
            solver_max_iterations: Maximum BiCGStab iterations per polarization.
            frequency_continuation: Reuse the previous higher-frequency field
                as the next lower-frequency initial guess. If omitted, enabled
                for ``"legacy"`` and disabled for ``"explicit_air"``.
            background_resistivity: Optional regional background resistivity
                used for padding and remote boundary excitation. If omitted,
                it is estimated from the model boundary cells.
            padding_layers: Number of geometrically expanding cells on each
                lateral side and below the model.
            padding_growth: Cell-width multiplier between adjacent padding
                layers. Must be greater than one.

        Notes:
            This module is forward-only. The CUDA extension does not implement
            an autograd backward pass.
        """
        super(MTForward3D, self).__init__()
        self.freqs = _normalize_frequency_list(freqs)
        self.last_freqs = None
        self.dx = _validate_positive_finite_scalar(dx, "dx")
        self.dy = _validate_positive_finite_scalar(dy, "dy")
        self.dz = _validate_positive_finite_scalar(dz, "dz")
        if not isinstance(boundary_mode, str) or boundary_mode not in BOUNDARY_MODES:
            choices = ", ".join(sorted(BOUNDARY_MODES))
            raise ValueError(
                f"boundary_mode must be one of {{{choices}}}, got {boundary_mode!r}."
            )
        if isinstance(air_layers, bool) or not isinstance(air_layers, int):
            raise TypeError("air_layers must be an integer.")
        if air_layers <= 0:
            raise ValueError("air_layers must be positive.")
        self.boundary_mode = boundary_mode
        self.air_layers = air_layers
        self.air_resistivity = _validate_positive_finite_scalar(
            air_resistivity, "air_resistivity"
        )
        if solver_tolerance is None:
            self.solver_tolerance = (
                1.0e-8 if boundary_mode == "explicit_air" else 1.0e-7
            )
        else:
            self.solver_tolerance = _validate_positive_finite_scalar(
                solver_tolerance, "solver_tolerance"
            )
        if (
            isinstance(solver_max_iterations, bool)
            or not isinstance(solver_max_iterations, int)
        ):
            raise TypeError("solver_max_iterations must be an integer.")
        if solver_max_iterations <= 0:
            raise ValueError("solver_max_iterations must be positive.")
        if frequency_continuation is None:
            frequency_continuation = boundary_mode != "explicit_air"
        elif not isinstance(frequency_continuation, bool):
            raise TypeError("frequency_continuation must be a bool or None.")
        self.solver_max_iterations = solver_max_iterations
        self.frequency_continuation = frequency_continuation
        self.background_resistivity = (
            None
            if background_resistivity is None
            else _validate_positive_finite_scalar(
                background_resistivity, "background_resistivity"
            )
        )
        if isinstance(padding_layers, bool) or not isinstance(padding_layers, int):
            raise TypeError("padding_layers must be an integer.")
        if padding_layers <= 0:
            raise ValueError("padding_layers must be positive.")
        self.padding_layers = padding_layers
        self.padding_growth = _validate_positive_finite_scalar(
            padding_growth, "padding_growth"
        )
        if self.padding_growth <= 1.0:
            raise ValueError("padding_growth must be greater than 1.0.")

    def forward(self, rho_tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.
            phase: Phase tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.

        Raises:
            RuntimeError: If `rho_tensor.requires_grad` is true because this
                forward-only extension has no autograd implementation.
        """
        _validate_rho_tensor(rho_tensor)
        if rho_tensor.requires_grad:
            raise RuntimeError(
                "MTForward3D is forward-only and does not implement autograd. "
                "Pass a tensor that does not require gradients."
            )
        if mt_forward_cuda is None:
            raise ImportError(
                "MT forward modeling requires the compiled mt_forward_cuda extension. "
                "Build Electrical/forward_modeling before running the MT solver, or pass --skip_mt "
                "when using the multiphysics pipeline without MT."
            )
        rho_solver_input = rho_tensor.to(torch.float64).contiguous()
        if self.background_resistivity is None:
            background_rho = _estimate_boundary_background_rho(rho_solver_input)
        else:
            background_rho = self.background_resistivity
        if self.freqs:
            freqs = list(self.freqs)
        else:
            if self.background_resistivity is None:
                freqs, background_rho, _, _ = _resolve_auto_mt_frequencies(
                    rho_solver_input, self.dz
                )
            else:
                skin_depth_min = self.dz * 2.0
                skin_depth_max = rho_solver_input.shape[2] * self.dz / 1.5
                f_max = background_rho * (503.0 / skin_depth_min) ** 2
                f_min = background_rho * (503.0 / skin_depth_max) ** 2
                freqs = _generate_phoenix_frequencies(f_min, f_max)

        self.last_freqs = tuple(freqs)
        app_res, phase = mt_forward_cuda._compute_mt_3d_validated(
            rho_solver_input,
            self.dx,
            self.dy,
            self.dz,
            freqs,
            background_rho,
            self.air_layers if self.boundary_mode == "explicit_air" else 0,
            self.air_resistivity,
            self.solver_tolerance,
            self.solver_max_iterations,
            self.frequency_continuation,
            self.padding_layers,
            self.padding_growth,
        )
        return app_res, phase
