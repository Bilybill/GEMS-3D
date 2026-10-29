import math
import sys
from pathlib import Path

import pytest
import torch


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.mt_forward import (  # noqa: E402
    MTForward3D,
    generate_mt_frequencies,
    resolve_auto_mt_frequencies,
)


def test_generate_mt_frequencies_accepts_forward_and_reverse_ranges():
    forward = generate_mt_frequencies(1.0, 100.0)
    reverse = generate_mt_frequencies(100.0, 1.0)

    assert forward == reverse
    assert forward == sorted(forward, reverse=True)
    assert all(1.0 <= freq <= 100.0 for freq in forward)


def test_equal_non_phoenix_frequency_is_not_duplicated():
    assert generate_mt_frequencies(5.0, 5.0) == [5.0]


def test_generate_mt_frequencies_handles_largest_finite_bound():
    freqs = generate_mt_frequencies(1.0e307, sys.float_info.max)

    assert freqs
    assert all(math.isfinite(freq) for freq in freqs)
    assert all(1.0e307 <= freq <= sys.float_info.max for freq in freqs)


@pytest.mark.parametrize(
    ("f_min", "f_max"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (math.nan, 1.0),
        (1.0, math.nan),
        (1.0, math.inf),
    ],
)
def test_generate_mt_frequencies_rejects_invalid_bounds(f_min, f_max):
    with pytest.raises(ValueError, match="positive and finite"):
        generate_mt_frequencies(f_min, f_max)


def test_auto_frequency_depth_rule_matches_cpp():
    rho = torch.full((4, 3, 10), 100.0, dtype=torch.float64)
    dz = 20.0

    freqs, bg_rho, f_min, f_max = resolve_auto_mt_frequencies(rho, dz)

    expected_max = 100.0 * (503.0 / (2.0 * dz)) ** 2
    expected_min = 100.0 * (503.0 / ((rho.shape[2] * dz) / 1.5)) ** 2
    assert bg_rho == pytest.approx(100.0)
    assert f_min == pytest.approx(expected_min)
    assert f_max == pytest.approx(expected_max)
    assert freqs == sorted(freqs, reverse=True)


def test_auto_frequencies_accept_positive_integer_resistivity():
    _, bg_rho, _, _ = resolve_auto_mt_frequencies(
        torch.full((2, 2, 4), 100, dtype=torch.int64),
        1.0,
    )

    assert bg_rho == pytest.approx(100.0)


@pytest.mark.parametrize(
    "rho",
    [
        torch.zeros((2, 2, 2), dtype=torch.float64),
        -torch.ones((2, 2, 2), dtype=torch.float64),
        torch.tensor([[[1.0, math.nan]]], dtype=torch.float64),
        torch.tensor([[[1.0, math.inf]]], dtype=torch.float64),
    ],
)
def test_auto_frequencies_reject_invalid_resistivity(rho):
    with pytest.raises(ValueError):
        resolve_auto_mt_frequencies(rho, 1.0)


@pytest.mark.parametrize(
    "rho",
    [
        torch.ones((2, 2), dtype=torch.float64),
        torch.empty((0, 2, 2), dtype=torch.float64),
    ],
)
def test_auto_frequencies_reject_invalid_dimensions(rho):
    with pytest.raises(ValueError):
        resolve_auto_mt_frequencies(rho, 1.0)


@pytest.mark.parametrize("dz", [0.0, -1.0, math.nan, math.inf])
def test_auto_frequencies_reject_invalid_dz(dz):
    with pytest.raises(ValueError, match="positive and finite"):
        resolve_auto_mt_frequencies(torch.ones((2, 2, 2)), dz)


@pytest.mark.parametrize("spacing_name", ["dx", "dy", "dz"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_solver_rejects_invalid_grid_spacing(spacing_name, value):
    kwargs = {"dx": 1.0, "dy": 1.0, "dz": 1.0}
    kwargs[spacing_name] = value

    with pytest.raises(ValueError, match="positive and finite"):
        MTForward3D(freqs=[1.0], **kwargs)


@pytest.mark.parametrize("freq", [0.0, -1.0, math.nan, math.inf])
def test_solver_rejects_invalid_explicit_frequency(freq):
    with pytest.raises(ValueError, match="positive and finite"):
        MTForward3D(freqs=[freq])


@pytest.mark.parametrize("mode", ["air", "dirichlet", "", None, []])
def test_solver_rejects_unknown_boundary_mode(mode):
    with pytest.raises(ValueError, match="boundary_mode"):
        MTForward3D(freqs=[1.0], boundary_mode=mode)


@pytest.mark.parametrize("air_layers", [0, -1])
def test_solver_rejects_nonpositive_air_layer_count(air_layers):
    with pytest.raises(ValueError, match="air_layers"):
        MTForward3D(freqs=[1.0], air_layers=air_layers)


@pytest.mark.parametrize("air_layers", [1.5, True, "8"])
def test_solver_rejects_noninteger_air_layer_count(air_layers):
    with pytest.raises(TypeError, match="air_layers"):
        MTForward3D(freqs=[1.0], air_layers=air_layers)


@pytest.mark.parametrize("air_resistivity", [0.0, -1.0, math.nan, math.inf])
def test_solver_rejects_invalid_air_resistivity(air_resistivity):
    with pytest.raises(ValueError, match="positive and finite"):
        MTForward3D(freqs=[1.0], air_resistivity=air_resistivity)


@pytest.mark.parametrize("solver_tolerance", [0.0, -1.0, math.nan, math.inf])
def test_solver_rejects_invalid_tolerance(solver_tolerance):
    with pytest.raises(ValueError, match="solver_tolerance"):
        MTForward3D(freqs=[1.0], solver_tolerance=solver_tolerance)


@pytest.mark.parametrize(
    "background_resistivity", [0.0, -1.0, math.nan, math.inf, True]
)
def test_solver_rejects_invalid_background_resistivity(background_resistivity):
    error = TypeError if background_resistivity is True else ValueError
    with pytest.raises(error, match="background_resistivity"):
        MTForward3D(
            freqs=[1.0], background_resistivity=background_resistivity
        )


@pytest.mark.parametrize("solver_max_iterations", [0, -1])
def test_solver_rejects_nonpositive_iteration_limit(solver_max_iterations):
    with pytest.raises(ValueError, match="solver_max_iterations"):
        MTForward3D(freqs=[1.0], solver_max_iterations=solver_max_iterations)


@pytest.mark.parametrize("solver_max_iterations", [1.5, True, "50000"])
def test_solver_rejects_noninteger_iteration_limit(solver_max_iterations):
    with pytest.raises(TypeError, match="solver_max_iterations"):
        MTForward3D(freqs=[1.0], solver_max_iterations=solver_max_iterations)


@pytest.mark.parametrize("frequency_continuation", [0, 1, "true"])
def test_solver_rejects_nonboolean_frequency_continuation(frequency_continuation):
    with pytest.raises(TypeError, match="frequency_continuation"):
        MTForward3D(freqs=[1.0], frequency_continuation=frequency_continuation)


@pytest.mark.parametrize("padding_layers", [0, -1])
def test_solver_rejects_nonpositive_padding_layer_count(padding_layers):
    with pytest.raises(ValueError, match="padding_layers"):
        MTForward3D(freqs=[1.0], padding_layers=padding_layers)


@pytest.mark.parametrize("padding_layers", [1.5, True, "12"])
def test_solver_rejects_noninteger_padding_layer_count(padding_layers):
    with pytest.raises(TypeError, match="padding_layers"):
        MTForward3D(freqs=[1.0], padding_layers=padding_layers)


@pytest.mark.parametrize(
    "padding_growth", [0.0, 1.0, -1.0, math.nan, math.inf]
)
def test_solver_rejects_invalid_padding_growth(padding_growth):
    with pytest.raises(ValueError, match="padding_growth"):
        MTForward3D(freqs=[1.0], padding_growth=padding_growth)


def test_boundary_modes_select_reproducibility_appropriate_default_tolerance():
    legacy = MTForward3D(freqs=[1.0], boundary_mode="legacy")
    explicit_air = MTForward3D(freqs=[1.0], boundary_mode="explicit_air")

    assert legacy.solver_tolerance == 1.0e-7
    assert explicit_air.solver_tolerance == 1.0e-8
    assert legacy.frequency_continuation is True
    assert explicit_air.frequency_continuation is False
    assert legacy.background_resistivity is None
    assert legacy.padding_layers == explicit_air.padding_layers == 10
    assert legacy.padding_growth == explicit_air.padding_growth == pytest.approx(1.4)


def test_forward_rejects_autograd_input_before_loading_extension():
    solver = MTForward3D(freqs=[1.0])
    rho = torch.ones((2, 2, 2), dtype=torch.float64, requires_grad=True)

    with pytest.raises(RuntimeError, match="forward-only.*autograd"):
        solver(rho)


@pytest.mark.parametrize(
    "rho",
    [
        torch.ones((2, 2), dtype=torch.float64),
        torch.zeros((2, 2, 2), dtype=torch.float64),
        torch.tensor([[[math.nan]]], dtype=torch.float64),
    ],
)
def test_forward_validates_model_before_loading_extension(rho):
    solver = MTForward3D(freqs=[1.0])

    with pytest.raises(ValueError):
        solver(rho)
