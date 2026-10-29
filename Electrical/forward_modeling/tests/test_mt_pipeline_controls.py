import inspect
import sys
from pathlib import Path

import pytest


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from core.forward_modeling.electrical import ElectricalForwardSolver  # noqa: E402
from run_multiphysics_forward import run_forward_pipeline_from_model  # noqa: E402


def test_electrical_solver_exposes_effective_explicit_air_settings():
    solver = ElectricalForwardSolver(
        freqs=[1.0],
        dx=100.0,
        dy=120.0,
        dz=80.0,
        boundary_mode="explicit_air",
        air_layers=6,
        air_resistivity=2.0e8,
        solver_tolerance=1.0e-9,
        solver_max_iterations=1234,
        frequency_continuation=True,
        padding_layers=12,
        padding_growth=1.25,
    )

    assert solver.settings == {
        "boundary_mode": "explicit_air",
        "air_layers": 6,
        "air_resistivity": 2.0e8,
        "solver_tolerance": 1.0e-9,
        "solver_max_iterations": 1234,
        "frequency_continuation": True,
        "padding_layers": 12,
        "padding_growth": 1.25,
    }


def test_electrical_solver_reports_legacy_effective_air_layer_count_as_zero():
    solver = ElectricalForwardSolver(
        freqs=[1.0],
        boundary_mode="legacy",
        air_layers=8,
    )

    assert solver.settings["boundary_mode"] == "legacy"
    assert solver.settings["air_layers"] == 0
    assert solver.settings["solver_tolerance"] == pytest.approx(1.0e-7)
    assert solver.settings["frequency_continuation"] is True
    assert solver.settings["padding_layers"] == 10
    assert solver.settings["padding_growth"] == pytest.approx(1.4)


def test_multiphysics_pipeline_defaults_to_reproducible_explicit_air_mode():
    parameters = inspect.signature(run_forward_pipeline_from_model).parameters

    assert parameters["mt_boundary_mode"].default == "explicit_air"
    assert parameters["mt_solver_tolerance"].default is None
    assert parameters["mt_frequency_continuation"].default is None
    assert parameters["mt_padding_layers"].default == 10
    assert parameters["mt_padding_growth"].default == pytest.approx(1.4)
