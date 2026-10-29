import sys
from pathlib import Path

import numpy as np
import pytest


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.examples.run_datfolder_mt_demo import (
    _boundary_mean,
    _geometric_padding_extent,
    _refine_vertical_model,
)
from Electrical.forward_modeling.examples.visualize_mt_cigvis import _load_results


def test_vertical_refinement_preserves_values_and_physical_extent():
    model = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1.0
    spacing = (50.0, 60.0, 80.0)

    refined, refined_spacing = _refine_vertical_model(model, spacing, 2)

    assert refined.shape == (2, 3, 8)
    np.testing.assert_array_equal(refined[:, :, 0::2], model)
    np.testing.assert_array_equal(refined[:, :, 1::2], model)
    assert refined_spacing == (50.0, 60.0, 40.0)
    assert refined.shape[2] * refined_spacing[2] == model.shape[2] * spacing[2]


def test_source_boundary_mean_is_not_recomputed_after_vertical_refinement():
    model = np.ones((3, 3, 3), dtype=np.float64)
    model[1, 1, 0] = 100.0
    refined, _ = _refine_vertical_model(model, (1.0, 1.0, 2.0), 2)

    assert _boundary_mean(model) != pytest.approx(_boundary_mean(refined))


def test_geometric_padding_extent_matches_explicit_sum():
    assert _geometric_padding_extent(64.0, 12, 1.4) == pytest.approx(
        sum(64.0 * 1.4**index for index in range(1, 13))
    )


def test_visualizer_prefers_saved_solver_background(tmp_path):
    path = tmp_path / "results.npz"
    np.savez(
        path,
        frequencies_hz=np.asarray([1000.0]),
        spacing_m=np.asarray([50.0, 50.0, 64.0]),
        resistivity_model=np.ones((2, 2, 2), dtype=np.float32),
        apparent_resistivity=np.ones((1, 2, 2, 2), dtype=np.float32),
        phase_deg=np.zeros((1, 2, 2, 2), dtype=np.float32),
        background_resistivity_ohm_m=np.float64(2990.5),
    )

    loaded = _load_results(path)

    assert loaded["background_resistivity_ohm_m"] == pytest.approx(2990.5)


@pytest.mark.parametrize("factor", [0, -1])
def test_vertical_refinement_rejects_nonpositive_factor(factor):
    with pytest.raises(ValueError, match="vertical_refinement"):
        _refine_vertical_model(np.ones((1, 1, 1)), (1.0, 1.0, 1.0), factor)


@pytest.mark.parametrize("factor", [True, 1.5, "2"])
def test_vertical_refinement_rejects_noninteger_factor(factor):
    with pytest.raises(TypeError, match="vertical_refinement"):
        _refine_vertical_model(np.ones((1, 1, 1)), (1.0, 1.0, 1.0), factor)
