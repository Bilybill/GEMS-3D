import json
import sys
from pathlib import Path

import pytest


GEMS_ROOT = Path(__file__).resolve().parents[3]
if str(GEMS_ROOT) not in sys.path:
    sys.path.insert(0, str(GEMS_ROOT))

from Electrical.forward_modeling.examples.cross_validate_mt3d_mfem import (  # noqa: E402
    _read_comp_metadata,
    _validate_reused_mfem_metadata,
)


def test_comp_periods_are_checked_in_frequency_units(tmp_path):
    comp_path = tmp_path / "case.comp"
    comp_path.write_text("2\n1 10\n2 0.25\n\n0\n")

    frequencies, sites = _read_comp_metadata(comp_path)

    assert frequencies == pytest.approx([0.1, 4.0])
    assert sites == {}


def test_reused_mfem_metadata_rejects_frequency_mismatch(tmp_path):
    summary = {
        "shape": [4, 4, 6],
        "spacing_m": [10.0, 10.0, 5.0],
        "frequencies_hz": [4.0, 1.0],
        "air_layers": 3,
        "background_rho_ohm_m": 100.0,
        "anomaly_rho_ohm_m": 10.0,
        "matched_side_bottom_padding": {"layers": 2, "growth": 1.4},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="frequencies"):
        _validate_reused_mfem_metadata(
            tmp_path,
            (4, 4, 6),
            (10.0, 10.0, 5.0),
            [4.0, 2.0],
            3,
            100.0,
            10.0,
            2,
            1.4,
        )
