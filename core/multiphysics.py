import numpy as np

from .builder import DatasetBuilder
from .petrophysics.rock_physics import PetrophysicsConverter


def build_multiphysics_model(vp_bg, label_vol, anomalies, dx, dy, dz, converter=None):
    """
    Build a consistent multi-physics model by:
    1. generating facies-aware background properties from the Vp volume, then
    2. applying each anomaly through its own `apply_properties()` implementation.

    Internal property keys follow the anomaly API:
      - `vp`
      - `rho`     (g/cm^3)
      - `resist`  (Ohm-m)
      - `chi`     (SI)
    """
    vp_bg = np.asarray(vp_bg, dtype=np.float32)
    label_arr = None if label_vol is None else np.asarray(label_vol, dtype=np.int32)
    anomaly_list = list(anomalies or [])

    if converter is None:
        converter = PetrophysicsConverter()

    rho_bg, resist_bg, chi_bg = converter.generate_background(vp_bg, label_vol=label_arr)
    background_state = converter.get_last_background_state()

    props_bg = {
        "vp": vp_bg.copy(),
        "rho": rho_bg.copy(),
        "resist": resist_bg.copy(),
        "chi": chi_bg.copy(),
    }

    builder = DatasetBuilder(dx, dy, dz)
    props_final, anomaly_label, X, Y, Z = builder.inject_properties(props_bg, anomaly_list)

    return {
        "vp": props_final["vp"],
        "rho": props_final["rho"],
        "resist": props_final["resist"],
        "chi": props_final["chi"],
        "rho_bg": rho_bg,
        "resist_bg": resist_bg,
        "chi_bg": chi_bg,
        "facies_bg": None if background_state is None else background_state.get("facies"),
        "background_state": background_state,
        "anomaly_label": anomaly_label,
        "X": X,
        "Y": Y,
        "Z": Z,
    }
