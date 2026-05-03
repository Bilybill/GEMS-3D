from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from typing import Callable
import numpy as np
import os

try:
    import yaml
except ImportError:
    yaml = None

try:
    import segyio
except ImportError:
    segyio = None

from .anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from .anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from .anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from .anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
from .anomalies.salt_dome_anomaly import SaltDomeAnomaly
from .anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from .anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams


DEFAULT_SEGY_SPACING = (10.0, 10.0, 25.0)


@dataclass(frozen=True)
class PresetBuildContext:
    vp_bg: object
    label_vol: object
    spacing: tuple[float, float, float]
    source_relpath: str | None = None
    variant_index: int = 0
    seed_offset: int = 0
    randomization_config: dict | None = None

    @property
    def shape(self):
        return self.vp_bg.shape


@dataclass(frozen=True)
class AnomalyPreset:
    key: str
    name_en: str
    name_zh: str
    anomaly: object
    rng_seed: int | None = None
    params_dict: dict | None = None


@dataclass(frozen=True)
class RegisteredAnomaly:
    key: str
    name_en: str
    name_zh: str
    factory: Callable[[PresetBuildContext], object]
    include_in_forward: bool = False
    include_in_default_viz: bool = True
    requires_explicit_include: bool = False


def read_segy_volume(path, spacing=DEFAULT_SEGY_SPACING, verbose=True):
    if segyio is None:
        raise ImportError("segyio is required to read SEGY volumes.")
    if verbose:
        print(f"Reading SEGY: {path}")
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception:
            f.mmap()
            vol = segyio.tools.cube(f)
    dx, dy, dz = spacing
    return vol, (float(dx), float(dy), float(dz))


def load_anomaly_randomization_config(config_path: str | None):
    if not config_path:
        return None
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Anomaly randomization config not found: {config_path}")
    if yaml is None:
        raise ImportError("PyYAML is required to read anomaly randomization config files.")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Anomaly randomization config must be a mapping.")
    return data


def _stable_seed(*parts) -> int:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest()[:8], byteorder="little", signed=False) % (2**32 - 1)


def _make_rng(ctx: PresetBuildContext, key: str):
    seed = _stable_seed(
        "GMESDataset",
        "anomaly-preset",
        key,
        ctx.source_relpath or "<no-source>",
        ctx.shape,
        ctx.spacing,
        int(ctx.variant_index),
        int(ctx.seed_offset),
    )
    return np.random.default_rng(seed), seed


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _integer(rng: np.random.Generator, low: int, high_inclusive: int) -> int:
    return int(rng.integers(low, high_inclusive + 1))


def _clip_float(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _grid_spans_m(ctx: PresetBuildContext):
    nx, ny, nz = ctx.shape
    dx, dy, dz = ctx.spacing
    x_span_m = max((nx - 1) * dx, dx)
    y_span_m = max((ny - 1) * dy, dy)
    z_span_m = max((nz - 1) * dz, dz)
    return float(x_span_m), float(y_span_m), float(z_span_m)


def _sample_override_value(rng: np.random.Generator, spec):
    if isinstance(spec, dict):
        if "value" in spec:
            return spec["value"]
        if "choices" in spec:
            choices = list(spec["choices"])
            if not choices:
                raise ValueError("Override 'choices' must not be empty.")
            idx = int(rng.integers(0, len(choices)))
            return choices[idx]
        if "prob_true" in spec:
            return bool(rng.random() < float(spec["prob_true"]))
        if "min" in spec and "max" in spec:
            low = spec["min"]
            high = spec["max"]
            value_type = str(spec.get("type", "float")).lower()
            if value_type == "int":
                return int(rng.integers(int(low), int(high) + 1))
            return float(rng.uniform(float(low), float(high)))
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("Override sequence must not be empty.")
        if len(spec) == 1:
            return spec[0]
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            low, high = spec
            if any(isinstance(v, float) for v in spec):
                return float(rng.uniform(float(low), float(high)))
            return int(rng.integers(int(low), int(high) + 1))
        idx = int(rng.integers(0, len(spec)))
        return spec[idx]
    return spec


def _merged_randomization_overrides(ctx: PresetBuildContext, anomaly_key: str):
    cfg = ctx.randomization_config or {}
    if not isinstance(cfg, dict):
        return {}

    merged = {}
    global_cfg = cfg.get("global", {})
    if isinstance(global_cfg, dict):
        merged.update(global_cfg)

    anomalies_cfg = cfg.get("anomalies", cfg)
    if isinstance(anomalies_cfg, dict):
        per_anomaly_cfg = anomalies_cfg.get(anomaly_key, {})
        if isinstance(per_anomaly_cfg, dict):
            merged.update(per_anomaly_cfg)
    return merged


def _apply_randomization_overrides(params, ctx: PresetBuildContext, anomaly_key: str, rng: np.random.Generator):
    overrides = _merged_randomization_overrides(ctx, anomaly_key)
    if not overrides:
        return params
    for field_name, spec in overrides.items():
        if hasattr(params, field_name):
            setattr(params, field_name, _sample_override_value(rng, spec))
    return params


def _build_igneous_swarm(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "igneous_swarm")
    x_span_m, y_span_m, z_span_m = _grid_spans_m(ctx)
    lateral_span_m = max(x_span_m, y_span_m)
    min_span_m = min(x_span_m, y_span_m)

    center_x_m = 0.5 * x_span_m + _uniform(rng, -0.12, 0.12) * x_span_m
    center_y_m = 0.5 * y_span_m + _uniform(rng, -0.12, 0.12) * y_span_m
    center_x_m = _clip_float(center_x_m, 0.18 * x_span_m, 0.82 * x_span_m)
    center_y_m = _clip_float(center_y_m, 0.18 * y_span_m, 0.82 * y_span_m)

    base_swarm_count = int(np.clip(round(min_span_m / 220.0), 7, 14))
    swarm_count = int(np.clip(base_swarm_count + _integer(rng, -2, 2), 7, 16))
    spacing_m = _clip_float(min_span_m / max(swarm_count + _integer(rng, 2, 5), 1), 70.0, 220.0)

    length_max_m = _clip_float(lateral_span_m * _uniform(rng, 0.80, 1.02), 1600.0, 10000.0)
    length_min_m = _clip_float(length_max_m * _uniform(rng, 0.38, 0.72), 800.0, max(length_max_m - 120.0, 800.0))
    vertical_base_m = _clip_float(z_span_m * _uniform(rng, 0.72, 0.95), 800.0, z_span_m * 0.98)
    swarm_thickness_min_m = _uniform(rng, 20.0, 24.0)
    swarm_thickness_max_m = max(_uniform(rng, 26.0, 32.0), swarm_thickness_min_m + 2.0)
    params = IgneousIntrusionParams(
        kind="swarm",
        dyke_x0_m=center_x_m,
        dyke_y0_m=center_y_m,
        dyke_z0_m=0.5 * vertical_base_m,
        dyke_thickness_m=_uniform(rng, 20.0, 30.0),
        dyke_length_m=length_max_m,
        dyke_width_m=vertical_base_m,
        dyke_strike_deg=_uniform(rng, 20.0, 70.0),
        dyke_dip_deg=_uniform(rng, 85.0, 89.5),
        swarm_count=swarm_count,
        swarm_spacing_m=spacing_m,
        swarm_fan_deg=_uniform(rng, 2.0, 9.0),
        swarm_spacing_jitter_frac=_uniform(rng, 0.08, 0.20),
        swarm_parallel_jitter_deg=_uniform(rng, 0.4, 1.8),
        swarm_dip_jitter_deg=_uniform(rng, 0.3, 1.2),
        swarm_thickness_min_m=swarm_thickness_min_m,
        swarm_thickness_max_m=swarm_thickness_max_m,
        swarm_length_min_m=length_min_m,
        swarm_length_max_m=length_max_m,
        swarm_echelon_zone_frac=_uniform(rng, 0.35, 0.65),
        swarm_echelon_step_m=spacing_m * _uniform(rng, 0.55, 0.95),
        swarm_top_z_m=0.0,
        swarm_base_z_m=vertical_base_m,
        vp_intr_mps=_uniform(rng, 4800.0, 5300.0),
        aureole_enable=True,
        aureole_thickness_m=_uniform(rng, 20.0, 60.0),
        edge_width_m=_uniform(rng, 1.0, 3.0),
    )
    params = _apply_randomization_overrides(params, ctx, "igneous_swarm", rng)
    return IgneousIntrusion(params=params, layer_labels=None, rng_seed=seed)


def _build_igneous_stock(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "igneous_stock")
    x_span_m, y_span_m, z_span_m = _grid_spans_m(ctx)
    params = IgneousIntrusionParams(
        kind="stock",
        stock_xc_m=_clip_float(0.5 * x_span_m + _uniform(rng, -0.15, 0.15) * x_span_m, 0.20 * x_span_m, 0.80 * x_span_m),
        stock_yc_m=_clip_float(0.5 * y_span_m + _uniform(rng, -0.15, 0.15) * y_span_m, 0.20 * y_span_m, 0.80 * y_span_m),
        stock_z_top_m=_uniform(rng, 0.18 * z_span_m, 0.40 * z_span_m),
        stock_z_base_m=_uniform(rng, 0.62 * z_span_m, 0.92 * z_span_m),
        stock_radius_m=_uniform(rng, 260.0, 620.0),
        vp_intr_mps=_uniform(rng, 5400.0, 6200.0),
        aureole_enable=True,
        aureole_thickness_m=_uniform(rng, 40.0, 120.0),
    )
    params = _apply_randomization_overrides(params, ctx, "igneous_stock", rng)
    return IgneousIntrusion(params=params, layer_labels=None, rng_seed=seed)


def _build_gas(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "gas")
    x_span_m, y_span_m, _ = _grid_spans_m(ctx)
    params = HydrocarbonHydrateParams(
        kind="gas",
        layer_id=-1,
        center_x_m=_clip_float(0.5 * x_span_m + _uniform(rng, -0.18, 0.18) * x_span_m, 0.15 * x_span_m, 0.85 * x_span_m),
        center_y_m=_clip_float(0.5 * y_span_m + _uniform(rng, -0.18, 0.18) * y_span_m, 0.15 * y_span_m, 0.85 * y_span_m),
        lens_extent_x_m=_uniform(rng, 800.0, 1600.0),
        lens_extent_y_m=_uniform(rng, 450.0, 950.0),
        lens_thickness_m=_uniform(rng, 90.0, 180.0),
        vp_gas_mps=_uniform(rng, 1600.0, 2200.0),
        gas_enable_chimney=True,
        chimney_height_m=_uniform(rng, 800.0, 1800.0),
        rng_seed=seed,
    )
    params = _apply_randomization_overrides(params, ctx, "gas", rng)
    return HydrocarbonHydrate(params=params, layer_labels=ctx.label_vol)


def _build_hydrate(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "hydrate")
    x_span_m, y_span_m, _ = _grid_spans_m(ctx)
    params = HydrocarbonHydrateParams(
        kind="hydrate",
        layer_id=-1,
        center_x_m=_clip_float(0.5 * x_span_m + _uniform(rng, -0.18, 0.18) * x_span_m, 0.15 * x_span_m, 0.85 * x_span_m),
        center_y_m=_clip_float(0.5 * y_span_m + _uniform(rng, -0.18, 0.18) * y_span_m, 0.15 * y_span_m, 0.85 * y_span_m),
        hydrate_offset_above_m=_uniform(rng, 20.0, 80.0),
        hydrate_thickness_m=_uniform(rng, 45.0, 110.0),
        vp_hydrate_mps=_uniform(rng, 3500.0, 4200.0),
        hard_gate_to_layer=True,
        hydrate_enable_patchy=bool(rng.random() < 0.35),
        rng_seed=seed,
    )
    params = _apply_randomization_overrides(params, ctx, "hydrate", rng)
    return HydrocarbonHydrate(params=params, layer_labels=ctx.label_vol)


def _build_brine_fault(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "brine_fault")
    params = BrineFaultZoneParams(
        top_k=2 if rng.random() < 0.45 else 1,
        fault_quantile=_uniform(rng, 0.994, 0.9985),
        smooth_sigma_m=_uniform(rng, 15.0, 40.0),
        min_component_voxels=_integer(rng, 2500, 9000),
        core_thickness_m=_uniform(rng, 20.0, 60.0),
        damage_width_m=_uniform(rng, 120.0, 320.0),
        edge_width_m=_uniform(rng, 12.0, 32.0),
        patch_enable=True,
        patch_sigma_m=_uniform(rng, 120.0, 280.0),
        patch_strength=_uniform(rng, 0.45, 0.92),
        patch_threshold=_uniform(rng, -0.12, 0.15),
        patch_soft=_uniform(rng, 0.12, 0.35),
        rho_core_ohmm=_uniform(rng, 0.2, 1.2),
        rho_damage_factor=_uniform(rng, 0.14, 0.40),
        vp_delta_frac_in_damage=_uniform(rng, -0.03, -0.005),
        vp_delta_frac_in_core=_uniform(rng, -0.06, -0.015),
    )
    params = _apply_randomization_overrides(params, ctx, "brine_fault", rng)
    return BrineFaultZone(params=params, vp_ref=ctx.vp_bg, rng_seed=seed)


def _build_sediment_basement(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "sediment_basement")
    params = SedimentBasementParams(
        use_layer_labels=True,
        basement_layer_id=-2 if rng.random() < 0.75 else -3,
        vp_basement_mps=_uniform(rng, 5900.0, 6800.0),
        rng_seed=seed,
    )
    params = _apply_randomization_overrides(params, ctx, "sediment_basement", rng)
    return SedimentBasementInterface(params=params, layer_labels=ctx.label_vol)


def _build_serpentinized(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "serpentinized")
    x_span_m, y_span_m, z_span_m = _grid_spans_m(ctx)
    params = SerpentinizedZoneParams(
        mode="corridor",
        use_layer_labels=True,
        center_x_m=_clip_float(0.5 * x_span_m + _uniform(rng, -0.15, 0.15) * x_span_m, 0.18 * x_span_m, 0.82 * x_span_m),
        center_y_m=_clip_float(0.5 * y_span_m + _uniform(rng, -0.15, 0.15) * y_span_m, 0.18 * y_span_m, 0.82 * y_span_m),
        corridor_length_m=_clip_float(max(x_span_m, y_span_m) * _uniform(rng, 0.55, 0.95), 1200.0, 4200.0),
        corridor_halfwidth_m=_uniform(rng, 140.0, 320.0),
        thickness_m=_clip_float(z_span_m * _uniform(rng, 0.12, 0.26), 260.0, 720.0),
        halo_thickness_m=_uniform(rng, 140.0, 280.0),
        fault_gate_radius_m=_uniform(rng, 180.0, 380.0),
        matrix_volume_frac=_uniform(rng, 0.55, 0.68),
        block_target_frac=_uniform(rng, 0.22, 0.35),
        resolved_block_count=_integer(rng, 18, 40),
        block_diameter_min_m=5.0,
        block_diameter_max_m=30.0,
        vein_spacing_m=_uniform(rng, 35.0, 70.0),
        vein_irregularity_frac=_uniform(rng, 0.20, 0.45),
        vein_intensity_frac=_uniform(rng, 0.40, 0.70),
        vp_delta_frac=_uniform(rng, -0.32, -0.18),
        rho_delta_frac=_uniform(rng, -0.16, -0.08),
        chi_add_SI=_uniform(rng, 0.008, 0.03),
        resist_delta_frac=_uniform(rng, -0.45, -0.20),
        rng_seed=seed,
    )
    params = _apply_randomization_overrides(params, ctx, "serpentinized", rng)
    return SerpentinizedZone(params=params, layer_labels=ctx.label_vol)


def _build_massive_sulfide(ctx: PresetBuildContext):
    rng, seed = _make_rng(ctx, "massive_sulfide")
    x_span_m, y_span_m, _ = _grid_spans_m(ctx)
    lens_extent_x_m = _clip_float(min(x_span_m, y_span_m) * _uniform(rng, 0.16, 0.32), 320.0, 900.0)
    lens_extent_y_m = _clip_float(lens_extent_x_m * _uniform(rng, 0.55, 0.92), 220.0, 760.0)
    params = MassiveSulfideParams(
        layer_id=-1,
        center_x_m=0.0,
        center_y_m=0.0,
        anchor_alpha=_uniform(rng, 0.02, 0.14),
        anchor_undulation_amp_m=_uniform(rng, 4.0, 18.0),
        lens_extent_x_m=lens_extent_x_m,
        lens_extent_y_m=lens_extent_y_m,
        lens_window_irregularity=_uniform(rng, 0.18, 0.45),
        lens_thickness_m=_uniform(rng, 80.0, 180.0),
        lens_thickness_var_frac=_uniform(rng, 0.16, 0.34),
        mound_bulge_amp_m=_uniform(rng, 35.0, 90.0),
        mound_sigma_frac=_uniform(rng, 0.30, 0.55),
        chimney_enable=True,
        chimney_height_m=_uniform(rng, 650.0, 1400.0),
        chimney_radius_top_m=_uniform(rng, 25.0, 55.0),
        chimney_radius_base_m=_uniform(rng, 55.0, 105.0),
        chimney_drift_m=_uniform(rng, 40.0, 180.0),
        stockwork_enable=True,
        stockwork_height_m=_uniform(rng, 550.0, 1400.0),
        stockwork_radius_top_m=_uniform(rng, 260.0, 520.0),
        stockwork_radius_base_m=_uniform(rng, 120.0, 260.0),
        stockwork_vein_threshold=_uniform(rng, 0.58, 0.78),
        stockwork_vein_softness=_uniform(rng, 0.08, 0.18),
        stockwork_anisotropy_ratio=_uniform(rng, 1.8, 3.6),
        stockwork_anisotropy_angle_deg=_uniform(rng, 0.0, 180.0),
        halo_thickness_m=_uniform(rng, 110.0, 240.0),
        vp_massive_mps=_uniform(rng, 5900.0, 6500.0),
        vp_chimney_mps=_uniform(rng, 5500.0, 6100.0),
        vp_stockwork_mps=_uniform(rng, 5000.0, 5500.0),
        rho_massive_gcc=_uniform(rng, 3.7, 4.3),
        rho_chimney_gcc=_uniform(rng, 3.2, 3.8),
        rho_stockwork_gcc=_uniform(rng, 2.8, 3.3),
        resist_massive_ohmm=_uniform(rng, 0.2, 1.0),
        resist_chimney_ohmm=_uniform(rng, 1.0, 4.0),
        resist_stockwork_ohmm=_uniform(rng, 6.0, 20.0),
        chi_massive_si=_uniform(rng, 0.009, 0.016),
        chi_chimney_si=_uniform(rng, 0.0035, 0.008),
        chi_stockwork_si=_uniform(rng, 8.0e-4, 0.0025),
        halo_chi_add_si=_uniform(rng, 3.0e-4, 8.0e-4),
        rng_seed=seed,
    )
    params = _apply_randomization_overrides(params, ctx, "massive_sulfide", rng)
    return MassiveSulfide(params=params, layer_labels=ctx.label_vol)


def _build_salt_dome(ctx: PresetBuildContext):
    nx, ny, nz = ctx.shape
    dx, dy, dz = ctx.spacing
    _, seed = _make_rng(ctx, "salt_dome")
    params = SaltDomeAnomaly.create_random_params((nx, ny, nz), (dx, dy, dz), seed=seed)
    params = _apply_randomization_overrides(params, ctx, "salt_dome", np.random.default_rng(seed + 17))
    return SaltDomeAnomaly(
        type="salt_dome",
        strength=0.0,
        edge_width_m=float(params.edge_width_m),
        params=params,
        rng_seed=seed,
    )


ANOMALY_REGISTRY = (
    RegisteredAnomaly(
        key="igneous_swarm",
        name_en="Igneous_Swarm",
        name_zh="岩墙群",
        factory=_build_igneous_swarm,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="igneous_stock",
        name_en="Igneous_Stock",
        name_zh="岩株",
        factory=_build_igneous_stock,
        requires_explicit_include=True,
    ),
    RegisteredAnomaly(
        key="gas",
        name_en="Hydrocarbon_Gas",
        name_zh="气藏",
        factory=_build_gas,
    ),
    RegisteredAnomaly(
        key="hydrate",
        name_en="Hydrocarbon_Hydrate",
        name_zh="天然气水合物",
        factory=_build_hydrate,
    ),
    RegisteredAnomaly(
        key="brine_fault",
        name_en="Brine_Fault",
        name_zh="含卤水断层",
        factory=_build_brine_fault,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="massive_sulfide",
        name_en="Massive_Sulfide",
        name_zh="块状硫化物",
        factory=_build_massive_sulfide,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="salt_dome",
        name_en="Salt_Dome",
        name_zh="盐丘",
        factory=_build_salt_dome,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="sediment_basement",
        name_en="Sediment_Basement",
        name_zh="沉积-基底界面",
        factory=_build_sediment_basement,
    ),
    RegisteredAnomaly(
        key="serpentinized",
        name_en="Serpentinized_Zone",
        name_zh="蛇纹岩化带",
        factory=_build_serpentinized,
        include_in_forward=True,
    ),
)

ANOMALY_REGISTRY_BY_KEY = {entry.key: entry for entry in ANOMALY_REGISTRY}
FORWARD_ANOMALY_TYPES = tuple(entry.key for entry in ANOMALY_REGISTRY if entry.include_in_forward)
REGISTERED_ANOMALY_TYPES = tuple(entry.key for entry in ANOMALY_REGISTRY)


def _build_preset_from_entry(entry: RegisteredAnomaly, ctx: PresetBuildContext):
    anomaly = entry.factory(ctx)
    params_obj = getattr(anomaly, "params", None)
    params_dict = asdict(params_obj) if is_dataclass(params_obj) else None
    rng_seed = getattr(anomaly, "rng_seed", None)
    return AnomalyPreset(
        key=entry.key,
        name_en=entry.name_en,
        name_zh=entry.name_zh,
        anomaly=anomaly,
        rng_seed=None if rng_seed is None else int(rng_seed),
        params_dict=params_dict,
    )


def build_registered_presets(vp_bg, label_vol, spacing, keys, source_relpath=None, variant_index=0, seed_offset=0, randomization_config=None):
    ctx = PresetBuildContext(
        vp_bg=vp_bg,
        label_vol=label_vol,
        spacing=spacing,
        source_relpath=source_relpath,
        variant_index=int(variant_index),
        seed_offset=int(seed_offset),
        randomization_config=randomization_config,
    )
    presets = []
    for key in keys:
        entry = ANOMALY_REGISTRY_BY_KEY.get(key)
        if entry is None:
            raise ValueError(f"Unsupported anomaly type: {key}")
        presets.append(_build_preset_from_entry(entry, ctx))
    return presets


def build_named_anomaly_preset(anomaly_type, vp_bg, label_vol, spacing, source_relpath=None, variant_index=0, seed_offset=0, randomization_config=None):
    return build_registered_presets(
        vp_bg,
        label_vol,
        spacing,
        [anomaly_type],
        source_relpath=source_relpath,
        variant_index=variant_index,
        seed_offset=seed_offset,
        randomization_config=randomization_config,
    )[0]


def build_default_viz_presets(vp_bg, label_vol, spacing, include_stock=False, source_relpath=None, variant_index=0, seed_offset=0, randomization_config=None):
    keys = []
    for entry in ANOMALY_REGISTRY:
        if not entry.include_in_default_viz:
            continue
        if entry.requires_explicit_include and not include_stock:
            continue
        keys.append(entry.key)
    return build_registered_presets(
        vp_bg,
        label_vol,
        spacing,
        keys,
        source_relpath=source_relpath,
        variant_index=variant_index,
        seed_offset=seed_offset,
        randomization_config=randomization_config,
    )


def build_all_anomalies(vp_bg, label_vol, spacing, include_stock=True, source_relpath=None, variant_index=0, seed_offset=0, randomization_config=None):
    keys = []
    for entry in ANOMALY_REGISTRY:
        if entry.requires_explicit_include and not include_stock:
            continue
        keys.append(entry.key)
    return [
        preset.anomaly
        for preset in build_registered_presets(
            vp_bg,
            label_vol,
            spacing,
            keys,
            source_relpath=source_relpath,
            variant_index=variant_index,
            seed_offset=seed_offset,
            randomization_config=randomization_config,
        )
    ]
