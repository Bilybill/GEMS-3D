import argparse
import hashlib
import json
import os
import time
from typing import Iterable
import numpy as np

from core.presets import (
    DEFAULT_SEGY_SPACING,
    FORWARD_ANOMALY_TYPES,
    REGISTERED_ANOMALY_TYPES,
    load_anomaly_randomization_config,
)
from core.label_volume import load_label_volume_from_sample_npz
from model_qc_report import write_model_qc_report
from run_multiphysics_forward import (
    SEISMIC_PRESETS,
    generate_model_from_volumes,
    save_model_bundle,
    run_forward_pipeline_from_model,
)
from Seismic.forward_modeling.utils import load_velocity_volume


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VELOCITY_ROOT = os.path.join(SCRIPT_DIR, "data", "background_velocity")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs", "PretrainDataset")
DEFAULT_SAMPLE_ROOT = os.path.join(SCRIPT_DIR, "data", "samples")
DEFAULT_MODEL_SHAPE = (256, 256, 256)
DEFAULT_ANOMALY_RANDOM_CONFIG = os.path.join(SCRIPT_DIR, "configs", "pretraining_anomaly_randomization.yaml")


def _iter_velocity_files(root: str, split_dirs: Iterable[str] | None = None):
    root = os.path.abspath(root)
    split_prefixes = tuple(sorted(split_dirs or []))
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if not filename.endswith(".bin"):
                continue
            abs_path = os.path.join(current_root, filename)
            rel_path = os.path.relpath(abs_path, root)
            if split_prefixes and not rel_path.startswith(split_prefixes):
                continue
            yield abs_path, rel_path


def _stable_seed(*parts) -> int:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest()[:8], byteorder="little", signed=False) % (2**32 - 1)


def _select_anomaly_types_for_sample(anomaly_types, rel_path: str, selection_mode: str, seed_offset: int):
    if selection_mode == "all":
        return list(anomaly_types), None
    if not anomaly_types:
        return [], None
    selection_seed = _stable_seed(
        "GMESDataset",
        "anomaly-selection",
        rel_path,
        tuple(anomaly_types),
        int(seed_offset),
    )
    rng = np.random.default_rng(selection_seed)
    picked_idx = int(rng.integers(0, len(anomaly_types)))
    return [anomaly_types[picked_idx]], selection_seed


def _resolve_label_path(label_root: str | None, rel_path: str):
    if not label_root:
        return None
    candidate = os.path.join(label_root, rel_path)
    return candidate if os.path.exists(candidate) else None


def _resolve_sample_npz_path(sample_root: str | None, rel_path: str):
    if not sample_root:
        return None
    sample_rel_path = os.path.splitext(rel_path)[0] + ".npz"
    candidate = os.path.join(sample_root, sample_rel_path)
    return candidate if os.path.exists(candidate) else None


def _output_dir(output_root: str, rel_path: str, anomaly_type: str, variant_index: int = 0, variants_per_model: int = 1):
    rel_stem = os.path.splitext(rel_path)[0]
    out_dir = os.path.join(output_root, rel_stem, anomaly_type)
    if int(variants_per_model) > 1:
        out_dir = os.path.join(out_dir, f"variant_{int(variant_index):03d}")
    return out_dir


def _task_complete(task_output_dir: str, stage: str):
    model_bundle = os.path.join(task_output_dir, "model_bundle.npz")
    forward_bundle = os.path.join(task_output_dir, "forward_bundle.npz")
    if stage == "models":
        return os.path.exists(model_bundle)
    return os.path.exists(model_bundle) and os.path.exists(forward_bundle)


def _append_manifest(manifest_path: str, record: dict):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _array_qc(arr):
    arr = np.asarray(arr)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _build_model_qc(model: dict):
    qc = {
        "label_source_kind": str(model.get("label_source_kind", "none")),
        "has_label_volume": bool(model.get("label_vol") is not None),
        "anomaly_variant_index": int(model.get("anomaly_variant_index", 0)),
    }
    if model.get("anomaly_seed") is not None:
        qc["anomaly_seed"] = int(model["anomaly_seed"])
    for prefix, key in (("vp", "vp"), ("rho", "rho"), ("res", "res"), ("chi", "chi")):
        arr_stats = _array_qc(model[key])
        for stat_name, value in arr_stats.items():
            qc[f"{prefix}_{stat_name}"] = value
        qc[f"{prefix}_has_nan"] = bool(np.isnan(model[key]).any())
        qc[f"{prefix}_has_inf"] = bool(np.isinf(model[key]).any())

    anomaly_label = np.asarray(model["anomaly_label"])
    anomaly_voxels = int(np.count_nonzero(anomaly_label))
    qc["anomaly_voxels"] = anomaly_voxels
    qc["anomaly_fraction"] = float(anomaly_voxels / anomaly_label.size)

    label_vol = model.get("label_vol")
    if label_vol is not None:
        label_unique = np.unique(label_vol.astype(np.int32))
        qc["label_unique_count"] = int(label_unique.size)
        qc["label_min"] = int(label_unique.min())
        qc["label_max"] = int(label_unique.max())
    else:
        qc["label_unique_count"] = 0
        qc["label_min"] = 0
        qc["label_max"] = 0

    facies_bg = model.get("facies_bg")
    qc["facies_unique_count"] = int(np.unique(facies_bg).size) if facies_bg is not None else 0

    label_levels = model.get("label_levels")
    qc["label_level_count"] = int(np.asarray(label_levels).size) if label_levels is not None else 0
    return qc


def _resolve_label_inputs(args, rel_path: str, shape: tuple[int, int, int]):
    label_path = _resolve_label_path(args.label_root, rel_path)
    sample_npz_path = _resolve_sample_npz_path(args.sample_root, rel_path)

    resolved_label_path = None
    resolved_sample_npz_path = None
    label_vol = None
    label_levels = None
    label_source_kind = "none"

    if args.label_source_mode in {"auto", "precomputed"} and label_path is not None:
        resolved_label_path = label_path
        label_source_kind = "precomputed_label"
        label_vol = load_velocity_volume(label_path, list(shape)).astype(np.int32, copy=False)
    elif args.label_source_mode in {"auto", "samples"} and sample_npz_path is not None:
        resolved_sample_npz_path = sample_npz_path
        label_source_kind = "sample_gtime_digitized"
        label_vol, label_levels = load_label_volume_from_sample_npz(
            sample_npz_path,
            contour_num=args.label_contour_num,
        )
        label_vol = label_vol.astype(np.int32, copy=False)
        label_levels = np.asarray(label_levels, dtype=np.float32)
    else:
        if args.label_source_mode == "precomputed":
            raise FileNotFoundError(f"Precomputed label volume not found for {rel_path}")
        if args.label_source_mode == "samples":
            raise FileNotFoundError(f"Sample npz not found for {rel_path}")

    return {
        "label_path": None if resolved_label_path is None else os.path.abspath(resolved_label_path),
        "sample_npz_path": None if resolved_sample_npz_path is None else os.path.abspath(resolved_sample_npz_path),
        "label_source_kind": label_source_kind,
        "label_vol": label_vol,
        "label_levels": label_levels,
    }


def _load_forward_qc(bundle_path: str):
    qc = {}
    with np.load(bundle_path, allow_pickle=True) as data:
        for key in ("mt_status", "gravity_status", "magnetic_status", "seismic_status"):
            if key in data.files:
                qc[key] = str(np.asarray(data[key]).item())
        if "mt_freqs_hz" in data.files:
            qc["mt_freq_count"] = int(np.asarray(data["mt_freqs_hz"]).shape[0])
        if "gravity_data" in data.files:
            gravity_data = np.asarray(data["gravity_data"])
            qc["gravity_absmax"] = float(np.max(np.abs(gravity_data)))
        if "magnetic_data" in data.files:
            magnetic_data = np.asarray(data["magnetic_data"])
            qc["magnetic_absmax"] = float(np.max(np.abs(magnetic_data)))
        if "seismic_source_locations" in data.files:
            qc["seismic_shot_count"] = int(np.asarray(data["seismic_source_locations"]).shape[0])
        if "seismic_receiver_locations" in data.files:
            qc["seismic_receiver_count_per_shot"] = int(np.asarray(data["seismic_receiver_locations"]).shape[1])
        if "seismic_nt" in data.files:
            qc["seismic_nt"] = int(np.asarray(data["seismic_nt"]).item())
    return qc


def _summarize_numeric(records, field):
    values = [float(rec[field]) for rec in records if field in rec]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _write_qc_summary(output_root: str, records: list[dict], filename: str = "qc_summary.json"):
    summary = {
        "total_records": len(records),
        "status_counts": {},
        "anomaly_type_counts": {},
        "label_source_kind_counts": {},
        "forward_status_counts": {
            "mt_status": {},
            "gravity_status": {},
            "magnetic_status": {},
            "seismic_status": {},
        },
        "numeric_qc": {},
    }

    for record in records:
        status = record.get("status", "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

        anomaly_type = record.get("anomaly_type", "unknown")
        summary["anomaly_type_counts"][anomaly_type] = summary["anomaly_type_counts"].get(anomaly_type, 0) + 1

        qc = record.get("qc", {})
        label_source_kind = qc.get("label_source_kind")
        if label_source_kind:
            summary["label_source_kind_counts"][label_source_kind] = summary["label_source_kind_counts"].get(label_source_kind, 0) + 1

        for field in ("mt_status", "gravity_status", "magnetic_status", "seismic_status"):
            if field in qc:
                bucket = summary["forward_status_counts"][field]
                bucket[qc[field]] = bucket.get(qc[field], 0) + 1

    for field in (
        "label_unique_count",
        "label_level_count",
        "anomaly_fraction",
        "vp_min",
        "vp_max",
        "rho_min",
        "rho_max",
        "res_min",
        "res_max",
        "chi_min",
        "chi_max",
        "gravity_absmax",
        "magnetic_absmax",
    ):
        flat_qc = [rec["qc"] for rec in records if "qc" in rec and field in rec["qc"]]
        stats = _summarize_numeric(flat_qc, field)
        if stats is not None:
            summary["numeric_qc"][field] = stats

    summary_path = os.path.join(output_root, filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build large-scale GMESDataset pretraining data from background velocity volumes.")
    parser.add_argument("--velocity-root", type=str, default=DEFAULT_VELOCITY_ROOT, help="Root directory containing background velocity .bin files.")
    parser.add_argument("--label-root", type=str, default=None, help="Optional root directory of label volumes that mirror the velocity directory structure.")
    parser.add_argument("--sample-root", type=str, default=DEFAULT_SAMPLE_ROOT, help="Root directory of mirrored sample npz files used to generate label volumes from gtime.")
    parser.add_argument("--label-source-mode", type=str, default="auto", choices=["auto", "samples", "precomputed", "none"], help="How to obtain the stratigraphic label volume used for petrophysical conversion.")
    parser.add_argument("--label-contour-num", type=int, default=12, help="Number of internal contour levels used to convert sample gtime into label classes.")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for generated model/forward bundles.")
    parser.add_argument("--stage", type=str, default="models", choices=["models", "full"], help="models: only build 4-property models; full: also run forward. For large-scale production, prefer models here and forward later with run_pretraining_forward_from_models.py.")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=DEFAULT_MODEL_SHAPE, help="Velocity volume shape for raw .bin input.")
    parser.add_argument("--spacing", type=float, nargs=3, metavar=("DX", "DY", "DZ"), default=DEFAULT_SEGY_SPACING, help="Grid spacing in meters.")
    parser.add_argument("--anomaly-types", type=str, nargs="+", default=list(FORWARD_ANOMALY_TYPES), choices=REGISTERED_ANOMALY_TYPES, help="Anomaly presets to inject for each background velocity volume.")
    parser.add_argument("--anomaly-selection-mode", type=str, default="all", choices=["all", "random_one"], help="all: build every requested anomaly for each background; random_one: deterministically pick one anomaly type per background velocity model.")
    parser.add_argument("--variants-per-model", type=int, default=1, help="Number of randomized anomaly realizations to generate for each background velocity model and anomaly type.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Optional integer mixed into the deterministic anomaly seed to create a new randomized campaign.")
    parser.add_argument("--anomaly-random-config", type=str, default=DEFAULT_ANOMALY_RANDOM_CONFIG, help="YAML file controlling anomaly-specific randomization ranges.")
    parser.add_argument("--split-dirs", type=str, nargs="*", default=None, help="Optional top-level subdirectories to include, e.g. train-river tests-river.")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum number of background velocity files to process. 0 means all files.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks whose outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned task count without generating data.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one task fails.")
    parser.add_argument("--save-previews", action="store_true", help="Save preview PNGs during forward modeling. Disabled by default for large-scale production.")
    parser.add_argument("--manifest-name", type=str, default="build_manifest.jsonl", help="JSONL manifest file name written under output-root.")

    parser.add_argument("--skip_gravity", action="store_true")
    parser.add_argument("--skip_magnetic", action="store_true")
    parser.add_argument("--skip_mt", action="store_true")
    parser.add_argument("--skip_seismic", action="store_true")
    parser.add_argument("--anomaly_mode", dest="gravity_anomaly_mode", type=str, default="background", choices=["absolute", "background", "constant"])
    parser.add_argument("--bg_density", dest="gravity_bg_density", type=float, default=2.67)
    parser.add_argument("--gravity-algorithm", dest="gravity_algorithm", type=str, default="prism_exact", choices=["point_mass_fast", "prism_exact"])
    parser.add_argument("--device", dest="torch_device_preference", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mt-freq-min", dest="mt_freq_min", type=float, default=None)
    parser.add_argument("--mt-freq-max", dest="mt_freq_max", type=float, default=None)
    parser.add_argument("--mt-boundary-mode", dest="mt_boundary_mode", choices=["legacy", "explicit_air"], default="explicit_air")
    parser.add_argument("--mt-air-layers", dest="mt_air_layers", type=int, default=8)
    parser.add_argument("--mt-air-resistivity", dest="mt_air_resistivity", type=float, default=1.0e8)
    parser.add_argument("--mt-solver-tolerance", dest="mt_solver_tolerance", type=float, default=None)
    parser.add_argument("--mt-solver-max-iterations", dest="mt_solver_max_iterations", type=int, default=50000)
    parser.add_argument("--mt-padding-layers", dest="mt_padding_layers", type=int, default=10)
    parser.add_argument("--mt-padding-growth", dest="mt_padding_growth", type=float, default=1.4)
    mt_continuation_group = parser.add_mutually_exclusive_group()
    mt_continuation_group.add_argument("--mt-frequency-continuation", dest="mt_frequency_continuation", action="store_true")
    mt_continuation_group.add_argument("--no-mt-frequency-continuation", dest="mt_frequency_continuation", action="store_false")
    parser.set_defaults(mt_frequency_continuation=None)
    parser.add_argument("--seismic-preset", dest="seismic_preset", type=str, default="light", choices=SEISMIC_PRESETS)
    parser.add_argument("--seismic-batch-size", dest="seismic_batch_size", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = os.path.join(args.output_root, args.manifest_name)
    anomaly_random_config = None
    if args.anomaly_random_config and os.path.exists(args.anomaly_random_config):
        anomaly_random_config = load_anomaly_randomization_config(args.anomaly_random_config)
    elif args.anomaly_random_config:
        print(f"Anomaly cfg   : {os.path.abspath(args.anomaly_random_config)} (not found, using built-in defaults)")

    velocity_tasks = list(_iter_velocity_files(args.velocity_root, split_dirs=args.split_dirs))
    if args.max_samples > 0:
        velocity_tasks = velocity_tasks[: args.max_samples]

    anomaly_multiplier = len(args.anomaly_types) if args.anomaly_selection_mode == "all" else min(len(args.anomaly_types), 1)
    total_tasks = len(velocity_tasks) * anomaly_multiplier * max(int(args.variants_per_model), 1)
    print("====== GMESDataset Pretraining Dataset Builder ======")
    print(f"Velocity root : {os.path.abspath(args.velocity_root)}")
    print(f"Output root   : {os.path.abspath(args.output_root)}")
    print(f"Stage         : {args.stage}")
    print(f"Shape         : {tuple(args.shape)}")
    print(f"Spacing (m)   : {tuple(args.spacing)}")
    print(f"Anomaly types : {', '.join(args.anomaly_types)}")
    print(f"Anomaly mode  : {args.anomaly_selection_mode}")
    print(f"Variants/model: {int(args.variants_per_model)}")
    print(f"Seed offset   : {int(args.seed_offset)}")
    if anomaly_random_config is not None:
        print(f"Anomaly cfg   : {os.path.abspath(args.anomaly_random_config)}")
    if args.split_dirs:
        print(f"Split filter  : {', '.join(args.split_dirs)}")
    print(f"Velocity files: {len(velocity_tasks)}")
    print(f"Total tasks   : {total_tasks}")

    if args.dry_run:
        print("Dry run requested, no data will be generated.")
        return

    os.makedirs(args.output_root, exist_ok=True)

    started = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    completed_records = []

    for sample_idx, (velocity_path, rel_path) in enumerate(velocity_tasks, start=1):
        print(f"\n[{sample_idx}/{len(velocity_tasks)}] Background velocity: {rel_path}")
        sample_anomaly_types, selection_seed = _select_anomaly_types_for_sample(
            args.anomaly_types,
            rel_path,
            args.anomaly_selection_mode,
            args.seed_offset,
        )
        if args.anomaly_selection_mode == "random_one" and sample_anomaly_types:
            print(
                f"  --> Randomly selected anomaly type for this background: "
                f"{sample_anomaly_types[0]} (seed={selection_seed})"
            )
        try:
            vp_bg = load_velocity_volume(velocity_path, list(args.shape)).astype(np.float32, copy=False)
            label_info = _resolve_label_inputs(args, rel_path, tuple(args.shape))
            if label_info["label_vol"] is not None and tuple(label_info["label_vol"].shape) != tuple(vp_bg.shape):
                raise ValueError(
                    f"Label volume shape {label_info['label_vol'].shape} does not match velocity shape {vp_bg.shape} "
                    f"for {rel_path}"
                )
        except Exception as exc:
            print(f"  - failed to load background/labels for {rel_path}: {exc}")
            if args.stop_on_error:
                raise
            fail_count += len(sample_anomaly_types) * max(int(args.variants_per_model), 1)
            continue

        for anomaly_type in sample_anomaly_types:
            for variant_index in range(max(int(args.variants_per_model), 1)):
                task_start = time.time()
                task_output_dir = _output_dir(
                    args.output_root,
                    rel_path,
                    anomaly_type,
                    variant_index=variant_index,
                    variants_per_model=args.variants_per_model,
                )
                record = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": args.stage,
                    "status": "unknown",
                    "velocity_path": os.path.abspath(velocity_path),
                    "label_path": label_info["label_path"],
                    "sample_npz_path": label_info["sample_npz_path"],
                    "source_relpath": rel_path,
                    "anomaly_type": anomaly_type,
                    "anomaly_selection_mode": args.anomaly_selection_mode,
                    "anomaly_selection_seed": None if selection_seed is None else int(selection_seed),
                    "variant_index": int(variant_index),
                    "output_dir": os.path.abspath(task_output_dir),
                }

                if args.resume and _task_complete(task_output_dir, args.stage):
                    print(f"  - [{anomaly_type} | variant {variant_index}] skipped (resume)")
                    record["status"] = "skipped"
                    record["duration_sec"] = 0.0
                    _append_manifest(manifest_path, record)
                    completed_records.append(record)
                    skip_count += 1
                    continue

                try:
                    print(f"  - [{anomaly_type} | variant {variant_index}] building model")
                    model = generate_model_from_volumes(
                        vp_bg,
                        label_vol=label_info["label_vol"],
                        spacing=tuple(args.spacing),
                        anomaly_type=anomaly_type,
                        source_meta={
                            "source_velocity_path": os.path.abspath(velocity_path),
                            "source_label_path": label_info["label_path"],
                            "label_source_path": label_info["sample_npz_path"],
                            "label_source_kind": label_info["label_source_kind"],
                            "label_contour_num": int(args.label_contour_num),
                            "label_levels": label_info["label_levels"],
                            "source_relpath": rel_path,
                            "source_format": os.path.splitext(velocity_path)[1].lower(),
                            "anomaly_variant_index": int(variant_index),
                            "anomaly_seed_offset": int(args.seed_offset),
                            "anomaly_random_config_path": os.path.abspath(args.anomaly_random_config) if anomaly_random_config is not None else None,
                        },
                        anomaly_random_config=anomaly_random_config,
                    )

                    model_qc = _build_model_qc(model)
                    record["qc"] = model_qc
                    if model.get("anomaly_seed") is not None:
                        record["anomaly_seed"] = int(model["anomaly_seed"])
                    save_model_bundle(
                        model,
                        task_output_dir,
                        gravity_algorithm=args.gravity_algorithm,
                        seismic_preset=args.seismic_preset,
                        seismic_batch_size=args.seismic_batch_size,
                    )

                    if args.stage == "full":
                        print(f"  - [{anomaly_type} | variant {variant_index}] running forward modeling")
                        bundle_path = run_forward_pipeline_from_model(
                            task_output_dir,
                            model,
                            run_gravity=not args.skip_gravity,
                            run_magnetic=not args.skip_magnetic,
                            run_electrical=not args.skip_mt,
                            run_seismic=not args.skip_seismic,
                            gravity_anomaly_mode=args.gravity_anomaly_mode,
                            gravity_bg_density=args.gravity_bg_density,
                            gravity_algorithm=args.gravity_algorithm,
                            torch_device_preference=args.torch_device_preference,
                            seismic_preset=args.seismic_preset,
                            seismic_batch_size=args.seismic_batch_size,
                            mt_freq_min=args.mt_freq_min,
                            mt_freq_max=args.mt_freq_max,
                            mt_boundary_mode=args.mt_boundary_mode,
                            mt_air_layers=args.mt_air_layers,
                            mt_air_resistivity=args.mt_air_resistivity,
                            mt_solver_tolerance=args.mt_solver_tolerance,
                            mt_solver_max_iterations=args.mt_solver_max_iterations,
                            mt_frequency_continuation=args.mt_frequency_continuation,
                            mt_padding_layers=args.mt_padding_layers,
                            mt_padding_growth=args.mt_padding_growth,
                            save_previews=args.save_previews,
                        )
                        record["qc"].update(_load_forward_qc(bundle_path))

                    record["status"] = "ok"
                    success_count += 1
                except Exception as exc:
                    fail_count += 1
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    print(f"  - [{anomaly_type} | variant {variant_index}] failed: {exc}")
                    if args.stop_on_error:
                        record["duration_sec"] = round(time.time() - task_start, 3)
                        _append_manifest(manifest_path, record)
                        raise
                finally:
                    record["duration_sec"] = round(time.time() - task_start, 3)
                    _append_manifest(manifest_path, record)
                    completed_records.append(record)

    elapsed = time.time() - started
    qc_summary_path = _write_qc_summary(args.output_root, completed_records)
    model_qc_png, model_qc_md = write_model_qc_report(args.output_root, completed_records)
    print("\n====== Dataset Build Summary ======")
    print(f"Success : {success_count}")
    print(f"Skipped : {skip_count}")
    print(f"Failed  : {fail_count}")
    print(f"Elapsed : {elapsed / 60.0:.2f} min")
    print(f"Manifest: {manifest_path}")
    print(f"QC Summ.: {qc_summary_path}")
    print(f"QC Plot : {model_qc_png}")
    print(f"QC Note : {model_qc_md}")


if __name__ == "__main__":
    main()
