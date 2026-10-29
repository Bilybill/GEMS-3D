import argparse
import os
import time

from build_pretraining_dataset import _append_manifest, _load_forward_qc, _write_qc_summary
from run_multiphysics_forward import SEISMIC_PRESETS, load_model_bundle, run_forward_pipeline_from_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ROOT = os.path.join(SCRIPT_DIR, "outputs", "PretrainDataset")


def _iter_model_bundles(root: str, split_dirs=None):
    root = os.path.abspath(root)
    split_prefixes = tuple(sorted(split_dirs or []))
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        if "model_bundle.npz" not in filenames:
            continue
        rel_dir = os.path.relpath(current_root, root)
        if split_prefixes and not rel_dir.startswith(split_prefixes):
            continue
        yield os.path.join(current_root, "model_bundle.npz"), rel_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run gravity/magnetic/MT/seismic forward modeling from previously generated model_bundle.npz files.")
    parser.add_argument("--model-root", type=str, default=DEFAULT_MODEL_ROOT, help="Root directory containing model_bundle.npz files.")
    parser.add_argument("--forward-root", type=str, default=None, help="Optional separate output root for forward_bundle.npz. Defaults to writing next to each model_bundle.")
    parser.add_argument("--split-dirs", type=str, nargs="*", default=None, help="Optional top-level subdirectories to include, e.g. train-river tests-river.")
    parser.add_argument("--anomaly-types", type=str, nargs="*", default=None, help="Optional anomaly type filter. If unset, all model bundles are processed.")
    parser.add_argument("--max-bundles", type=int, default=0, help="Maximum number of model bundles to process. 0 means all bundles.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks whose forward_bundle already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned task count without running forward modeling.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one task fails.")
    parser.add_argument("--save-previews", action="store_true", help="Save preview PNGs during forward modeling.")
    parser.add_argument("--manifest-name", type=str, default="forward_manifest.jsonl", help="JSONL manifest file name written under the effective output root.")

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
    parser.add_argument("--seismic-freq-min", dest="seismic_freq_min", type=float, default=None)
    parser.add_argument("--seismic-freq-max", dest="seismic_freq_max", type=float, default=None)
    parser.add_argument("--seismic-preset", dest="seismic_preset", type=str, default="light", choices=SEISMIC_PRESETS)
    parser.add_argument("--seismic-batch-size", dest="seismic_batch_size", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    forward_root = os.path.abspath(args.forward_root or args.model_root)
    manifest_path = os.path.join(forward_root, args.manifest_name)

    tasks = list(_iter_model_bundles(args.model_root, split_dirs=args.split_dirs))
    if args.max_bundles > 0:
        tasks = tasks[: args.max_bundles]

    print("====== GMESDataset Forward-Only Runner ======")
    print(f"Model root    : {os.path.abspath(args.model_root)}")
    print(f"Forward root  : {forward_root}")
    if args.split_dirs:
        print(f"Split filter  : {', '.join(args.split_dirs)}")
    if args.anomaly_types:
        print(f"Anomaly filter: {', '.join(args.anomaly_types)}")
    print(f"Model bundles : {len(tasks)}")

    if args.dry_run:
        print("Dry run requested, no forward modeling will be executed.")
        return

    os.makedirs(forward_root, exist_ok=True)

    started = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    completed_records = []

    for task_idx, (model_bundle_path, rel_dir) in enumerate(tasks, start=1):
        task_output_dir = os.path.join(forward_root, rel_dir) if args.forward_root else os.path.dirname(model_bundle_path)
        forward_bundle_path = os.path.join(task_output_dir, "forward_bundle.npz")
        task_start = time.time()

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "forward_only",
            "status": "unknown",
            "model_bundle_path": os.path.abspath(model_bundle_path),
            "output_dir": os.path.abspath(task_output_dir),
        }

        if args.resume and os.path.exists(forward_bundle_path):
            print(f"[{task_idx}/{len(tasks)}] skipped (resume): {rel_dir}")
            record["status"] = "skipped"
            record["duration_sec"] = 0.0
            _append_manifest(manifest_path, record)
            completed_records.append(record)
            skip_count += 1
            continue

        try:
            model = load_model_bundle(model_bundle_path)
            record["source_relpath"] = str(model.get("source_relpath", rel_dir))
            record["anomaly_type"] = str(model.get("anomaly_type", "unknown"))
            record["variant_index"] = int(model.get("anomaly_variant_index", 0))
            if model.get("anomaly_seed") is not None:
                record["anomaly_seed"] = int(model["anomaly_seed"])

            if args.anomaly_types and record["anomaly_type"] not in set(args.anomaly_types):
                print(f"[{task_idx}/{len(tasks)}] skipped (anomaly filter): {rel_dir}")
                record["status"] = "skipped"
                record["duration_sec"] = 0.0
                _append_manifest(manifest_path, record)
                completed_records.append(record)
                skip_count += 1
                continue

            print(
                f"[{task_idx}/{len(tasks)}] forward: {record['source_relpath']} | "
                f"{record['anomaly_type']} | variant {record['variant_index']}"
            )
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
                seismic_freq_min=args.seismic_freq_min,
                seismic_freq_max=args.seismic_freq_max,
                save_previews=args.save_previews,
            )
            record["qc"] = _load_forward_qc(bundle_path)
            record["status"] = "ok"
            success_count += 1
        except Exception as exc:
            fail_count += 1
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"  - forward failed for {rel_dir}: {exc}")
            if args.stop_on_error:
                record["duration_sec"] = round(time.time() - task_start, 3)
                _append_manifest(manifest_path, record)
                raise
        finally:
            record["duration_sec"] = round(time.time() - task_start, 3)
            _append_manifest(manifest_path, record)
            completed_records.append(record)

    qc_summary_path = _write_qc_summary(forward_root, completed_records, filename="forward_qc_summary.json")
    elapsed = time.time() - started
    print("\n====== Forward Summary ======")
    print(f"Success : {success_count}")
    print(f"Skipped : {skip_count}")
    print(f"Failed  : {fail_count}")
    print(f"Elapsed : {elapsed / 60.0:.2f} min")
    print(f"Manifest: {manifest_path}")
    print(f"QC Summ.: {qc_summary_path}")


if __name__ == "__main__":
    main()
