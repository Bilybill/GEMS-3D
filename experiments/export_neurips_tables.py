from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


MODEL_ORDER = ("unet", "pinn", "deeponet", "fno", "gnot")
FORWARD_TASK_ORDER = ("rho_to_gravity", "chi_to_magnetic", "res_to_mt", "vp_to_seismic")
INVERSE_TASK_ORDER = (
    "gravity_to_rho",
    "magnetic_to_chi",
    "mt_to_phi",
    "seismic_to_vp",
    "gravity_magnetic_to_rho_chi",
    "gravity_magnetic_mt_to_rho_chi_phi",
    "gravity_magnetic_mt_seismic_to_vp_rho_phi_chi",
)
COMPLETION_TARGET_ORDER = ("gravity", "magnetic", "mt", "seismic")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_base() -> Path:
    root = _project_root()
    if root.name == "GEMS-3D":
        return root / "outputs" / "ExperimentRuns"
    return root / "DATAFOLDER" / "ExperimentRuns"


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _iter_json(root: Path, filename: str):
    if not root.exists():
        return
    for path in sorted(root.rglob(filename)):
        payload = _load_json(path)
        if payload is not None:
            yield path, payload


def _summary(payload: dict, split: str) -> dict:
    if "summary" in payload and split in payload["summary"]:
        return payload["summary"][split]
    if "summary" in payload and isinstance(payload["summary"], dict):
        return payload["summary"]
    return payload


def _metric_stats(summary: dict, key: str) -> tuple[float, float | None]:
    stats = summary.get("metric_stats", {}).get(key, {})
    mean = stats.get("mean", summary.get(key))
    std = stats.get("std")
    try:
        mean = float(mean)
    except (TypeError, ValueError):
        mean = float("nan")
    try:
        std = float(std) if std is not None else None
    except (TypeError, ValueError):
        std = None
    return mean, std


def _fmt_number(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_pm(summary: dict) -> str:
    r_mean, r_std = _metric_stats(summary, "pearson_r")
    mae_mean, mae_std = _metric_stats(summary, "mae")
    r = _fmt_number(r_mean) if r_std is None else f"{_fmt_number(r_mean)}±{_fmt_number(r_std)}"
    mae = _fmt_number(mae_mean) if mae_std is None else f"{_fmt_number(mae_mean)}±{_fmt_number(mae_std)}"
    return f"{r} / {mae}"


def _task_key(modalities: list[str], targets: list[str]) -> str:
    return f"{'_'.join(modalities)}_to_{'_'.join(targets)}"


def collect_forward(root: Path) -> dict:
    rows = {}
    for path, payload in _iter_json(root, "metrics.json"):
        task = payload.get("task")
        model = payload.get("model")
        if task == "vp_source_to_seismic_shot" and model == "shot_film":
            task = "vp_to_seismic"
            model = "unet"
        if task not in FORWARD_TASK_ORDER or model not in MODEL_ORDER:
            continue
        heldout_payload = _load_json(path.parent / "heldout_eval.json")
        heldout_summary = heldout_payload.get("summary", {}) if heldout_payload else _summary(payload, "heldout_at_best_validation")
        rows[(task, model, "validation")] = _fmt_pm(_summary(payload, "best_validation"))
        rows[(task, model, "heldout")] = _fmt_pm(heldout_summary)
        rows[(task, model, "time_ms")] = _fmt_number(float(heldout_summary.get("inference_time_ms", float("nan"))), digits=1)
    return rows


def collect_inverse(root: Path) -> dict:
    rows = {}
    for path, payload in _iter_json(root, "metrics.json"):
        if payload.get("task") != "inverse_regression":
            continue
        model = payload.get("model")
        key = _task_key(payload.get("modalities", []), payload.get("target_properties", []))
        if model not in MODEL_ORDER:
            continue
        heldout_payload = _load_json(path.parent / "heldout_eval.json")
        heldout_summary = heldout_payload.get("summary", {}) if heldout_payload else _summary(payload, "heldout_at_best_validation")
        rows[(key, model, "validation")] = _fmt_pm(_summary(payload, "best_validation"))
        rows[(key, model, "heldout")] = _fmt_pm(heldout_summary)
    return rows


def collect_completion(root: Path) -> dict:
    rows = {}
    for path, payload in _iter_json(root, "metrics.json"):
        if payload.get("task") != "modality_completion":
            continue
        model = payload.get("model")
        target = payload.get("target_modality")
        if model not in ("unet", "fno"):
            continue
        heldout_payload = _load_json(path.parent / "heldout_eval.json")
        heldout_summary = heldout_payload.get("summary", {}) if heldout_payload else _summary(payload, "heldout_at_best_validation")
        rows[(target, model)] = _fmt_pm(heldout_summary)
    return rows


def collect_degraded(root: Path) -> dict:
    rows = {}
    if not root.exists():
        return rows
    for path in sorted(root.glob("degraded_inversion_*.json")):
        payload = _load_json(path)
        if not payload:
            continue
        model = payload.get("model")
        if model not in ("unet", "fno"):
            continue
        for corruption, entry in payload.get("degraded", {}).items():
            retention = entry.get("retention", {}).get("pearson_r", float("nan"))
            rows[(corruption, model)] = f"{_fmt_pm(entry.get('summary', {}))} + {_fmt_number(retention)}"
    return rows


def _write_markdown(path: Path, forward: dict, inverse: dict, completion: dict, degraded: dict):
    with path.open("w", encoding="utf-8") as f:
        f.write("# NeurIPS Table Export\n\n")
        f.write("## Forward Modeling\n\n")
        f.write("| Task | Report | " + " | ".join(MODEL_ORDER) + " |\n")
        f.write("|" + "|".join(["---"] * (2 + len(MODEL_ORDER))) + "|\n")
        for task in FORWARD_TASK_ORDER:
            for report in ("validation", "heldout", "time_ms"):
                values = [forward.get((task, model, report), "--") for model in MODEL_ORDER]
                f.write(f"| {task} | {report} | " + " | ".join(values) + " |\n")
        f.write("\n## Inverse Regression\n\n")
        f.write("| Task | Report | " + " | ".join(MODEL_ORDER) + " |\n")
        f.write("|" + "|".join(["---"] * (2 + len(MODEL_ORDER))) + "|\n")
        for task in INVERSE_TASK_ORDER:
            for report in ("validation", "heldout"):
                values = [inverse.get((task, model, report), "--") for model in MODEL_ORDER]
                f.write(f"| {task} | {report} | " + " | ".join(values) + " |\n")
        f.write("\n## Modality Completion\n\n")
        f.write("| Missing target | U-Net | FNO |\n|---|---|---|\n")
        for target in COMPLETION_TARGET_ORDER:
            f.write(f"| {target} | {completion.get((target, 'unet'), '--')} | {completion.get((target, 'fno'), '--')} |\n")
        f.write("\n## Degraded Inversion\n\n")
        f.write("| Corruption | U-Net | FNO |\n|---|---|---|\n")
        for corruption in ("reduced_sampling", "additive_noise", "amplitude_perturbation"):
            f.write(f"| {corruption} | {degraded.get((corruption, 'unet'), '--')} | {degraded.get((corruption, 'fno'), '--')} |\n")


def parse_args():
    base = _default_base()
    parser = argparse.ArgumentParser(description="Export GMES-3D experiment metrics into manuscript-oriented tables.")
    parser.add_argument("--forward-root", type=Path, default=base / "forward_exp1_braided_crossed")
    parser.add_argument("--inverse-root", type=Path, default=base / "inverse_regression")
    parser.add_argument("--completion-root", type=Path, default=base / "modality_completion")
    parser.add_argument("--robustness-root", type=Path, default=base / "robustness")
    parser.add_argument("--output-dir", type=Path, default=base / "neurips_table_exports")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forward = collect_forward(args.forward_root)
    inverse = collect_inverse(args.inverse_root)
    completion = collect_completion(args.completion_root)
    degraded = collect_degraded(args.robustness_root)
    summary = {
        "forward": {str(key): value for key, value in forward.items()},
        "inverse": {str(key): value for key, value in inverse.items()},
        "completion": {str(key): value for key, value in completion.items()},
        "degraded": {str(key): value for key, value in degraded.items()},
    }
    with (args.output_dir / "neurips_table_export.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    _write_markdown(args.output_dir / "neurips_table_export.md", forward, inverse, completion, degraded)
    print(f"[+] Wrote {args.output_dir / 'neurips_table_export.md'}")
    print(f"[+] Wrote {args.output_dir / 'neurips_table_export.json'}")


if __name__ == "__main__":
    main()
