from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Iterable


DEFAULT_TASK_ORDER = [
    "rho_to_gravity",
    "chi_to_magnetic",
    "res_to_mt",
    "vp_to_seismic",
    "vp_source_to_seismic_shot",
    "joint_multiphysics",
]

DEFAULT_MODEL_ORDER = [
    "unet",
    "pinn",
    "deeponet",
    "fno",
    "gnot",
    "shot_film",
]


def _load_json_if_exists(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_run_directories(root: str) -> list[str]:
    run_dirs = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        if entry == "logs":
            continue
        if os.path.exists(os.path.join(path, "metrics.json")) or os.path.exists(os.path.join(path, "heldout_eval.json")):
            run_dirs.append(path)
    return run_dirs


def _get_nested(mapping: dict | None, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_summary_row(run_dir: str) -> dict:
    metrics_path = os.path.join(run_dir, "metrics.json")
    heldout_eval_path = os.path.join(run_dir, "heldout_eval.json")
    metrics = _load_json_if_exists(metrics_path) or {}
    heldout_eval = _load_json_if_exists(heldout_eval_path) or {}

    task = metrics.get("task") or heldout_eval.get("task") or os.path.basename(run_dir)
    model = metrics.get("model") or heldout_eval.get("model") or "unknown"
    summary = metrics.get("summary", {})

    best_val = summary.get("best_validation", {})
    heldout_at_best = summary.get("heldout_at_best_validation", {})
    final_val = summary.get("final_validation", {})
    final_heldout = summary.get("final_heldout", {})
    heldout_summary = heldout_eval.get("summary", {}) or heldout_at_best or final_heldout

    row = {
        "task": task,
        "model": model,
        "run_dir": os.path.abspath(run_dir),
        "metrics_json": os.path.abspath(metrics_path) if os.path.exists(metrics_path) else "",
        "heldout_eval_json": os.path.abspath(heldout_eval_path) if os.path.exists(heldout_eval_path) else "",
        "best_epoch": _get_nested(metrics, "selection", "best_epoch"),
        "train_size": _get_nested(metrics, "splits", "train_size"),
        "val_size": _get_nested(metrics, "splits", "val_size"),
        "heldout_size": heldout_eval.get("num_samples", _get_nested(metrics, "splits", "heldout_size")),
        "val_mae": _float_or_none(best_val.get("mae")),
        "val_r": _float_or_none(best_val.get("pearson_r")),
        "val_nonzero_rl2": _float_or_none(best_val.get("nonzero_relative_l2")),
        "val_rl2": _float_or_none(best_val.get("relative_l2")),
        "heldout_mae": _float_or_none(heldout_summary.get("mae")),
        "heldout_r": _float_or_none(heldout_summary.get("pearson_r")),
        "heldout_nonzero_rl2": _float_or_none(heldout_summary.get("nonzero_relative_l2")),
        "heldout_rl2": _float_or_none(heldout_summary.get("relative_l2")),
        "heldout_inference_time_ms": _float_or_none(heldout_summary.get("inference_time_ms")),
        "final_val_mae": _float_or_none(final_val.get("mae")),
        "final_val_r": _float_or_none(final_val.get("pearson_r")),
        "final_val_nonzero_rl2": _float_or_none(final_val.get("nonzero_relative_l2")),
        "final_val_rl2": _float_or_none(final_val.get("relative_l2")),
        "final_heldout_mae": _float_or_none(final_heldout.get("mae")),
        "final_heldout_r": _float_or_none(final_heldout.get("pearson_r")),
        "final_heldout_nonzero_rl2": _float_or_none(final_heldout.get("nonzero_relative_l2")),
        "final_heldout_rl2": _float_or_none(final_heldout.get("relative_l2")),
    }
    return row


def _sort_key(row: dict):
    task = row.get("task", "")
    model = row.get("model", "")
    task_rank = DEFAULT_TASK_ORDER.index(task) if task in DEFAULT_TASK_ORDER else len(DEFAULT_TASK_ORDER)
    model_rank = DEFAULT_MODEL_ORDER.index(model) if model in DEFAULT_MODEL_ORDER else len(DEFAULT_MODEL_ORDER)
    return (task_rank, task, model_rank, model)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_json(path: str, rows: list[dict]):
    payload = {
        "num_runs": len(rows),
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: str, rows: list[dict]):
    headers = [
        "Task",
        "Model",
        "Best Epoch",
        "Val MAE",
        "Val R",
        "Val nonzero-RL2",
        "Held-out MAE",
        "Held-out R",
        "Held-out nonzero-RL2",
        "Held-out RL2",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [
                row.get("task", ""),
                row.get("model", ""),
                row.get("best_epoch", ""),
                _format_value(row.get("val_mae")),
                _format_value(row.get("val_r")),
                _format_value(row.get("val_nonzero_rl2")),
                _format_value(row.get("heldout_mae")),
                _format_value(row.get("heldout_r")),
                _format_value(row.get("heldout_nonzero_rl2")),
                _format_value(row.get("heldout_rl2")),
            ]
            f.write("| " + " | ".join(str(v) for v in values) + " |\n")


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(
        os.path.join(script_dir, "..", "DATAFOLDER", "ExperimentRuns", "forward_exp1_braided_crossed")
    )
    parser = argparse.ArgumentParser(description="Summarize forward-modeling experiment results into JSON/CSV/Markdown tables.")
    parser.add_argument("--root", type=str, default=default_root, help="Root directory containing per-run forward experiment outputs.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="forward_results_summary",
        help="Output file prefix written under --root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Result root does not exist: {root}")

    run_dirs = _collect_run_directories(root)
    rows = [_build_summary_row(run_dir) for run_dir in run_dirs]
    rows.sort(key=_sort_key)

    output_prefix = os.path.join(root, args.output_prefix)
    json_path = output_prefix + ".json"
    csv_path = output_prefix + ".csv"
    md_path = output_prefix + ".md"

    fieldnames = [
        "task",
        "model",
        "best_epoch",
        "train_size",
        "val_size",
        "heldout_size",
        "val_mae",
        "val_r",
        "val_nonzero_rl2",
        "val_rl2",
        "heldout_mae",
        "heldout_r",
        "heldout_nonzero_rl2",
        "heldout_rl2",
        "heldout_inference_time_ms",
        "final_val_mae",
        "final_val_r",
        "final_val_nonzero_rl2",
        "final_val_rl2",
        "final_heldout_mae",
        "final_heldout_r",
        "final_heldout_nonzero_rl2",
        "final_heldout_rl2",
        "run_dir",
        "metrics_json",
        "heldout_eval_json",
    ]

    _write_json(json_path, rows)
    _write_csv(csv_path, rows, fieldnames=fieldnames)
    _write_markdown(md_path, rows)

    print(f"[+] Found {len(rows)} runs under {root}")
    print(f"[+] Wrote JSON summary to {json_path}")
    print(f"[+] Wrote CSV summary to {csv_path}")
    print(f"[+] Wrote Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
