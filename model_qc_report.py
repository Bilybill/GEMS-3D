import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-gmesdataset"))

import matplotlib.pyplot as plt
import numpy as np


def _ok_records_with_qc(records):
    return [rec for rec in records if rec.get("status") == "ok" and isinstance(rec.get("qc"), dict)]


def _group_by_anomaly(ok_records):
    grouped = {}
    for rec in ok_records:
        key = rec.get("anomaly_type", "unknown")
        grouped.setdefault(key, []).append(rec)
    return grouped


def _safe_log10(values, floor=1.0e-8):
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, floor, None)
    return np.log10(arr)


def _top_records(ok_records, field, topn=5, reverse=True):
    ranked = [rec for rec in ok_records if field in rec.get("qc", {})]
    ranked.sort(key=lambda rec: float(rec["qc"][field]), reverse=reverse)
    return ranked[:topn]


def write_model_qc_report(output_root: str, records: list[dict], prefix: str = "model_qc_report"):
    ok_records = _ok_records_with_qc(records)
    grouped = _group_by_anomaly(ok_records)

    png_path = os.path.join(output_root, f"{prefix}.png")
    md_path = os.path.join(output_root, f"{prefix}.md")

    if not records:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No records available", ha="center", va="center", fontsize=14)
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model QC Report\n\nNo records available.\n")
        return png_path, md_path

    status_counts = {}
    for rec in records:
        status = rec.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    anomaly_counts = {k: len(v) for k, v in grouped.items()}
    anomaly_names = list(grouped.keys()) or ["none"]

    anomaly_fraction_data = [np.asarray([float(rec["qc"].get("anomaly_fraction", 0.0)) for rec in grouped[key]], dtype=np.float64) for key in anomaly_names] if grouped else [np.asarray([0.0])]
    log_res_max_data = [np.asarray(_safe_log10([rec["qc"].get("res_max", 1.0) for rec in grouped[key]]), dtype=np.float64) for key in anomaly_names] if grouped else [np.asarray([0.0])]
    label_unique = np.asarray([float(rec["qc"].get("label_unique_count", 0.0)) for rec in ok_records], dtype=np.float64) if ok_records else np.asarray([0.0])
    vp_range = np.asarray([
        float(rec["qc"].get("vp_max", 0.0)) - float(rec["qc"].get("vp_min", 0.0))
        for rec in ok_records
    ], dtype=np.float64) if ok_records else np.asarray([0.0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes[0, 0]
    labels = list(status_counts.keys())
    values = [status_counts[k] for k in labels]
    ax.bar(labels, values, color=["#4CAF50" if k == "ok" else "#FFC107" if k == "skipped" else "#F44336" for k in labels])
    ax.set_title("Build Status Counts")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    if anomaly_counts:
        names = list(anomaly_counts.keys())
        counts = [anomaly_counts[k] for k in names]
        ax.bar(names, counts, color="#5B8FF9")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title("Samples per Anomaly Type")
    ax.set_ylabel("Count")

    ax = axes[0, 2]
    if grouped:
        ax.boxplot(anomaly_fraction_data, tick_labels=anomaly_names, patch_artist=True)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Anomaly Fraction by Type")
    ax.set_ylabel("Fraction")

    ax = axes[1, 0]
    if grouped:
        ax.boxplot(log_res_max_data, tick_labels=anomaly_names, patch_artist=True)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("log10(Resistivity Max) by Type")
    ax.set_ylabel("log10(Ohm-m)")

    ax = axes[1, 1]
    ax.hist(label_unique, bins=min(12, max(4, int(np.unique(label_unique).size))), color="#61DDAA", edgecolor="black")
    ax.set_title("Label Unique Count Distribution")
    ax.set_xlabel("Unique label count")
    ax.set_ylabel("Sample count")

    ax = axes[1, 2]
    ax.axis("off")
    text_lines = [
        f"Total records: {len(records)}",
        f"Successful models: {len(ok_records)}",
        f"Anomaly types: {len(grouped)}",
        f"Mean label_unique_count: {float(np.mean(label_unique)):.2f}",
        f"Mean Vp range: {float(np.mean(vp_range)):.1f} m/s",
    ]
    top_frac = _top_records(ok_records, "anomaly_fraction", topn=3, reverse=True)
    if top_frac:
        text_lines.append("")
        text_lines.append("Top anomaly_fraction samples:")
        for rec in top_frac:
            text_lines.append(
                f"- {rec.get('anomaly_type', 'unknown')} | v{rec.get('variant_index', 0):03d} | "
                f"{rec.get('source_relpath', '')} | {float(rec['qc']['anomaly_fraction']):.4f}"
            )
    ax.text(0.02, 0.98, "\n".join(text_lines), ha="left", va="top", fontsize=10, family="monospace")

    fig.suptitle("GMESDataset Model QC Report", fontsize=16)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    failed_records = [rec for rec in records if rec.get("status") == "failed"]
    top_res = _top_records(ok_records, "res_max", topn=5, reverse=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model QC Report\n\n")
        f.write(f"- Total records: {len(records)}\n")
        f.write(f"- Successful models: {len(ok_records)}\n")
        f.write(f"- Failed models: {len(failed_records)}\n")
        f.write(f"- Report figure: `{os.path.basename(png_path)}`\n\n")

        f.write("## Status Counts\n\n")
        for key, value in status_counts.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Anomaly Counts\n\n")
        for key, value in anomaly_counts.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Top Anomaly Fraction Samples\n\n")
        for rec in _top_records(ok_records, "anomaly_fraction", topn=10, reverse=True):
            f.write(
                f"- `{rec.get('source_relpath', '')}` | `{rec.get('anomaly_type', 'unknown')}` | "
                f"`variant_{int(rec.get('variant_index', 0)):03d}` | "
                f"`anomaly_fraction={float(rec['qc']['anomaly_fraction']):.6f}`\n"
            )

        f.write("\n## Top Resistivity-Max Samples\n\n")
        for rec in top_res:
            f.write(
                f"- `{rec.get('source_relpath', '')}` | `{rec.get('anomaly_type', 'unknown')}` | "
                f"`variant_{int(rec.get('variant_index', 0)):03d}` | "
                f"`res_max={float(rec['qc']['res_max']):.3f}`\n"
            )

        if failed_records:
            f.write("\n## Failed Records\n\n")
            for rec in failed_records[:20]:
                f.write(
                    f"- `{rec.get('source_relpath', '')}` | `{rec.get('anomaly_type', 'unknown')}` | "
                    f"`variant_{int(rec.get('variant_index', 0)):03d}` | "
                    f"`{rec.get('error', 'unknown error')}`\n"
                )

        f.write("\n## Raw Summary Snippet\n\n")
        summary = {
            "status_counts": status_counts,
            "anomaly_counts": anomaly_counts,
            "mean_label_unique_count": float(np.mean(label_unique)) if label_unique.size else 0.0,
            "mean_vp_range": float(np.mean(vp_range)) if vp_range.size else 0.0,
        }
        f.write("```json\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n```\n")

    return png_path, md_path
