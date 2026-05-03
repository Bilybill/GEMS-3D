from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np


DEFAULT_THRESHOLDS = (1.0, 0.99, 0.95, 0.90, 0.75, 0.50, 0.33, 0.20, 0.10, 0.05, 0.01)


def _round_freqs(values) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in values)


def _collect_mt_frequency_stats(root: str):
    sequence_counter = Counter()
    frequency_counter = Counter()
    count_counter = Counter()
    top_level_counter = Counter()
    example_by_sequence = {}
    frequency_sets = []

    total_samples = 0
    for dirpath, _, filenames in os.walk(root):
        if "forward_bundle.npz" not in filenames:
            continue
        bundle_path = os.path.join(dirpath, "forward_bundle.npz")
        relpath = os.path.relpath(dirpath, root)
        top_level = relpath.split(os.sep, 1)[0]

        with np.load(bundle_path) as bundle:
            if "mt_freqs_hz" not in bundle:
                continue
            freqs = _round_freqs(np.asarray(bundle["mt_freqs_hz"], dtype=np.float64).tolist())

        total_samples += 1
        sequence_counter[freqs] += 1
        count_counter[len(freqs)] += 1
        top_level_counter[top_level] += 1
        example_by_sequence.setdefault(freqs, relpath)
        frequency_sets.append(set(freqs))

        for freq in freqs:
            frequency_counter[freq] += 1

    union_freqs = sorted(set.union(*frequency_sets), reverse=True) if frequency_sets else []
    intersection_freqs = sorted(set.intersection(*frequency_sets), reverse=True) if frequency_sets else []

    candidate_tables = {}
    for threshold in DEFAULT_THRESHOLDS:
        freqs = [
            freq
            for freq in sorted(frequency_counter, reverse=True)
            if frequency_counter[freq] / max(total_samples, 1) >= threshold
        ]
        candidate_tables[f"support_ge_{threshold:.2f}"] = freqs

    recommended_key = "support_ge_0.50"
    recommended_table = candidate_tables.get(recommended_key, [])

    return {
        "root": os.path.abspath(root),
        "num_samples": total_samples,
        "count_distribution": {str(k): int(v) for k, v in sorted(count_counter.items())},
        "top_level_counts": dict(sorted(top_level_counter.items())),
        "top_sequences": [
            {
                "count": int(count),
                "num_freqs": len(freqs),
                "freqs_hz": list(freqs),
                "example_relpath": example_by_sequence[freqs],
            }
            for freqs, count in sequence_counter.most_common(20)
        ],
        "frequency_support": [
            {
                "freq_hz": float(freq),
                "count": int(frequency_counter[freq]),
                "fraction": float(frequency_counter[freq] / max(total_samples, 1)),
            }
            for freq in sorted(frequency_counter, reverse=True)
        ],
        "union_freqs_hz": union_freqs,
        "intersection_freqs_hz": intersection_freqs,
        "candidate_tables_hz": candidate_tables,
        "recommended_table_key": recommended_key,
        "recommended_table_hz": recommended_table,
    }


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(
        os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward", "train-river")
    )
    parser = argparse.ArgumentParser(description="Summarize MT frequency-axis coverage for GMES-3D forward data.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = _collect_mt_frequency_stats(args.root)
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[+] wrote MT frequency summary to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
