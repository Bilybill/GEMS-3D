from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES, build_forward_index


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))
    parser = argparse.ArgumentParser(description="Audit the GMES-3D benchmark dataset.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument(
        "--include-top-levels",
        type=str,
        nargs="*",
        default=list(DEFAULT_PARTITION_ALIASES),
    )
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    records = build_forward_index(
        args.root,
        include_top_levels=args.include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
    )

    partition_counts = Counter(record["partition"] for record in records)
    anomaly_counts = Counter(record["anomaly_type"] for record in records)
    modality_status = defaultdict(Counter)
    for record in records:
        modality_status["gravity"][record["gravity_status"]] += 1
        modality_status["magnetic"][record["magnetic_status"]] += 1
        modality_status["mt"][record["mt_status"]] += 1
        modality_status["seismic"][record["seismic_status"]] += 1

    summary = {
        "num_records": len(records),
        "partition_counts": dict(sorted(partition_counts.items())),
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "modality_status": {name: dict(sorted(counter.items())) for name, counter in modality_status.items()},
    }

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
