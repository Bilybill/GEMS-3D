from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import (
    DEFAULT_PARTITION_ALIASES,
    build_forward_index,
)
from experiments.datasets.gmes_forward_dataset import GMESForwardDataset, infer_output_spec_from_sample
from experiments.models.registry import build_forward_model
from experiments.train_forward_surrogate import (
    _filter_records_for_task,
    _format_metric_block,
    _infer_condition_dim,
    _infer_in_channels,
    _resolve_device,
    run_epoch,
)
from experiments.utils.splits import split_records_by_background


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))

    parser = argparse.ArgumentParser(description="Evaluate a trained GMES-3D forward surrogate.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument(
        "--include-top-levels",
        type=str,
        nargs="*",
        default=list(DEFAULT_PARTITION_ALIASES),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string, e.g. auto, cpu, cuda, cuda:0, cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--development-source-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Optional source_relpath prefixes used to restrict the development set.",
    )
    parser.add_argument(
        "--heldout-source-prefixes",
        type=str,
        nargs="*",
        default=None,
        help="Optional source_relpath prefixes used to restrict the held-out set.",
    )
    parser.add_argument("--split", type=str, default="heldout", choices=["val", "heldout"])
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    target_scales = checkpoint.get("target_scales")
    device = _resolve_device(args.device)
    development_source_prefixes = (
        args.development_source_prefixes
        if args.development_source_prefixes is not None
        else ckpt_args.get("development_source_prefixes")
    )
    heldout_source_prefixes = (
        args.heldout_source_prefixes
        if args.heldout_source_prefixes is not None
        else ckpt_args.get("heldout_source_prefixes")
    )

    records = build_forward_index(
        args.root,
        include_top_levels=args.include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
        development_source_prefixes=development_source_prefixes,
        heldout_source_prefixes=heldout_source_prefixes,
    )
    records = _filter_records_for_task(records, ckpt_args["task"])
    _, val_records, heldout_records = split_records_by_background(
        records,
        validation_fraction=args.val_fraction,
        seed=args.seed,
    )
    eval_records = val_records if args.split == "val" else heldout_records
    dataset = GMESForwardDataset(eval_records, task_name=ckpt_args["task"], target_scales=target_scales)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = dataset[0]
    output_specs = infer_output_spec_from_sample(sample["targets"])
    in_channels = _infer_in_channels(sample["inputs"])
    condition_dim = _infer_condition_dim(sample["inputs"])
    if "default" in output_specs:
        model = build_forward_model(
            model_name=ckpt_args["model"],
            in_channels=in_channels,
            out_channels=int(output_specs["default"]["out_channels"]),
            output_shape=tuple(output_specs["default"]["output_shape"]),
            condition_dim=condition_dim,
        ).to(device)
    else:
        model = build_forward_model(
            model_name=ckpt_args["model"],
            in_channels=in_channels,
            output_specs=output_specs,
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        metrics = run_epoch(model, loader, optimizer=None, device=device, training=False, target_scales=target_scales)

    payload = {
        "task": ckpt_args["task"],
        "model": ckpt_args["model"],
        "split": args.split,
        "num_samples": len(dataset),
        "device": str(device),
        "checkpoint": os.path.abspath(args.checkpoint),
        "summary": _format_metric_block(metrics),
    }
    if "per_target" in metrics:
        payload["per_target"] = {
            name: _format_metric_block(per_target_metrics)
            for name, per_target_metrics in metrics["per_target"].items()
        }
    if target_scales is not None:
        payload["target_scales"] = target_scales

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
