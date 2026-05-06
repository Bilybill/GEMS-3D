from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES
from experiments.datasets.gmes_reconstruction_dataset import GMESPropertyReconstructionDataset
from experiments.models.reconstruction import build_property_reconstruction_model
from experiments.train_forward_surrogate import _format_metric_block, _resolve_device
from experiments.train_inverse_regression import _default_root, build_inverse_splits, run_inverse_epoch
from experiments.utils.dataloading import build_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GMES-3D inverse property-reconstruction model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default=_default_root())
    parser.add_argument("--include-top-levels", type=str, nargs="*", default=list(DEFAULT_PARTITION_ALIASES))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="heldout", choices=["val", "heldout"])
    parser.add_argument("--development-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--heldout-source-prefixes", type=str, nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    modalities = tuple(checkpoint["modalities"])
    target_properties = tuple(checkpoint["target_properties"])
    target_shape = tuple(int(x) for x in checkpoint["target_shape"])
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
    _, val_records, heldout_records = build_inverse_splits(
        root=args.root,
        include_top_levels=args.include_top_levels,
        modalities=modalities,
        seed=args.seed,
        val_fraction=args.val_fraction,
        development_source_prefixes=development_source_prefixes,
        heldout_source_prefixes=heldout_source_prefixes,
    )
    records = val_records if args.split == "val" else heldout_records
    dataset = GMESPropertyReconstructionDataset(records, modalities, target_properties, target_shape)
    loader = build_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_property_reconstruction_model(
        model_name=ckpt_args["model"],
        modalities=modalities,
        target_properties=target_properties,
        target_shape=target_shape,
        embedding_dim=ckpt_args["embedding_dim"],
        fusion_hidden_dim=ckpt_args["fusion_hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        metrics = run_inverse_epoch(model, loader, optimizer=None, device=device, training=False)
    payload = {
        "task": "inverse_regression",
        "model": ckpt_args["model"],
        "modalities": list(modalities),
        "target_properties": list(target_properties),
        "target_shape": list(target_shape),
        "split": args.split,
        "num_samples": len(dataset),
        "device": str(device),
        "checkpoint": os.path.abspath(args.checkpoint),
        "summary": _format_metric_block(metrics),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
