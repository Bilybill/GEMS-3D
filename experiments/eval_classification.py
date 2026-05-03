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

from experiments.datasets.gmes_inverse_dataset import GMESInverseDataset
from experiments.models.fusion import LateFusionClassifier
from experiments.train_classification import (
    _resolve_device,
    build_classification_splits,
    run_classification_epoch,
)


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))

    parser = argparse.ArgumentParser(description="Evaluate a trained GMES-3D anomaly-family classifier.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string, e.g. auto, cpu, cuda, cuda:0, cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="heldout", choices=["val", "heldout"])
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    modalities = tuple(checkpoint["modalities"])
    anomaly_to_index = checkpoint["anomaly_to_index"]
    device = _resolve_device(args.device)

    _, val_records, heldout_records = build_classification_splits(
        root=args.root,
        include_top_levels=ckpt_args["include_top_levels"],
        modalities=modalities,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    records = val_records if args.split == "val" else heldout_records
    dataset = GMESInverseDataset(records, modalities=modalities, anomaly_to_index=anomaly_to_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LateFusionClassifier(
        modalities=modalities,
        num_classes=len(anomaly_to_index),
        embedding_dim=ckpt_args["embedding_dim"],
        fusion_hidden_dim=ckpt_args["fusion_hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        metrics = run_classification_epoch(model, loader, optimizer=None, device=device, training=False)
    print(json.dumps({"split": args.split, "metrics": metrics}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
