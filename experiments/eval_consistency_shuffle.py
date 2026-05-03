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

from experiments.datasets.gmes_inverse_dataset import GMESInverseDataset, normalize_modalities
from experiments.models.fusion import LateFusionClassifier
from experiments.train_classification import _resolve_device, build_classification_splits
from experiments.utils.metrics_classification import summarize_classification_metrics


def _evaluate_with_shuffle(model, loader, device, shuffle_modalities: tuple[str, ...], seed: int):
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)
    logits_list = []
    labels_list = []

    for batch in loader:
        modalities = {
            name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
            for name, tensor in batch["modalities"].items()
        }
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
        batch_size = int(labels.shape[0])
        for modality in shuffle_modalities:
            if modality in modalities and batch_size > 1:
                permutation = torch.randperm(batch_size, generator=generator, device=modalities[modality].device)
                modalities[modality] = modalities[modality][permutation]
        logits = model(modalities)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    probabilities = torch.softmax(torch.cat(logits_list, dim=0), dim=1).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return summarize_classification_metrics(labels, probabilities, num_classes=probabilities.shape[1])


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))

    parser = argparse.ArgumentParser(description="Evaluate modality-shuffled controls for GMES-3D classifiers.")
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
    parser.add_argument("--shuffle-modalities", type=str, nargs="+", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    modalities = tuple(checkpoint["modalities"])
    anomaly_to_index = checkpoint["anomaly_to_index"]
    shuffle_modalities = normalize_modalities(args.shuffle_modalities)
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
        metrics = _evaluate_with_shuffle(model, loader, device, shuffle_modalities=shuffle_modalities, seed=args.seed)
    print(
        json.dumps(
            {"split": args.split, "shuffle_modalities": list(shuffle_modalities), "metrics": metrics},
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
