from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES, build_forward_index
from experiments.datasets.gmes_inverse_dataset import (
    GMESInverseDataset,
    build_anomaly_label_mapping,
    filter_records_for_modalities,
    normalize_modalities,
)
from experiments.models.fusion import LateFusionClassifier
from experiments.utils.metrics_classification import summarize_classification_metrics
from experiments.utils.splits import split_records_by_background


def _resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid device '{device_arg}'. Use values such as auto, cpu, cuda, or cuda:1.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is out of range. "
                f"Visible CUDA device count: {torch.cuda.device_count()}."
            )
    return device


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_modalities_to_device(modalities: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        for name, tensor in modalities.items()
    }


def build_classification_splits(
    root: str,
    include_top_levels: list[str],
    modalities: tuple[str, ...],
    seed: int,
    val_fraction: float,
):
    records = build_forward_index(
        root,
        include_top_levels=include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
    )
    records = filter_records_for_modalities(records, modalities)
    return split_records_by_background(records, validation_fraction=val_fraction, seed=seed)


def run_classification_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    loss_meter = 0.0
    sample_count = 0
    logits_list = []
    labels_list = []

    for batch in loader:
        modalities = _move_modalities_to_device(batch["modalities"], device)
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(modalities)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(labels.shape[0])
        loss_meter += float(loss.item()) * batch_size
        sample_count += batch_size
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    probabilities = torch.softmax(torch.cat(logits_list, dim=0), dim=1).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    metrics = summarize_classification_metrics(labels, probabilities, num_classes=probabilities.shape[1])
    metrics["loss"] = loss_meter / max(sample_count, 1)
    return metrics


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))
    default_output_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "ExperimentRuns", "classification"))

    parser = argparse.ArgumentParser(description="Train GMES-3D anomaly-family classifiers.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument(
        "--include-top-levels",
        type=str,
        nargs="*",
        default=list(DEFAULT_PARTITION_ALIASES),
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        help="Modalities to use: gravity magnetic mt seismic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string, e.g. auto, cpu, cuda, cuda:0, cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-heldout-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=default_output_root)
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = normalize_modalities(args.modalities)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    train_records, val_records, heldout_records = build_classification_splits(
        root=args.root,
        include_top_levels=args.include_top_levels,
        modalities=modalities,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_records = val_records[: args.max_val_samples]
    if args.max_heldout_samples > 0:
        heldout_records = heldout_records[: args.max_heldout_samples]

    if not train_records or not val_records or not heldout_records:
        raise RuntimeError("Train/val/held-out splits must all be non-empty.")

    anomaly_to_index = build_anomaly_label_mapping(train_records + val_records + heldout_records)
    train_dataset = GMESInverseDataset(train_records, modalities=modalities, anomaly_to_index=anomaly_to_index)
    val_dataset = GMESInverseDataset(val_records, modalities=modalities, anomaly_to_index=anomaly_to_index)
    heldout_dataset = GMESInverseDataset(heldout_records, modalities=modalities, anomaly_to_index=anomaly_to_index)

    model = LateFusionClassifier(
        modalities=modalities,
        num_classes=len(anomaly_to_index),
        embedding_dim=args.embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    heldout_loader = DataLoader(heldout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = "_".join(modalities)
    output_dir = os.path.abspath(os.path.join(args.output_root, run_name))
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")

    history = []
    best_val = -1.0
    print("====== GMES-3D Classification Training ======")
    print(f"Modalities: {modalities}")
    print(f"Device    : {device}")
    print(f"Train/Val : {len(train_dataset)} / {len(val_dataset)}")
    print(f"Held-out  : {len(heldout_dataset)}")
    print(f"Output dir: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_classification_epoch(model, train_loader, optimizer, device, training=True)
        with torch.no_grad():
            val_metrics = run_classification_epoch(model, val_loader, optimizer=None, device=device, training=False)
            heldout_metrics = run_classification_epoch(model, heldout_loader, optimizer=None, device=device, training=False)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "heldout": heldout_metrics,
        }
        history.append(record)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"heldout_macro_f1={heldout_metrics['macro_f1']:.4f}"
        )

        state = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "modalities": list(modalities),
            "anomaly_to_index": anomaly_to_index,
            "history": history,
        }
        torch.save(state, last_ckpt_path)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            torch.save(state, best_ckpt_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modalities": list(modalities),
                "anomaly_to_index": anomaly_to_index,
                "history": history,
                "best_val_macro_f1": best_val,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    print(f"[+] Saved metrics to {metrics_path}")
    print(f"[+] Saved best checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
