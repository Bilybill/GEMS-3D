from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES, build_forward_index
from experiments.datasets.gmes_inverse_dataset import MODALITY_STATUS_KEYS, normalize_modalities
from experiments.datasets.gmes_reconstruction_dataset import GMESModalityCompletionDataset
from experiments.models.reconstruction import AVAILABLE_RECONSTRUCTION_MODELS, build_modality_completion_model
from experiments.train_forward_surrogate import _format_metric_block, _resolve_device
from experiments.train_inverse_regression import _default_root
from experiments.utils.dataloading import build_data_loader
from experiments.utils.metrics_forward import (
    AverageMeter,
    build_forward_metric_meters,
    finalize_forward_metric_meters,
    summarize_forward_metrics,
    update_forward_metric_meters,
)
from experiments.utils.splits import split_records_by_background


def _default_output_root() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.basename(project_root) == "GEMS-3D":
        return os.path.join(project_root, "outputs", "ExperimentRuns", "modality_completion")
    return os.path.join(project_root, "DATAFOLDER", "ExperimentRuns", "modality_completion")


def _filter_records(records: list[dict], input_modalities: tuple[str, ...], target_modality: str) -> list[dict]:
    required = tuple(dict.fromkeys((*input_modalities, target_modality)))
    return [
        record
        for record in records
        if all(record.get(MODALITY_STATUS_KEYS[modality]) == "ok" for modality in required)
    ]


def build_completion_splits(
    root: str,
    include_top_levels: list[str],
    input_modalities: tuple[str, ...],
    target_modality: str,
    seed: int,
    val_fraction: float,
    development_source_prefixes: list[str] | None = None,
    heldout_source_prefixes: list[str] | None = None,
):
    records = build_forward_index(
        root,
        include_top_levels=include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
        development_source_prefixes=development_source_prefixes,
        heldout_source_prefixes=heldout_source_prefixes,
    )
    records = _filter_records(records, input_modalities, target_modality)
    return split_records_by_background(records, validation_fraction=val_fraction, seed=seed)


def _move_batch(batch: dict, device: torch.device):
    inputs = {
        name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        for name, tensor in batch["inputs"].items()
    }
    targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
    raw_targets = batch["raw_targets"].to(device=device, dtype=torch.float32, non_blocking=True)
    masks = batch["target_masks"].to(device=device, dtype=torch.float32, non_blocking=True)
    return inputs, targets, raw_targets, masks


def run_completion_epoch(model, loader, optimizer, device, training: bool):
    model.train(training)
    loss_meter = AverageMeter()
    metric_meters = build_forward_metric_meters()
    for batch in loader:
        inputs, targets, raw_targets, masks = _move_batch(batch, device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            predictions = model(inputs)
            loss = ((predictions - targets).pow(2) * masks).sum() / masks.sum().clamp_min(1.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = int(targets.shape[0])
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(batch_size, 1)
        batch_metrics = summarize_forward_metrics(predictions.detach(), raw_targets, mask=masks)
        loss_meter.update(loss.item(), batch_size)
        update_forward_metric_meters(metric_meters, batch_metrics, batch_size)
        metric_meters["inference_time_ms"].update(elapsed_ms, batch_size)
    result = {"loss": loss_meter.average}
    result.update(finalize_forward_metric_meters(metric_meters))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train GMES-3D modality-completion baselines.")
    parser.add_argument("--root", type=str, default=_default_root())
    parser.add_argument("--include-top-levels", type=str, nargs="*", default=list(DEFAULT_PARTITION_ALIASES))
    parser.add_argument("--input-modalities", type=str, nargs="+", required=True)
    parser.add_argument("--target-modality", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=AVAILABLE_RECONSTRUCTION_MODELS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-heldout-samples", type=int, default=0)
    parser.add_argument("--development-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--heldout-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--output-root", type=str, default=_default_output_root())
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device)
    input_modalities = normalize_modalities(args.input_modalities)
    target_modality = normalize_modalities((args.target_modality,))[0]

    train_records, val_records, heldout_records = build_completion_splits(
        root=args.root,
        include_top_levels=args.include_top_levels,
        input_modalities=input_modalities,
        target_modality=target_modality,
        seed=args.seed,
        val_fraction=args.val_fraction,
        development_source_prefixes=args.development_source_prefixes,
        heldout_source_prefixes=args.heldout_source_prefixes,
    )
    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_records = val_records[: args.max_val_samples]
    if args.max_heldout_samples > 0:
        heldout_records = heldout_records[: args.max_heldout_samples]
    if not train_records or not val_records or not heldout_records:
        raise RuntimeError("Train/val/held-out splits must all be non-empty.")

    train_dataset = GMESModalityCompletionDataset(train_records, input_modalities, target_modality)
    val_dataset = GMESModalityCompletionDataset(val_records, input_modalities, target_modality)
    heldout_dataset = GMESModalityCompletionDataset(heldout_records, input_modalities, target_modality)
    sample = train_dataset[0]
    out_channels = int(sample["targets"].shape[0])
    output_shape = tuple(int(x) for x in sample["targets"].shape[-2:])

    model = build_modality_completion_model(
        model_name=args.model,
        input_modalities=input_modalities,
        target_modality=target_modality,
        out_channels=out_channels,
        output_shape=output_shape,
        embedding_dim=args.embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_data_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        prefetch_factor=args.prefetch_factor,
    )
    heldout_loader = build_data_loader(
        heldout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        prefetch_factor=args.prefetch_factor,
    )

    run_name = f"{'_'.join(input_modalities)}_to_{target_modality}_{args.model}"
    output_dir = os.path.abspath(os.path.join(args.output_root, run_name))
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_epoch_record = None

    print("====== GMES-3D Modality Completion Training ======")
    print(f"Inputs    : {input_modalities}")
    print(f"Target    : {target_modality} ({out_channels}, {output_shape})")
    print(f"Model     : {args.model}")
    print(f"Train/Val : {len(train_dataset)} / {len(val_dataset)}")
    print(f"Held-out  : {len(heldout_dataset)}")
    print(f"Output dir: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_completion_epoch(model, train_loader, optimizer, device, training=True)
        with torch.no_grad():
            val_metrics = run_completion_epoch(model, val_loader, optimizer=None, device=device, training=False)
            heldout_metrics = run_completion_epoch(model, heldout_loader, optimizer=None, device=device, training=False)
        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "heldout": heldout_metrics}
        history.append(epoch_record)
        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['loss']:.6f} "
            f"val_rl2={val_metrics['relative_l2']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"val_r={val_metrics['pearson_r']:.6f} heldout_r={heldout_metrics['pearson_r']:.6f}"
        )
        state = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "input_modalities": list(input_modalities),
            "target_modality": target_modality,
            "out_channels": out_channels,
            "output_shape": list(output_shape),
            "history": history,
        }
        torch.save(state, last_ckpt_path)
        if val_metrics["relative_l2"] < best_val:
            best_val = val_metrics["relative_l2"]
            best_epoch = epoch
            best_epoch_record = epoch_record
            torch.save(state, best_ckpt_path)

    final_epoch_record = history[-1]
    results = {
        "task": "modality_completion",
        "model": args.model,
        "input_modalities": list(input_modalities),
        "target_modality": target_modality,
        "device": str(device),
        "splits": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "heldout_size": len(heldout_dataset),
        },
        "paths": {"output_dir": output_dir, "best_checkpoint": best_ckpt_path, "last_checkpoint": last_ckpt_path},
        "selection": {"metric": "val.relative_l2", "best_epoch": best_epoch},
        "summary": {
            "best_validation": _format_metric_block(best_epoch_record["val"]) if best_epoch_record is not None else {},
            "heldout_at_best_validation": _format_metric_block(best_epoch_record["heldout"]) if best_epoch_record is not None else {},
            "final_validation": _format_metric_block(final_epoch_record["val"]),
            "final_heldout": _format_metric_block(final_epoch_record["heldout"]),
        },
        "history": history,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"[+] Saved metrics to {metrics_path}")
    print(f"[+] Saved best checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
