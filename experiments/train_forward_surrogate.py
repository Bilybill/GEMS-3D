from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import (
    DEFAULT_PARTITION_ALIASES,
    build_forward_index,
)
from experiments.datasets.gmes_forward_dataset import (
    FORWARD_TASK_SPECS,
    GMESForwardDataset,
    infer_output_spec_from_sample,
)
from experiments.models.registry import AVAILABLE_FORWARD_MODELS, build_forward_model
from experiments.utils.metrics_forward import (
    AverageMeter,
    build_forward_metric_meters,
    finalize_forward_metric_meters,
    summarize_forward_metrics,
    update_forward_metric_meters,
)
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_tree_to_device(tree, device: torch.device):
    if isinstance(tree, dict):
        return {key: _move_tree_to_device(value, device) for key, value in tree.items()}
    return tree.to(device=device, dtype=torch.float32, non_blocking=True)


def _move_batch(batch: dict, device: torch.device):
    inputs = _move_tree_to_device(batch["inputs"], device)
    targets = _move_tree_to_device(batch["targets"], device)
    raw_targets = _move_tree_to_device(batch["raw_targets"], device)
    target_masks = _move_tree_to_device(batch["target_masks"], device)
    return inputs, targets, raw_targets, target_masks


def _compute_forward_loss(predictions, targets, target_masks=None):
    if isinstance(predictions, dict):
        losses = [
            _compute_forward_loss(
                predictions[name],
                targets[name],
                target_masks.get(name) if isinstance(target_masks, dict) else None,
            )
            for name in sorted(predictions)
        ]
        return sum(losses) / max(len(losses), 1)
    if target_masks is not None:
        mask = target_masks.to(dtype=predictions.dtype)
        squared_error = (predictions - targets).pow(2) * mask
        normalizer = mask.sum().clamp_min(1.0)
        return squared_error.sum() / normalizer
    return torch.nn.functional.mse_loss(predictions, targets)


def _batch_size_from_inputs(inputs) -> int:
    if isinstance(inputs, dict):
        first = next(iter(inputs.values()))
        return int(first.shape[0])
    return int(inputs.shape[0])


def _shape_tree(value):
    if isinstance(value, dict):
        return {key: _shape_tree(child) for key, child in value.items()}
    return tuple(value.shape)


def _infer_in_channels(inputs) -> int:
    if isinstance(inputs, dict):
        volume = inputs["volume"]
        if volume.ndim == 3:
            return 1
        if volume.ndim == 4:
            return int(volume.shape[0])
        if volume.ndim == 5:
            return int(volume.shape[1])
        raise ValueError(f"Cannot infer volume channels from shape {tuple(volume.shape)}.")
    return int(inputs.shape[0])


def _infer_condition_dim(inputs) -> int | None:
    if isinstance(inputs, dict) and "condition" in inputs:
        return int(inputs["condition"].shape[-1])
    return None


def _rescale_prediction_tree(predictions, target_scales):
    if target_scales is None:
        return predictions
    if isinstance(predictions, dict):
        return {
            key: _rescale_prediction_tree(
                predictions[key],
                target_scales.get(key) if isinstance(target_scales, dict) else None,
            )
            for key in predictions
        }
    if isinstance(target_scales, dict):
        target_scales = target_scales.get("default")
    scale = float(target_scales) if target_scales not in (None, 0.0) else 1.0
    return predictions * scale


def _compute_bundle_array_global_scale(records: list[dict], bundle_key: str) -> float:
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    for record in records:
        with np.load(record["bundle_path"], allow_pickle=True) as bundle:
            values = np.asarray(bundle[bundle_key], dtype=np.float64)
        total_count += values.size
        total_sum += float(values.sum())
        total_sum_sq += float(np.square(values).sum())
    if total_count == 0:
        return 1.0
    mean = total_sum / total_count
    variance = max(total_sum_sq / total_count - mean * mean, 0.0)
    scale = float(np.sqrt(variance))
    return scale if scale > 1.0e-6 else 1.0


def _build_target_scales(train_records: list[dict], task_name: str):
    if task_name == "rho_to_gravity":
        return {"default": _compute_bundle_array_global_scale(train_records, "gravity_data")}
    if task_name == "chi_to_magnetic":
        return {"default": _compute_bundle_array_global_scale(train_records, "magnetic_data")}
    if task_name == "joint_multiphysics":
        return {
            "gravity": _compute_bundle_array_global_scale(train_records, "gravity_data"),
            "magnetic": _compute_bundle_array_global_scale(train_records, "magnetic_data"),
        }
    return None


def _format_metric_block(metrics: dict) -> dict:
    ordered = {}
    for key in ("mae", "pearson_r", "nonzero_relative_l2", "relative_l2", "inference_time_ms", "loss"):
        if key in metrics:
            ordered[key] = metrics[key]
    for key, value in metrics.items():
        if key == "per_target":
            continue
        if key not in ordered:
            ordered[key] = value
    return ordered


def _filter_records_for_task(records: list[dict], task_name: str) -> list[dict]:
    spec = FORWARD_TASK_SPECS[task_name]
    if spec.required_status_key == "has_all_modalities":
        return [record for record in records if bool(record.get("has_all_modalities"))]
    return [record for record in records if record.get(spec.required_status_key) == "ok"]


def run_epoch(model, loader, optimizer, device, training: bool, target_scales=None):
    model.train(training)
    loss_meter = AverageMeter()
    metrics_meter = None

    for batch in loader:
        inputs, targets, raw_targets, target_masks = _move_batch(batch, device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            predictions = model(inputs)
            loss = _compute_forward_loss(predictions, targets, target_masks=target_masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = _batch_size_from_inputs(inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(batch_size, 1)
        raw_predictions = _rescale_prediction_tree(predictions.detach(), target_scales)
        batch_metrics = summarize_forward_metrics(raw_predictions, raw_targets, mask=target_masks)

        if metrics_meter is None:
            target_names = sorted(batch_metrics.get("per_target", {})) if isinstance(batch_metrics, dict) else None
            metrics_meter = build_forward_metric_meters(target_names=target_names)

        loss_meter.update(loss.item(), batch_size)
        update_forward_metric_meters(metrics_meter, batch_metrics, batch_size)
        metrics_meter["inference_time_ms"].update(elapsed_ms, batch_size)

    if metrics_meter is None:
        metrics_meter = build_forward_metric_meters()
    result = {"loss": loss_meter.average}
    result.update(finalize_forward_metric_meters(metrics_meter))
    return result


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))
    default_output_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "ExperimentRuns", "forward_surrogates"))

    parser = argparse.ArgumentParser(description="Train forward-operator surrogates for GMES-3D.")
    parser.add_argument("--root", type=str, default=default_root, help="Root directory containing forward_bundle.npz files.")
    parser.add_argument(
        "--include-top-levels",
        type=str,
        nargs="*",
        default=list(DEFAULT_PARTITION_ALIASES),
        help="Raw top-level directories to include in the forward benchmark index.",
    )
    parser.add_argument("--task", type=str, required=True, choices=sorted(FORWARD_TASK_SPECS))
    parser.add_argument("--model", type=str, required=True, choices=list(AVAILABLE_FORWARD_MODELS))
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string, e.g. auto, cpu, cuda, cuda:0, cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
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
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional training-set cap for smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional validation-set cap for smoke tests.")
    parser.add_argument("--max-heldout-samples", type=int, default=0, help="Optional held-out-set cap for smoke tests.")
    parser.add_argument("--output-root", type=str, default=default_output_root)
    return parser.parse_args()


def main():
    args = parse_args()
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    records = build_forward_index(
        args.root,
        include_top_levels=args.include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
        development_source_prefixes=args.development_source_prefixes,
        heldout_source_prefixes=args.heldout_source_prefixes,
    )
    records = _filter_records_for_task(records, args.task)
    train_records, val_records, heldout_records = split_records_by_background(
        records,
        validation_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_records = val_records[: args.max_val_samples]
    if args.max_heldout_samples > 0:
        heldout_records = heldout_records[: args.max_heldout_samples]

    if not train_records:
        raise RuntimeError("No training records were selected.")
    if not val_records:
        raise RuntimeError("No validation records were selected.")
    if not heldout_records:
        raise RuntimeError("No held-out records were selected.")

    target_scales = _build_target_scales(train_records, args.task)
    train_dataset = GMESForwardDataset(train_records, task_name=args.task, target_scales=target_scales)
    val_dataset = GMESForwardDataset(val_records, task_name=args.task, target_scales=target_scales)
    heldout_dataset = GMESForwardDataset(heldout_records, task_name=args.task, target_scales=target_scales)

    sample = train_dataset[0]
    in_channels = _infer_in_channels(sample["inputs"])
    condition_dim = _infer_condition_dim(sample["inputs"])
    output_specs = infer_output_spec_from_sample(sample["targets"])
    if "default" in output_specs:
        model = build_forward_model(
            model_name=args.model,
            in_channels=in_channels,
            out_channels=int(output_specs["default"]["out_channels"]),
            output_shape=tuple(output_specs["default"]["output_shape"]),
            condition_dim=condition_dim,
        ).to(device)
    else:
        model = build_forward_model(
            model_name=args.model,
            in_channels=in_channels,
            output_specs=output_specs,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    heldout_loader = DataLoader(heldout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = f"{args.task}_{args.model}"
    output_dir = os.path.abspath(os.path.join(args.output_root, run_name))
    os.makedirs(output_dir, exist_ok=True)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_epoch_record = None
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")

    print("====== GMES-3D Forward Surrogate Training ======")
    print(f"Task      : {args.task}")
    print(f"Model     : {args.model}")
    print(f"Device    : {device}")
    print(f"Train/Val : {len(train_dataset)} / {len(val_dataset)}")
    print(f"Held-out  : {len(heldout_dataset)}")
    print(f"Input     : {json.dumps(_shape_tree(sample['inputs']), ensure_ascii=True)}")
    print(f"Targets   : {json.dumps(output_specs, ensure_ascii=True)}")
    if target_scales is not None:
        print(f"TargetSc. : {json.dumps(target_scales, ensure_ascii=True)}")
    print(f"Output dir: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, training=True, target_scales=target_scales)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, training=False, target_scales=target_scales)
            heldout_metrics = run_epoch(model, heldout_loader, optimizer=None, device=device, training=False, target_scales=target_scales)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "heldout": heldout_metrics,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_rl2={val_metrics['relative_l2']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_r={val_metrics['pearson_r']:.6f} "
            f"val_nonzero_rl2={val_metrics['nonzero_relative_l2']:.6f} "
            f"heldout_rl2={heldout_metrics['relative_l2']:.6f} "
            f"heldout_mae={heldout_metrics['mae']:.6f} "
            f"heldout_r={heldout_metrics['pearson_r']:.6f} "
            f"heldout_nonzero_rl2={heldout_metrics['nonzero_relative_l2']:.6f}"
        )

        state = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "history": history,
            "sample_shapes": {
                "inputs": _shape_tree(sample["inputs"]),
                "targets": output_specs,
            },
            "target_scales": target_scales,
        }
        torch.save(state, last_ckpt_path)

        if val_metrics["relative_l2"] < best_val:
            best_val = val_metrics["relative_l2"]
            best_epoch = epoch
            best_epoch_record = epoch_record
            torch.save(state, best_ckpt_path)

    final_epoch_record = history[-1]
    results = {
        "task": args.task,
        "model": args.model,
        "device": str(device),
        "splits": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "heldout_size": len(heldout_dataset),
        },
        "sample_shapes": {
            "inputs": _shape_tree(sample["inputs"]),
            "targets": output_specs,
        },
        "target_scales": target_scales,
        "paths": {
            "output_dir": output_dir,
            "best_checkpoint": best_ckpt_path,
            "last_checkpoint": last_ckpt_path,
        },
        "selection": {
            "metric": "val.relative_l2",
            "best_epoch": best_epoch,
        },
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
