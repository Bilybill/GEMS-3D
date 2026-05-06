from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES
from experiments.datasets.gmes_reconstruction_dataset import GMESPropertyReconstructionDataset
from experiments.models.reconstruction import build_property_reconstruction_model
from experiments.train_forward_surrogate import _format_metric_block, _resolve_device
from experiments.train_inverse_regression import _default_root, build_inverse_splits
from experiments.utils.dataloading import build_data_loader
from experiments.utils.metrics_forward import (
    AverageMeter,
    build_forward_metric_meters,
    finalize_forward_metric_meters,
    summarize_forward_metrics,
    update_forward_metric_meters,
)


def _move_batch(batch: dict, device: torch.device):
    inputs = {
        name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        for name, tensor in batch["inputs"].items()
    }
    targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
    raw_targets = batch["raw_targets"].to(device=device, dtype=torch.float32, non_blocking=True)
    masks = batch["target_masks"].to(device=device, dtype=torch.float32, non_blocking=True)
    return inputs, targets, raw_targets, masks


def _reduced_sampling(tensor: torch.Tensor, stride: int):
    mask = torch.zeros_like(tensor)
    slices = [slice(None)] * tensor.ndim
    slices[-1] = slice(None, None, max(int(stride), 1))
    slices[-2] = slice(None, None, max(int(stride), 1))
    mask[tuple(slices)] = 1.0
    return tensor * mask


def _add_noise(tensor: torch.Tensor, noise_std: float):
    reduce_dims = tuple(range(1, tensor.ndim))
    scale = tensor.flatten(start_dim=1).std(dim=1, keepdim=True).reshape((tensor.shape[0],) + (1,) * (tensor.ndim - 1))
    return tensor + torch.randn_like(tensor) * scale.clamp_min(1.0e-6) * float(noise_std)


def corrupt_inputs(inputs: dict[str, torch.Tensor], corruption: str, sampling_stride: int, noise_std: float, amplitude_scale: float):
    corrupted = {}
    for name, tensor in inputs.items():
        if corruption == "reduced_sampling":
            corrupted[name] = _reduced_sampling(tensor, sampling_stride)
        elif corruption == "additive_noise":
            corrupted[name] = _add_noise(tensor, noise_std)
        elif corruption == "amplitude_perturbation":
            corrupted[name] = tensor * float(amplitude_scale)
        elif corruption == "clean":
            corrupted[name] = tensor
        else:
            raise ValueError(f"Unsupported corruption '{corruption}'.")
    return corrupted


def evaluate_with_corruption(model, loader, device, corruption: str, sampling_stride: int, noise_std: float, amplitude_scale: float):
    model.eval()
    loss_meter = AverageMeter()
    metric_meters = build_forward_metric_meters()
    for batch in loader:
        inputs, targets, raw_targets, masks = _move_batch(batch, device)
        inputs = corrupt_inputs(inputs, corruption, sampling_stride, noise_std, amplitude_scale)
        start = time.perf_counter()
        predictions = model(inputs)
        loss = ((predictions - targets).pow(2) * masks).sum() / masks.sum().clamp_min(1.0)
        batch_size = int(targets.shape[0])
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(batch_size, 1)
        batch_metrics = summarize_forward_metrics(predictions.detach(), raw_targets, mask=masks)
        loss_meter.update(loss.item(), batch_size)
        update_forward_metric_meters(metric_meters, batch_metrics, batch_size)
        metric_meters["inference_time_ms"].update(elapsed_ms, batch_size)
    result = {"loss": loss_meter.average}
    result.update(finalize_forward_metric_meters(metric_meters))
    return result


def _retention(clean: dict, degraded: dict) -> dict:
    clean_r = float(clean.get("pearson_r", float("nan")))
    degraded_r = float(degraded.get("pearson_r", float("nan")))
    clean_mae = float(clean.get("mae", float("nan")))
    degraded_mae = float(degraded.get("mae", float("nan")))
    return {
        "pearson_r": degraded_r / clean_r if clean_r != 0.0 and not math.isnan(clean_r) else float("nan"),
        "mae": clean_mae / degraded_mae if degraded_mae != 0.0 and not math.isnan(degraded_mae) else float("nan"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate degraded-observation inverse regression for GMES-3D.")
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
    parser.add_argument("--corruptions", type=str, nargs="*", default=["reduced_sampling", "additive_noise", "amplitude_perturbation"])
    parser.add_argument("--sampling-stride", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--amplitude-scale", type=float, default=0.75)
    parser.add_argument("--development-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--heldout-source-prefixes", type=str, nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
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
        clean = evaluate_with_corruption(
            model, loader, device, "clean", args.sampling_stride, args.noise_std, args.amplitude_scale
        )
        degraded = {}
        for corruption in args.corruptions:
            metrics = evaluate_with_corruption(
                model, loader, device, corruption, args.sampling_stride, args.noise_std, args.amplitude_scale
            )
            degraded[corruption] = {
                "summary": _format_metric_block(metrics),
                "retention": _retention(clean, metrics),
            }
    payload = {
        "task": "degraded_inversion",
        "model": ckpt_args["model"],
        "modalities": list(modalities),
        "target_properties": list(target_properties),
        "target_shape": list(target_shape),
        "split": args.split,
        "num_samples": len(dataset),
        "device": str(device),
        "checkpoint": os.path.abspath(args.checkpoint),
        "clean_summary": _format_metric_block(clean),
        "degraded": degraded,
        "corruption_config": {
            "sampling_stride": args.sampling_stride,
            "noise_std": args.noise_std,
            "amplitude_scale": args.amplitude_scale,
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
