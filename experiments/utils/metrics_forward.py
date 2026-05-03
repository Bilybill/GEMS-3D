from __future__ import annotations

import math


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def relative_l2(prediction, target, eps: float = 1.0e-8):
    numerator = (prediction - target).pow(2).sum(dim=tuple(range(1, prediction.ndim)))
    denominator = target.pow(2).sum(dim=tuple(range(1, target.ndim))).clamp_min(eps)
    return (numerator / denominator).sqrt()


def masked_relative_l2(prediction, target, mask, eps: float = 1.0e-8):
    mask = mask.to(dtype=prediction.dtype)
    reduce_dims = tuple(range(1, prediction.ndim))
    numerator = ((prediction - target).pow(2) * mask).sum(dim=reduce_dims)
    denominator = (target.pow(2) * mask).sum(dim=reduce_dims).clamp_min(eps)
    valid = mask.sum(dim=reduce_dims) > 0
    values = (numerator / denominator).sqrt()
    return torch.where(valid, values, torch.full_like(values, float("nan")))


def pearson_r(prediction, target, eps: float = 1.0e-8):
    pred = prediction.flatten(start_dim=1)
    tgt = target.flatten(start_dim=1)
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
    numerator = (pred_centered * tgt_centered).sum(dim=1)
    denominator = pred_centered.norm(dim=1).clamp_min(eps) * tgt_centered.norm(dim=1).clamp_min(eps)
    return numerator / denominator


def masked_pearson_r(prediction, target, mask, eps: float = 1.0e-8):
    pred = prediction.flatten(start_dim=1)
    tgt = target.flatten(start_dim=1)
    weights = mask.flatten(start_dim=1).to(dtype=prediction.dtype)
    valid = weights.sum(dim=1)
    safe_valid = valid.clamp_min(1.0)

    pred_mean = (pred * weights).sum(dim=1, keepdim=True) / safe_valid.unsqueeze(1)
    tgt_mean = (tgt * weights).sum(dim=1, keepdim=True) / safe_valid.unsqueeze(1)

    pred_centered = (pred - pred_mean) * weights
    tgt_centered = (tgt - tgt_mean) * weights
    numerator = (pred_centered * tgt_centered).sum(dim=1)
    denominator = (
        pred_centered.pow(2).sum(dim=1).sqrt().clamp_min(eps)
        * tgt_centered.pow(2).sum(dim=1).sqrt().clamp_min(eps)
    )
    values = numerator / denominator
    return torch.where(valid > 1.0, values, torch.full_like(values, float("nan")))


def mean_absolute_error(prediction, target):
    return (prediction - target).abs().mean(dim=tuple(range(1, prediction.ndim)))


def masked_mean_absolute_error(prediction, target, mask):
    mask = mask.to(dtype=prediction.dtype)
    reduce_dims = tuple(range(1, prediction.ndim))
    numerator = ((prediction - target).abs() * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims).clamp_min(1.0)
    valid = mask.sum(dim=reduce_dims) > 0
    values = numerator / denominator
    return torch.where(valid, values, torch.full_like(values, float("nan")))


def nonzero_target_relative_l2(prediction, target, eps: float = 1.0e-8, target_norm_eps: float = 1.0e-8):
    rl2 = relative_l2(prediction, target, eps=eps)
    target_norm = target.pow(2).sum(dim=tuple(range(1, target.ndim))).sqrt()
    mask = target_norm > target_norm_eps
    if mask.any():
        return rl2[mask].mean()
    return rl2.new_tensor(float("nan"))


def masked_nonzero_target_relative_l2(
    prediction,
    target,
    mask,
    eps: float = 1.0e-8,
    target_norm_eps: float = 1.0e-8,
):
    rl2 = masked_relative_l2(prediction, target, mask, eps=eps)
    target_norm = (target.pow(2) * mask.to(dtype=target.dtype)).sum(dim=tuple(range(1, target.ndim))).sqrt()
    valid = target_norm > target_norm_eps
    if valid.any():
        return rl2[valid].mean()
    return rl2.new_tensor(float("nan"))


def _nanmean(values):
    valid = ~torch.isnan(values)
    if valid.any():
        return float(values[valid].mean().item())
    return float("nan")


def summarize_forward_metrics(prediction, target, mask=None):
    if isinstance(prediction, dict):
        per_target = {
            name: summarize_forward_metrics(
                prediction[name],
                target[name],
                mask.get(name) if isinstance(mask, dict) else None,
            )
            for name in sorted(prediction)
        }
        keys = ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2")
        aggregate = {
            key: float(
                sum(metrics[key] for metrics in per_target.values() if not math.isnan(metrics[key]))
                / max(sum(0 if math.isnan(metrics[key]) else 1 for metrics in per_target.values()), 1)
            )
            for key in keys
        }
        return {
            **aggregate,
            "per_target": per_target,
        }

    if mask is None:
        rl2 = relative_l2(prediction, target)
        r = pearson_r(prediction, target)
        mae = mean_absolute_error(prediction, target)
        nonzero_rl2 = nonzero_target_relative_l2(prediction, target)
    else:
        rl2 = masked_relative_l2(prediction, target, mask)
        r = masked_pearson_r(prediction, target, mask)
        mae = masked_mean_absolute_error(prediction, target, mask)
        nonzero_rl2 = masked_nonzero_target_relative_l2(prediction, target, mask)
    return {
        "relative_l2": _nanmean(rl2),
        "pearson_r": _nanmean(r),
        "mae": _nanmean(mae),
        "nonzero_relative_l2": float(nonzero_rl2.item()),
    }


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return math.nan
        return self.total / self.count


def build_forward_metric_meters(target_names: list[str] | None = None) -> dict:
    base = {
        "relative_l2": AverageMeter(),
        "pearson_r": AverageMeter(),
        "mae": AverageMeter(),
        "nonzero_relative_l2": AverageMeter(),
        "inference_time_ms": AverageMeter(),
    }
    if not target_names:
        return base
    base["per_target"] = {
        name: {
            "relative_l2": AverageMeter(),
            "pearson_r": AverageMeter(),
            "mae": AverageMeter(),
            "nonzero_relative_l2": AverageMeter(),
        }
        for name in target_names
    }
    return base


def update_forward_metric_meters(metric_meters: dict, batch_metrics: dict, batch_size: int):
    for key in ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2"):
        value = batch_metrics[key]
        if not math.isnan(value):
            metric_meters[key].update(value, batch_size)
    if "per_target" not in metric_meters or "per_target" not in batch_metrics:
        return
    for target_name, per_target_metrics in batch_metrics["per_target"].items():
        for key in ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2"):
            value = per_target_metrics[key]
            if not math.isnan(value):
                metric_meters["per_target"][target_name][key].update(value, batch_size)


def finalize_forward_metric_meters(metric_meters: dict) -> dict[str, float]:
    result = {
        "relative_l2": metric_meters["relative_l2"].average,
        "pearson_r": metric_meters["pearson_r"].average,
        "mae": metric_meters["mae"].average,
        "nonzero_relative_l2": metric_meters["nonzero_relative_l2"].average,
        "inference_time_ms": metric_meters["inference_time_ms"].average,
    }
    if "per_target" in metric_meters:
        result["per_target"] = {
            target_name: {
                key: meters[key].average
                for key in ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2")
            }
            for target_name, meters in metric_meters["per_target"].items()
        }
    return result
