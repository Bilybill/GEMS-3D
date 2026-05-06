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


def _nanstd(values):
    valid = ~torch.isnan(values)
    count = int(valid.sum().item())
    if count <= 1:
        return 0.0 if count == 1 else float("nan")
    return float(values[valid].std(unbiased=True).item())


def _stats_from_values(values) -> dict[str, float | int]:
    valid = ~torch.isnan(values)
    count = int(valid.sum().item())
    if count == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    selected = values[valid]
    std = float(selected.std(unbiased=True).item()) if count > 1 else 0.0
    return {
        "mean": float(selected.mean().item()),
        "std": std,
        "count": count,
    }


def _nanmean_rows(values):
    valid = ~torch.isnan(values)
    count = valid.sum(dim=1)
    safe = torch.where(valid, values, torch.zeros_like(values)).sum(dim=1) / count.clamp_min(1)
    return torch.where(count > 0, safe, torch.full_like(safe, float("nan")))


def _metric_values(prediction, target, mask=None) -> dict:
    if mask is None:
        rl2 = relative_l2(prediction, target)
        r = pearson_r(prediction, target)
        mae = mean_absolute_error(prediction, target)
        target_norm = target.pow(2).sum(dim=tuple(range(1, target.ndim))).sqrt()
        nonzero_rl2 = torch.where(target_norm > 1.0e-8, rl2, torch.full_like(rl2, float("nan")))
    else:
        rl2 = masked_relative_l2(prediction, target, mask)
        r = masked_pearson_r(prediction, target, mask)
        mae = masked_mean_absolute_error(prediction, target, mask)
        target_norm = (target.pow(2) * mask.to(dtype=target.dtype)).sum(dim=tuple(range(1, target.ndim))).sqrt()
        nonzero_rl2 = torch.where(target_norm > 1.0e-8, rl2, torch.full_like(rl2, float("nan")))
    return {
        "relative_l2": rl2,
        "pearson_r": r,
        "mae": mae,
        "nonzero_relative_l2": nonzero_rl2,
    }


def _summarize_values(values_by_key: dict) -> dict:
    metric_stats = {
        key: _stats_from_values(values)
        for key, values in values_by_key.items()
    }
    return {
        **{key: stats["mean"] for key, stats in metric_stats.items()},
        "metric_stats": metric_stats,
    }


def _summarize_forward_metrics_with_values(prediction, target, mask=None):
    if isinstance(prediction, dict):
        per_target = {}
        per_target_values = {}
        for name in sorted(prediction):
            summary, values = _summarize_forward_metrics_with_values(
                prediction[name],
                target[name],
                mask.get(name) if isinstance(mask, dict) else None,
            )
            per_target[name] = summary
            per_target_values[name] = values
        keys = ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2")
        aggregate_values = {}
        for key in keys:
            tensors = [values[key] for values in per_target_values.values() if key in values]
            if tensors:
                aggregate_values[key] = _nanmean_rows(torch.stack(tensors, dim=1))
        summary = _summarize_values(aggregate_values)
        summary["per_target"] = per_target
        return summary, aggregate_values

    values = _metric_values(prediction, target, mask=mask)
    return _summarize_values(values), values


def summarize_forward_metrics(prediction, target, mask=None):
    summary, _ = _summarize_forward_metrics_with_values(prediction, target, mask=mask)
    return summary


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


class MetricStatsMeter:
    def __init__(self):
        self.total = 0.0
        self.total_sq = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        if math.isnan(float(value)):
            return
        self.total += float(value) * int(n)
        self.total_sq += float(value) * float(value) * int(n)
        self.count += int(n)

    def update_stats(self, stats: dict):
        count = int(stats.get("count", 0) or 0)
        if count <= 0:
            return
        mean = float(stats.get("mean", float("nan")))
        std = float(stats.get("std", 0.0) or 0.0)
        if math.isnan(mean):
            return
        self.total += mean * count
        if count > 1:
            self.total_sq += (count - 1) * std * std + count * mean * mean
        else:
            self.total_sq += mean * mean
        self.count += count

    @property
    def average(self) -> float:
        if self.count == 0:
            return math.nan
        return self.total / self.count

    @property
    def std(self) -> float:
        if self.count == 0:
            return math.nan
        if self.count == 1:
            return 0.0
        variance = max((self.total_sq - (self.total * self.total) / self.count) / (self.count - 1), 0.0)
        return math.sqrt(variance)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.average,
            "std": self.std,
            "count": self.count,
        }


def build_forward_metric_meters(target_names: list[str] | None = None) -> dict:
    base = {
        "relative_l2": MetricStatsMeter(),
        "pearson_r": MetricStatsMeter(),
        "mae": MetricStatsMeter(),
        "nonzero_relative_l2": MetricStatsMeter(),
        "inference_time_ms": AverageMeter(),
    }
    if not target_names:
        return base
    base["per_target"] = {
        name: {
            "relative_l2": MetricStatsMeter(),
            "pearson_r": MetricStatsMeter(),
            "mae": MetricStatsMeter(),
            "nonzero_relative_l2": MetricStatsMeter(),
        }
        for name in target_names
    }
    return base


def update_forward_metric_meters(metric_meters: dict, batch_metrics: dict, batch_size: int):
    for key in ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2"):
        stats = batch_metrics.get("metric_stats", {}).get(key)
        if stats is not None and hasattr(metric_meters[key], "update_stats"):
            metric_meters[key].update_stats(stats)
        else:
            value = batch_metrics[key]
            if not math.isnan(value):
                metric_meters[key].update(value, batch_size)
    if "per_target" not in metric_meters or "per_target" not in batch_metrics:
        return
    for target_name, per_target_metrics in batch_metrics["per_target"].items():
        for key in ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2"):
            stats = per_target_metrics.get("metric_stats", {}).get(key)
            if stats is not None and hasattr(metric_meters["per_target"][target_name][key], "update_stats"):
                metric_meters["per_target"][target_name][key].update_stats(stats)
            else:
                value = per_target_metrics[key]
                if not math.isnan(value):
                    metric_meters["per_target"][target_name][key].update(value, batch_size)


def finalize_forward_metric_meters(metric_meters: dict) -> dict[str, float]:
    metric_keys = ("relative_l2", "pearson_r", "mae", "nonzero_relative_l2")
    result = {
        key: metric_meters[key].average
        for key in metric_keys
    }
    result.update({
        "inference_time_ms": metric_meters["inference_time_ms"].average,
        "metric_stats": {
            key: metric_meters[key].as_dict()
            for key in metric_keys
            if hasattr(metric_meters[key], "as_dict")
        },
    })
    if "per_target" in metric_meters:
        result["per_target"] = {
            target_name: {
                **{
                    key: meters[key].average
                    for key in metric_keys
                },
                "metric_stats": {
                    key: meters[key].as_dict()
                    for key in metric_keys
                    if hasattr(meters[key], "as_dict")
                },
            }
            for target_name, meters in metric_meters["per_target"].items()
        }
    return result
