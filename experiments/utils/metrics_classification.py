from __future__ import annotations

import math

import numpy as np


def confusion_matrix_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(labels.astype(np.int64), predictions.astype(np.int64), strict=False):
        matrix[target, pred] += 1
    return matrix


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    f1_scores = []
    for class_index in range(confusion.shape[0]):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        f1_scores.append(0.0 if denominator == 0.0 else (2.0 * tp) / denominator)
    return float(np.mean(f1_scores)) if f1_scores else math.nan


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> float:
    recalls = []
    for class_index in range(confusion.shape[0]):
        tp = float(confusion[class_index, class_index])
        positives = float(confusion[class_index, :].sum())
        recalls.append(0.0 if positives == 0.0 else tp / positives)
    return float(np.mean(recalls)) if recalls else math.nan


def accuracy_from_confusion(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    if total == 0.0:
        return math.nan
    return float(np.trace(confusion) / total)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    positive_count = int(labels.sum())
    negative_count = int(labels.shape[0] - positive_count)
    if positive_count == 0 or negative_count == 0:
        return math.nan

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ranks_sorted[start:end].mean()
        ranks_sorted[start:end] = average_rank
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    positive_rank_sum = ranks[labels == 1].sum()
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(auc)


def macro_auroc_from_probabilities(labels: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    aucs = []
    for class_index in range(num_classes):
        binary_labels = (labels == class_index).astype(np.int64)
        auc = _binary_auc(binary_labels, probabilities[:, class_index])
        if not math.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else math.nan


def summarize_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float | list[list[int]]]:
    predictions = probabilities.argmax(axis=1)
    confusion = confusion_matrix_from_predictions(labels, predictions, num_classes=num_classes)
    return {
        "accuracy": accuracy_from_confusion(confusion),
        "macro_f1": macro_f1_from_confusion(confusion),
        "balanced_accuracy": balanced_accuracy_from_confusion(confusion),
        "macro_auroc": macro_auroc_from_probabilities(labels, probabilities, num_classes=num_classes),
        "confusion_matrix": confusion.tolist(),
    }
