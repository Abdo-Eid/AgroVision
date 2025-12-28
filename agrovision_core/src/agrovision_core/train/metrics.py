"""Metrics for semantic segmentation."""

from __future__ import annotations

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def fast_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute a confusion matrix using torch.bincount."""
    if preds.ndim > 1:
        preds = preds.reshape(-1)
    if targets.ndim > 1:
        targets = targets.reshape(-1)

    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    if preds.numel() == 0:
        return torch.zeros((num_classes, num_classes), device=preds.device, dtype=torch.int64)

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def iou_per_class(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """Compute IoU for each class from a confusion matrix."""
    tp = torch.diag(confusion_matrix).to(torch.float32)
    fp = confusion_matrix.sum(dim=0).to(torch.float32) - tp
    fn = confusion_matrix.sum(dim=1).to(torch.float32) - tp
    denom = tp + fp + fn
    return _safe_divide(tp, denom)


def mean_iou(confusion_matrix: torch.Tensor) -> float:
    """Compute mean IoU over classes with any support."""
    tp = torch.diag(confusion_matrix).to(torch.float32)
    fp = confusion_matrix.sum(dim=0).to(torch.float32) - tp
    fn = confusion_matrix.sum(dim=1).to(torch.float32) - tp
    denom = tp + fp + fn
    valid = denom > 0
    if valid.any():
        return float(_safe_divide(tp, denom)[valid].mean().item())
    return 0.0


def per_class_f1(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """Compute per-class F1 scores."""
    tp = torch.diag(confusion_matrix).to(torch.float32)
    fp = confusion_matrix.sum(dim=0).to(torch.float32) - tp
    fn = confusion_matrix.sum(dim=1).to(torch.float32) - tp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return _safe_divide(2 * precision * recall, precision + recall)


def macro_f1(confusion_matrix: torch.Tensor) -> float:
    """Compute macro F1 score over classes with any support."""
    tp = torch.diag(confusion_matrix).to(torch.float32)
    fp = confusion_matrix.sum(dim=0).to(torch.float32) - tp
    fn = confusion_matrix.sum(dim=1).to(torch.float32) - tp
    support = tp + fn
    f1 = per_class_f1(confusion_matrix)
    valid = support > 0
    if valid.any():
        return float(f1[valid].mean().item())
    return 0.0


def pixel_accuracy(confusion_matrix: torch.Tensor) -> float:
    """Compute overall pixel accuracy."""
    total = confusion_matrix.sum().item()
    if total == 0:
        return 0.0
    correct = torch.diag(confusion_matrix).sum().item()
    return float(correct / total)
