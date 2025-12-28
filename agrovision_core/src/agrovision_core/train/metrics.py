"""Segmentation metrics with ignore_index handling."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Elementwise divide with 0 when denominator is 0."""
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute confusion matrix for segmentation.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class indices, shape (N, H, W) or flat.
    targets : torch.Tensor
        Ground truth class indices, shape (N, H, W) or flat.
    num_classes : int
        Number of classes (including ignore index if present).
    ignore_index : int, optional
        Class index to ignore.
    """
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()

    if ignore_index is not None:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]

    if preds.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=preds.device)

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)


def segmentation_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Dict[str, object]:
    """Compute metrics from logits tensor."""
    preds = torch.argmax(logits, dim=1)
    return segmentation_metrics(preds, targets, num_classes, ignore_index)


def segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute mIoU, per-class IoU, and macro F1.

    Returns
    -------
    dict with keys: mIoU, per_class_iou, macro_f1, per_class_f1, confusion_matrix
    """
    cm = compute_confusion_matrix(preds, targets, num_classes, ignore_index)
    cm = cm.to(dtype=torch.float32)

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    iou = _safe_divide(tp, tp + fp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)

    valid = torch.ones(num_classes, dtype=torch.bool, device=cm.device)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid[ignore_index] = False

    if valid.any():
        miou = iou[valid].mean().item()
        macro_f1 = f1[valid].mean().item()
    else:
        miou = 0.0
        macro_f1 = 0.0

    return {
        "mIoU": miou,
        "per_class_iou": iou.detach().cpu().tolist(),
        "macro_f1": macro_f1,
        "per_class_f1": f1.detach().cpu().tolist(),
        "confusion_matrix": cm.detach().cpu().to(dtype=torch.long).tolist(),
    }


def segmentation_metrics_from_confusion_matrix(
    cm: torch.Tensor,
    ignore_index: Optional[int] = None,
) -> Dict[str, object]:
    """Compute metrics from a pre-aggregated confusion matrix."""
    cm = cm.to(dtype=torch.float32)
    num_classes = cm.shape[0]

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    iou = _safe_divide(tp, tp + fp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)

    valid = torch.ones(num_classes, dtype=torch.bool, device=cm.device)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid[ignore_index] = False

    if valid.any():
        miou = iou[valid].mean().item()
        macro_f1 = f1[valid].mean().item()
    else:
        miou = 0.0
        macro_f1 = 0.0

    return {
        "mIoU": miou,
        "per_class_iou": iou.detach().cpu().tolist(),
        "macro_f1": macro_f1,
        "per_class_f1": f1.detach().cpu().tolist(),
        "confusion_matrix": cm.detach().cpu().to(dtype=torch.long).tolist(),
    }
