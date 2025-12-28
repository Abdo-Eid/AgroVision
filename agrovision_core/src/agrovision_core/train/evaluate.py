"""Evaluation utilities for semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from ..data.dataset import CropDataset
from .metrics import (
    fast_confusion_matrix,
    iou_per_class,
    macro_f1,
    mean_iou,
    per_class_f1,
    pixel_accuracy,
)
from .modeling import build_model
from ..utils.io import resolve_path


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    device_name = str(cfg.get("model", {}).get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _evaluate_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int,
) -> Dict[str, Any]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            # If masks use raw class IDs (e.g., 36) convert them to contiguous training indices
            # using the dataset-provided mapping `raw_to_contig` (if available).
            raw_to_contig = getattr(dataloader.dataset, "raw_to_contig", None)
            contig_ignore = ignore_index
            if raw_to_contig:
                import numpy as _np

                mask_np = masks.cpu().numpy()
                # Build a small lookup array up to the max observed raw id in this batch
                max_val = int(mask_np.max()) if mask_np.size else 0
                lookup = _np.full((max_val + 1,), -1, dtype=_np.int64)
                for r, c in raw_to_contig.items():
                    if r <= max_val:
                        lookup[int(r)] = int(c)
                mapped = lookup[mask_np]
                masks = torch.from_numpy(mapped).to(device)

                # Map ignore index from raw id to contiguous id if possible
                if isinstance(ignore_index, int):
                    contig_ignore = raw_to_contig.get(int(ignore_index), ignore_index)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            batch_cm = fast_confusion_matrix(
                preds=preds,
                targets=masks,
                num_classes=num_classes,
                ignore_index=contig_ignore,
            )
            confusion += batch_cm.cpu()

    iou = iou_per_class(confusion)
    f1 = per_class_f1(confusion)
    metrics = {
        "confusion_matrix": confusion,
        "per_class_iou": iou,
        "per_class_f1": f1,
        "miou": mean_iou(confusion),
        "macro_f1": macro_f1(confusion),
        "pixel_accuracy": pixel_accuracy(confusion),
    }
    return metrics


def evaluate(
    model: torch.nn.Module | None,
    cfg: Dict[str, Any],
    checkpoint_path: str | None = None,
) -> Dict[str, Any]:
    """Evaluate a model on the validation split."""
    device = _resolve_device(cfg)
    if model is None:
        model = build_model(cfg)

    if checkpoint_path is not None:
        _load_checkpoint(model, resolve_path(checkpoint_path), device)

    model.to(device)

    data_dir = resolve_path(cfg["paths"]["data_processed"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {data_dir}\n"
            "Run the dataset preparation script to generate `*_images.npy`/`*_masks.npy`."
        )
    val_dataset = CropDataset(data_dir, split="val")
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    # Avoid DataLoader multiprocessing issues in interactive kernels
    try:
        import sys, os

        if num_workers > 0 and ("ipykernel" in sys.modules or os.name == "nt"):
            print(
                "Interactive or Windows environment detected; setting num_workers=0 to avoid DataLoader spawn errors."
            )
            num_workers = 0
    except Exception:
        pass

    pin_memory = device.type == "cuda"
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    ignore_index = int(cfg.get("training", {}).get("ignore_index", 255))
    num_classes = int(val_dataset.num_classes)
    raw_metrics = _evaluate_loader(model, val_loader, num_classes, device, ignore_index)

    per_class_iou = {}
    per_class_f1_scores = {}
    for idx in range(num_classes):
        raw_id = val_dataset.index_to_raw.get(idx, idx)
        per_class_iou[int(raw_id)] = float(raw_metrics["per_class_iou"][idx].item())
        per_class_f1_scores[int(raw_id)] = float(raw_metrics["per_class_f1"][idx].item())

    class_names = {
        int(raw_id): cfg.get("classes", {}).get(raw_id, {}).get("name")
        for raw_id in per_class_iou.keys()
    }

    return {
        "miou": raw_metrics["miou"],
        "macro_f1": raw_metrics["macro_f1"],
        "pixel_accuracy": raw_metrics["pixel_accuracy"],
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1_scores,
        "class_names": class_names,
        "confusion_matrix": raw_metrics["confusion_matrix"].tolist(),
    }
