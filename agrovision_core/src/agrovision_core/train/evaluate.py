"""Evaluation entry point for AgroVision segmentation models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from ..data.dataset import CropDataset
from .metrics import compute_confusion_matrix, segmentation_metrics_from_confusion_matrix
from .train import build_model, load_config, select_device


def build_val_loader(data_dir: str, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    dataset = CropDataset(data_dir, split="val", transforms=None)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_class_map(data_dir: str) -> Dict[str, Any]:
    class_map_path = Path(data_dir) / "class_map.json"
    if class_map_path.exists():
        with open(class_map_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_metrics(
    output_dir: Path,
    metrics: Dict[str, Any],
    class_map: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    per_class_path = output_dir / "per_class_metrics.csv"
    contig_to_raw = {
        int(k): int(v) for k, v in class_map.get("contig_to_raw", {}).items()
    }
    classes = class_map.get("classes", {})

    with open(per_class_path, "w", encoding="utf-8") as f:
        f.write("class_id,raw_id,name,is_ignore,iou,f1\n")
        for idx, (iou, f1_score) in enumerate(
            zip(metrics["per_class_iou"], metrics["per_class_f1"])
        ):
            raw_id = contig_to_raw.get(idx, idx)
            name = classes.get(str(raw_id), {}).get("name", "")
            is_ignore = "1" if idx == metrics.get("ignore_index", -1) else "0"
            f.write(f"{idx},{raw_id},{name},{is_ignore},{iou:.6f},{f1_score:.6f}\n")


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    device = select_device(model_cfg.get("device", "cpu"))

    val_loader = build_val_loader(
        config["paths"]["data_processed"],
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 4),
    )

    dataset = val_loader.dataset
    in_channels = dataset.num_channels
    num_classes = dataset.num_classes
    ignore_index = training_cfg.get("ignore_index", None)
    if ignore_index is not None:
        ignore_index = int(ignore_index)
        if ignore_index <= 0:
            ignore_index = None

    model = build_model(args.model, in_channels, num_classes, model_cfg)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    all_cm: torch.Tensor | None = None
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            cm = compute_confusion_matrix(preds, masks, num_classes, ignore_index)
            cm = cm.to(device)
            all_cm = cm if all_cm is None else all_cm + cm

    if all_cm is None:
        metrics = {
            "mIoU": 0.0,
            "mIoU_fg": 0.0,
            "macro_f1": 0.0,
            "Dice_fg": 0.0,
            "per_class_iou": [0.0] * num_classes,
            "per_class_f1": [0.0] * num_classes,
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }
    else:
        metrics = segmentation_metrics_from_confusion_matrix(all_cm, ignore_index)

    metrics["num_classes"] = num_classes
    metrics["ignore_index"] = -1 if ignore_index is None else ignore_index
    metrics["checkpoint"] = str(args.checkpoint)

    output_root = Path("outputs/runs")
    run_name = args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / run_name

    class_map = load_class_map(config["paths"]["data_processed"])
    write_metrics(output_dir, metrics, class_map)

    print(
        f"mIoU: {metrics['mIoU']:.4f} | mIoU_fg: {metrics['mIoU_fg']:.4f} | "
        f"Dice_fg: {metrics['Dice_fg']:.4f} | macro_f1: {metrics['macro_f1']:.4f}"
    )
    print(f"Saved metrics to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AgroVision segmentation model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/runs/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--model", default="unet_baseline", help="Model name")
    parser.add_argument("--run-name", default=None, help="Eval run name under outputs/runs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
