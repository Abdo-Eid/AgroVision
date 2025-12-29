"""Training entry point for AgroVision segmentation models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from ..data.dataset import CropDataset
from ..models.unet_baseline import UNet
from .losses import FocalCrossEntropyLoss
from .metrics import compute_confusion_matrix, segmentation_metrics_from_confusion_matrix


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve device name with fallback."""
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    model_config: Dict[str, Any],
) -> torch.nn.Module:
    """Construct a model from config."""
    if model_name == "unet_baseline":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=model_config.get("base_channels", 64),
            depth=model_config.get("depth", 4),
            dropout=model_config.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CropDataset]:
    """Create train/val dataloaders with augmentation on train only."""
    from ..data.dataset import RandomFlipRotate

    train_dataset = CropDataset(data_dir, split="train", transforms=RandomFlipRotate())
    val_dataset = CropDataset(data_dir, split="val", transforms=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: Optional[int],
) -> Dict[str, Any]:
    """Run evaluation for one epoch."""
    model.eval()
    all_cm: torch.Tensor | None = None

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            cm = compute_confusion_matrix(preds, masks, num_classes, ignore_index)
            all_cm = cm if all_cm is None else all_cm + cm

    if all_cm is None:
        return {
            "mIoU": 0.0,
            "mIoU_fg": 0.0,
            "macro_f1": 0.0,
            "Dice_fg": 0.0,
            "per_class_iou": [0.0] * num_classes,
            "per_class_f1": [0.0] * num_classes,
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }

    return segmentation_metrics_from_confusion_matrix(all_cm, ignore_index)


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    seed_everything(training_cfg.get("random_seed", 42))
    device = select_device(model_cfg.get("device", "cpu"))

    train_loader, val_loader, train_dataset = create_dataloaders(
        config["paths"]["data_processed"],
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 4),
    )

    in_channels = train_dataset.num_channels
    num_classes = train_dataset.num_classes
    ignore_index = training_cfg.get("ignore_index", None)
    if ignore_index is not None:
        ignore_index = int(ignore_index)
        if ignore_index <= 0:
            ignore_index = None
    min_labeled_fraction = float(training_cfg.get("min_labeled_fraction", 0.0))

    model = build_model(args.model, in_channels, num_classes, model_cfg)
    model.to(device)

    class_weights_cfg = training_cfg.get("class_weights", None)
    if class_weights_cfg is not None:
        weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
    else:
        weights = train_dataset.get_class_weights(normalize=True)
    if weights.numel() != num_classes:
        raise ValueError("class_weights must match num_classes.")
    weights = weights.to(device)
    background_weight = float(training_cfg.get("background_weight", 0.05))
    weights[0] = background_weight  # keep background low to reduce imbalance dominance
    gamma = float(training_cfg.get("focal_gamma", 2.0))
    criterion = FocalCrossEntropyLoss(gamma=gamma, alpha=weights)

    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    optimizer_name = training_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    start_epoch = 0
    best_miou = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))

    base_run_dir = Path("outputs/runs")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(config["paths"].get("model_checkpoint", "outputs/runs/best_model.pth"))
    last_model_path = run_dir / "last_model.pth"
    metrics_path = run_dir / "metrics.jsonl"

    epochs = int(training_cfg.get("epochs", 20))

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if min_labeled_fraction > 0.0 and ignore_index is not None:
                labeled_fraction = (masks != ignore_index).float().mean().item()
                if labeled_fraction < min_labeled_fraction:
                    continue

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device, num_classes, ignore_index)
        val_miou = float(val_metrics["mIoU"])

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_mIoU": val_miou,
            "val_mIoU_fg": float(val_metrics["mIoU_fg"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_Dice_fg": float(val_metrics["Dice_fg"]),
            "val_per_class_iou": val_metrics["per_class_iou"],
            "val_per_class_f1": val_metrics["per_class_f1"],
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "model_name": args.model,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "config_path": args.config,
        }
        save_checkpoint(checkpoint, last_model_path)

        if val_miou > best_miou:
            best_miou = val_miou
            checkpoint["best_miou"] = best_miou
            save_checkpoint(checkpoint, best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AgroVision segmentation model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="unet_baseline", help="Model name")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--run-name", default=None, help="Run name for outputs/runs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
