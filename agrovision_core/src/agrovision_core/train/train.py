"""Training entry point for AgroVision segmentation models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import WeightedRandomSampler

from ..data.dataset import CropDataset
from ..models.unet_baseline import UNet
from .losses import CombinedLoss, FieldLoss, FocalCrossEntropyLoss
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
    use_weighted_sampling: bool = False,
    weight_power: float = 1.0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CropDataset]:
    """Create train/val dataloaders with augmentation on train only.

    Parameters
    ----------
    data_dir : str
        Path to processed data directory.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data loading workers.
    use_weighted_sampling : bool
        If True, oversample tiles with more labeled pixels.
    weight_power : float
        Exponent for sampling weight computation (higher = stronger preference).

    Returns
    -------
    tuple
        (train_loader, val_loader, train_dataset)
    """
    from ..data.dataset import RandomFlipRotate

    train_dataset = CropDataset(data_dir, split="train", transforms=RandomFlipRotate())
    val_dataset = CropDataset(data_dir, split="val", transforms=None)

    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights(power=weight_power)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
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


def compute_field_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    field_ids: torch.Tensor,
) -> Dict[str, Any]:
    """Compute field-level accuracy and cross-entropy.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape (B, C, H, W).
    masks : torch.Tensor
        Ground truth labels of shape (B, H, W).
    field_ids : torch.Tensor
        Field ID per pixel of shape (B, H, W).

    Returns
    -------
    dict
        Dictionary with field_accuracy, field_ce, and num_fields.
    """
    correct = 0
    total = 0
    ce_losses = []

    for b in range(logits.shape[0]):
        unique_fields = field_ids[b].unique()

        for fid in unique_fields:
            if fid == 0:  # Skip non-field pixels
                continue

            field_mask = field_ids[b] == fid
            field_logits = logits[b, :, field_mask].mean(dim=1)  # [C]

            # Get field label (majority vote excluding background)
            field_labels = masks[b][field_mask]
            valid_labels = field_labels[field_labels > 0]
            if valid_labels.numel() == 0:
                field_label = field_labels.mode().values
            else:
                field_label = valid_labels.mode().values

            # Prediction
            pred = field_logits.argmax()

            # Metrics
            correct += (pred == field_label).item()
            total += 1
            ce_losses.append(
                F.cross_entropy(field_logits.unsqueeze(0), field_label.unsqueeze(0))
            )

    field_acc = correct / total if total > 0 else 0.0
    field_ce = torch.stack(ce_losses).mean().item() if ce_losses else 0.0

    return {"field_accuracy": field_acc, "field_ce": field_ce, "num_fields": total}


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: Optional[int],
    compute_field_level: bool = True,
) -> Dict[str, Any]:
    """Run evaluation for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    loader : DataLoader
        Validation data loader.
    device : torch.device
        Device to run on.
    num_classes : int
        Number of classes.
    ignore_index : int, optional
        Class index to ignore.
    compute_field_level : bool
        Whether to compute field-level metrics.

    Returns
    -------
    dict
        Dictionary with pixel-level and optionally field-level metrics.
    """
    model.eval()
    all_cm: torch.Tensor | None = None

    # Field-level metrics accumulators
    total_fields = 0
    total_correct = 0
    all_field_ce = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            # Pixel-level confusion matrix
            cm = compute_confusion_matrix(preds, masks, num_classes, ignore_index)
            all_cm = cm if all_cm is None else all_cm + cm

            # Field-level metrics (if field_ids available)
            if compute_field_level and "field_ids" in batch:
                field_ids = batch["field_ids"].to(device, non_blocking=True)
                field_metrics = compute_field_metrics(logits, masks, field_ids)
                total_fields += field_metrics["num_fields"]
                total_correct += int(
                    field_metrics["field_accuracy"] * field_metrics["num_fields"]
                )
                if field_metrics["num_fields"] > 0:
                    all_field_ce.append(field_metrics["field_ce"])

    if all_cm is None:
        result = {
            "mIoU": 0.0,
            "mIoU_fg": 0.0,
            "macro_f1": 0.0,
            "Dice_fg": 0.0,
            "per_class_iou": [0.0] * num_classes,
            "per_class_f1": [0.0] * num_classes,
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }
    else:
        result = segmentation_metrics_from_confusion_matrix(all_cm, ignore_index)

    # Add field-level metrics
    if compute_field_level:
        result["field_accuracy"] = total_correct / total_fields if total_fields > 0 else 0.0
        result["field_ce"] = sum(all_field_ce) / len(all_field_ce) if all_field_ce else 0.0
        result["num_fields"] = total_fields

    return result


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    seed_everything(training_cfg.get("random_seed", 42))
    device = select_device(model_cfg.get("device", "cpu"))

    # Weighted sampling configuration
    use_weighted_sampling = training_cfg.get("use_weighted_sampling", False)
    weight_power = float(training_cfg.get("weight_power", 1.0))

    train_loader, val_loader, train_dataset = create_dataloaders(
        config["paths"]["data_processed"],
        batch_size=training_cfg.get("batch_size", 32),
        num_workers=training_cfg.get("num_workers", 4),
        use_weighted_sampling=use_weighted_sampling,
        weight_power=weight_power,
    )

    in_channels = train_dataset.num_channels
    num_classes = train_dataset.num_classes
    ignore_index = training_cfg.get("ignore_index", None)
    if ignore_index is not None:
        ignore_index = int(ignore_index)
        # Only invalidate truly negative values (e.g., -1 used as sentinel)
        # 0 is a VALID ignore_index for background/unlabeled class
        if ignore_index < 0:
            ignore_index = None
    min_labeled_fraction = float(training_cfg.get("min_labeled_fraction", 0.0))

    # Loss weights for combined loss
    lambda_pixel = float(training_cfg.get("lambda_pixel", 0.2))
    lambda_field = float(training_cfg.get("lambda_field", 1.0))
    use_field_loss = training_cfg.get("use_field_loss", True)

    model = build_model(args.model, in_channels, num_classes, model_cfg)
    model.to(device)

    # Class weights for pixel loss
    class_weights_cfg = training_cfg.get("class_weights", None)
    if class_weights_cfg is not None:
        weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
    else:
        weights = train_dataset.get_class_weights(normalize=True)
    if weights.numel() != num_classes:
        raise ValueError("class_weights must match num_classes.")
    weights = weights.to(device)
    background_weight = float(training_cfg.get("background_weight", 0.0))
    weights[0] = background_weight  # Set to 0 since background is ignored anyway

    gamma = float(training_cfg.get("focal_gamma", 2.0))
    pixel_loss = FocalCrossEntropyLoss(gamma=gamma, alpha=weights, ignore_index=ignore_index)

    # Create combined loss if field loss is enabled
    if use_field_loss:
        field_loss = FieldLoss()
        criterion = CombinedLoss(
            pixel_loss=pixel_loss,
            field_loss=field_loss,
            lambda_pixel=lambda_pixel,
            lambda_field=lambda_field,
        )
    else:
        criterion = pixel_loss

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
    best_field_ce = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", -1.0))
        best_field_ce = float(checkpoint.get("best_field_ce", float("inf")))

    base_run_dir = Path("outputs/runs")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(config["paths"].get("model_checkpoint", "outputs/runs/best_model.pth"))
    last_model_path = run_dir / "last_model.pth"
    metrics_path = run_dir / "metrics.jsonl"

    epochs = int(training_cfg.get("epochs", 20))

    # Model selection criterion
    model_selection = training_cfg.get("model_selection", "field_ce")  # "field_ce" or "mIoU"

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            field_ids = batch["field_ids"].to(device, non_blocking=True)

            # Skip batches with insufficient labeled pixels
            if min_labeled_fraction > 0.0:
                if ignore_index is not None:
                    labeled_fraction = (masks != ignore_index).float().mean().item()
                else:
                    labeled_fraction = (masks != 0).float().mean().item()
                if labeled_fraction < min_labeled_fraction:
                    continue

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)

            # Compute loss (CombinedLoss needs field_ids, FocalCE doesn't)
            if use_field_loss:
                loss = criterion(logits, masks, field_ids)
            else:
                loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device, num_classes, ignore_index, compute_field_level=True)
        val_miou = float(val_metrics["mIoU"])
        val_field_acc = float(val_metrics.get("field_accuracy", 0.0))
        val_field_ce = float(val_metrics.get("field_ce", float("inf")))

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_mIoU": val_miou,
            "val_mIoU_fg": float(val_metrics["mIoU_fg"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_Dice_fg": float(val_metrics["Dice_fg"]),
            "val_field_accuracy": val_field_acc,
            "val_field_ce": val_field_ce,
            "val_num_fields": int(val_metrics.get("num_fields", 0)),
            "val_per_class_iou": val_metrics["per_class_iou"],
            "val_per_class_f1": val_metrics["per_class_f1"],
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} | val_field_acc={val_field_acc:.4f} | "
            f"val_field_ce={val_field_ce:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "best_field_ce": best_field_ce,
            "model_name": args.model,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "config_path": args.config,
        }
        save_checkpoint(checkpoint, last_model_path)

        # Model selection based on configured criterion
        if model_selection == "field_ce":
            # Lower field CE is better
            if val_field_ce < best_field_ce:
                best_field_ce = val_field_ce
                best_miou = val_miou
                checkpoint["best_field_ce"] = best_field_ce
                checkpoint["best_miou"] = best_miou
                save_checkpoint(checkpoint, best_model_path)
                print(f"  -> New best model! field_ce={best_field_ce:.4f}")
        else:
            # Higher mIoU is better
            if val_miou > best_miou:
                best_miou = val_miou
                best_field_ce = val_field_ce
                checkpoint["best_miou"] = best_miou
                checkpoint["best_field_ce"] = best_field_ce
                save_checkpoint(checkpoint, best_model_path)
                print(f"  -> New best model! mIoU={best_miou:.4f}")


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
