"""Training loop for semantic segmentation."""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import CropDataset
from src.train.evaluate import _evaluate_loader, evaluate
from src.train.modeling import build_model
from src.utils.io import ensure_dir, resolve_path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    device_name = str(cfg.get("model", {}).get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def _apply_training_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    training_cfg = cfg.setdefault("training", {})
    defaults = {
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "amp": True,
        "scheduler": "cosine",
        "use_dice": False,
        "dice_weight": 0.5,
        "ignore_index": 255,
        "grad_clip_norm": None,
        "log_every": 20,
    }
    for key, value in defaults.items():
        training_cfg.setdefault(key, value)
    return training_cfg


def _dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    mask = targets != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)

    targets_clamped = targets.clone()
    targets_clamped[~mask] = 0
    one_hot = torch.nn.functional.one_hot(targets_clamped, num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    mask = mask.unsqueeze(1)
    probs = probs * mask
    one_hot = one_hot * mask

    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    metrics: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
    }
    torch.save(payload, path)


def train(cfg: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a segmentation model and return the model plus metrics."""
    training_cfg = _apply_training_defaults(cfg)
    device = _resolve_device(cfg)
    seed = int(training_cfg.get("random_seed", 42))
    _set_seed(seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(resolve_path("outputs") / "runs" / run_id)

    config_path = run_dir / "config_used.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    train_dataset = CropDataset("train", cfg)
    val_dataset = CropDataset("val", cfg)

    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)

    ignore_index = int(training_cfg.get("ignore_index", 255))
    num_classes = len(train_dataset.class_map)

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr")),
        weight_decay=float(training_cfg.get("weight_decay")),
    )

    scheduler = None
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()
    epochs = int(training_cfg.get("epochs", 1))
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        step_size = max(1, epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    use_amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_dice = bool(training_cfg.get("use_dice", False))
    dice_weight = float(training_cfg.get("dice_weight", 0.5))
    grad_clip_norm = training_cfg.get("grad_clip_norm")
    log_every = int(training_cfg.get("log_every", 20))

    best_miou = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] | None = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
                if use_dice:
                    loss = loss + dice_weight * _dice_loss(
                        logits, masks, num_classes=num_classes, ignore_index=ignore_index
                    )

            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())

            if log_every > 0 and step % log_every == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch} Step {step}: loss={avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / max(1, len(train_loader))
        val_metrics = _evaluate_loader(
            model=model,
            dataloader=val_loader,
            num_classes=num_classes,
            device=device,
            ignore_index=ignore_index,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_miou": val_metrics["miou"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch}/{epochs} - loss={epoch_loss:.4f} "
            f"- val_mIoU={val_metrics['miou']:.4f} "
            f"- val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        last_checkpoint = run_dir / "last_model.pth"
        _save_checkpoint(model, last_checkpoint, epoch, epoch_summary, cfg)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_epoch = epoch
            best_metrics = copy.deepcopy(epoch_summary)
            best_checkpoint = run_dir / "best_model.pth"
            _save_checkpoint(model, best_checkpoint, epoch, epoch_summary, cfg)

            stable_checkpoint = resolve_path(cfg["paths"]["model_checkpoint"])
            ensure_dir(stable_checkpoint.parent)
            _save_checkpoint(model, stable_checkpoint, epoch, epoch_summary, cfg)

    metrics_payload = {
        "history": history,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }
    metrics_path = run_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    eval_metrics = evaluate(
        model=None,
        cfg=cfg,
        checkpoint_path=str(run_dir / "best_model.pth"),
    )
    eval_path = run_dir / "eval_metrics.json"
    eval_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    return model, metrics_payload
