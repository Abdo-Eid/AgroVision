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
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.dataset import CropDataset
from .evaluate import _evaluate_loader, evaluate
from .modeling import build_model
from ..utils.io import ensure_dir, resolve_path


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
        "use_dice": True,
        "dice_weight": 0.5,
        "use_focal": False,
        "focal_gamma": 2.0,
        "focal_weight": 1.0,
        "ignore_index": 0,
        "class_weights_power": 1.0,  # >1.0 strengthens minority up-weighting
        "grad_clip_norm": None,
        "log_every": 20,
        "class_weights": "auto",  # "auto", None, or list/tensor of weights
        "sample_foreground": False,  # enable weighted sampling toward FG tiles
        "foreground_boost": 10.0,  # additional weight multiplier for FG tiles
        "foreground_min_fraction": 0.0005,  # minimum FG fraction to count as FG tile
        "min_labeled_fraction": 0.05,
    }
    for key, value in defaults.items():
        training_cfg.setdefault(key, value)
    return training_cfg


def _load_class_weights(
    cfg: Dict[str, Any],
    class_map: Dict[int, int],
    power: float = 1.0,
) -> torch.Tensor | None:
    """
    Derive per-class weights from class_counts in processed class_map.json.

    Returns weights in contiguous ID order (len == num_classes), normalized to mean=1.
    """
    class_map_path = resolve_path(cfg["paths"]["data_processed"]) / "class_map.json"
    if not class_map_path.exists():
        return None

    try:
        with class_map_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return None

    class_counts = raw.get("class_counts") if isinstance(raw, dict) else None
    if not isinstance(class_counts, dict) or not class_counts:
        return None

    # Sort by contiguous id to align with training target ids
    by_contig = sorted(class_map.items(), key=lambda kv: kv[1])
    counts = []
    for raw_id, _ in by_contig:
        cnt = int(class_counts.get(str(raw_id), 0))
        counts.append(cnt if cnt > 0 else 1)

    total = float(sum(counts))
    if total <= 0:
        return None

    weights = [(total / c) ** max(power, 1e-6) for c in counts]
    mean_w = float(sum(weights)) / len(weights)
    weights = [w / mean_w for w in weights]  # normalize to mean=1 to keep LR stable
    return torch.tensor(weights, dtype=torch.float32)


def _compute_sample_weights_from_masks(
    masks: np.ndarray,
    min_fg_fraction: float = 0.001,
    foreground_boost: float = 10.0,
) -> list[float]:
    """
    Compute per-sample weights for a dataset based on foreground fraction.

    Any tile with foreground fraction >= min_fg_fraction gets an additional
    multiplicative boost to increase its sampling probability.
    """
    num_samples = masks.shape[0]
    weights: list[float] = []
    total_pixels = masks.shape[1] * masks.shape[2]
    for i in range(num_samples):
        mask = masks[i]
        fg_frac = float(np.count_nonzero(mask)) / float(total_pixels)
        if fg_frac >= min_fg_fraction:
            weights.append(1.0 + foreground_boost * fg_frac / max(min_fg_fraction, 1e-6))
        else:
            weights.append(1.0)
    return weights


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

    if 0 <= ignore_index < num_classes:
        valid_classes = torch.ones_like(dice, dtype=torch.bool)
        valid_classes[ignore_index] = False
        dice = dice[valid_classes]

    if dice.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    return 1.0 - dice.mean()


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multi-class focal loss with optional per-class alpha weights.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    # mask ignore_index
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)

    targets_clamped = targets.clone()
    targets_clamped[~valid_mask] = 0

    log_pt = log_probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)

    focal_factor = (1.0 - pt) ** gamma

    if alpha is not None:
        alpha_t = alpha.to(logits.device)[targets_clamped]
    else:
        alpha_t = 1.0

    loss = -alpha_t * focal_factor * log_pt
    loss = loss * valid_mask
    return loss.sum() / valid_mask.sum()


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

    # Fix: use configured processed data directory and explicit split names
    data_dir = resolve_path(cfg["paths"]["data_processed"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {data_dir}\n"
            "Run the dataset preparation script to generate `*_images.npy`/`*_masks.npy`."
        )

    train_dataset = CropDataset(data_dir, split="train")
    val_dataset = CropDataset(data_dir, split="val")

    num_classes = int(train_dataset.num_classes)
    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("num_classes") != num_classes:
        model_cfg["num_classes"] = num_classes

    config_path = run_dir / "config_used.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    ignore_index = int(training_cfg.get("ignore_index", 0))
    min_labeled_fraction = float(training_cfg.get("min_labeled_fraction", 0.05))

    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(training_cfg.get("num_workers", 0))

    # Avoid multiprocessing worker spawn issues in interactive environments (notebooks)
    # and commonly problematic Windows spawn contexts. If we're running inside an
    # IPython kernel or similar interactive session, force num_workers to 0.
    try:
        import sys, os

        if num_workers > 0 and ("ipykernel" in sys.modules or os.name == "nt"):
            print(
                "Interactive or Windows environment detected; setting num_workers=0 to avoid DataLoader spawn errors."
            )
            num_workers = 0
            training_cfg["num_workers"] = 0
    except Exception:
        # Be conservative: if detection fails, keep the configured value
        pass

    pin_memory = device.type == "cuda"

    sampler = None
    if bool(training_cfg.get("sample_foreground", False)):
        try:
            fg_boost = float(training_cfg.get("foreground_boost", 10.0))
            fg_min_frac = float(training_cfg.get("foreground_min_fraction", 0.0005))
            sample_weights = _compute_sample_weights_from_masks(
                train_dataset.masks,
                min_fg_fraction=fg_min_frac,
                foreground_boost=fg_boost,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(
                f"Foreground-aware sampling enabled: boost={fg_boost}, "
                f"min_fg_fraction={fg_min_frac}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: foreground sampler disabled due to error: {exc}")
            sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
    class_weights_cfg = training_cfg.get("class_weights", None)
    class_weights_power = float(training_cfg.get("class_weights_power", 1.0))
    class_weights: torch.Tensor | None = None
    if isinstance(class_weights_cfg, (list, tuple)):
        if len(class_weights_cfg) == num_classes:
            class_weights = torch.tensor(class_weights_cfg, dtype=torch.float32)
    elif isinstance(class_weights_cfg, str) and class_weights_cfg.lower() == "auto":
        class_weights = _load_class_weights(
            cfg,
            train_dataset.raw_to_contig,
            power=class_weights_power,
        )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        if class_weights.numel() != num_classes:
            print(
                f"Warning: class_weights length {class_weights.numel()} does not match "
                f"num_classes {num_classes}; disabling class weights."
            )
            class_weights = None
        else:
            if 0 <= ignore_index < num_classes:
                class_weights = class_weights.clone()
                class_weights[ignore_index] = 0.0
                non_ignored = torch.ones_like(class_weights, dtype=torch.bool)
                non_ignored[ignore_index] = False
                non_ignored_weights = class_weights[non_ignored]
                if non_ignored_weights.numel() > 0:
                    mean_w = non_ignored_weights.mean()
                    if mean_w > 0:
                        class_weights[non_ignored] = non_ignored_weights / mean_w
            print(
                f"Using class weights (normalized mean=1): {class_weights.cpu().tolist()}"
            )
    if class_weights is None:
        print("Class weights: None")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
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
    try:
        scaler = torch.amp.GradScaler(enabled=use_amp, device_type=device.type)
    except TypeError:
        # Older PyTorch versions do not accept `device_type` kwarg
        scaler = torch.amp.GradScaler(enabled=use_amp)
    use_dice = bool(training_cfg.get("use_dice", False))
    dice_weight = float(training_cfg.get("dice_weight", 0.5))
    use_focal = bool(training_cfg.get("use_focal", False))
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    focal_weight = float(training_cfg.get("focal_weight", 1.0))
    grad_clip_norm = training_cfg.get("grad_clip_norm")
    log_every = int(training_cfg.get("log_every", 20))

    best_miou = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] | None = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            labeled = (masks != ignore_index).sum().item()
            if labeled < min_labeled_fraction * masks.numel():
                continue

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
                if use_focal:
                    focal = _focal_loss(
                        logits,
                        masks,
                        ignore_index=ignore_index,
                        gamma=focal_gamma,
                        alpha=class_weights,
                    )
                    loss = loss + focal_weight * focal
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
            train_steps += 1

            running_loss += float(loss.item())

            if log_every > 0 and step % log_every == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch} Step {step}: loss={avg_loss:.4f}")

        # Only step the scheduler if we performed at least one optimizer step this epoch.
        if scheduler is not None and train_steps > 0:
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
