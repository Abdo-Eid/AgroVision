"""Model construction helpers."""

from __future__ import annotations

from typing import Any, Dict

from src.models.unet_baseline import UNet


def build_model(cfg: Dict[str, Any]) -> UNet:
    model_cfg = cfg.get("model", {})
    bands = cfg.get("bands", [])
    in_channels = len(bands) if bands else int(model_cfg.get("in_channels", 3))

    classes_cfg = cfg.get("classes", {})
    if classes_cfg:
        num_classes = len(classes_cfg)
    else:
        num_classes = int(model_cfg.get("num_classes", 1))

    arch = str(model_cfg.get("arch", "unet_baseline")).lower()
    base_channels = int(model_cfg.get("base_channels", 64))
    if arch in {"unet_baseline", "unet"}:
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            bilinear=bool(model_cfg.get("bilinear", True)),
        )

    raise ValueError(f"Unsupported model architecture '{arch}'.")
