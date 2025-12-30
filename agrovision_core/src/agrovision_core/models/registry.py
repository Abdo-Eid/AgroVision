"""Model registry and factory for AgroVision."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Type

from torch import nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
_AUTOLOADED_PACKAGES: set[str] = set()


def register_model(name: str):
    """Decorator to register a model class by name."""

    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def _clean_model_cfg(model_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Strip non-constructor keys from model config before instantiation."""
    cfg = dict(model_cfg or {})
    for key in ("name", "device", "num_classes", "in_channels", "tile_limit", "input_size"):
        cfg.pop(key, None)
    return cfg


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    model_cfg: Dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a registered model."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model not registered: {name}. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[name]
    cfg = _clean_model_cfg(model_cfg)
    return cls(in_channels=in_channels, num_classes=num_classes, **cfg)


def autoload_models(package_name: str) -> None:
    """Import all modules in a package so they self-register."""
    if package_name in _AUTOLOADED_PACKAGES:
        return
    # Importing modules triggers @register_model decorators.
    package = importlib.import_module(package_name)
    prefix = package.__name__ + "."
    for _, name, _ in pkgutil.iter_modules(package.__path__, prefix):
        importlib.import_module(name)
    _AUTOLOADED_PACKAGES.add(package_name)


def ensure_models_loaded() -> None:
    """Load models from the default AgroVision package."""
    # Single call site to populate the registry before building models.
    autoload_models("agrovision_core.models")
