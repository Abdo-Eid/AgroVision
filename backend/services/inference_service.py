from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from agrovision_core.models.unet_baseline import UNet
from agrovision_core.utils.io import load_config, resolve_path


class InferenceError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class InferenceResult:
    overlay_image: str | None
    legend: list[dict[str, Any]]
    stats: list[dict[str, Any]]
    runtime_ms: int
    is_mock: bool


class InferenceService:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = config_path
        self._config: dict[str, Any] | None = None
        self._class_map: dict[str, Any] | None = None
        self._norm_stats: dict[str, Any] | None = None
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._num_classes: int | None = None
        self._logger = logging.getLogger("agrovision.backend.inference")

    def _load_config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_config(self._config_path)
            self._logger.info("Loaded config from %s", self._config_path)
        return self._config

    def _get_backend_cfg(self) -> dict[str, Any]:
        return self._load_config().get("backend", {})

    def _data_dir(self) -> Path:
        cfg = self._load_config()
        data_dir = cfg.get("paths", {}).get("data_processed", "data/processed")
        return resolve_path(data_dir)

    def _load_class_map(self) -> dict[str, Any]:
        if self._class_map is None:
            class_map_path = self._data_dir() / "class_map.json"
            if not class_map_path.exists():
                raise InferenceError(
                    "class_map_missing",
                    f"Missing class_map.json at {class_map_path}",
                    status_code=500,
                )
            with class_map_path.open("r", encoding="utf-8") as handle:
                self._class_map = json.load(handle)
            self._logger.info("Loaded class map from %s", class_map_path)
        return self._class_map

    def _load_norm_stats(self) -> dict[str, Any]:
        if self._norm_stats is None:
            stats_path = self._data_dir() / "normalization_stats.json"
            if not stats_path.exists():
                raise InferenceError(
                    "normalization_missing",
                    f"Missing normalization_stats.json at {stats_path}",
                    status_code=500,
                )
            with stats_path.open("r", encoding="utf-8") as handle:
                self._norm_stats = json.load(handle)
            self._logger.info("Loaded normalization stats from %s", stats_path)
        return self._norm_stats

    def _select_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _resolve_model_path(self) -> Path:
        cfg = self._load_config()
        candidates: list[str] = []
        paths_cfg = cfg.get("paths", {})
        if paths_cfg.get("model_checkpoint"):
            candidates.append(paths_cfg["model_checkpoint"])
        if cfg.get("model_path"):
            candidates.append(cfg["model_path"])
        backend_cfg = self._get_backend_cfg()
        if backend_cfg.get("model_path"):
            candidates.append(backend_cfg["model_path"])
        candidates.extend(
            [
                "outputs/runs/best_model.pth",
                "outputs/models/unet_baseline_best_model.pth",
            ]
        )

        for candidate in candidates:
            path = resolve_path(candidate)
            if path.exists():
                return path
        raise InferenceError(
            "model_checkpoint_missing",
            "Model checkpoint not found. Train the model or update config paths.",
            status_code=500,
        )

    def _build_model(
        self, model_name: str, in_channels: int, num_classes: int, model_cfg: dict[str, Any]
    ) -> torch.nn.Module:
        if model_name != "unet_baseline":
            raise InferenceError(
                "model_not_supported",
                f"Unsupported model_name: {model_name}",
                status_code=500,
            )
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=model_cfg.get("base_channels", 64),
            depth=model_cfg.get("depth", 4),
            dropout=model_cfg.get("dropout", 0.0),
        )

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        cfg = self._load_config()
        model_cfg = cfg.get("model", {})
        device_name = model_cfg.get("device", "cpu")
        self._device = self._select_device(device_name)

        checkpoint_path = self._resolve_model_path()
        self._logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._logger.info("Checkpoint loaded into memory (%s)", checkpoint_path.name)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            meta = checkpoint
        else:
            state_dict = checkpoint
            meta = {}

        class_map = self._load_class_map()
        num_classes = int(
            meta.get("num_classes")
            or class_map.get("num_classes", 0)
            or model_cfg.get("num_classes", 14)
        )
        bands = cfg.get("bands", [])
        in_channels = int(
            meta.get("in_channels") or len(bands) or model_cfg.get("in_channels", 12)
        )
        model_name = meta.get("model_name", model_cfg.get("name", "unet_baseline"))

        model = self._build_model(model_name, in_channels, num_classes, model_cfg)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise InferenceError(
                "checkpoint_mismatch",
                f"Model checkpoint incompatible: {exc}",
                status_code=500,
            ) from exc

        model.to(self._device)
        model.eval()
        self._model = model
        self._num_classes = num_classes
        self._logger.info(
            "Model ready | name=%s device=%s in_channels=%s num_classes=%s",
            model_name,
            self._device,
            in_channels,
            num_classes,
        )
        return model

    def _band_names(self) -> list[str]:
        bands = self._load_config().get("bands", [])
        return [band.get("name", "") for band in bands if band.get("name")]

    def _load_demo_sample(self) -> tuple[np.ndarray, np.ndarray]:
        backend_cfg = self._get_backend_cfg()
        split = backend_cfg.get("demo_split", "val")
        index = int(backend_cfg.get("demo_tile_index", 0))

        data_dir = self._data_dir()
        images_path = data_dir / f"{split}_images.npy"
        masks_path = data_dir / f"{split}_masks.npy"

        if not images_path.exists():
            raise InferenceError(
                "demo_images_missing",
                f"Missing {images_path}. Run dataset preparation.",
                status_code=500,
            )
        if not masks_path.exists():
            raise InferenceError(
                "demo_masks_missing",
                f"Missing {masks_path}. Run dataset preparation.",
                status_code=500,
            )

        images = np.load(images_path, mmap_mode="r")
        masks = np.load(masks_path, mmap_mode="r")
        if images.shape[0] == 0:
            raise InferenceError(
                "demo_empty",
                f"No samples found in {images_path}.",
                status_code=500,
            )
        if index >= images.shape[0] or index < 0:
            index = 0
        image = np.array(images[index])
        mask = np.array(masks[index])
        return image, mask

    def _contig_to_raw(self, class_map: dict[str, Any], num_classes: int) -> dict[int, int]:
        mapping = {int(k): int(v) for k, v in class_map.get("contig_to_raw", {}).items()}
        if not mapping:
            return {idx: idx for idx in range(num_classes)}
        for idx in range(num_classes):
            mapping.setdefault(idx, idx)
        return mapping

    def _class_info(self, class_map: dict[str, Any]) -> dict[int, dict[str, Any]]:
        classes = class_map.get("classes", {})
        info: dict[int, dict[str, Any]] = {}
        for key, value in classes.items():
            cls_id = int(key)
            info[cls_id] = {
                "name": value.get("name", f"Class {cls_id}"),
                "color": value.get("color", [0, 0, 0]),
            }
        return info

    def _legend(self, class_map: dict[str, Any]) -> list[dict[str, Any]]:
        info = self._class_info(class_map)
        legend = []
        for cls_id in sorted(info.keys()):
            color = info[cls_id]["color"]
            legend.append(
                {
                    "id": cls_id,
                    "name": info[cls_id]["name"],
                    "color": self._rgb_to_hex(color),
                }
            )
        return legend

    def _rgb_to_hex(self, color: Iterable[int]) -> str:
        r, g, b = [int(x) for x in color]
        return f"#{r:02X}{g:02X}{b:02X}"

    def _build_colormap(
        self, class_map: dict[str, Any], num_classes: int
    ) -> np.ndarray:
        info = self._class_info(class_map)
        contig_to_raw = self._contig_to_raw(class_map, num_classes)
        colormap = np.zeros((num_classes, 3), dtype=np.uint8)
        for idx in range(num_classes):
            raw_id = contig_to_raw.get(idx, idx)
            color = info.get(raw_id, {}).get("color", [0, 0, 0])
            colormap[idx] = np.array(color, dtype=np.uint8)
        return colormap

    def _make_rgb(self, image: np.ndarray) -> np.ndarray:
        band_names = self._band_names()
        norm_stats = self._load_norm_stats()
        rgb_bands = ["B04", "B03", "B02"]
        indices = []
        for name in rgb_bands:
            if name in band_names:
                indices.append(band_names.index(name))
        if len(indices) != 3:
            indices = list(range(min(3, image.shape[0])))

        rgb = []
        for i, idx in enumerate(indices):
            band_name = rgb_bands[i] if i < len(rgb_bands) else ""
            band = image[idx].astype(np.float32)
            stats = norm_stats.get(band_name)
            if stats:
                band = band * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))
            rgb.append(band)
        rgb_stack = np.stack(rgb, axis=-1)
        return self._percentile_stretch(rgb_stack)

    def _percentile_stretch(self, rgb: np.ndarray) -> np.ndarray:
        stretched = np.zeros_like(rgb, dtype=np.float32)
        for channel in range(rgb.shape[-1]):
            values = rgb[..., channel]
            low, high = np.percentile(values, [2, 98])
            if high - low < 1e-6:
                stretched[..., channel] = 0.0
            else:
                stretched[..., channel] = (values - low) / (high - low)
        stretched = np.clip(stretched, 0.0, 1.0)
        return (stretched * 255).astype(np.uint8)

    def _render_overlay(
        self,
        rgb: np.ndarray,
        pred_mask: np.ndarray,
        class_map: dict[str, Any],
        alpha: float,
    ) -> str:
        num_classes = int(self._num_classes or pred_mask.max() + 1)
        colormap = self._build_colormap(class_map, num_classes)
        contig_to_raw = self._contig_to_raw(class_map, num_classes)
        raw_ids = np.array([contig_to_raw.get(i, i) for i in range(num_classes)])
        raw_mask = raw_ids[pred_mask]

        mask_rgb = colormap[pred_mask].astype(np.float32)
        overlay = rgb.astype(np.float32)
        valid = raw_mask != 0
        overlay[valid] = (1.0 - alpha) * overlay[valid] + alpha * mask_rgb[valid]
        overlay = overlay.astype(np.uint8)

        image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _compute_stats(
        self,
        pred_mask: np.ndarray,
        probs: np.ndarray | None,
        class_map: dict[str, Any],
        include_confidence: bool,
    ) -> list[dict[str, Any]]:
        num_classes = int(self._num_classes or pred_mask.max() + 1)
        contig_to_raw = self._contig_to_raw(class_map, num_classes)
        info = self._class_info(class_map)

        counts = np.bincount(pred_mask.reshape(-1), minlength=num_classes)
        valid_indices = [
            idx for idx in range(num_classes) if contig_to_raw.get(idx, idx) != 0
        ]
        total_pixels = int(sum(counts[idx] for idx in valid_indices))

        stats: list[dict[str, Any]] = []
        for idx in valid_indices:
            raw_id = contig_to_raw.get(idx, idx)
            name = info.get(raw_id, {}).get("name", f"Class {raw_id}")
            pixel_count = int(counts[idx])
            percent = (pixel_count / total_pixels * 100.0) if total_pixels > 0 else 0.0
            if include_confidence:
                if pixel_count == 0:
                    confidence = 0.0
                elif probs is None:
                    confidence = 1.0
                else:
                    confidence = float(probs[idx][pred_mask == idx].mean())
            else:
                confidence = 0.0
            stats.append(
                {
                    "id": raw_id,
                    "name": name,
                    "pixels": pixel_count,
                    "percent": percent,
                    "confidence": confidence,
                }
            )
        return stats

    def get_legend(self) -> list[dict[str, Any]]:
        class_map = self._load_class_map()
        return self._legend(class_map)

    def run_inference(
        self, *, tile_count: int, include_confidence: bool
    ) -> InferenceResult:
        cfg = self._load_config()
        tile_limit = int(cfg.get("model", {}).get("tile_limit", 9))
        if tile_count > tile_limit:
            raise InferenceError(
                "tile_limit_exceeded",
                f"Tile limit exceeded: {tile_count} > {tile_limit}. Zoom in to continue.",
                status_code=400,
            )

        backend_cfg = self._get_backend_cfg()
        is_mock = bool(backend_cfg.get("mock_inference", True))
        self._logger.info(
            "Inference request | tileCount=%s tileLimit=%s includeConfidence=%s mock=%s",
            tile_count,
            tile_limit,
            include_confidence,
            is_mock,
        )
        start = time.perf_counter()
        if is_mock:
            result = self._run_mock(include_confidence)
        else:
            result = self._run_model(include_confidence)
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return InferenceResult(
            overlay_image=result.overlay_image,
            legend=result.legend,
            stats=result.stats,
            runtime_ms=runtime_ms,
            is_mock=is_mock,
        )

    def _run_mock(self, include_confidence: bool) -> InferenceResult:
        image, mask = self._load_demo_sample()
        class_map = self._load_class_map()
        rgb = self._make_rgb(image)
        overlay_alpha = float(self._get_backend_cfg().get("overlay_alpha", 0.45))
        overlay = self._render_overlay(rgb, mask, class_map, overlay_alpha)
        stats = self._compute_stats(mask, None, class_map, include_confidence)
        return InferenceResult(
            overlay_image=overlay,
            legend=self._legend(class_map),
            stats=stats,
            runtime_ms=0,
            is_mock=True,
        )

    def _run_model(self, include_confidence: bool) -> InferenceResult:
        model = self._load_model()
        image, _ = self._load_demo_sample()
        class_map = self._load_class_map()
        rgb = self._make_rgb(image)

        tensor = torch.from_numpy(image).unsqueeze(0).float()
        if self._device is None:
            self._device = torch.device("cpu")
        tensor = tensor.to(self._device)
        with torch.inference_mode():
            logits = model(tensor)
            if logits.ndim != 4:
                raise InferenceError(
                    "bad_logits",
                    f"Expected logits with 4 dims, got shape {tuple(logits.shape)}",
                    status_code=500,
                )
            probs_tensor = torch.softmax(logits, dim=1).cpu()[0]
        probs = probs_tensor.numpy()
        pred_mask = np.argmax(probs, axis=0).astype(np.int64)
        num_classes = probs.shape[0]
        self._num_classes = num_classes

        overlay_alpha = float(self._get_backend_cfg().get("overlay_alpha", 0.45))
        overlay = self._render_overlay(rgb, pred_mask, class_map, overlay_alpha)
        stats = self._compute_stats(pred_mask, probs, class_map, include_confidence)
        return InferenceResult(
            overlay_image=overlay,
            legend=self._legend(class_map),
            stats=stats,
            runtime_ms=0,
            is_mock=False,
        )
