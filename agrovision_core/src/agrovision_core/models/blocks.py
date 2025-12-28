"""Reusable CNN blocks for segmentation models."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two-layer Conv-BN-ReLU block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Conv block followed by max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class UpBlock(nn.Module):
    """Upsample, concatenate skip, then Conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    @staticmethod
    def _center_crop(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        _, _, h, w = x.shape
        target_h, target_w = target_hw
        if h == target_h and w == target_w:
            return x
        start_h = max((h - target_h) // 2, 0)
        start_w = max((w - target_w) // 2, 0)
        return x[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = self._center_crop(skip, x.shape[-2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
