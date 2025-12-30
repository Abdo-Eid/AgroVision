"""Reusable CNN blocks for segmentation models."""

from __future__ import annotations

from typing import Tuple, List

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

# ==========================
# LiteFieldSeg blocks
# ==========================

class ConvBNAct(nn.Module):
    """Conv -> BN -> activation. (Used for 1x1 and regular convs.)"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        groups: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise 3x3 + Pointwise 1x1.
    Much cheaper than a full 3x3 conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "silu") -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=stride, groups=in_ch, act=act)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (channel attention).
    Efficient attention: global pooling + tiny MLP.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pooling
        s = F.adaptive_avg_pool2d(x, 1)
        # Excite: channel gates
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class InvertedResidualSE(nn.Module):
    """
    MobileNetV2/EfficientNet-inspired inverted residual block with SE attention.
    - expand (1x1) -> depthwise (3x3) -> SE -> project (1x1)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int = 2) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, mid, k=1, s=1, p=0))

        # depthwise
        layers.append(ConvBNAct(mid, mid, k=3, s=stride, groups=mid))

        # attention (channel)
        layers.append(SEBlock(mid))

        # project
        layers.append(nn.Conv2d(mid, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_res else y


class CBAMLite(nn.Module):
    """
    Lightweight CBAM-style attention for refinement:
    - Channel attention (SE-like)
    - Spatial attention (avg/max over channels -> small conv -> sigmoid)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = SEBlock(channels, reduction=4)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)

        # Spatial gate: pool along channel dimension
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * a
