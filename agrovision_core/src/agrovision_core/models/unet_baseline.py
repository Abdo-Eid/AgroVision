"""Baseline U-Net implementation."""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from .blocks import ConvBlock, DownBlock, UpBlock
from .registry import register_model


# Registered so it can be constructed via the model registry by name.
@register_model("unet_baseline")
class UNet(nn.Module):
    """U-Net baseline for semantic segmentation."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth

        encoder_channels: List[int] = [base_channels * (2**i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        for idx, out_channels in enumerate(encoder_channels):
            in_ch = in_channels if idx == 0 else encoder_channels[idx - 1]
            self.encoders.append(DownBlock(in_ch, out_channels))

        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels)

        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))
        in_ch = bottleneck_channels
        for out_ch in decoder_channels:
            self.decoders.append(UpBlock(in_ch, out_ch))
            in_ch = out_ch

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)
        x = self.drop(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)


def _smoke_test() -> None:
    model = UNet(in_channels=12, num_classes=14)
    x = torch.randn(2, 12, 256, 256)
    y = model(x)
    print("input:", x.shape, "output:", y.shape)


if __name__ == "__main__":
    _smoke_test()
