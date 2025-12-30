"""
LiteFieldSeg: a lightweight segmentation network for sparse-labeled AgriFieldNet.

Design goals:
- Faster + lower memory than U-Net (depthwise convs, skip-add not concat)
- Attention modules included (SE in encoder, CBAM-lite in decoder)
- Output is pixel-wise logits [B, num_classes, H, W] for overlay + field-aware training
"""
from __future__ import annotations
from dataclasses import dataclass

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ConvBNAct, InvertedResidualSE, DepthwiseSeparableConv, CBAMLite

from .registry import register_model

@dataclass
class EncoderStageCfg:
    out_ch: int
    num_blocks: int
    stride: int
    expand: int

@register_model("litefieldseg")
class LiteFieldSeg(nn.Module):
    """
    LiteFieldSeg
    - Encoder: inverted residual + SE (MobileNetV2/EfficientNet inspired)
    - Decoder: FPN-like top-down fusion with skip-add (not concat) + CBAM-lite refine
    - Head: produces pixel logits [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stem_ch: int = 16,
        dec_ch: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Early downsample reduces compute quickly
        self.stem = ConvBNAct(in_channels, stem_ch, k=3, s=2)  # /2

        # Encoder configuration (keep it modest for speed/RAM)
        cfgs = [
            EncoderStageCfg(out_ch=24, num_blocks=2, stride=2, expand=2),  # /4
            EncoderStageCfg(out_ch=32, num_blocks=3, stride=2, expand=2),  # /8
            EncoderStageCfg(out_ch=64, num_blocks=3, stride=2, expand=2),  # /16
            EncoderStageCfg(out_ch=96, num_blocks=2, stride=1, expand=2),  # /16 refine
        ]

        self.stages = nn.ModuleList()
        in_ch = stem_ch
        for c in cfgs:
            blocks: List[nn.Module] = []
            # first block may downsample
            blocks.append(InvertedResidualSE(in_ch, c.out_ch, stride=c.stride, expand=c.expand))
            in_ch = c.out_ch
            # remaining blocks keep resolution
            for _ in range(c.num_blocks - 1):
                blocks.append(InvertedResidualSE(in_ch, c.out_ch, stride=1, expand=c.expand))
            self.stages.append(nn.Sequential(*blocks))

        # Decoder (FPN-style): unify channels then fuse by addition (cheaper than concat)
        self.lat4 = ConvBNAct(24, dec_ch, k=1, s=1, p=0)
        self.lat8 = ConvBNAct(32, dec_ch, k=1, s=1, p=0)
        self.lat16 = ConvBNAct(96, dec_ch, k=1, s=1, p=0)

        self.fuse8 = DepthwiseSeparableConv(dec_ch, dec_ch)
        self.fuse4 = DepthwiseSeparableConv(dec_ch, dec_ch)

        # Attention refine near high-res output
        self.refine = CBAMLite(dec_ch)

        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.pred = nn.Conv2d(dec_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits: [B, num_classes, H, W]
        """
        b, _, h, w = x.shape

        x = self.stem(x)          # /2
        x4 = self.stages[0](x)    # /4  (24ch)
        x8 = self.stages[1](x4)   # /8  (32ch)
        x16 = self.stages[2](x8)  # /16 (64ch)
        x16 = self.stages[3](x16) # /16 (96ch)

        # Top-down fusion (FPN-like)
        p16 = self.lat16(x16)  # [B, dec_ch, H/16, W/16]

        p8 = self.lat8(x8) + F.interpolate(p16, size=x8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.fuse8(p8)

        p4 = self.lat4(x4) + F.interpolate(p8, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.fuse4(p4)

        # Attention refinement at higher resolution
        p4 = self.refine(p4)
        p4 = self.drop(p4)

        # Upsample back to input size
        p = F.interpolate(p4, size=(h, w), mode="bilinear", align_corners=False)
        logits = self.pred(p)
        return logits


# -------------------------
# Optional helper: field aggregation (useful for eval/debug)
# -------------------------

@torch.no_grad()
def aggregate_field_logits(
    logits: torch.Tensor, reveals: torch.Tensor, field_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate pixel logits to per-field logits by mean over pixels.

    Args:
        logits: [B, C, H, W]
        field_ids: [B, H, W] int, 0 = no-field

    Returns:
        field_logits: [N_fields, C]
        field_batch_ids: [N_fields] which batch index each field came from
    """
    b, c, h, w = logits.shape
    flat_logits = logits.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]
    flat_ids = field_ids.reshape(b, h * w)                        # [B, HW]

    all_field_logits = []
    all_batch_ids = []

    for bi in range(b):
        ids = flat_ids[bi]
        lg = flat_logits[bi]

        uniq = torch.unique(ids)
        uniq = uniq[uniq > 0]  # ignore background/no-field

        for fid in uniq:
            mask = (ids == fid)
            if mask.any():
                all_field_logits.append(lg[mask].mean(dim=0))
                all_batch_ids.append(torch.tensor(bi, device=logits.device))

    if len(all_field_logits) == 0:
        return torch.empty(0, c, device=logits.device), torch.empty(0, device=logits.device, dtype=torch.long)

    return torch.stack(all_field_logits, dim=0), torch.stack(all_batch_ids, dim=0)


def _smoke_test() -> None:
    model = LiteFieldSeg(in_channels=12, num_classes=14, dropout=0.1)
    x = torch.randn(2, 12, 256, 256)
    y = model(x)
    print("input:", x.shape, "output:", y.shape)


if __name__ == "__main__":
    _smoke_test()
