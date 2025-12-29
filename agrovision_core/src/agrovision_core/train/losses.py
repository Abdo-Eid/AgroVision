"""Loss functions for segmentation training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class focal cross-entropy on logits with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.dim() != 1:
                raise ValueError("alpha must be a 1D tensor of class weights.")
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets = targets.long()

        valid = (targets >= 0) & (targets < num_classes)
        if not valid.any():
            return logits.sum() * 0.0

        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        log_pt = log_pt[valid]
        pt = log_pt.exp()

        if self.alpha is None:
            loss = -(1.0 - pt).pow(self.gamma) * log_pt
        else:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.numel() != num_classes:
                raise ValueError("alpha must match number of classes.")
            alpha_t = alpha[safe_targets[valid]]
            loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")
