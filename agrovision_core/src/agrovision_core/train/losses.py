"""Loss functions for segmentation training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class focal cross-entropy with ignore_index and class weights.

    Parameters
    ----------
    gamma : float
        Focal loss focusing parameter. Higher values down-weight easy examples.
    alpha : torch.Tensor, optional
        Per-class weights of shape (num_classes,).
    ignore_index : int, optional
        Class index to ignore in loss computation (e.g., 0 for unlabeled).
    reduction : str
        'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.dim() != 1:
                raise ValueError("alpha must be a 1D tensor of class weights.")
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal cross-entropy loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model outputs of shape (N, C, H, W).
        targets : torch.Tensor
            Ground truth labels of shape (N, H, W).

        Returns
        -------
        torch.Tensor
            Scalar loss if reduction is 'mean' or 'sum', else (N*H*W,) tensor.
        """
        num_classes = logits.shape[1]
        targets = targets.long()

        # Build validity mask
        valid = (targets >= 0) & (targets < num_classes)

        # Exclude ignore_index from loss computation
        if self.ignore_index is not None:
            valid = valid & (targets != self.ignore_index)

        if not valid.any():
            # Return zero loss if no valid pixels
            return logits.sum() * 0.0

        # Safe gather: replace invalid indices with 0 temporarily
        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        # Apply validity mask
        log_pt = log_pt[valid]
        pt = log_pt.exp()

        # Compute focal weight
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is None:
            loss = -focal_weight * log_pt
        else:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.numel() != num_classes:
                raise ValueError("alpha must match number of classes.")
            alpha_t = alpha[safe_targets[valid]]
            loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class FieldLoss(nn.Module):
    """Field-level cross-entropy loss.

    Aggregates pixel logits per field and computes CE at the field level.
    This aligns training with the AgriFieldNet challenge scoring which is per-field.

    Parameters
    ----------
    reduction : str
        'mean' to average over fields, 'sum' to sum.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute field-level cross-entropy loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model outputs of shape (B, C, H, W).
        masks : torch.Tensor
            Ground truth labels of shape (B, H, W).
        field_ids : torch.Tensor
            Field ID per pixel of shape (B, H, W). 0 = no field.

        Returns
        -------
        torch.Tensor
            Scalar loss averaged (or summed) over all fields in batch.
        """
        losses = []

        for b in range(logits.shape[0]):
            unique_fields = field_ids[b].unique()

            for fid in unique_fields:
                if fid == 0:  # Skip non-field pixels
                    continue

                # Get pixels for this field
                field_mask = field_ids[b] == fid

                # Average logits over field pixels → [C]
                field_logits = logits[b, :, field_mask].mean(dim=1)

                # Get field label (majority vote from pixel labels)
                field_labels = masks[b][field_mask]
                # Filter out background (class 0) for majority vote
                valid_labels = field_labels[field_labels > 0]
                if valid_labels.numel() == 0:
                    # If all pixels are background, use the mode of all labels
                    field_label = field_labels.mode().values
                else:
                    field_label = valid_labels.mode().values

                # CE for this field
                loss = F.cross_entropy(
                    field_logits.unsqueeze(0),
                    field_label.unsqueeze(0),
                )
                losses.append(loss)

        if len(losses) == 0:
            return logits.sum() * 0.0  # No fields in batch

        stacked = torch.stack(losses)
        if self.reduction == "mean":
            return stacked.mean()
        if self.reduction == "sum":
            return stacked.sum()
        return stacked


class CombinedLoss(nn.Module):
    """Combined pixel-level and field-level loss.

    L_total = λ_pixel * L_pixel + λ_field * L_field

    The pixel loss maintains clean segmentation boundaries for visual overlay,
    while the field loss aligns training with AgriFieldNet challenge scoring.

    Parameters
    ----------
    pixel_loss : nn.Module
        Pixel-level loss function (e.g., FocalCrossEntropyLoss).
    field_loss : nn.Module
        Field-level loss function (e.g., FieldLoss).
    lambda_pixel : float
        Weight for pixel loss. Default 0.2.
    lambda_field : float
        Weight for field loss. Default 1.0.
    """

    def __init__(
        self,
        pixel_loss: nn.Module,
        field_loss: nn.Module,
        lambda_pixel: float = 0.2,
        lambda_field: float = 1.0,
    ) -> None:
        super().__init__()
        self.pixel_loss = pixel_loss
        self.field_loss = field_loss
        self.lambda_pixel = lambda_pixel
        self.lambda_field = lambda_field

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model outputs of shape (B, C, H, W).
        masks : torch.Tensor
            Ground truth labels of shape (B, H, W).
        field_ids : torch.Tensor
            Field ID per pixel of shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Combined scalar loss.
        """
        L_pixel = self.pixel_loss(logits, masks)
        L_field = self.field_loss(logits, masks, field_ids)
        return self.lambda_pixel * L_pixel + self.lambda_field * L_field
