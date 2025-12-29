# Field-Aware Training for AgriFieldNet

This document explains the field-aware training implementation for the AgroVision crop mapping system. It covers the problem we faced, our solution, and implementation details for future developers.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution: Dual Loss Training](#2-solution-dual-loss-training)
3. [Implementation Details](#3-implementation-details)
4. [Bug Fixes](#4-bug-fixes)
5. [Configuration Reference](#5-configuration-reference)
6. [Metrics & Expected Results](#6-metrics--expected-results)
7. [File Reference](#7-file-reference)

---

## 1. Problem Statement

### 1.1 The Sparse Label Challenge

AgriFieldNet is a crop classification dataset derived from Sentinel-2 satellite imagery. Unlike typical segmentation datasets where most pixels are labeled, **AgriFieldNet only labels surveyed agricultural fields**.

```
Typical segmentation dataset:     AgriFieldNet dataset:
┌─────────────────────────┐       ┌─────────────────────────┐
│█████████████████████████│       │                         │
│█████████████████████████│       │    ██                   │
│█████████████████████████│       │         ███             │
│█████████████████████████│       │                  █      │
│█████████████████████████│       │              ██         │
└─────────────────────────┘       └─────────────────────────┘
     ~100% labeled                    ~0.25% labeled
```

**Key statistics:**
- Only ~0.25% of pixels have crop labels
- ~99.75% of pixels are background/unlabeled (class 0)
- This is **by design** - only surveyed fields are annotated

### 1.2 The Objective Mismatch

The AgriFieldNet challenge evaluates predictions **per-field**, not per-pixel:

| Training Objective | Evaluation Metric |
|-------------------|-------------------|
| Pixel-level cross-entropy | Field-level accuracy |

This creates a fundamental mismatch:
- **Pixel-level training** optimizes for individual pixel predictions
- **Challenge scoring** aggregates predictions per field and compares to ground truth

**Result:** A model trained purely on pixel loss will:
- Plateau at low mIoU (~0.07)
- Have poor field-level accuracy
- Overfit to the dominant background class

---

## 2. Solution: Dual Loss Training

### 2.1 The Combined Loss Function

We implement a dual loss that optimizes both objectives:

```
L_total = λ_pixel × L_pixel + λ_field × L_field
        = 0.2 × FocalCrossEntropy + 1.0 × FieldCrossEntropy
```

### 2.2 Pixel Loss (λ = 0.2)

**Purpose:** Maintain clean segmentation boundaries for overlay visualization.

**Implementation:** `FocalCrossEntropyLoss` with `ignore_index=0`

```python
# From losses.py:12-106
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=None, reduction="mean"):
        # Focal loss down-weights easy examples
        # ignore_index=0 excludes background from loss
```

**Why Focal Loss?**
- Down-weights easy/well-classified pixels (focal term: `(1-p)^γ`)
- Focuses learning on hard examples near field boundaries
- `γ=2.0` provides good balance between easy/hard examples

### 2.3 Field Loss (λ = 1.0)

**Purpose:** Align training with the challenge scoring metric.

**Implementation:** `FieldLoss` class aggregates pixel predictions per field.

```python
# From losses.py:109-187
class FieldLoss(nn.Module):
    def forward(self, logits, masks, field_ids):
        for each field_id in batch:
            1. Get all pixels belonging to this field
            2. Average logits over field pixels → field prediction
            3. Majority vote (excluding background) → field ground truth
            4. Compute cross-entropy for this field
        return mean(field_losses)
```

**How it works:**

```
Input tile with 3 fields:
┌─────────────────────────┐
│  Field A    Field B     │
│  (wheat)    (maize)     │
│                         │
│      Field C            │
│      (rice)             │
└─────────────────────────┘

Step 1: Group pixels by field_id
  Field A: pixels at positions [(0,0), (0,1), ...]
  Field B: pixels at positions [(0,10), (0,11), ...]
  Field C: pixels at positions [(3,5), (3,6), ...]

Step 2: Average logits per field
  Field A logits: mean([logits[0,0], logits[0,1], ...]) → [0.1, 0.8, 0.05, ...]

Step 3: Get ground truth (majority vote, excluding class 0)
  Field A labels: [1, 1, 1, 0, 1] → majority = 1 (wheat)

Step 4: Cross-entropy per field
  CE(predicted=[0.1, 0.8, ...], target=1)
```

### 2.4 Weight Rationale

| Weight | Value | Reasoning |
|--------|-------|-----------|
| `λ_pixel` | 0.2 | Lower weight - secondary objective for visualization |
| `λ_field` | 1.0 | Higher weight - primary objective matching evaluation |

**Why this ratio?**
- Field loss dominates because it matches the evaluation metric
- Pixel loss prevents boundary degradation in visual overlays
- Empirically validated: these weights balance visualization quality with metric optimization

---

## 3. Implementation Details

### 3.1 Data Pipeline

```
┌──────────────────┐     ┌─────────────┐     ┌───────────┐
│ prepare_dataset  │     │  dataset    │     │   train   │
│      .py         │     │    .py      │     │    .py    │
└────────┬─────────┘     └──────┬──────┘     └─────┬─────┘
         │                      │                   │
         ▼                      ▼                   ▼
  Load field_ids.tiff    Load .npy files     Pass to FieldLoss
         │                      │                   │
         ▼                      ▼                   ▼
  Save field_ids.npy     Return in batch     Compute field CE
```

### 3.2 Field ID Extraction (`prepare_dataset.py`)

**Location:** `agrovision_core/src/agrovision_core/data/prepare_dataset.py`

The AgriFieldNet dataset includes `field_ids.tiff` files that identify which pixels belong to which field. Each unique non-zero value represents a distinct field.

```python
# Lines 245-251: Load field IDs from TIFF
field_ids_path = labels_dir / "field_ids.tiff"
if field_ids_path.exists():
    with rasterio.open(field_ids_path) as src:
        field_ids = src.read(1)  # Shape: (256, 256)
else:
    field_ids = np.zeros((256, 256), dtype=np.int32)
```

**Output files:**
- `data/processed/train_field_ids.npy` - Shape: `(N_train, 256, 256)`
- `data/processed/val_field_ids.npy` - Shape: `(N_val, 256, 256)`

### 3.3 Dataset Integration (`dataset.py`)

**Location:** `agrovision_core/src/agrovision_core/data/dataset.py`

The `CropDataset` class loads field IDs alongside images and masks:

```python
# Lines 74, 86-91: Load field_ids
field_ids_path = data_dir / f"{split}_field_ids.npy"
if field_ids_path.exists():
    if mmap_mode:
        self.field_ids = np.load(field_ids_path, mmap_mode="r")
    else:
        self.field_ids = np.load(field_ids_path)
```

**Batch output:**
```python
# Lines 150-154: Return field_ids in batch dict
return {
    "image": image_tensor,      # (C, H, W)
    "mask": mask_tensor,        # (H, W)
    "field_ids": field_ids,     # (H, W) - int64
}
```

**Augmentation handling:**

The `RandomFlipRotate` transform applies identical transformations to `field_ids` as it does to images and masks, ensuring spatial consistency.

### 3.4 Weighted Sampling

**Problem:** Most tiles have few or no labeled pixels, leading to:
- Wasted computation on uninformative tiles
- Bias toward predicting background

**Solution:** Oversample tiles with more labeled pixels.

```python
# dataset.py:226-261
def get_sample_weights(self, power: float = 1.0) -> torch.Tensor:
    """Compute per-tile sampling weights based on labeled pixel fraction."""
    fractions = []
    for mask in self.masks:
        labeled = np.sum(mask != 0)  # Non-background pixels
        total = mask.size
        fractions.append(labeled / total)

    # Apply power transform: higher power = stronger preference
    weights = (fractions + 1e-6) ** power
    return weights / weights.sum() * len(weights)
```

**Usage in training:**
```python
# train.py:87-119
if use_weighted_sampling:
    sample_weights = train_dataset.get_sample_weights(power=weight_power)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(dataset, sampler=sampler, ...)
```

**Effect of `weight_power`:**

| Power | Effect |
|-------|--------|
| 0.0 | Uniform sampling (no preference) |
| 1.0 | Linear preference for labeled tiles |
| 1.5 | Moderate preference (default) |
| 2.0+ | Strong preference for highly-labeled tiles |

### 3.5 Training Loop Changes (`train.py`)

**Location:** `agrovision_core/src/agrovision_core/train/train.py`

**Loss setup:**
```python
# Lines 327-343
pixel_loss = FocalCrossEntropyLoss(
    gamma=focal_gamma,
    alpha=class_weights,
    ignore_index=ignore_index,
)

if use_field_loss:
    field_loss = FieldLoss(reduction="mean")
    criterion = CombinedLoss(
        pixel_loss=pixel_loss,
        field_loss=field_loss,
        lambda_pixel=lambda_pixel,
        lambda_field=lambda_field,
    )
else:
    criterion = pixel_loss
```

**Training step:**
```python
# Lines 406-428
for batch in train_loader:
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    field_ids = batch["field_ids"].to(device)  # NEW

    logits = model(images)

    if use_field_loss:
        loss = criterion(logits, masks, field_ids)  # Combined loss
    else:
        loss = criterion(logits, masks)  # Pixel-only loss
```

**Model selection:**
```python
# Lines 458-476
if model_selection == "field_ce":
    # Lower is better
    is_best = val_metrics["field_ce"] < best_metric
elif model_selection == "mIoU":
    # Higher is better
    is_best = val_metrics["mIoU"] > best_metric
```

---

## 4. Bug Fixes

### 4.1 The `ignore_index=0` Bug

**Location:** `evaluate.py:83-85`, `train.py`

**The Problem:**

```python
# OLD CODE - INCORRECT
ignore_index = training_cfg.get("ignore_index", None)
if ignore_index is not None:
    ignore_index = int(ignore_index)
    if ignore_index <= 0:  # BUG: This disables ignore_index=0!
        ignore_index = None
```

**Why this was wrong:**
- Class 0 is background/unlabeled in AgriFieldNet
- We WANT to ignore class 0 in loss computation
- The condition `ignore_index <= 0` incorrectly treated 0 as invalid

**The Fix:**

```python
# NEW CODE - CORRECT
ignore_index = training_cfg.get("ignore_index", None)
if ignore_index is not None:
    ignore_index = int(ignore_index)
    # Only invalidate truly negative values (e.g., -1 used as sentinel)
    # 0 is a VALID ignore_index for background/unlabeled class
    if ignore_index < 0:
        ignore_index = None
```

**Impact:**
- Before fix: Model trained on background pixels, wasting capacity
- After fix: Model focuses on actual crop classes

---

## 5. Configuration Reference

All field-aware training parameters are in `config/config.yaml`:

```yaml
training:
  # Loss configuration
  use_field_loss: true        # Enable combined loss
  lambda_pixel: 0.2           # Weight for pixel-level loss
  lambda_field: 1.0           # Weight for field-level loss

  # Masking
  ignore_index: 0             # Background class - IGNORE in loss
  min_labeled_fraction: 0.001 # Skip batches with <0.1% labels
  background_weight: 0.0      # Class weight for background
  focal_gamma: 2.0            # Focal loss focusing parameter

  # Weighted sampling
  use_weighted_sampling: true # Oversample tiles with more labels
  weight_power: 1.5           # Preference strength (higher = stronger)

  # Model selection
  model_selection: field_ce   # "field_ce" (lower better) or "mIoU" (higher better)
```

**Parameter Details:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_field_loss` | bool | `true` | Enable combined pixel + field loss |
| `lambda_pixel` | float | `0.2` | Weight for pixel loss in combined loss |
| `lambda_field` | float | `1.0` | Weight for field loss in combined loss |
| `ignore_index` | int | `0` | Class index to exclude from loss computation |
| `min_labeled_fraction` | float | `0.001` | Skip batches with fewer labeled pixels |
| `background_weight` | float | `0.0` | Class weight for background (0 = ignore) |
| `focal_gamma` | float | `2.0` | Focal loss gamma (higher = more focus on hard examples) |
| `use_weighted_sampling` | bool | `true` | Enable weighted tile sampling |
| `weight_power` | float | `1.5` | Power for sampling weight calculation |
| `model_selection` | str | `"field_ce"` | Criterion for saving best model |

---

## 6. Metrics & Expected Results

### 6.1 Training Metrics

The training loop logs these metrics per epoch:

| Metric | Description | Direction |
|--------|-------------|-----------|
| `train_loss` | Combined loss (pixel + field) | Lower is better |
| `val_mIoU` | Pixel-level mean Intersection over Union | Higher is better |
| `val_field_accuracy` | Fraction of fields correctly classified | Higher is better |
| `val_field_ce` | Field-level cross-entropy loss | Lower is better |
| `val_num_fields` | Number of fields in validation set | Info only |

### 6.2 Expected Values by Training Stage

| Metric | Epoch 1-5 | Epoch 10-20 | Epoch 30-50 |
|--------|-----------|-------------|-------------|
| `train_loss` | 2.0 - 3.0 | 1.0 - 1.5 | 0.5 - 1.0 |
| `val_field_ce` | 2.5 - 3.0 | 1.5 - 2.0 | 1.2 - 1.8 |
| `val_field_acc` | 0.10 - 0.20 | 0.30 - 0.45 | 0.40 - 0.55 |
| `val_mIoU` | 0.05 - 0.10 | 0.15 - 0.25 | 0.20 - 0.35 |

### 6.3 Healthy Training Indicators

**Good signs:**
- `train_loss` decreasing steadily
- `val_field_ce` decreasing (this is the primary metric)
- `val_field_acc` increasing toward 0.4-0.5+

**Warning signs:**
- `val_field_ce` stuck or increasing after epoch 10 → possible overfitting
- `val_field_acc` stuck at ~0.08 → model predicting single class
- Large gap between `train_loss` and validation metrics → overfitting

### 6.4 Comparison to Baselines

| Approach | Expected Field Accuracy |
|----------|------------------------|
| Random guessing (13 classes) | ~7.7% |
| Pixel-only training (old) | ~15-25% |
| Field-aware training (new) | ~40-55% |
| 1st place ensemble (competition) | ~56% |

---

## 7. File Reference

### Modified Files

| File | Path | Key Changes |
|------|------|-------------|
| `losses.py` | `agrovision_core/src/.../train/losses.py` | Added `FieldLoss`, `CombinedLoss` classes; added `ignore_index` to `FocalCrossEntropyLoss` |
| `train.py` | `agrovision_core/src/.../train/train.py` | Dual loss training loop, field metrics computation, weighted sampling, model selection |
| `dataset.py` | `agrovision_core/src/.../data/dataset.py` | Load `field_ids`, `get_sample_weights()` method, transform updates |
| `prepare_dataset.py` | `agrovision_core/src/.../data/prepare_dataset.py` | Extract and save `field_ids.npy` files |
| `evaluate.py` | `agrovision_core/src/.../train/evaluate.py` | Fixed `ignore_index=0` bug |
| `config.yaml` | `config/config.yaml` | Added field-aware training parameters |

### New Files

| File | Purpose |
|------|---------|
| `AgroVision_Training_Colab.ipynb` | Google Colab notebook for GPU training |
| `data/processed/train_field_ids.npy` | Field IDs for training tiles |
| `data/processed/val_field_ids.npy` | Field IDs for validation tiles |

### Code Location Quick Reference

| Component | File | Lines |
|-----------|------|-------|
| `FocalCrossEntropyLoss` | `losses.py` | 12-106 |
| `FieldLoss` | `losses.py` | 109-187 |
| `CombinedLoss` | `losses.py` | 190-247 |
| `get_sample_weights()` | `dataset.py` | 226-261 |
| `compute_field_metrics()` | `train.py` | 158-194 |
| Weighted sampling setup | `train.py` | 87-119 |
| Training loop with field loss | `train.py` | 406-428 |

---

## Summary

The field-aware training implementation addresses the fundamental mismatch between pixel-level training and field-level evaluation in the AgriFieldNet challenge. By combining pixel and field losses with appropriate weights (0.2 and 1.0 respectively), we:

1. **Align training with evaluation** - Field loss matches challenge scoring
2. **Maintain visualization quality** - Pixel loss preserves clean boundaries
3. **Handle sparse labels** - Weighted sampling focuses on informative tiles
4. **Fix critical bugs** - `ignore_index=0` now works correctly

Expected improvement: Field accuracy from ~15-25% (pixel-only) to ~40-55% (field-aware).
