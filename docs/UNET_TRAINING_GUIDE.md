# U-Net Training Guide for AgroVision

**For:** Student B/C (Model Implementation & Training)
**Prepared By:** Student A (Data Pipeline)
**Date:** December 2024

---

## 1. What You're Building

A **semantic segmentation model** that takes satellite imagery and outputs a per-pixel crop classification map.

```
INPUT                           OUTPUT
┌─────────────────┐            ┌─────────────────┐
│ Satellite Image │            │ Segmentation    │
│ 12 bands        │  ──U-Net──►│ Mask            │
│ 256×256 pixels  │            │ 14 classes      │
└─────────────────┘            └─────────────────┘
     (12, 256, 256)                (14, 256, 256)
```

---

## 2. What Processing Was Done (Data Pipeline Summary)

### 2.1 Raw Data Source

**AgriFieldNet Competition Dataset** from Radiant MLHub:
- **Location:** Uttar Pradesh & Rajasthan, India
- **Satellite:** Sentinel-2 (ESA)
- **Time:** Single timestamp per tile (cloud-free composite)
- **Original format:** GeoTIFF files organized by tile ID

### 2.2 Band Extraction & Stacking

All **12 Sentinel-2 spectral bands** were extracted and stacked in this order:

| Index | Band | Wavelength | What It Captures |
|-------|------|------------|------------------|
| 0 | B01 | 443 nm | Coastal/Aerosol (atmospheric correction) |
| 1 | B02 | 490 nm | **Blue** (visible) |
| 2 | B03 | 560 nm | **Green** (visible) |
| 3 | B04 | 665 nm | **Red** (visible, chlorophyll absorption) |
| 4 | B05 | 705 nm | Red Edge 1 (vegetation stress) |
| 5 | B06 | 740 nm | Red Edge 2 (canopy structure) |
| 6 | B07 | 783 nm | Red Edge 3 (leaf area index) |
| 7 | B08 | 842 nm | **NIR** (vegetation health, NDVI) |
| 8 | B8A | 865 nm | NIR Narrow (fine vegetation detail) |
| 9 | B09 | 945 nm | Water Vapor |
| 10 | B11 | 1610 nm | **SWIR 1** (soil moisture, crop water) |
| 11 | B12 | 2190 nm | **SWIR 2** (mineral content, drought stress) |

**Why 12 bands instead of RGB?**
- Different crops have unique **spectral signatures** beyond visible light
- NIR bands reveal vegetation health invisible to human eyes
- SWIR bands detect water content and soil properties
- Red Edge bands are highly sensitive to crop type differences

### 2.3 Spatial Resampling

All bands resampled to **256×256 pixels** using:
- **Bilinear interpolation** for spectral data (preserves continuous values)
- **Nearest-neighbor** for labels (preserves discrete class IDs)

### 2.4 Normalization (Z-Score)

Each band was normalized independently using **training set statistics**:

```
normalized_pixel = (raw_pixel - band_mean) / (band_std + 1e-6)
```

**Computed statistics** (from 932 training tiles × 256 × 256 pixels):

| Band | Mean | Std Dev |
|------|------|---------|
| B01 | 42.89 | 3.06 |
| B02 | 38.34 | 3.76 |
| B03 | 41.05 | 4.89 |
| B04 | 42.68 | 6.25 |
| B08 (NIR) | 59.75 | 7.08 |
| B11 (SWIR) | 66.24 | 12.87 |
| B12 (SWIR) | 47.66 | 14.16 |

**Why Z-score normalization?**
- Centers data around 0 (faster convergence)
- Scales all bands to similar ranges (prevents band dominance)
- Standard practice for deep learning

### 2.5 Train/Validation Split

| Split | Tiles | Percentage |
|-------|-------|------------|
| Training | 932 | 80% |
| Validation | 233 | 20% |

- **Split method:** Random at tile level (seed=42 for reproducibility)
- **Important:** Normalization stats computed from training set ONLY (no data leakage)

---

## 3. What the U-Net Will Learn

### 3.1 The Classification Task

The model learns to predict **which crop is growing at each pixel** based on its spectral signature.

**14 Classes:**

| ID | Crop | Training Pixels | % of Data |
|----|------|-----------------|-----------|
| 0 | Background | 76,161,212 | 99.43% |
| 1 | Wheat | 75,118 | 0.098% |
| 2 | Mustard | 46,818 | 0.061% |
| 4 | No Crop/Fallow | 36,397 | 0.047% |
| 9 | Maize | 8,773 | 0.011% |
| 6 | Sugarcane | 5,820 | 0.008% |
| 13 | Gram (Chickpea) | 3,503 | 0.005% |
| 36 | Rice | 3,410 | 0.004% |
| 8 | Garlic | 3,150 | 0.004% |
| 3 | Lentil | 2,883 | 0.004% |
| 15 | Potato | 886 | 0.001% |
| 14 | Coriander | 678 | 0.001% |
| 5 | Green Pea | 531 | 0.001% |
| 16 | Berseem (Clover) | 261 | <0.001% |

### 3.2 What Spectral Patterns the Model Learns

The U-Net will learn to recognize crops by their **spectral signatures**:

```
Spectral Signature Example (simplified):

Wavelength →  Blue  Green  Red   NIR   SWIR
              │     │      │     │     │
Wheat:        ▂▂▂▂▂▃▃▃▃▃▂▂▂▂▂████▅▅▅▅▅
Rice:         ▃▃▃▃▃▄▄▄▄▄▃▃▃▃▃██████████
Sugarcane:    ▂▂▂▂▂▄▄▄▄▄▂▂▂▂▂██████▃▃▃▃▃

Each crop reflects light differently at each wavelength!
```

**Key spectral features:**
- **Chlorophyll** absorbs red (B04), reflects NIR (B08) → healthy vegetation
- **Water content** affects SWIR bands (B11, B12)
- **Leaf structure** influences Red Edge bands (B05-B07)
- **Soil exposure** visible in SWIR when crops are sparse

### 3.3 Spatial Patterns the Model Learns

Beyond spectral signatures, U-Net learns **spatial context**:
- Field boundaries and shapes
- Texture patterns within fields
- Neighboring pixel relationships
- Edge detection between crop types

---

## 4. Critical Challenge: Class Imbalance

**The biggest challenge:** Background is 99.43% of pixels!

### Why This Matters
- Without handling, model will predict "background" everywhere (99.43% accuracy, but useless)
- Rare classes like Berseem (0.0003%) will never be learned

### Solutions Implemented

**1. Class Weights (provided by dataset)**
```python
weights = train_dataset.get_class_weights()
# Background: ~0.007 (low weight)
# Berseem: ~465 (high weight)

criterion = nn.CrossEntropyLoss(weight=weights)
```

**2. Recommended Additional Strategies**
- **Focal Loss:** Down-weights easy examples, focuses on hard ones
- **Dice Loss:** Directly optimizes overlap (IoU-like)
- **Oversampling:** Sample tiles with rare classes more frequently
- **Data Augmentation:** Flip/rotate to increase effective dataset size

---

## 5. How to Use the Dataset

### 5.1 Basic Usage

```python
from backend.src.data.dataset import CropDataset, get_dataloaders

# Create datasets
train_dataset = CropDataset("data/processed", split="train")
val_dataset = CropDataset("data/processed", split="val")

# Check properties
print(f"Training samples: {len(train_dataset)}")      # 932
print(f"Validation samples: {len(val_dataset)}")      # 233
print(f"Input channels: {train_dataset.num_channels}") # 12
print(f"Output classes: {train_dataset.num_classes}")  # 14
```

### 5.2 DataLoader Setup

```python
# Simple way (recommended)
train_loader, val_loader = get_dataloaders(
    "data/processed",
    batch_size=32,
    num_workers=0  # MUST be 0 on Windows/Jupyter!
)

# Manual way (more control)
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True  # Drop incomplete batches
)
```

### 5.3 Batch Format

```python
batch = next(iter(train_loader))

images = batch['image']  # Shape: (B, 12, 256, 256), dtype: float32
masks = batch['mask']    # Shape: (B, 256, 256), dtype: int64

# B = batch size (e.g., 32)
# 12 = spectral bands (channels)
# 256×256 = spatial dimensions
# Mask values: integers 0-36 (class IDs)
```

### 5.4 Data Augmentation

```python
from backend.src.data.dataset import RandomFlipRotate

# Create augmented dataset
transforms = RandomFlipRotate(p_flip=0.5, p_rotate=0.5)
train_dataset = CropDataset(
    "data/processed",
    split="train",
    transforms=transforms
)
```

**Augmentations applied:**
- Horizontal flip (50% chance)
- Vertical flip (50% chance)
- 90° rotations (50% chance, random 1-3 rotations)

### 5.5 Class Weights for Loss

```python
import torch.nn as nn

# Get inverse-frequency weights
weights = train_dataset.get_class_weights()
weights_tensor = torch.tensor(list(weights.values()))

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
```

---

## 6. Model Architecture Requirements

### Input/Output Specification

```python
class UNet(nn.Module):
    def __init__(self):
        # Input: (B, 12, 256, 256) - 12 spectral bands
        # Output: (B, 14, 256, 256) - 14 class logits per pixel
        pass

    def forward(self, x):
        # x shape: (batch, 12, 256, 256)
        # return shape: (batch, 14, 256, 256)
        pass
```

### Architecture Notes

- **First conv layer:** Must accept 12 input channels (not 3!)
- **Final conv layer:** Must output 14 channels (one per class)
- **Output activation:** None (CrossEntropyLoss expects raw logits)
- **Spatial size:** Input and output must both be 256×256

---

## 7. Recommended Training Setup

### Loss Functions (choose one or combine)

```python
# Option 1: Weighted Cross-Entropy
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Focal Loss (better for extreme imbalance)
# Implement or use: kornia.losses.FocalLoss

# Option 3: Dice Loss + CE (common in segmentation)
# loss = 0.5 * dice_loss + 0.5 * ce_loss
```

### Metrics to Track

```python
# Per-class IoU (Intersection over Union)
# Mean IoU (mIoU) - average across all classes
# Per-class F1 score
# Overall pixel accuracy (less useful due to imbalance)
# Confusion matrix
```

### Suggested Hyperparameters (starting point)

```python
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training
batch_size = 32  # Adjust based on GPU memory
epochs = 100     # With early stopping
```

---

## 8. File Locations Reference

| Purpose | Path |
|---------|------|
| Processed images | `data/processed/train_images.npy` |
| Processed masks | `data/processed/train_masks.npy` |
| Normalization stats | `data/processed/normalization_stats.json` |
| Class definitions | `data/processed/class_map.json` |
| Dataset class | `backend/src/data/dataset.py` |
| Config | `config/config.yaml` |

---

## 9. Quick Verification Script

Run this to verify everything is working:

```python
import torch
from backend.src.data.dataset import CropDataset, get_dataloaders

# Load data
train_loader, val_loader = get_dataloaders("data/processed", batch_size=4, num_workers=0)
train_dataset = CropDataset("data/processed", split="train")

# Get batch
batch = next(iter(train_loader))
print(f"Image batch shape: {batch['image'].shape}")  # (4, 12, 256, 256)
print(f"Mask batch shape: {batch['mask'].shape}")    # (4, 256, 256)
print(f"Image dtype: {batch['image'].dtype}")        # float32
print(f"Mask dtype: {batch['mask'].dtype}")          # int64
print(f"Mask unique values: {torch.unique(batch['mask'])}")  # Class IDs

# Class info
print(f"\nClasses: {train_dataset.get_class_names()}")
print(f"Class weights: {train_dataset.get_class_weights()}")
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Task** | Semantic segmentation (pixel-wise classification) |
| **Input** | 12 Sentinel-2 bands, 256×256, normalized |
| **Output** | 14 class probabilities per pixel |
| **Main Challenge** | Extreme class imbalance (99.4% background) |
| **Key Solution** | Class weights + Focal/Dice loss |
| **Dataset** | 932 train / 233 val tiles |

**You're ready to build the U-Net!**
