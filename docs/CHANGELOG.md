# Changelog

All notable changes to the AgroVision project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - Field-Aware Training

This release implements field-aware training to align the U-Net model training with the AgriFieldNet challenge scoring metric.

### Added

#### Loss Functions (`losses.py`)
- **`FieldLoss`** class - Field-level cross-entropy loss that aggregates pixel predictions per field and computes CE at the field level
- **`CombinedLoss`** class - Wrapper combining pixel and field losses with configurable weights (`L = 0.2 * L_pixel + 1.0 * L_field`)
- **`ignore_index`** parameter to `FocalCrossEntropyLoss` - Allows excluding background class (0) from loss computation

#### Data Pipeline (`dataset.py`, `prepare_dataset.py`)
- **Field ID extraction** - Load `field_ids.tiff` from AgriFieldNet labels directory
- **`train_field_ids.npy`** and **`val_field_ids.npy`** - Preprocessed field ID arrays
- **`get_sample_weights()`** method in `CropDataset` - Computes per-tile sampling weights based on labeled pixel fraction
- **Field ID augmentation** - `RandomFlipRotate` now applies identical transforms to field_ids

#### Training (`train.py`)
- **Weighted sampling** - `WeightedRandomSampler` to oversample tiles with more labeled pixels
- **`compute_field_metrics()`** function - Computes field-level accuracy and cross-entropy
- **Field metrics logging** - `val_field_accuracy`, `val_field_ce`, `val_num_fields` per epoch
- **Configurable model selection** - Choose best model by `field_ce` (lower better) or `mIoU` (higher better)

#### Configuration (`config.yaml`)
- `use_field_loss` - Enable/disable combined loss
- `lambda_pixel` - Weight for pixel loss (default: 0.2)
- `lambda_field` - Weight for field loss (default: 1.0)
- `use_weighted_sampling` - Enable/disable weighted tile sampling
- `weight_power` - Strength of sampling preference (default: 1.5)
- `model_selection` - Criterion for best model (`field_ce` or `mIoU`)

#### Colab Support
- **`AgroVision_Training_Colab.ipynb`** - Complete notebook for training on Google Colab with T4 GPU

### Changed

#### Data Pipeline
- **`CropDataset.__getitem__`** - Now returns `field_ids` tensor in batch dictionary
- **`load_tile()`** in `prepare_dataset.py` - Returns tuple of `(image, mask, field_ids)` instead of `(image, mask)`
- **`generate_npy_files()`** - Creates additional `{split}_field_ids.npy` files

#### Training Loop
- **Loss computation** - Now passes `field_ids` to `CombinedLoss` when `use_field_loss=True`
- **Validation** - Extended `evaluate()` to compute and return field-level metrics
- **Epoch logging** - Includes field accuracy, field CE, and number of fields

### Fixed

#### Critical Bug: `ignore_index=0` Disabled
**Location:** `evaluate.py:83-85`, `train.py`

**Problem:** The condition `if ignore_index <= 0` incorrectly disabled `ignore_index` when set to 0 (background class).

**Before:**
```python
if ignore_index <= 0:
    ignore_index = None  # BUG: Disables valid ignore_index=0
```

**After:**
```python
if ignore_index < 0:
    ignore_index = None  # Only disable negative sentinels like -1
```

**Impact:** Model now correctly ignores background pixels during loss computation, focusing learning capacity on actual crop classes.

---

## [Previous] - Initial Implementation

### Features
- U-Net semantic segmentation model
- Sentinel-2 12-band input support
- FocalCrossEntropyLoss for class imbalance
- FastAPI backend with inference endpoints
- React + shadcn frontend for visualization

---

## Migration Guide

### Updating from Pixel-Only Training

1. **Regenerate preprocessed data** to include field IDs:
   ```bash
   uv run python -m agrovision_core.data.prepare_dataset
   ```

2. **Update config.yaml** (or use new defaults):
   ```yaml
   training:
     use_field_loss: true
     lambda_pixel: 0.2
     lambda_field: 1.0
     use_weighted_sampling: true
     model_selection: field_ce
   ```

3. **Retrain the model**:
   ```bash
   uv run python -m agrovision_core.train.train --run-name field_aware_v1
   ```

### Expected Improvements

| Metric | Pixel-Only | Field-Aware |
|--------|------------|-------------|
| Field Accuracy | 15-25% | 40-55% |
| Field CE | 2.5+ | 1.2-1.8 |
| mIoU | 5-10% | 20-35% |
