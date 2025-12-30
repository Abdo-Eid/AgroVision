# Design Decisions and Comprehensive Q&A

This document covers the **design rationale** behind all data pipeline decisions and provides a comprehensive **interview preparation guide**.

---

## Table of Contents

1. [Major Design Decisions](#1-major-design-decisions)
2. [Alternative Approaches Not Taken](#2-alternative-approaches-not-taken)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Potential Interview Questions by Topic](#4-potential-interview-questions-by-topic)
5. [Common "What If" Scenarios](#5-common-what-if-scenarios)
6. [Debugging and Troubleshooting](#6-debugging-and-troubleshooting)
7. [Quick Reference Cheat Sheet](#7-quick-reference-cheat-sheet)

---

## 1. Major Design Decisions

### Decision 1: Pre-process to .npy Instead of Loading GeoTIFFs On-the-fly

**Choice**: Pre-process all data to `.npy` files before training.

| Approach | Pros | Cons |
|----------|------|------|
| **On-the-fly (TorchGeo)** | No preprocessing step; Always fresh data | Slow (~50ms/sample); Needs rasterio |
| **Pre-processed (.npy)** | Fast (~5ms/sample); Simple dependencies | Storage overhead; Preprocessing step |

**Why we chose pre-processing**:
1. **Training speed**: 10× faster data loading
2. **Simplicity**: No complex geospatial dependencies during training
3. **Reproducibility**: Fixed normalization, fixed splits
4. **Portability**: `.npy` files work anywhere with numpy

**Trade-off accepted**: Requires ~4GB disk space for processed data.

---

### Decision 2: Z-Score Normalization Per Band

**Choice**: Normalize each band to mean=0, std=1 using training statistics.

**Alternatives considered**:

| Method | Formula | Issues |
|--------|---------|--------|
| **Min-Max [0,1]** | `(x-min)/(max-min)` | Sensitive to outliers |
| **Min-Max [-1,1]** | Same, rescaled | Same issue |
| **Z-Score** | `(x-mean)/std` | Unbounded range |
| **Per-image norm** | Normalize each image independently | Inconsistent across tiles |

**Why Z-Score**:
1. **Robust to outliers**: Mean/std less affected than min/max
2. **Standard for NNs**: Expected input distribution for most architectures
3. **Consistent**: Same normalization applied to train, val, and inference

**Critical**: Statistics computed from **training set only** to prevent data leakage.

---

### Decision 3: Contiguous Class ID Remapping

**Choice**: Remap raw class IDs `[0,1,2,3,4,5,6,8,9,13,14,15,16,36]` to `[0,1,2,...,13]`.

**Problem with raw IDs**:
```python
# Raw IDs go up to 36, but only 14 classes exist
num_classes = 37  # Wasteful!
model_output = (batch, 37, 256, 256)  # 23 channels never used
```

**Solution**:
```python
# Contiguous IDs: 0 to 13
num_classes = 14  # Efficient!
model_output = (batch, 14, 256, 256)  # All channels used
```

**Implementation**: Look-Up Table (LUT) for O(1) remapping
```python
lut[36] = 13  # Rice maps to index 13
mask = lut[mask]  # Vectorized, fast
```

---

### Decision 4: Welford's Algorithm for Statistics

**Choice**: Use Welford's online algorithm for computing mean/std.

**Why not naive approach**:
```python
# Naive: Requires all data in memory
all_pixels = np.concatenate([load_tile(t).flatten() for t in tiles])
mean = np.mean(all_pixels)  # 3GB+ memory per band!
```

**Welford's algorithm**:
```python
# Online: O(1) memory
for x in data:
    count += 1
    delta = x - mean
    mean += delta / count
    M2 += delta * (x - mean)
variance = M2 / (count - 1)
```

**Benefits**:
1. Constant memory usage
2. Numerically stable (avoids catastrophic cancellation)
3. Single pass through data

---

### Decision 5: Nearest-Neighbor Resampling for Labels

**Choice**: Use `nearest` for labels, `bilinear` for bands.

**Problem with bilinear for labels**:
```
Original: [1, 2]    Bilinear: [1, 1.5, 2]  ← Invalid class 1.5!
```

**Nearest-neighbor preserves discrete values**:
```
Original: [1, 2]    Nearest: [1, 1, 2, 2]  ← All valid classes
```

---

### Decision 6: Store Field IDs Separately

**Choice**: Save `field_ids` arrays alongside `masks`.

**Why**:
- AgriFieldNet challenge evaluates **per-field**, not per-pixel
- Each field has a unique ID
- Needed for field-level loss calculation

**Usage**:
```python
# Per-pixel predictions
pixel_loss = CrossEntropyLoss(predictions, masks)

# Per-field predictions (majority vote per field)
field_predictions = majority_vote(predictions, field_ids)
field_loss = CrossEntropyLoss(field_predictions, field_labels)
```

---

### Decision 7: 80/20 Train/Val Split by Tile

**Choice**: Split at tile level, not pixel level.

**Why not pixel-level split**:
- Pixels in same tile are spatially correlated
- Model would "memorize" tile patterns
- Overfit metric, not generalizable

**Tile-level split**:
- Validation tiles are completely unseen
- Better generalization estimate
- Simulates real-world inference

---

## 2. Alternative Approaches Not Taken

### Alternative 1: Use PyTorch DataLoader with GeoTIFF

**TorchGeo approach**:
```python
from torchgeo.datasets import AgriFieldNet
dataset = AgriFieldNet(paths="data/", ...)
```

**Why not used for production**:
- Slower (50ms vs 5ms per sample)
- Requires rasterio/GDAL during training
- More complex spatial sampling logic

**When TorchGeo is better**:
- Quick experiments
- Need geospatial sampling (random patches from large images)
- Multi-temporal data

---

### Alternative 2: Memory-Mapped Arrays Throughout

**Could have used**:
```python
images = np.load("train_images.npy", mmap_mode="r")
```

**Why we don't by default**:
- Slower access (disk I/O each read)
- Dataset fits in RAM (~4GB)
- Training would be bottlenecked on I/O

**When to use mmap**:
- Dataset > available RAM
- Just validating a few samples
- Memory-constrained environment

---

### Alternative 3: HDF5 or Zarr Instead of .npy

**Could have used**:
```python
import h5py
with h5py.File("data.h5") as f:
    images = f["train_images"][:]
```

**Why .npy is sufficient**:
- Simple, no additional dependencies
- Fast for our data size
- Easy to debug (just `np.load()`)

**When HDF5/Zarr is better**:
- Very large datasets (>100GB)
- Need partial reads (chunks)
- Complex hierarchical data

---

### Alternative 4: Compute Statistics On-the-fly

**Could have done**:
```python
# In DataLoader
def normalize(batch):
    return (batch - batch.mean()) / batch.std()
```

**Why we pre-compute**:
- Per-image normalization is inconsistent
- Batch statistics vary with batch composition
- Need consistent normalization for inference

---

## 3. Data Flow Diagram

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DATA PIPELINE OVERVIEW                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 1: DATA DOWNLOAD                                                         │
│  ════════════════════════════════════════════════════                           │
│                                                                                  │
│    Source Cooperative (Internet)                                                │
│              │                                                                   │
│              ▼                                                                   │
│    ┌─────────────────────────────────────────────┐                              │
│    │  TorchGeo / azcopy                          │                              │
│    │  (download=True)                            │                              │
│    └─────────────────────────────────────────────┘                              │
│              │                                                                   │
│              ▼                                                                   │
│    data/agrifieldnet/                                                           │
│    ├── source/            (12 bands × 1217 tiles)                               │
│    └── train_labels/      (raster + field_ids)                                  │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 2: PREPROCESSING                                                         │
│  ════════════════════════════════════════════════                               │
│                                                                                  │
│    config/config.yaml                                                           │
│    ├── paths                                                                    │
│    ├── bands: [B01, B02, ..., B12]                                             │
│    └── classes: {0: Background, 1: Wheat, ...}                                 │
│              │                                                                   │
│              ▼                                                                   │
│    ┌─────────────────────────────────────────────┐                              │
│    │  prepare_dataset.py                         │                              │
│    │                                             │                              │
│    │  1. Discover valid tiles                    │                              │
│    │  2. Split train/val (80/20, seed=42)       │                              │
│    │  3. Compute band statistics (train only!)  │                              │
│    │  4. For each tile:                         │                              │
│    │     a. Load 12 bands                       │                              │
│    │     b. Resample to 256×256                 │                              │
│    │     c. Z-score normalize                   │                              │
│    │     d. Remap class IDs                     │                              │
│    │  5. Save to .npy                           │                              │
│    └─────────────────────────────────────────────┘                              │
│              │                                                                   │
│              ▼                                                                   │
│    data/processed/                                                              │
│    ├── train_images.npy    (N, 12, 256, 256)                                   │
│    ├── train_masks.npy     (N, 256, 256)                                       │
│    ├── train_field_ids.npy (N, 256, 256)                                       │
│    ├── val_*.npy           (same structure)                                    │
│    ├── normalization_stats.json                                                │
│    └── class_map.json                                                          │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 3: TRAINING                                                              │
│  ════════════════════════════════════════════════                               │
│                                                                                  │
│    ┌─────────────────────────────────────────────┐                              │
│    │  CropDataset                                │                              │
│    │                                             │                              │
│    │  - Loads .npy files                        │                              │
│    │  - Applies transforms (flip/rotate)        │                              │
│    │  - Returns torch tensors                   │                              │
│    └─────────────────────────────────────────────┘                              │
│              │                                                                   │
│              ▼                                                                   │
│    ┌─────────────────────────────────────────────┐                              │
│    │  DataLoader                                 │                              │
│    │                                             │                              │
│    │  - Batches samples                         │                              │
│    │  - Shuffles training                       │                              │
│    │  - Parallel loading (num_workers)          │                              │
│    └─────────────────────────────────────────────┘                              │
│              │                                                                   │
│              ▼                                                                   │
│    Training Loop → Model → Loss → Optimizer                                    │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STAGE 4: INFERENCE                                                             │
│  ════════════════════════════════════════════════                               │
│                                                                                  │
│    New Sentinel-2 imagery                                                       │
│              │                                                                   │
│              ▼                                                                   │
│    ┌─────────────────────────────────────────────┐                              │
│    │  Inference Service                          │                              │
│    │                                             │                              │
│    │  1. Load normalization_stats.json          │                              │
│    │  2. Normalize new data (same stats!)       │                              │
│    │  3. Run model                              │                              │
│    │  4. Map output indices → raw class IDs     │                              │
│    └─────────────────────────────────────────────┘                              │
│              │                                                                   │
│              ▼                                                                   │
│    Prediction map + Statistics                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Potential Interview Questions by Topic

### Data Loading & I/O

| Question | Key Points |
|----------|------------|
| Why use rasterio for GeoTIFFs? | Standard for geospatial; handles CRS, multi-band |
| Why lazy import of rasterio? | Optional dependency; module works without it |
| What's a context manager? | `with` statement; auto-cleanup on exit |
| Why Path over string paths? | OOP, cross-platform, cleaner syntax |

### Normalization

| Question | Key Points |
|----------|------------|
| Why Z-score, not min-max? | Robust to outliers; standard for neural nets |
| Why train stats only? | Prevent data leakage; simulate real inference |
| What's Welford's algorithm? | Online mean/var; O(1) memory; numerically stable |
| What if std=0? | Add epsilon: `(x - mean) / (std + 1e-6)` |

### Class Handling

| Question | Key Points |
|----------|------------|
| Why remap class IDs? | Contiguous for efficient model output |
| What's a LUT? | Look-up table; O(1) array indexing |
| How handle class imbalance? | Class weights in loss; weighted sampling |
| Why int64 for masks? | PyTorch CrossEntropyLoss expects LongTensor |

### Resampling

| Question | Key Points |
|----------|------------|
| Why different methods for bands vs labels? | Bands=continuous (bilinear); Labels=discrete (nearest) |
| What if labels used bilinear? | Creates fractional class IDs; corrupts labels |
| Why 256×256 target? | Matches model input; all bands same size |

### PyTorch Integration

| Question | Key Points |
|----------|------------|
| Why inherit from Dataset? | DataLoader compatibility; __len__, __getitem__ |
| Why return dict, not tuple? | Self-documenting; easier to access by name |
| Why .copy() in __getitem__? | Prevent modifying original array |
| What's pin_memory? | Faster CPU→GPU transfer; pinned RAM |

---

## 5. Common "What If" Scenarios

### What if the download fails?

```python
# Error:
RuntimeError: Failed to download AgriFieldNet

# Solutions:
1. Check internet connection
2. Delete partial download: rm -rf data/agrifieldnet
3. Retry with download=True
4. Manually download from Source Cooperative
```

### What if preprocessing produces wrong shapes?

```python
# Check:
>>> train_images.shape
(880, 12, 256, 256)  # Expected

# If wrong:
# - Check config bands list
# - Verify GeoTIFF files aren't corrupted
# - Check resample target_size
```

### What if normalization is off?

```python
# Check:
>>> train_images.mean(axis=(0,2,3))  # Per-band mean
array([-0.001, 0.002, ...])  # Should be ~0

# If far from 0:
# - Stats computed on different data?
# - Applied wrong stats file?
# - Order of bands changed?
```

### What if classes are missing?

```python
# Check:
>>> np.unique(train_masks)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # All 14

# If missing:
# - Class remapping issue
# - Source data incomplete
# - Label files corrupted
```

### What if training is slow?

```python
# Check DataLoader settings:
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,    # Increase if CPU-bound
    pin_memory=True,  # Enable for GPU
)

# If still slow:
# - Use load_to_memory=True
# - Check disk I/O bottleneck
# - Profile with torch.utils.bottleneck
```

### What if validation loss doesn't decrease?

```python
# Potential data pipeline issues:
1. Data leakage: val stats used in normalization?
2. Wrong split: val and train overlap?
3. Label corruption: masks contain wrong values?
4. Normalization: different stats for train vs val?

# Debug:
>>> train_dataset.norm_stats == val_dataset.norm_stats
True  # Should match!
```

---

## 6. Debugging and Troubleshooting

### Debug Checklist

```
□ Config loads correctly
  >>> config = load_config()
  >>> config["paths"]["data_raw"]

□ Raw data exists
  >>> Path("data/agrifieldnet/source").exists()
  True

□ Preprocessing completed
  >>> Path("data/processed/train_images.npy").exists()
  True

□ Shapes are correct
  >>> np.load("data/processed/train_images.npy").shape
  (N, 12, 256, 256)

□ Normalization is valid
  >>> train_images.mean()
  ~0.0
  >>> train_images.std()
  ~1.0

□ Classes are contiguous
  >>> np.unique(train_masks)
  [0, 1, 2, ..., 13]

□ Dataset loads
  >>> dataset = CropDataset("data/processed")
  >>> len(dataset)
  880

□ DataLoader works
  >>> loader = DataLoader(dataset, batch_size=4)
  >>> batch = next(iter(loader))
  >>> batch["image"].shape
  torch.Size([4, 12, 256, 256])
```

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: config.yaml` | Wrong working directory | `cd` to repo root |
| `UnicodeDecodeError` | Missing encoding | Add `encoding="utf-8"` |
| `ModuleNotFoundError: rasterio` | Not installed | `pip install rasterio` |
| `RuntimeError: num_workers` | Windows multiprocessing | Use `num_workers=0` |
| `CUDA out of memory` | Batch too large | Reduce batch_size |

---

## 7. Quick Reference Cheat Sheet

### File Locations

| File | Purpose |
|------|---------|
| `config/config.yaml` | Master configuration |
| `agrovision_core/.../utils/io.py` | GeoTIFF I/O functions |
| `agrovision_core/.../data/prepare_dataset.py` | Preprocessing script |
| `agrovision_core/.../data/dataset.py` | PyTorch Dataset |
| `data/processed/*.npy` | Preprocessed arrays |
| `data/processed/class_map.json` | Class definitions |
| `data/processed/normalization_stats.json` | Band statistics |

### Key Functions

| Function | Purpose | Location |
|----------|---------|----------|
| `load_band_tiff()` | Load single GeoTIFF band | `io.py` |
| `load_label_tiff()` | Load label mask | `io.py` |
| `load_config()` | Load YAML config | `io.py` |
| `compute_band_statistics()` | Welford's algorithm | `prepare_dataset.py` |
| `build_raw_to_contig_map()` | Class ID mapping | `prepare_dataset.py` |
| `CropDataset.__getitem__()` | Get training sample | `dataset.py` |
| `get_dataloaders()` | Create DataLoaders | `dataset.py` |

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `target_size` | `(256, 256)` | Tile dimensions |
| `num_classes` | `14` | 13 crops + background |
| `num_channels` | `12` | Sentinel-2 bands |
| `val_split` | `0.2` | 20% validation |
| `random_seed` | `42` | Reproducibility |

### Data Shapes

| Array | Shape | Dtype |
|-------|-------|-------|
| `train_images` | `(N, 12, 256, 256)` | `float32` |
| `train_masks` | `(N, 256, 256)` | `int64` |
| `train_field_ids` | `(N, 256, 256)` | `int64` |
| batch `image` | `(B, 12, 256, 256)` | `torch.float32` |
| batch `mask` | `(B, 256, 256)` | `torch.int64` |

### Common Commands

```bash
# Run preprocessing
python -m agrovision_core.data.prepare_dataset

# Start Jupyter
jupyter notebook notebooks/data-pipeline.ipynb

# Check data sizes
du -sh data/processed/*.npy
```

```python
# Quick data check
import numpy as np
images = np.load("data/processed/train_images.npy")
print(f"Shape: {images.shape}")
print(f"Mean: {images.mean():.4f}")
print(f"Std: {images.std():.4f}")
```

---

## Final Summary

### Your Data Pipeline Responsibilities (Student A)

1. **I/O Utilities** (`io.py`): Loading GeoTIFFs, resampling, config management
2. **Preprocessing** (`prepare_dataset.py`): Convert raw data to .npy format
3. **Dataset Class** (`dataset.py`): PyTorch integration for training
4. **Configuration** (`config.yaml`): Centralized settings
5. **Notebook** (`data-pipeline.ipynb`): Documentation and demonstration

### Key Takeaways

1. **Speed**: Pre-processing saves 10× training time
2. **Correctness**: Proper normalization and class remapping are critical
3. **Reproducibility**: Fixed seeds and saved splits ensure consistent results
4. **Simplicity**: Minimal dependencies for production training

Good luck with your interview!
