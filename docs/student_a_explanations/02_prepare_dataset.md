# Dataset Preparation Pipeline - Complete Documentation

**File**: `agrovision_core/src/agrovision_core/data/prepare_dataset.py`

**Purpose**: Converts raw AgriFieldNet GeoTIFF files into PyTorch-ready `.npy` files with proper normalization and class remapping.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Function: `load_config()`](#3-function-load_config)
4. [Function: `build_raw_to_contig_map()`](#4-function-build_raw_to_contig_map)
5. [Function: `get_valid_tile_ids()`](#5-function-get_valid_tile_ids)
6. [Function: `create_train_val_split()`](#6-function-create_train_val_split)
7. [Function: `compute_band_statistics()`](#7-function-compute_band_statistics)
8. [Function: `load_tile()`](#8-function-load_tile)
9. [Function: `generate_npy_files()`](#9-function-generate_npy_files)
10. [Function: `save_splits()`](#10-function-save_splits)
11. [Function: `save_metadata()`](#11-function-save_metadata)
12. [Function: `main()`](#12-function-main)
13. [Complete Pipeline Flow](#13-complete-pipeline-flow)
14. [Interview Questions & Answers](#14-interview-questions--answers)

---

## 1. Module Overview

### What This Script Does

This is the **main preprocessing pipeline** that transforms raw satellite data into ML-ready format:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           INPUT (Raw Data)                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  data/agrifieldnet/                                                          │
│  ├── source/                    # 1,217 tile directories                     │
│  │   └── ref_agrifieldnet_..._001c1/                                        │
│  │       ├── ..._B01_10m.tif   # 12 spectral bands per tile                 │
│  │       ├── ..._B02_10m.tif                                                │
│  │       └── ... (12 bands)                                                 │
│  └── train_labels/                                                          │
│      ├── ..._001c1.tif         # Crop class labels (0, 1, 2, ... 36)       │
│      └── ..._001c1_field_ids.tif  # Field boundary IDs                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING STEPS                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  1. Discover valid tiles (have both imagery + labels)                        │
│  2. Split into train/val sets (80/20, seed=42)                              │
│  3. Compute normalization statistics from training set ONLY                  │
│  4. Load each tile → Stack 12 bands → Z-score normalize                     │
│  5. Remap class IDs: [0,1,2,3,4,5,6,8,9,13,14,15,16,36] → [0,1,2,...,13]  │
│  6. Save as .npy arrays for fast loading                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT (Processed Data)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  data/processed/                                                             │
│  ├── train_images.npy     # (N_train, 12, 256, 256) float32 normalized      │
│  ├── train_masks.npy      # (N_train, 256, 256) int64 contiguous IDs        │
│  ├── train_field_ids.npy  # (N_train, 256, 256) int64 field boundaries      │
│  ├── val_images.npy       # (N_val, 12, 256, 256) float32                   │
│  ├── val_masks.npy        # (N_val, 256, 256) int64                         │
│  ├── val_field_ids.npy    # (N_val, 256, 256) int64                         │
│  ├── normalization_stats.json  # Per-band mean & std                        │
│  └── class_map.json       # Class definitions & ID mappings                 │
│                                                                              │
│  data/splits/                                                                │
│  ├── train_ids.csv        # List of training tile IDs                       │
│  └── val_ids.csv          # List of validation tile IDs                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why This Script Exists

| Problem | Solution |
|---------|----------|
| GeoTIFFs are slow to load | Pre-process into `.npy` (10-100× faster) |
| Bands have different scales | Z-score normalization per band |
| Class IDs are non-contiguous (0,1,2,...,8,...,36) | Remap to contiguous (0-13) |
| Need reproducible splits | Fixed random seed, saved to CSV |
| Memory management | Process tiles one at a time |

---

## 2. Imports and Dependencies

```python
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm

from ..utils.io import (
    get_band_filepath,
    get_label_filepath,
    get_tile_ids_from_source,
    load_band_tiff,
    load_label_tiff,
)
```

### Import Analysis

| Import | Source | Purpose |
|--------|--------|---------|
| `json` | stdlib | Write metadata JSON files |
| `Path` | stdlib | Cross-platform path handling |
| `Optional` | stdlib | Type hint for optional parameters |
| `numpy` | third-party | Array operations |
| `yaml` | third-party | Read config.yaml |
| `tqdm` | third-party | Progress bars during processing |
| `..utils.io.*` | local | GeoTIFF loading functions (from io.py) |

### Why `tqdm`?

```python
# Without tqdm - no progress feedback
for tile_id in tile_ids:
    process(tile_id)
# User sees nothing for potentially 30+ minutes!

# With tqdm - progress bar
for tile_id in tqdm(tile_ids, desc="Processing"):
    process(tile_id)
# Output: Processing: 45%|████████████              | 547/1217 [12:34<15:23]
```

---

## 3. Function: `load_config()`

```python
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
```

### Why Default Path?

The default `"config/config.yaml"` assumes the script is run from the **repository root**:
```bash
cd AgroVision/
python -m agrovision_core.data.prepare_dataset
```

### Why `encoding="utf-8"`?

**Critical for Windows!** The config contains Arabic text:
```yaml
classes:
  1:
    name: Wheat
    name_ar: قمح  # Arabic - requires UTF-8
```

Without explicit encoding on Windows:
```
UnicodeDecodeError: 'charmap' codec can't decode byte 0xd9...
```

### Interview Question

**Q: Why not use the `load_config()` from `io.py`?**

A: This is actually a slight code duplication. The `io.py` version uses `resolve_path()` for flexibility, while this one is simpler. In a refactored version, you could:
```python
from ..utils.io import load_config  # Reuse from io.py
```

---

## 4. Function: `build_raw_to_contig_map()`

```python
def build_raw_to_contig_map(config: dict) -> dict[int, int]:
    """
    Build a mapping from raw class IDs to contiguous IDs 0..N-1.

    Example:
      raw_ids: [0,1,2,3,4,5,6,8,9,13,14,15,16,36]
      contig : [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    """
    raw_ids = sorted(int(k) for k in config["classes"].keys())
    return {raw_id: i for i, raw_id in enumerate(raw_ids)}
```

### The Problem: Non-Contiguous Class IDs

AgriFieldNet uses these raw class IDs:
```
0 (Background), 1 (Wheat), 2 (Mustard), 3 (Lentil), 4 (No Crop),
5 (Green Pea), 6 (Sugarcane), 8 (Garlic), 9 (Maize), 13 (Gram),
14 (Coriander), 15 (Potato), 16 (Berseem), 36 (Rice)
```

**Notice**: IDs 7, 10, 11, 12, 17-35 don't exist!

### Why Is This a Problem?

```python
# PyTorch CrossEntropyLoss expects classes 0 to C-1
num_classes = 37  # If we used raw IDs, need 0-36 range
predictions = model(images)  # Shape: (batch, 37, 256, 256)

# But we only have 14 actual classes!
# We'd waste memory on 23 never-used channels
```

### The Solution: Contiguous Remapping

```python
raw_ids = [0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36]
#          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓   ↓   ↓   ↓   ↓
contig  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

| Raw ID | Crop | Contiguous ID |
|--------|------|---------------|
| 0 | Background | 0 |
| 1 | Wheat | 1 |
| 2 | Mustard | 2 |
| 3 | Lentil | 3 |
| 4 | No Crop | 4 |
| 5 | Green Pea | 5 |
| 6 | Sugarcane | 6 |
| **8** | Garlic | **7** |
| **9** | Maize | **8** |
| **13** | Gram | **9** |
| **14** | Coriander | **10** |
| **15** | Potato | **11** |
| **16** | Berseem | **12** |
| **36** | Rice | **13** |

### How the Mapping is Built

```python
raw_ids = sorted(int(k) for k in config["classes"].keys())
# sorted() ensures consistent ordering: [0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36]

# enumerate gives (index, value) pairs:
# (0, 0), (1, 1), (2, 2), ..., (7, 8), (8, 9), ..., (13, 36)

return {raw_id: i for i, raw_id in enumerate(raw_ids)}
# {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 13: 9, 14: 10, 15: 11, 16: 12, 36: 13}
```

### Interview Questions

**Q: Why `sorted()` the keys?**

Dictionary iteration order is preserved in Python 3.7+, but config file order might vary. Sorting ensures:
1. **Consistent mapping** across runs
2. **Predictable class indices** (class 0 always maps to 0)

**Q: What if a new class is added?**

Adding a new class (e.g., ID 17) would:
1. Increase num_classes from 14 to 15
2. Insert a new mapping (17 → new_index)
3. Require model retraining (output layer changes)

**Q: Could you use a lookup table instead of a dictionary?**

Yes! And we do - see `generate_npy_files()` which creates a numpy LUT for faster remapping:
```python
lut = np.zeros(max_raw + 1, dtype=np.int64)
for raw_id, contig_id in raw_to_contig.items():
    lut[raw_id] = contig_id
# Now: lut[36] → 13 (O(1) lookup)
```

---

## 5. Function: `get_valid_tile_ids()`

```python
def get_valid_tile_ids(config: dict) -> list[str]:
    """
    Get tile IDs that have both source imagery and labels.
    """
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    labels_dir = Path(config["paths"]["data_raw"]) / "train_labels"

    # Get all tile IDs from source
    all_tile_ids = get_tile_ids_from_source(source_dir)

    # Filter to only those with labels
    valid_ids = []
    for tile_id in all_tile_ids:
        label_path = get_label_filepath(labels_dir, tile_id, "raster")
        if label_path.exists():
            valid_ids.append(tile_id)

    return valid_ids
```

### Why Filter for Valid Tiles?

Not all tiles have labels:
- **Source imagery**: 1,217 tiles (all tiles have satellite data)
- **Training labels**: ~1,100 tiles (some tiles are test-only)
- **Test labels**: ~200 tiles (no crop labels, only field IDs)

### Data Flow

```
all_tile_ids (from source dir)    valid_ids (have labels)
┌────────────────────────────┐    ┌────────────────────────┐
│ 001c1, 002a5, 003b7, ...  │    │ 001c1, 002a5, 003b7   │
│ test_001, test_002, ...   │ →  │ ... (only training    │
│ (total: 1217)             │    │      tiles remain)    │
└────────────────────────────┘    └────────────────────────┘
```

### Interview Question

**Q: Why not just iterate over the labels directory?**

You could! But this approach:
1. Uses `get_tile_ids_from_source()` which we already have
2. Validates that both imagery AND labels exist
3. Would catch cases where labels exist but imagery is missing

Alternative implementation:
```python
# Alternative: iterate labels directory directly
for label_file in labels_dir.glob("*_labels_train_*.tif"):
    if "_field_ids" not in label_file.name:
        tile_id = extract_tile_id(label_file.name)
        valid_ids.append(tile_id)
```

---

## 6. Function: `create_train_val_split()`

```python
def create_train_val_split(
    tile_ids: list[str],
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split tile IDs into training and validation sets.
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(tile_ids))

    n_val = int(len(tile_ids) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ids = [tile_ids[i] for i in train_indices]
    val_ids = [tile_ids[i] for i in val_indices]

    return train_ids, val_ids
```

### Step-by-Step Breakdown

```python
# Step 1: Set random seed for reproducibility
np.random.seed(42)

# Step 2: Create shuffled indices
# If we have 100 tiles: indices = [47, 12, 89, 3, 56, ...]
indices = np.random.permutation(len(tile_ids))

# Step 3: Calculate split sizes
# val_split=0.2 means 20% for validation
n_val = int(len(tile_ids) * val_split)  # e.g., 20 out of 100

# Step 4: Split indices
val_indices = indices[:n_val]      # First 20 shuffled indices → validation
train_indices = indices[n_val:]    # Remaining 80 → training

# Step 5: Map indices back to tile IDs
train_ids = [tile_ids[i] for i in train_indices]
val_ids = [tile_ids[i] for i in val_indices]
```

### Why Random Seed = 42?

`42` is a common convention (reference to "The Hitchhiker's Guide to the Galaxy"). The actual value doesn't matter, but using a **fixed seed ensures reproducibility**:

```python
# Run 1:
np.random.seed(42)
np.random.permutation(5)  # [2, 1, 4, 0, 3]

# Run 2 (same seed):
np.random.seed(42)
np.random.permutation(5)  # [2, 1, 4, 0, 3] - SAME!

# Run 3 (different seed):
np.random.seed(123)
np.random.permutation(5)  # [4, 0, 2, 3, 1] - Different
```

### Interview Questions

**Q: Why shuffle before splitting instead of just taking first 80%?**

Without shuffling, the split would be **deterministic based on file order**:
- First tiles alphabetically → training
- Last tiles alphabetically → validation

This could introduce **systematic bias** if tile IDs correlate with geographic location or time.

**Q: Why use `np.random.permutation()` instead of `random.shuffle()`?**

| Method | In-place? | Returns | NumPy integration |
|--------|-----------|---------|-------------------|
| `random.shuffle(list)` | Yes | None | No |
| `np.random.permutation(n)` | No | New array | Yes |

`permutation` returns a new array, which is cleaner for our use case.

**Q: What if we wanted stratified splitting (preserve class distribution)?**

Current approach: **Random split by tile** (simpler, faster)

Stratified split would require:
1. Load all labels first
2. Count classes per tile
3. Assign tiles to train/val while balancing class counts

```python
from sklearn.model_selection import train_test_split

# Example of stratified split by dominant class
tile_classes = [get_dominant_class(tid) for tid in tile_ids]
train_ids, val_ids = train_test_split(
    tile_ids, test_size=0.2, stratify=tile_classes, random_state=42
)
```

---

## 7. Function: `compute_band_statistics()`

```python
def compute_band_statistics(
    tile_ids: list[str],
    config: dict,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Compute per-band mean and standard deviation from training tiles.
    """
```

### Full Implementation

```python
def compute_band_statistics(
    tile_ids: list[str],
    config: dict,
    sample_size: Optional[int] = None,
) -> dict:
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    bands = [b["name"] for b in config["bands"]]

    # Sample tiles if requested (for faster computation)
    if sample_size and sample_size < len(tile_ids):
        np.random.seed(42)
        sampled_ids = np.random.choice(tile_ids, sample_size, replace=False)
    else:
        sampled_ids = tile_ids

    print(f"Computing statistics from {len(sampled_ids)} tiles...")

    # Accumulate statistics using Welford's online algorithm
    stats = {band: {"count": 0, "mean": 0.0, "M2": 0.0} for band in bands}

    for tile_id in tqdm(sampled_ids, desc="Computing band statistics"):
        for band in bands:
            band_path = get_band_filepath(source_dir, tile_id, band)
            if not band_path.exists():
                continue

            data = load_band_tiff(band_path)
            pixels = data.flatten()

            # Welford's online algorithm for mean and variance
            for x in pixels:
                stats[band]["count"] += 1
                delta = x - stats[band]["mean"]
                stats[band]["mean"] += delta / stats[band]["count"]
                delta2 = x - stats[band]["mean"]
                stats[band]["M2"] += delta * delta2

    # Finalize statistics
    result = {}
    for band in bands:
        count = stats[band]["count"]
        if count > 1:
            variance = stats[band]["M2"] / (count - 1)
            result[band] = {
                "mean": float(stats[band]["mean"]),
                "std": float(np.sqrt(variance)),
                "count": int(count),
            }
        else:
            result[band] = {"mean": 0.0, "std": 1.0, "count": 0}

    return result
```

### Welford's Online Algorithm Explained

**The Problem**: Computing mean and standard deviation of billions of pixels.

**Naive approach** (requires storing all values):
```python
all_pixels = []
for tile in tiles:
    all_pixels.extend(load_band_tiff(tile).flatten())
mean = np.mean(all_pixels)      # Need ALL pixels in memory
std = np.std(all_pixels)
```

For 1000 tiles × 256×256 × 12 bands × 4 bytes = **3 GB per band!**

**Welford's algorithm** (constant memory):
```python
count = 0
mean = 0.0
M2 = 0.0  # Sum of squared differences from mean

for x in data:
    count += 1
    delta = x - mean
    mean += delta / count  # Running mean
    delta2 = x - mean      # New delta after mean update
    M2 += delta * delta2   # Accumulate variance

variance = M2 / (count - 1)  # Unbiased variance (Bessel's correction)
std = sqrt(variance)
```

### Mathematical Proof of Welford's Algorithm

For running mean update:
```
After n observations, mean_n = (sum of first n) / n

When we add x_{n+1}:
  mean_{n+1} = (sum of first n + x_{n+1}) / (n+1)
             = (n * mean_n + x_{n+1}) / (n+1)
             = mean_n + (x_{n+1} - mean_n) / (n+1)
             = mean_n + delta / count  ✓
```

### Why Training Set Only?

**Critical**: Statistics computed from **training tiles only**, not validation!

If we included validation data:
1. **Data leakage**: Validation set influences normalization
2. **Inflated performance**: Model "sees" validation distribution
3. **Invalid evaluation**: Results don't generalize

```
Correct:                      Wrong (data leakage):
┌─────────────────────┐       ┌─────────────────────┐
│   Training Tiles    │       │     All Tiles       │
│  ↓                  │       │  ↓                  │
│  Compute mean/std   │       │  Compute mean/std   │ ← includes val!
│  ↓                  │       │  ↓                  │
│  Normalize train    │       │  Normalize all      │
│  ↓                  │       │  ↓                  │
│  Normalize val      │       │  Train & validate   │
│  (using train stats)│       │  (biased results)   │
└─────────────────────┘       └─────────────────────┘
```

### Example Output

```json
{
  "B01": {"mean": 1923.45, "std": 412.33, "count": 67108864},
  "B02": {"mean": 1456.78, "std": 523.67, "count": 67108864},
  "B03": {"mean": 1234.56, "std": 489.12, "count": 67108864},
  "B04": {"mean": 1567.89, "std": 612.45, "count": 67108864},
  ...
}
```

### Interview Questions

**Q: Why use Welford's algorithm instead of two-pass mean/std?**

| Method | Memory | Passes | Numerical Stability |
|--------|--------|--------|---------------------|
| Two-pass | O(n) | 2 | Good |
| Naive one-pass | O(1) | 1 | **Poor** (catastrophic cancellation) |
| Welford | O(1) | 1 | **Excellent** |

**Q: What is "catastrophic cancellation"?**

Naive variance: `E[X²] - E[X]²`

When values are large but variance is small:
```python
# Example: values around 1,000,000 with small variance
E[X²] = 1,000,000,000,000.1234
E[X]² = 1,000,000,000,000.0000
Variance = 0.1234

# But with floating point:
E[X²] ≈ 1,000,000,000,000.125  # Rounded
E[X]² ≈ 1,000,000,000,000.000
Variance ≈ 0.125  # Wrong!

# Or worse, negative variance due to rounding!
```

Welford avoids this by computing differences from the running mean.

**Q: What's the `sample_size` parameter for?**

For very large datasets, you can estimate statistics from a sample:
```python
stats = compute_band_statistics(train_ids, config, sample_size=100)
# Uses only 100 random tiles instead of all 1000
```

This trades accuracy for speed. In practice, sampling 10-20% of tiles gives reliable estimates.

---

## 8. Function: `load_tile()`

```python
def load_tile(
    tile_id: str,
    config: dict,
    norm_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and normalize a single tile.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (image, mask, field_ids) where:
        - image: (12, 256, 256) float32 normalized
        - mask: (256, 256) int64 class IDs
        - field_ids: (256, 256) int64 field IDs per pixel
    """
```

### Full Implementation

```python
def load_tile(
    tile_id: str,
    config: dict,
    norm_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    labels_dir = Path(config["paths"]["data_raw"]) / "train_labels"
    bands = [b["name"] for b in config["bands"]]
    target_size = (256, 256)

    # Load and normalize bands
    band_data = []
    for band in bands:
        band_path = get_band_filepath(source_dir, tile_id, band)
        if band_path.exists():
            data = load_band_tiff(band_path, target_size)
            # Z-score normalization
            mean = norm_stats[band]["mean"]
            std = norm_stats[band]["std"]
            data = (data - mean) / (std + 1e-6)
        else:
            # If band is missing, use zeros
            data = np.zeros(target_size, dtype=np.float32)
        band_data.append(data)

    image = np.stack(band_data, axis=0).astype(np.float32)  # (12, 256, 256)

    # Load label mask
    label_path = get_label_filepath(labels_dir, tile_id, "raster")
    mask = load_label_tiff(label_path, target_size)  # (256, 256)

    # Load field IDs
    field_ids_path = get_label_filepath(labels_dir, tile_id, "field_ids")
    if field_ids_path.exists():
        field_ids = load_label_tiff(field_ids_path, target_size)
    else:
        field_ids = np.zeros(target_size, dtype=np.int64)

    return image, mask, field_ids
```

### Z-Score Normalization Explained

```python
data = (data - mean) / (std + 1e-6)
```

**What it does**: Transforms data to have approximately mean=0 and std=1.

```
Before normalization:              After normalization:
Values: [1000, 2000, 3000]        Values: [-1.22, 0.0, 1.22]
Mean: 2000                         Mean: ~0
Std: 816.5                         Std: ~1
```

**Why `+ 1e-6`?**: Prevents division by zero if std=0 (constant band).

### Why Normalize at All?

| Without Normalization | With Normalization |
|----------------------|-------------------|
| B01 range: [0, 4000] | B01 range: [-2, +2] |
| B04 range: [0, 8000] | B04 range: [-2, +2] |
| Different gradients per band | Balanced gradients |
| Harder to train | Faster convergence |

Neural networks work best when inputs are centered around 0 with similar scales.

### Band Stacking

```python
image = np.stack(band_data, axis=0)
```

```
band_data = [B01_array, B02_array, ..., B12_array]
            (256,256)  (256,256)       (256,256)
                          ↓
                      np.stack(axis=0)
                          ↓
            ┌─────────────────────────────┐
            │        image                 │
            │   Shape: (12, 256, 256)      │
            │   Axis 0: Bands              │
            │   Axis 1: Height             │
            │   Axis 2: Width              │
            └─────────────────────────────┘
```

### Interview Questions

**Q: Why handle missing bands with zeros?**

Some tiles might have corrupted or missing band files. Options:
1. **Skip entire tile**: Lose data
2. **Fill with zeros**: Keep tile, model learns to handle
3. **Fill with mean**: Less impact on normalization

Zeros are simple and make the missing band obvious in the data.

**Q: What if we normalized min-max (0-1) instead of z-score?**

| Z-Score | Min-Max |
|---------|---------|
| `(x - mean) / std` | `(x - min) / (max - min)` |
| Unbounded range | Fixed [0, 1] range |
| Preserves distribution | Sensitive to outliers |
| Standard for NNs | Common for images |

Z-score is preferred for satellite data because:
- Different tiles have different min/max
- Outliers (clouds, shadows) would distort min-max scaling

---

## 9. Function: `generate_npy_files()`

```python
def generate_npy_files(
    tile_ids: list[str],
    config: dict,
    norm_stats: dict,
    output_prefix: str,
) -> tuple[int, dict]:
    """
    Generate .npy files for a set of tiles.

    Returns
    -------
    tuple[int, dict]
        (num_tiles, class_counts) - number of tiles and pixel counts per class
    """
```

### Full Implementation with Annotations

```python
def generate_npy_files(
    tile_ids: list[str],
    config: dict,
    norm_stats: dict,
    output_prefix: str,  # "train" or "val"
) -> tuple[int, dict]:
    output_dir = Path(config["paths"]["data_processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tiles = len(tile_ids)
    n_bands = len(config["bands"])  # 12
    target_size = 256

    # Pre-allocate arrays (more efficient than appending)
    images = np.zeros((n_tiles, n_bands, target_size, target_size), dtype=np.float32)
    masks = np.zeros((n_tiles, target_size, target_size), dtype=np.int64)
    field_ids_arr = np.zeros((n_tiles, target_size, target_size), dtype=np.int64)

    class_counts = {}

    # Build LUT for fast class remapping
    raw_to_contig = build_raw_to_contig_map(config)
    max_raw = max(raw_to_contig.keys())  # 36
    lut = np.zeros(max_raw + 1, dtype=np.int64)  # Size 37
    for raw_id, contig_id in raw_to_contig.items():
        lut[raw_id] = contig_id

    print(f"Processing {n_tiles} tiles for {output_prefix} set...")
    for i, tile_id in enumerate(tqdm(tile_ids, desc=f"Generating {output_prefix}")):
        # Load tile data
        image, mask, field_ids = load_tile(tile_id, config, norm_stats)

        # Remap mask from raw IDs -> contiguous IDs
        mask = lut[mask]  # Vectorized lookup!

        images[i] = image
        masks[i] = mask
        field_ids_arr[i] = field_ids

        # Count class pixels
        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            cls = int(cls)
            class_counts[cls] = class_counts.get(cls, 0) + int(cnt)

    # Save arrays
    np.save(output_dir / f"{output_prefix}_images.npy", images)
    np.save(output_dir / f"{output_prefix}_masks.npy", masks)
    np.save(output_dir / f"{output_prefix}_field_ids.npy", field_ids_arr)

    print(f"Saved {output_prefix}_images.npy: shape {images.shape}")
    print(f"Saved {output_prefix}_masks.npy: shape {masks.shape}")
    print(f"Saved {output_prefix}_field_ids.npy: shape {field_ids_arr.shape}")

    return n_tiles, class_counts
```

### LUT (Lookup Table) for Fast Remapping

**Without LUT** (slow):
```python
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        mask[i, j] = raw_to_contig[mask[i, j]]  # Dict lookup per pixel
# Time: O(H × W) dict lookups = 256 × 256 = 65,536 lookups
```

**With LUT** (fast):
```python
lut = np.zeros(max_raw + 1, dtype=np.int64)
for raw_id, contig_id in raw_to_contig.items():
    lut[raw_id] = contig_id

mask = lut[mask]  # Vectorized numpy indexing
# Time: O(1) numpy operation (internally optimized)
```

**Visualization**:
```
LUT array:
Index:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ..., 36]
Value:  [0, 1, 2, 3, 4, 5, 6, 0, 7, 8,  0,  0,  0,  9, 10, 11, 12, ..., 13]
                           ↑        ↑
                     unmapped IDs → 0 (but they don't occur in data)

Original mask:      [[36, 1, 8],      Remapped mask:    [[13, 1, 7],
                     [2, 36, 9],   →                     [2, 13, 8],
                     [6, 8, 1]]                          [6, 7, 1]]
```

### Memory Pre-allocation

```python
# Pre-allocation (efficient):
images = np.zeros((n_tiles, 12, 256, 256), dtype=np.float32)
for i in range(n_tiles):
    images[i] = load_tile(...)  # Direct assignment

# Appending (inefficient):
images = []
for i in range(n_tiles):
    images.append(load_tile(...))  # Memory reallocation each time!
images = np.stack(images)
```

Pre-allocation is **10-100× faster** for large arrays.

### Memory Calculation

For 1000 tiles:
```
images: 1000 × 12 × 256 × 256 × 4 bytes = 3.15 GB
masks:  1000 × 256 × 256 × 8 bytes      = 0.52 GB
field_ids: same as masks                 = 0.52 GB
Total: ~4.2 GB RAM needed
```

### Interview Questions

**Q: Why save as `.npy` instead of keeping GeoTIFFs?**

| Aspect | GeoTIFF | .npy |
|--------|---------|------|
| Load time | ~50ms per file | ~5ms per tile |
| Total for 1000 tiles | ~50 seconds | <1 second |
| Dependencies | rasterio, GDAL | numpy only |
| Geospatial info | Preserved | Lost |
| File size | Compressed | Uncompressed |

For training, we don't need geospatial info and speed matters most.

**Q: What's `np.unique(mask, return_counts=True)`?**

```python
mask = [[0, 1, 1],
        [1, 2, 2],
        [2, 2, 0]]

unique, counts = np.unique(mask, return_counts=True)
# unique = [0, 1, 2]
# counts = [2, 3, 4]  # Two 0s, three 1s, four 2s
```

Used for class distribution statistics.

---

## 10. Function: `save_splits()`

```python
def save_splits(
    train_ids: list[str],
    val_ids: list[str],
    config: dict,
) -> None:
    """Save train/val splits to CSV files."""
    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train_ids.csv", "w") as f:
        f.write("tile_id\n")
        for tile_id in train_ids:
            f.write(f"{tile_id}\n")

    with open(splits_dir / "val_ids.csv", "w") as f:
        f.write("tile_id\n")
        for tile_id in val_ids:
            f.write(f"{tile_id}\n")

    print(f"Saved splits: {len(train_ids)} train, {len(val_ids)} val")
```

### Output Example

**train_ids.csv**:
```csv
tile_id
001c1
002a5
003b7
...
```

### Why Save Splits?

1. **Reproducibility**: Know exactly which tiles were in each set
2. **Debugging**: Check if a specific tile was train or val
3. **Re-running**: Skip split logic if files exist
4. **Auditing**: Document data handling for publication

---

## 11. Function: `save_metadata()`

```python
def save_metadata(
    norm_stats: dict,
    class_counts: dict,
    config: dict,
) -> None:
    """Save normalization stats and class map."""
```

### Full Implementation

```python
def save_metadata(
    norm_stats: dict,
    class_counts: dict,
    config: dict,
) -> None:
    output_dir = Path(config["paths"]["data_processed"])

    # Save normalization stats
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    # Create class map from config
    raw_ids = sorted(int(k) for k in config["classes"].keys())
    raw_to_contig = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    contig_to_raw = {i: raw_id for raw_id, i in raw_to_contig.items()}

    class_map = {
        "classes": {},
        "raw_ids": raw_ids,
        "raw_to_contig": {str(k): int(v) for k, v in raw_to_contig.items()},
        "contig_to_raw": {str(k): int(v) for k, v in contig_to_raw.items()},
        "num_classes": len(raw_ids),
        "class_counts": class_counts,
    }

    for cls_id, cls_info in config["classes"].items():
        class_map["classes"][str(cls_id)] = {
            "id": int(cls_id),
            "name": cls_info["name"],
            "name_ar": cls_info.get("name_ar", ""),
            "color": cls_info["color"],
        }

    with open(output_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"Saved normalization_stats.json and class_map.json")
```

### class_map.json Structure

```json
{
  "classes": {
    "0": {"id": 0, "name": "Background", "name_ar": "خلفية", "color": [0, 0, 0]},
    "1": {"id": 1, "name": "Wheat", "name_ar": "قمح", "color": [255, 200, 0]},
    ...
  },
  "raw_ids": [0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36],
  "raw_to_contig": {"0": 0, "1": 1, ..., "36": 13},
  "contig_to_raw": {"0": 0, "1": 1, ..., "13": 36},
  "num_classes": 14,
  "class_counts": {"0": 45000000, "1": 12000000, ...}
}
```

### Why Both Mappings?

| Mapping | Use Case |
|---------|----------|
| `raw_to_contig` | Preprocessing: raw GeoTIFF → training mask |
| `contig_to_raw` | Inference: model output → original class ID |

---

## 12. Function: `main()`

```python
def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("AgroVision Data Preparation Pipeline")
    print("=" * 60)

    # 1. Load config
    config = load_config()

    # 2. Get valid tile IDs
    tile_ids = get_valid_tile_ids(config)

    # 3. Create train/val split
    val_split = config["training"]["val_split"]
    random_seed = config["training"]["random_seed"]
    train_ids, val_ids = create_train_val_split(tile_ids, val_split, random_seed)

    # 4. Save splits
    save_splits(train_ids, val_ids, config)

    # 5. Compute normalization statistics
    norm_stats = compute_band_statistics(train_ids, config)

    # 6. Generate training set
    n_train, train_class_counts = generate_npy_files(
        train_ids, config, norm_stats, "train"
    )

    # 7. Generate validation set
    n_val, val_class_counts = generate_npy_files(
        val_ids, config, norm_stats, "val"
    )

    # 8. Combine class counts and save metadata
    all_class_counts = {}
    for cls, cnt in train_class_counts.items():
        all_class_counts[cls] = all_class_counts.get(cls, 0) + cnt
    for cls, cnt in val_class_counts.items():
        all_class_counts[cls] = all_class_counts.get(cls, 0) + cnt

    save_metadata(norm_stats, all_class_counts, config)

    # 9. Print summary
    print("\nPipeline Complete!")
```

### Pipeline Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│  1. Load config.yaml                                        │
│     └─ Gets paths, band definitions, class definitions      │
├─────────────────────────────────────────────────────────────┤
│  2. Discover valid tiles                                    │
│     └─ ~1,100 tiles with both imagery and labels           │
├─────────────────────────────────────────────────────────────┤
│  3. Split into train/val (80/20)                           │
│     └─ ~880 train, ~220 val                                │
├─────────────────────────────────────────────────────────────┤
│  4. Save splits to CSV                                      │
│     └─ For reproducibility                                  │
├─────────────────────────────────────────────────────────────┤
│  5. Compute band statistics (TRAIN ONLY!)                   │
│     └─ Welford's algorithm for mean/std                     │
├─────────────────────────────────────────────────────────────┤
│  6. Generate train_*.npy files                             │
│     └─ Load → Normalize → Remap → Stack → Save             │
├─────────────────────────────────────────────────────────────┤
│  7. Generate val_*.npy files                               │
│     └─ Same process, same normalization stats              │
├─────────────────────────────────────────────────────────────┤
│  8. Save metadata (norm_stats.json, class_map.json)        │
│     └─ For inference pipeline                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Complete Pipeline Flow

```
                                    config/config.yaml
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main()                                          │
│                                                                              │
│  ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────┐ │
│  │  load_config()   │ → │get_valid_tile_ids()│ → │create_train_val_split│ │
│  └──────────────────┘    └───────────────────┘    └──────────────────────┘ │
│           │                       │                         │               │
│           │                       │                         ▼               │
│           │                       │              ┌──────────────────────┐   │
│           │                       │              │    save_splits()     │   │
│           │                       │              │  train_ids.csv       │   │
│           │                       │              │  val_ids.csv         │   │
│           │                       │              └──────────────────────┘   │
│           │                       │                                         │
│           ▼                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                compute_band_statistics(train_ids)                     │  │
│  │                                                                       │  │
│  │    for each band in 12 bands:                                        │  │
│  │        for each tile in train_ids:                                   │  │
│  │            load_band_tiff() → Welford's algorithm → running mean/std │  │
│  │                                                                       │  │
│  │    Output: {B01: {mean, std}, B02: {mean, std}, ...}                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │               generate_npy_files(train_ids, norm_stats)               │  │
│  │                                                                       │  │
│  │    Build LUT for class remapping: [0,1,2,...,8,...,36] → [0,1,...13] │  │
│  │                                                                       │  │
│  │    for each tile:                                                    │  │
│  │        load_tile() → image (12,256,256), mask (256,256), field_ids   │  │
│  │        mask = lut[mask]  ← vectorized remapping                      │  │
│  │        store in pre-allocated arrays                                 │  │
│  │                                                                       │  │
│  │    np.save(train_images.npy, train_masks.npy, train_field_ids.npy)   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │               generate_npy_files(val_ids, norm_stats)                 │  │
│  │                                                                       │  │
│  │    Same process as training, using SAME normalization stats          │  │
│  │                                                                       │  │
│  │    np.save(val_images.npy, val_masks.npy, val_field_ids.npy)         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         save_metadata()                               │  │
│  │                                                                       │  │
│  │    normalization_stats.json ← band means and stds                    │  │
│  │    class_map.json ← class definitions, ID mappings, counts           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Interview Questions & Answers

### Conceptual Questions

**Q1: Why preprocess into .npy files instead of loading GeoTIFFs on-the-fly?**

A: **Speed and simplicity**:
- GeoTIFF loading: ~50ms per tile × 12 bands = 600ms per sample
- .npy loading: ~5ms per tile
- 100× faster training iteration
- No need for rasterio/GDAL during training

**Q2: What would break if you computed normalization stats from validation set too?**

A: **Data leakage**:
- Model indirectly "sees" validation distribution
- Evaluation metrics would be artificially inflated
- Results wouldn't generalize to truly unseen data

**Q3: Why use contiguous class IDs instead of raw IDs?**

A: **Memory and correctness**:
- Raw IDs go up to 36 → need 37 output channels
- Only 14 classes actually exist → 23 wasted channels
- PyTorch CrossEntropyLoss works best with 0..N-1 indices

### Technical Questions

**Q4: Explain Welford's algorithm vs naive mean/std calculation.**

A: **Naive**: `mean = sum(x)/n`, `var = sum((x-mean)²)/n`
- Requires two passes or storing all values
- Catastrophic cancellation with large values

**Welford**: Updates running mean and M2 incrementally
- Single pass, O(1) memory
- Numerically stable for any value range

**Q5: What's the purpose of the LUT in `generate_npy_files()`?**

A: **Vectorized class remapping**:
```python
# Without LUT: O(H×W) dict lookups
# With LUT: O(1) numpy indexing
lut = [0, 1, 2, 3, 4, 5, 6, 0, 7, 8, 0, 0, 0, 9, 10, 11, 12, ..., 13]
mask = lut[mask]  # All pixels remapped in one operation
```

**Q6: Why is field_ids stored separately from masks?**

A: **Different purposes**:
- `mask`: Crop class per pixel (for training loss)
- `field_ids`: Field boundary IDs (for field-level evaluation)

The AgriFieldNet challenge evaluates per-field accuracy, not per-pixel.

### Debugging Questions

**Q7: A tile has all zeros after normalization. What happened?**

A: Possible causes:
1. Band files are missing → filled with zeros
2. All pixels have the same value → std=0 → division issue
3. GeoTIFF is corrupted

Check with:
```python
raw = load_band_tiff(path)
print(f"Raw range: [{raw.min()}, {raw.max()}]")
print(f"Std: {raw.std()}")
```

**Q8: Training loss is NaN. What could be wrong in preprocessing?**

A: Check:
1. **Division by zero**: `std + 1e-6` should prevent this
2. **Invalid class IDs**: Raw IDs not in mapping → LUT returns 0
3. **Corrupted files**: Some pixels might be NaN/Inf

Debugging:
```python
print(np.isnan(images).sum())  # Check for NaNs
print(np.isinf(images).sum())  # Check for Infs
print(np.unique(masks))        # Check class ID range
```

### Design Questions

**Q9: How would you parallelize this pipeline?**

A: Options:
1. **Multiprocessing per tile**:
   ```python
   from multiprocessing import Pool
   with Pool(8) as p:
       results = p.map(load_and_process_tile, tile_ids)
   ```

2. **Dask for larger-than-memory**:
   ```python
   import dask.array as da
   images = da.from_delayed(...)
   ```

3. **Band-parallel statistics**:
   ```python
   # Compute each band's stats independently
   with ThreadPoolExecutor(12) as ex:
       stats = list(ex.map(compute_band_stats, bands))
   ```

**Q10: What changes for a different dataset (e.g., EuroSAT)?**

A:
1. Update `get_tile_ids_from_source()` for new naming convention
2. Update `get_band_filepath()` for new directory structure
3. Modify band list in config (EuroSAT has different bands)
4. Update class definitions in config
5. Everything else stays the same!
