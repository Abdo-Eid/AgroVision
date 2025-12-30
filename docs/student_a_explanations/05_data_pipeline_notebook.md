# Data Pipeline Notebook - Complete Documentation

**File**: `notebooks/data-pipeline.ipynb`

**Purpose**: Interactive Jupyter notebook that demonstrates the complete data pipeline from raw AgriFieldNet data to PyTorch-ready datasets.

---

## Table of Contents

1. [Notebook Overview](#1-notebook-overview)
2. [Part 1: TorchGeo Integration](#2-part-1-torchgeo-integration)
3. [Part 2: Custom Preprocessing Pipeline](#3-part-2-custom-preprocessing-pipeline)
4. [Part 3: Data Verification](#4-part-3-data-verification)
5. [Part 4: Visualization](#5-part-4-visualization)
6. [Part 5: PyTorch Integration](#6-part-5-pytorch-integration)
7. [Cell-by-Cell Explanation](#7-cell-by-cell-explanation)
8. [Interview Questions & Answers](#8-interview-questions--answers)

---

## 1. Notebook Overview

### What This Notebook Does

This notebook serves as **both documentation and execution** of the data pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Notebook Flow                                         │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Download   │ →  │   Process   │ →  │   Verify    │ →  │  Visualize  │  │
│  │  AgriField  │    │   to .npy   │    │   Data      │    │  Samples    │  │
│  │  Net Data   │    │   files     │    │   Quality   │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
│  Key Outputs:                                                                │
│  ├── data/processed/train_images.npy                                        │
│  ├── data/processed/train_masks.npy                                         │
│  ├── data/processed/val_images.npy                                          │
│  └── data/processed/class_map.json                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Two-Part Structure

| Part | Purpose | Tools Used |
|------|---------|------------|
| Part 1 | Quick experimentation with TorchGeo | TorchGeo library |
| Part 2 | Production preprocessing | Custom `prepare_dataset.py` |

---

## 2. Part 1: TorchGeo Integration

### Header and Dataset Overview

```markdown
# AgriFieldNet Data Pipeline

## Dataset Overview

| Property | Value |
|----------|-------|
| **Source** | Sentinel-2 satellite imagery |
| **Resolution** | 10m (visible/NIR), 20m (red edge/SWIR) |
| **Tile Size** | 256 × 256 pixels |
| **Total Tiles** | 1,217 tiles |
| **Training Fields** | 5,551 fields |
| **Test Fields** | 1,530 fields |
| **Crop Classes** | 13 classes |
```

**Why include this table?**
- Quick reference for dataset properties
- Documents the expected data format
- Helps understand preprocessing choices

### Cell 1: Working Directory Setup

```python
from pathlib import Path
import os

parent = Path.cwd().parent
os.chdir(parent)
print(f"Changed working directory to parent: {Path.cwd()}")
```

**Expected Output**:
```
Changed working directory to parent: d:\Faculty\Level 4\CNN\AgroVision
```

**Why change to parent directory?**
- Notebooks are in `notebooks/` subdirectory
- Config and data paths are relative to repo root
- Ensures `config/config.yaml` can be found

### Cell 2: Import TorchGeo

```python
from torchgeo.datasets import AgriFieldNet
from torchgeo.datamodules import AgriFieldNetDataModule
```

**What TorchGeo provides**:
- Pre-built dataset classes for geospatial data
- Handles GeoTIFF loading, CRS, and geospatial sampling
- PyTorch Lightning integration

### Cell 3: Download Dataset

```python
dataset = AgriFieldNet(
    paths="data/agrifieldnet",
    download=True,  # This triggers the download
)
```

**What happens**:
1. Creates `data/agrifieldnet/` directory
2. Downloads from Source Cooperative via `azcopy`
3. Extracts Sentinel-2 bands and labels
4. Indexes all files for fast access

**Download size**: ~2-3 GB

### Cell 4: Verify Dataset

```python
print(f"Dataset CRS: {dataset.crs}")
print(f"Dataset bands: {dataset.bands}")
print(f"Number of files: {len(dataset.files)}")
```

**Expected Output**:
```
Dataset CRS: EPSG:32643
Dataset bands: ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
Number of files: 1217
```

**What to verify**:
| Property | Expected | Issue if Different |
|----------|----------|-------------------|
| CRS | EPSG:32643 (UTM Zone 43N) | Wrong projection |
| Bands | 12 bands | Missing spectral data |
| Files | ~1217 | Incomplete download |

### Cell 5: TorchGeo DataModule

```python
datamodule = AgriFieldNetDataModule(
    batch_size=32,
    patch_size=256,
    num_workers=4,
    paths="data/agrifieldnet",
)
datamodule.setup("fit")
```

**Parameters explained**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 32 | Samples per training step |
| `patch_size` | 256 | Size of extracted patches |
| `num_workers` | 4 | Parallel data loading (use 0 on Windows!) |
| `paths` | ... | Location of downloaded data |

**What `setup("fit")` does**:
1. Indexes all GeoTIFF files
2. Creates spatial index for efficient sampling
3. Prepares RandomGeoSampler for training

---

## 3. Part 2: Custom Preprocessing Pipeline

### Why Custom Pipeline?

The TorchGeo approach is great for **quick experiments**, but has limitations:

| TorchGeo | Custom Pipeline |
|----------|-----------------|
| Loads GeoTIFFs on-the-fly | Pre-processes to .npy |
| ~50ms per sample | ~5ms per sample |
| Requires rasterio at runtime | Pure numpy/PyTorch |
| Dynamic spatial sampling | Fixed tile-based splits |
| Complex geospatial features | Simple array indexing |

### Cell 6: Run Preprocessing

```python
import sys
sys.path.insert(0, ".")  # Add project root to path

from backend.src.data.prepare_dataset import main as run_preprocessing

run_preprocessing()
```

**What happens**:
```
============================================================
AgroVision Data Preparation Pipeline
============================================================

Loaded config from config/config.yaml

Scanning for valid tiles...
Found 1100 tiles with both imagery and labels

Split: 880 train, 220 val
Saved splits: 880 train, 220 val

Computing normalization statistics...
Computing statistics from 880 tiles...
Computing band statistics: 100%|██████████| 880/880 [05:23<00:00]

============================================================
Processing 880 tiles for train set...
Generating train: 100%|██████████| 880/880 [12:45<00:00]
Saved train_images.npy: shape (880, 12, 256, 256)
Saved train_masks.npy: shape (880, 256, 256)
Saved train_field_ids.npy: shape (880, 256, 256)

============================================================
Processing 220 tiles for val set...
Generating val: 100%|██████████| 220/220 [03:12<00:00]
Saved val_images.npy: shape (220, 12, 256, 256)

Saved normalization_stats.json and class_map.json

============================================================
Pipeline Complete!
============================================================
```

---

## 4. Part 3: Data Verification

### Cell 7: Load and Verify Processed Data

```python
import numpy as np
import json
from pathlib import Path

# Load processed data
data_dir = Path("data/processed")

train_images = np.load(data_dir / "train_images.npy")
train_masks = np.load(data_dir / "train_masks.npy")
val_images = np.load(data_dir / "val_images.npy")
val_masks = np.load(data_dir / "val_masks.npy")

# Print shapes and dtypes
print("=" * 50)
print("Dataset Shapes:")
print("=" * 50)
print(f"train_images: {train_images.shape} ({train_images.dtype})")
print(f"train_masks:  {train_masks.shape} ({train_masks.dtype})")
print(f"val_images:   {val_images.shape} ({val_images.dtype})")
print(f"val_masks:    {val_masks.shape} ({val_masks.dtype})")
```

**Expected Output**:
```
==================================================
Dataset Shapes:
==================================================
train_images: (880, 12, 256, 256) (float32)
train_masks:  (880, 256, 256) (int64)
val_images:   (220, 12, 256, 256) (float32)
val_masks:    (220, 256, 256) (int64)
```

**What to verify**:
| Check | Expected | Issue if Different |
|-------|----------|-------------------|
| train_images shape | `(N, 12, 256, 256)` | Wrong band count or tile size |
| dtype images | `float32` | Memory inefficiency or precision issues |
| dtype masks | `int64` | PyTorch CrossEntropyLoss compatibility |
| val/train ratio | ~20% / 80% | Split configuration issue |

### Normalization Verification

```python
print("Normalization Verification (per-band):")
print("=" * 50)
print(f"{'Band':<6} {'Mean':>10} {'Std':>10}")
print("-" * 28)
for i in range(train_images.shape[1]):
    mean = train_images[:, i, :, :].mean()
    std = train_images[:, i, :, :].std()
    print(f"Band {i:<2} {mean:>10.4f} {std:>10.4f}")
```

**Expected Output** (approximately):
```
Band 0     -0.0012     1.0023
Band 1      0.0008     0.9987
Band 2     -0.0003     1.0001
...
```

**What to verify**:
- Mean should be close to **0** (±0.01)
- Std should be close to **1** (±0.1)
- If far off, normalization statistics may be incorrect

### Class Distribution

```python
with open(data_dir / "class_map.json", encoding="utf-8") as f:
    class_map = json.load(f)

print("Class Distribution:")
print("=" * 50)
print(f"{'ID':<4} {'Name':<15} {'Pixels':>12} {'Percentage':>10}")
print("-" * 45)
total_pixels = sum(class_map["class_counts"].values())
for cls_id, count in sorted(class_map["class_counts"].items(), key=lambda x: -x[1]):
    cls_info = class_map["classes"].get(str(cls_id), {"name": "Unknown"})
    pct = count / total_pixels * 100
    print(f"{cls_id:<4} {cls_info['name']:<15} {count:>12,} {pct:>9.2f}%")
```

**Expected Output** (varies):
```
==================================================
Class Distribution:
==================================================
ID   Name            Pixels     Percentage
---------------------------------------------
0    Background     45,234,567     45.23%
1    Wheat          12,345,678     12.35%
2    Mustard         8,765,432      8.77%
...
13   Rice              567,890      0.57%
```

**What to verify**:
- Background usually largest (40-50%)
- All 14 classes should be present (0-13)
- Rare classes should still have reasonable counts

---

## 5. Part 4: Visualization

### Cell 8: Visualization Functions

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_rgb_composite(image, bands=(3, 2, 1)):
    """
    Create RGB composite from multi-band image.

    Parameters:
    - image: (C, H, W) normalized image array
    - bands: tuple of band indices for (R, G, B) - default is (B04, B03, B02)
    """
    rgb = np.stack([image[b] for b in bands], axis=-1)
    # Rescale from normalized to 0-1 for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return np.clip(rgb, 0, 1)
```

**Band Indices for RGB**:
```
Index 0: B01 (Coastal Aerosol)
Index 1: B02 (Blue)
Index 2: B03 (Green)
Index 3: B04 (Red)     ← R channel
...

True-color composite: R=B04 (index 3), G=B03 (index 2), B=B02 (index 1)
```

**Why rescale?**
- Normalized data has range approximately [-3, +3]
- Matplotlib expects [0, 1] for display
- Min-max rescaling maps to displayable range

### Mask Visualization Function

```python
def create_mask_rgb(mask, class_colors):
    """
    Create RGB visualization of class mask.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for cls_id, color in class_colors.items():
        cls_mask = mask == int(cls_id)
        for c in range(3):
            rgb[:, :, c] += cls_mask * (color[c] / 255.0)

    return rgb
```

**How it works**:
```
mask:                           rgb output:
┌─────────────────┐             ┌─────────────────┐
│ 0 │ 0 │ 1 │ 1 │             │ ■ │ ■ │ ▓ │ ▓ │
│ 0 │ 1 │ 1 │ 2 │     →       │ ■ │ ▓ │ ▓ │ ░ │
│ 2 │ 2 │ 2 │ 2 │             │ ░ │ ░ │ ░ │ ░ │
└─────────────────┘             └─────────────────┘

class_colors:
  0 → [0, 0, 0]    (black)    ■
  1 → [255, 200, 0] (gold)    ▓
  2 → [255, 150, 0] (orange)  ░
```

### Sample Visualization Grid

```python
# Get class colors from class_map
class_colors = {int(k): v["color"] for k, v in class_map["classes"].items()}

# Plot 4 sample tiles
fig, axes = plt.subplots(4, 2, figsize=(10, 16))
fig.suptitle("Sample Training Tiles", fontsize=14, fontweight="bold")

sample_indices = [0, len(train_images)//4, len(train_images)//2, 3*len(train_images)//4]

for row, idx in enumerate(sample_indices):
    # RGB composite (B04=Red, B03=Green, B02=Blue -> indices 3, 2, 1)
    rgb = create_rgb_composite(train_images[idx], bands=(3, 2, 1))
    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(f"Tile {idx}: RGB Composite")
    axes[row, 0].axis("off")

    # Mask visualization
    mask_rgb = create_mask_rgb(train_masks[idx], class_colors)
    axes[row, 1].imshow(mask_rgb)
    axes[row, 1].set_title(f"Tile {idx}: Crop Mask")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()
```

**Expected Output**:

```
┌─────────────────────────────────────────────────────────────┐
│              Sample Training Tiles                           │
├─────────────────────────────┬───────────────────────────────┤
│ Tile 0: RGB Composite       │ Tile 0: Crop Mask             │
│ ┌─────────────────────┐     │ ┌─────────────────────┐       │
│ │  🌾🌾🌾🌾🌾🌾🌾🌾   │     │ │  ■■▓▓▓▓░░░░░░    │       │
│ │  🌾🌾🌾🌿🌿🌿🌿    │     │ │  ■■▓▓▓▓▓▓░░░░    │       │
│ │  🌿🌿🌿🌿🌿🌿🌿🌿   │     │ │  ░░░░░░▒▒▒▒▒▒    │       │
│ └─────────────────────┘     │ └─────────────────────┘       │
├─────────────────────────────┼───────────────────────────────┤
│ Tile 220: RGB Composite     │ Tile 220: Crop Mask           │
│ ...                         │ ...                           │
└─────────────────────────────┴───────────────────────────────┘
```

---

## 6. Part 5: PyTorch Integration

### Cell 9: Test CropDataset

```python
from backend.src.data.dataset import CropDataset, get_dataloaders, RandomFlipRotate

# Load dataset using our custom class
train_dataset = CropDataset("data/processed", split="train")
val_dataset = CropDataset("data/processed", split="val")

print("=" * 50)
print("CropDataset Properties:")
print("=" * 50)
print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes:  {train_dataset.num_classes}")
print(f"Number of channels: {train_dataset.num_channels}")
```

**Expected Output**:
```
==================================================
CropDataset Properties:
==================================================
Training samples:   880
Validation samples: 220
Number of classes:  14
Number of channels: 12
```

### Test Single Sample

```python
sample = train_dataset[0]
print(f"\nSample batch structure:")
print(f"  image: {sample['image'].shape} ({sample['image'].dtype})")
print(f"  mask:  {sample['mask'].shape} ({sample['mask'].dtype})")
```

**Expected Output**:
```
Sample batch structure:
  image: torch.Size([12, 256, 256]) (torch.float32)
  mask:  torch.Size([256, 256]) (torch.int64)
```

### Test DataLoader

```python
# num_workers=0 required for Windows/Jupyter
train_loader, val_loader = get_dataloaders("data/processed", batch_size=8, num_workers=0)

batch = next(iter(train_loader))
print(f"\nDataLoader batch structure:")
print(f"  image: {batch['image'].shape} ({batch['image'].dtype})")
print(f"  mask:  {batch['mask'].shape} ({batch['mask'].dtype})")
```

**Expected Output**:
```
DataLoader batch structure:
  image: torch.Size([8, 12, 256, 256]) (torch.float32)
  mask:  torch.Size([8, 256, 256]) (torch.int64)
```

**Why `num_workers=0`?**

On Windows with Jupyter, multiprocessing can fail:
```python
# Error without num_workers=0:
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

### Test Class Weights

```python
weights = train_dataset.get_class_weights()
print(f"\nClass weights shape: {weights.shape}")
print(f"Class weights (first 5): {weights[:5].numpy()}")
```

**Expected Output**:
```
Class weights shape: torch.Size([14])
Class weights (first 5): [0.15 1.23 1.78 2.34 0.89]
```

**Interpretation**:
- Weight < 1: Class is over-represented (e.g., Background)
- Weight > 1: Class is under-represented (e.g., Rice)
- Used in `CrossEntropyLoss(weight=weights)` for balanced training

---

## 7. Cell-by-Cell Explanation

### Summary Table

| Cell | Purpose | Key Outputs |
|------|---------|-------------|
| 1 | Change to repo root | Working directory set |
| 2 | Import TorchGeo | Library loaded |
| 3 | Create data directory | `data/agrifieldnet/` exists |
| 4 | Download dataset | Raw GeoTIFFs downloaded |
| 5 | Verify download | CRS, bands, file count |
| 6 | Create DataModule | TorchGeo ready for use |
| 7 | Run preprocessing | `.npy` files created |
| 8 | Load & verify data | Shapes, dtypes confirmed |
| 9 | Check normalization | Mean≈0, Std≈1 |
| 10 | Check class distribution | All classes present |
| 11 | Define viz functions | Helpers for display |
| 12 | Visualize samples | RGB + mask plots |
| 13 | Test CropDataset | PyTorch integration |
| 14 | Test DataLoader | Batching works |
| 15 | Test class weights | Imbalance handling ready |

---

## 8. Interview Questions & Answers

### Notebook Design Questions

**Q1: Why have a notebook instead of just scripts?**

A: **Different purposes**:
| Script | Notebook |
|--------|----------|
| Automated execution | Interactive exploration |
| CI/CD pipelines | Documentation |
| Production use | Education/debugging |

The notebook serves as executable documentation.

**Q2: Why change the working directory at the start?**

A: Notebooks run from their own directory (`notebooks/`), but:
- Config is at `config/config.yaml` (relative to root)
- Data is at `data/` (relative to root)
- Imports use `backend.src.data...` (requires root in path)

```python
# Without chdir:
open("config/config.yaml")  # FileNotFoundError (looks in notebooks/)

# With chdir to parent:
open("config/config.yaml")  # Works (looks in AgroVision/)
```

**Q3: Why include both TorchGeo and custom approaches?**

A: **Different use cases**:
- TorchGeo: Quick prototyping, geospatial-aware sampling
- Custom: Production training, faster loading, simpler dependencies

The notebook demonstrates both for educational purposes.

### Verification Questions

**Q4: What would indicate a preprocessing bug?**

A: Check these:
| Symptom | Possible Cause |
|---------|---------------|
| Mean ≠ 0 | Wrong normalization stats |
| Std ≠ 1 | Applied wrong band stats |
| Missing classes | Class ID remapping error |
| All zeros | Missing band files |
| NaN values | Division by zero |

**Q5: Why verify normalization per-band?**

A: Each band has different statistics:
```
B01 (Coastal): mean=1923, std=412
B08 (NIR):     mean=3456, std=678
```

If all bands had the same mean/std after normalization, something went wrong.

**Q6: Why check class distribution?**

A: To detect:
- Missing classes (preprocessing dropped them)
- Extreme imbalance (might need weighted sampling)
- Wrong class mapping (unexpected class IDs)

### Visualization Questions

**Q7: Why use (3, 2, 1) for RGB instead of (0, 1, 2)?**

A: Band ordering in our data:
```
Index 0: B01 (Coastal Aerosol) - Not visible
Index 1: B02 (Blue)            - B for RGB
Index 2: B03 (Green)           - G for RGB
Index 3: B04 (Red)             - R for RGB
```

True-color RGB = (Red, Green, Blue) = (B04, B03, B02) = indices (3, 2, 1)

**Q8: Why clip RGB values to [0, 1]?**

A: After min-max normalization:
```python
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
```

Values should be [0, 1], but floating-point errors can cause tiny negatives or values > 1. Clipping ensures valid display values.

**Q9: What's a "false-color composite"?**

A: Using non-visible bands for RGB:
```python
# NIR-Red-Green composite (vegetation appears bright red):
create_rgb_composite(image, bands=(7, 3, 2))  # B08, B04, B03
```

Useful for vegetation analysis but not used in this notebook.

### Practical Questions

**Q10: What if the download fails midway?**

A: Options:
1. Delete `data/agrifieldnet/` and retry
2. Use TorchGeo's `checksum=True` to verify downloads
3. Manually download from Source Cooperative

**Q11: How long does preprocessing take?**

A: Rough estimates:
| Step | Time |
|------|------|
| Band statistics | ~10 min |
| Train set processing | ~15 min |
| Val set processing | ~5 min |
| **Total** | **~30 min** |

Varies based on disk speed and CPU.

**Q12: Can I run individual cells out of order?**

A: Some cells depend on previous:
- Cell 7 needs data from Cell 6
- Visualization needs data loaded
- CropDataset needs .npy files to exist

Run cells in order for first-time setup. After that, can skip download/preprocessing if files exist.

### Technical Questions

**Q13: Why `encoding="utf-8"` for JSON?**

A: The `class_map.json` contains Arabic text:
```json
{"name_ar": "قمح"}
```

Windows default encoding (cp1252) can't decode Arabic:
```python
open("class_map.json").read()  # UnicodeDecodeError!
open("class_map.json", encoding="utf-8").read()  # Works!
```

**Q14: What's `sys.path.insert(0, ".")`?**

A: Adds current directory to Python's import search path:
```python
sys.path = [".", ...other paths...]

# Now this import works:
from backend.src.data.prepare_dataset import main
```

Without it, Python wouldn't find the `backend` package.

**Q15: Why does the notebook print "Next steps" at the end?**

A: Documents the workflow for other team members:
```
Next steps:
  1. Student B: Implement U-Net model
  2. Student C: Implement training loop
  3. Student D: Implement inference API
```

This is project coordination embedded in code.
