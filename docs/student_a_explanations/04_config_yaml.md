# Configuration File - Complete Documentation

**File**: `config/config.yaml`

**Purpose**: Master configuration file that defines paths, Sentinel-2 band specifications, crop class definitions, and training hyperparameters.

---

## Table of Contents

1. [File Overview](#1-file-overview)
2. [Paths Configuration](#2-paths-configuration)
3. [Model Settings](#3-model-settings)
4. [Backend Settings](#4-backend-settings)
5. [Training Settings](#5-training-settings)
6. [Sentinel-2 Band Configuration](#6-sentinel-2-band-configuration)
7. [Crop Class Definitions](#7-crop-class-definitions)
8. [Why YAML?](#8-why-yaml)
9. [Interview Questions & Answers](#9-interview-questions--answers)

---

## 1. File Overview

### What This File Does

This configuration file is the **single source of truth** for the entire AgroVision project:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         config/config.yaml                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Used by:                                                           │ │
│  │                                                                     │ │
│  │  prepare_dataset.py  ─→  Paths, bands, classes, training settings  │ │
│  │  dataset.py          ─→  Class definitions (via class_map.json)    │ │
│  │  train.py            ─→  Training hyperparameters                   │ │
│  │  inference_service.py ─→  Model path, backend settings             │ │
│  │  Frontend             ─→  Tile limits, overlay settings            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```yaml
# 1. Paths - Where data lives
paths:
  data_raw: ...
  data_processed: ...
  ...

# 2. Model settings
model:
  device: ...
  tile_limit: ...
  ...

# 3. Backend settings
backend:
  demo_split: ...
  ...

# 4. Training settings
training:
  batch_size: ...
  epochs: ...
  ...

# 5. Band definitions
bands:
  - name: B01
    wavelength: ...
    ...

# 6. Class definitions
classes:
  0:
    name: Background
    ...
  1:
    name: Wheat
    ...
```

---

## 2. Paths Configuration

```yaml
paths:
  data_raw: data/agrifieldnet
  data_processed: data/processed
  splits_dir: data/splits
  model_checkpoint: outputs/models/unet_baseline_best_model.pth
```

### Explanation

| Key | Value | Purpose |
|-----|-------|---------|
| `data_raw` | `data/agrifieldnet` | Raw AgriFieldNet download (GeoTIFFs) |
| `data_processed` | `data/processed` | Preprocessed `.npy` files |
| `splits_dir` | `data/splits` | Train/val split CSV files |
| `model_checkpoint` | `outputs/models/...` | Best trained model weights |

### Directory Structure

```
AgroVision/
├── config/
│   └── config.yaml          ← This file
├── data/
│   ├── agrifieldnet/        ← data_raw
│   │   ├── source/          # Sentinel-2 bands
│   │   └── train_labels/    # Crop labels
│   ├── processed/           ← data_processed
│   │   ├── train_images.npy
│   │   ├── train_masks.npy
│   │   └── ...
│   └── splits/              ← splits_dir
│       ├── train_ids.csv
│       └── val_ids.csv
└── outputs/
    └── models/              ← model_checkpoint directory
        └── unet_baseline_best_model.pth
```

### Why Relative Paths?

All paths are relative to the **repository root**:
```python
# Assumed working directory when running scripts
cd AgroVision/
python -m agrovision_core.data.prepare_dataset
```

**Benefits**:
- Portable across machines
- Works on Windows and Linux
- Can be checked into git

---

## 3. Model Settings

```yaml
model:
  device: cuda
  tile_limit: 9
  input_size: 256
  num_classes: 14
```

### Explanation

| Key | Value | Purpose |
|-----|-------|---------|
| `device` | `cuda` | GPU for inference (`cuda` or `cpu`) |
| `tile_limit` | `9` | Maximum tiles per inference request |
| `input_size` | `256` | Tile dimensions (256×256 pixels) |
| `num_classes` | `14` | Output channels (13 crops + background) |

### Why `tile_limit: 9`?

**Memory constraint**: Each tile uses significant GPU memory

```
Per tile:
  Input:  12 × 256 × 256 × 4 bytes = 3.1 MB
  Model:  ~50 MB intermediate activations
  Output: 14 × 256 × 256 × 4 bytes = 3.7 MB
  Total: ~57 MB per tile

9 tiles × 57 MB = ~513 MB (safe for 4GB GPU)
```

**UX constraint**: More tiles = longer processing time

---

## 4. Backend Settings

```yaml
backend:
  demo_split: val
  demo_tile_index: 0
  overlay_alpha: 0.45
  sentinel2_stac_url: "https://planetarycomputer.microsoft.com/api/stac/v1"
  sentinel2_collection: "sentinel-2-l2a"
  sentinel2_date_window: "2023-01-01/2024-12-31"
  sentinel2_cloud_cover: 60
  sentinel2_max_items: 10
  sentinel2_scale: 0.01
```

### Explanation

| Key | Value | Purpose |
|-----|-------|---------|
| `demo_split` | `val` | Which split to use for demo/testing |
| `demo_tile_index` | `0` | Default tile to display |
| `overlay_alpha` | `0.45` | Transparency of prediction overlay |
| `sentinel2_stac_url` | Microsoft Planetary Computer | STAC API endpoint |
| `sentinel2_collection` | `sentinel-2-l2a` | Level-2A (atmospherically corrected) |
| `sentinel2_date_window` | `2023-01-01/2024-12-31` | Date range for imagery search |
| `sentinel2_cloud_cover` | `60` | Maximum cloud cover percentage |
| `sentinel2_max_items` | `10` | Maximum search results |
| `sentinel2_scale` | `0.01` | Reflectance scaling factor |

### Sentinel-2 STAC Integration

```
User provides: Coordinates (lat, lon) + date range
        ↓
Backend queries: Planetary Computer STAC API
        ↓
Returns: Available Sentinel-2 imagery matching criteria
        ↓
Inference: Download bands → Normalize → Predict → Overlay
```

---

## 5. Training Settings

```yaml
training:
  batch_size: 32
  num_workers: 2
  val_split: 0.2
  random_seed: 42
  epochs: 50

  # Loss configuration
  use_field_loss: true
  lambda_pixel: 0.2
  lambda_field: 1.0

  # Masking and class weighting
  ignore_index: 0
  min_labeled_fraction: 0.001
  background_weight: 0.0
  focal_gamma: 2.0

  # Weighted tile sampling
  use_weighted_sampling: true
  weight_power: 1.5

  # Optimizer
  learning_rate: 0.0003
  weight_decay: 0.0001
  optimizer: adamw

  # Model selection criterion
  model_selection: field_ce
```

### Basic Training Parameters

| Key | Value | Purpose |
|-----|-------|---------|
| `batch_size` | `32` | Samples per gradient update |
| `num_workers` | `2` | Data loading threads |
| `val_split` | `0.2` | 20% for validation |
| `random_seed` | `42` | Reproducibility |
| `epochs` | `50` | Training iterations |

### Loss Configuration

```yaml
use_field_loss: true
lambda_pixel: 0.2
lambda_field: 1.0
```

**AgriFieldNet Challenge**: Evaluated per-field, not per-pixel!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Combined Loss                                    │
│                                                                          │
│  L_total = lambda_pixel × L_pixel + lambda_field × L_field              │
│          = 0.2 × CrossEntropy(pixel) + 1.0 × CrossEntropy(field)        │
│                                                                          │
│  Why both?                                                               │
│  - L_pixel: Good for overlay visualization (smooth boundaries)           │
│  - L_field: Aligns with challenge scoring (main optimization target)    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Masking and Weighting

```yaml
ignore_index: 0
min_labeled_fraction: 0.001
background_weight: 0.0
focal_gamma: 2.0
```

| Key | Value | Purpose |
|-----|-------|---------|
| `ignore_index` | `0` | Don't compute loss for background |
| `min_labeled_fraction` | `0.001` | Skip batches with <0.1% labels |
| `background_weight` | `0.0` | Zero weight for background class |
| `focal_gamma` | `2.0` | Focal loss focusing parameter |

**Focal Loss**: Down-weights easy examples, focuses on hard ones
```
FL(p) = -(1-p)^γ × log(p)

When γ=2.0:
  Easy example (p=0.9): weight = (1-0.9)² = 0.01
  Hard example (p=0.1): weight = (1-0.1)² = 0.81
```

### Weighted Sampling

```yaml
use_weighted_sampling: true
weight_power: 1.5
```

Tiles with more labeled pixels are sampled more often:
```python
weight = (labeled_fraction + 1e-6) ** 1.5
```

### Optimizer Settings

```yaml
learning_rate: 0.0003
weight_decay: 0.0001
optimizer: adamw
```

| Key | Value | Explanation |
|-----|-------|-------------|
| `learning_rate` | `3e-4` | Step size for gradient descent |
| `weight_decay` | `1e-4` | L2 regularization strength |
| `optimizer` | `adamw` | Adam with decoupled weight decay |

**Why AdamW over Adam?**
- Adam applies weight decay to gradient moments (incorrect)
- AdamW applies weight decay directly to weights (correct)

### Model Selection

```yaml
model_selection: field_ce
```

Options:
- `field_ce`: Lower field-level cross-entropy is better
- `mIoU`: Higher mean Intersection over Union is better

---

## 6. Sentinel-2 Band Configuration

```yaml
bands:
  - name: B01
    wavelength: 443
    resolution: 60
    description: Coastal Aerosol
  - name: B02
    wavelength: 490
    resolution: 10
    description: Blue
  # ... (12 bands total)
```

### Complete Band Table

| Band | Wavelength (nm) | Resolution (m) | Description | Primary Use |
|------|-----------------|----------------|-------------|-------------|
| B01 | 443 | 60 | Coastal Aerosol | Atmospheric correction |
| B02 | 490 | 10 | Blue | Water, vegetation |
| B03 | 560 | 10 | Green | Vegetation health |
| B04 | 665 | 10 | Red | Chlorophyll absorption |
| B05 | 705 | 20 | Red Edge 1 | Vegetation stress |
| B06 | 740 | 20 | Red Edge 2 | Leaf area index |
| B07 | 783 | 20 | Red Edge 3 | Canopy structure |
| B08 | 842 | 10 | NIR | Biomass (NDVI) |
| B8A | 865 | 20 | NIR Narrow | Water vapor |
| B09 | 945 | 60 | Water Vapor | Atmospheric |
| B11 | 1610 | 20 | SWIR 1 | Moisture content |
| B12 | 2190 | 20 | SWIR 2 | Mineral/dry vegetation |

### Why These Specific Bands?

**Vegetation Indices** can be computed:
```
NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04)
```

**Crop Discrimination**:
- Red Edge bands (B05-B07): Distinguish healthy vs stressed vegetation
- SWIR bands (B11-B12): Detect water content, distinguish crop types

### Band Visualization

```
                    Electromagnetic Spectrum
    ←── UV ──┼────── Visible ──────┼──── NIR ────┼──── SWIR ────→

    400nm    500nm    600nm    700nm    800nm   1000nm  1500nm  2000nm
      │        │        │        │        │        │       │       │
      │   B02  │   B03  │   B04  │  B05   │   B08  │       │  B11  │  B12
      │  Blue  │ Green  │  Red   │Red Edge│   NIR  │       │ SWIR1 │ SWIR2
      │        │        │        │  B06   │   B8A  │       │       │
      │        │        │        │  B07   │   B09  │       │       │
      ▼        ▼        ▼        ▼        ▼        ▼       ▼       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                    Sentinel-2 Spectral Coverage                   │
   │  B01                                                              │
   │ (Coastal)                                                         │
   └──────────────────────────────────────────────────────────────────┘
```

### Interview Questions on Bands

**Q: Why not use B10?**

A: B10 (Cirrus, 1375nm) is for cirrus cloud detection, not surface observation. It's often excluded from analysis datasets.

**Q: Why different resolutions?**

A: Trade-off between spatial detail and signal-to-noise ratio:
- 10m bands: High spatial detail for vegetation structure
- 20m bands: Better spectral discrimination for vegetation type
- 60m bands: Atmospheric correction (don't need high resolution)

**Q: How are different resolutions handled?**

A: All bands are resampled to 256×256 pixels (10m equivalent):
- 10m bands: Direct read (256×256)
- 20m bands: Upsampled 2× with bilinear interpolation
- 60m bands: Upsampled ~6× with bilinear interpolation

---

## 7. Crop Class Definitions

```yaml
classes:
  0:
    name: Background
    name_ar: خلفية
    color: [0, 0, 0]
  1:
    name: Wheat
    name_ar: قمح
    color: [255, 200, 0]
  # ... (14 classes total)
```

### Complete Class Table

| ID | Name (English) | Name (Arabic) | Color RGB | Notes |
|----|----------------|---------------|-----------|-------|
| 0 | Background | خلفية | [0, 0, 0] | Unlabeled pixels |
| 1 | Wheat | قمح | [255, 200, 0] | Winter crop |
| 2 | Mustard | خردل | [255, 150, 0] | Oil seed |
| 3 | Lentil | عدس | [200, 100, 50] | Pulse crop |
| 4 | No Crop | بور | [128, 128, 128] | Fallow land |
| 5 | Green Pea | بسلة خضراء | [0, 200, 0] | Legume |
| 6 | Sugarcane | قصب السكر | [100, 200, 100] | Cash crop |
| 8 | Garlic | ثوم | [200, 150, 200] | Vegetable |
| 9 | Maize | ذرة صفراء | [255, 255, 0] | Cereal |
| 13 | Gram | حمص | [150, 100, 50] | Chickpea |
| 14 | Coriander | كزبرة | [0, 150, 100] | Spice |
| 15 | Potato | بطاطس | [200, 150, 100] | Vegetable |
| 16 | Berseem | برسيم | [0, 255, 150] | Fodder |
| 36 | Rice | أرز | [0, 100, 255] | Paddy |

### Why Non-Contiguous IDs?

**Original AgriFieldNet labeling** used arbitrary IDs. Notice gaps:
- Missing: 7, 10, 11, 12, 17-35
- Highest ID: 36 (Rice)

**Solution**: Remap to contiguous 0-13 during preprocessing.

### Color Scheme Design

Colors are chosen for:
1. **Visual distinction**: Each class has a unique color
2. **Semantic meaning**:
   - Greens for green plants (Pea, Sugarcane)
   - Yellows for cereals (Wheat, Maize)
   - Browns for legumes (Lentil, Gram)
   - Blue for water-intensive (Rice)

### Arabic Names (`name_ar`)

**Why include Arabic?**

AgriFieldNet is from **India**, but the frontend might be used in Arabic-speaking regions. Internationalization support built-in.

**Important**: Requires `encoding="utf-8"` when reading:
```python
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
```

---

## 8. Why YAML?

### YAML vs JSON vs Python Dict

| Aspect | YAML | JSON | Python |
|--------|------|------|--------|
| Comments | ✓ Yes | ✗ No | ✓ Yes |
| Readability | Best | Good | Good |
| Multi-line strings | Easy | Hard | Hard |
| Type support | Basic | Basic | Full |
| Standard | Cross-language | Cross-language | Python only |

### YAML Features Used

**Comments**:
```yaml
# This is a comment
training:
  batch_size: 32  # Samples per batch
```

**Nested structures**:
```yaml
classes:
  1:
    name: Wheat
    color: [255, 200, 0]
```

**Lists**:
```yaml
bands:
  - name: B01
    wavelength: 443
  - name: B02
    wavelength: 490
```

### Loading in Python

```python
import yaml

with open("config/config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Access values
batch_size = config["training"]["batch_size"]
bands = config["bands"]  # List of dicts
classes = config["classes"]  # Dict of dicts
```

---

## 9. Interview Questions & Answers

### Configuration Design Questions

**Q1: Why put all configuration in one file instead of separate files?**

A: **Single source of truth**:
- All settings in one place
- No inconsistencies between files
- Easy to version control
- Simple to copy for experiments

Alternative: Separate files for different concerns
```
config/
├── paths.yaml
├── model.yaml
├── training.yaml
└── classes.yaml
```

But this adds complexity for small projects.

**Q2: Why not use environment variables for paths?**

A: Environment variables are good for **secrets and deployment**, but:
- Config values are not secrets
- YAML is version-controlled (reproducibility)
- Complex structures (bands, classes) don't fit env vars

**Q3: How would you add a new crop class?**

A:
1. Add entry to `classes`:
   ```yaml
   17:
     name: Barley
     name_ar: شعير
     color: [180, 150, 100]
   ```
2. Update `num_classes` in model section
3. Rerun preprocessing (remapping changes)
4. Retrain model (output layer changes)

### Band Questions

**Q4: Why 12 bands instead of RGB (3 bands)?**

A: **Spectral information**:
- RGB can't distinguish crops with similar colors
- Red edge bands detect vegetation stress
- NIR/SWIR detect water content

```
                RGB Only                    Full Spectrum
                ─────────                   ──────────────
Wheat:          [200, 180, 100]             + Red edge, NIR, SWIR patterns
Mustard:        [200, 180, 90]              + Distinct spectral signature

Hard to distinguish with RGB alone!
```

**Q5: What if a band is corrupted for a tile?**

A: Current behavior in `load_tile()`:
```python
if band_path.exists():
    data = load_band_tiff(band_path, target_size)
else:
    data = np.zeros(target_size, dtype=np.float32)  # Fill with zeros
```

Model learns to handle missing bands (zeros after normalization become negative).

### Training Questions

**Q6: Why `random_seed: 42`?**

A: **Reproducibility**. Running the same code twice produces identical results:
- Same train/val split
- Same weight initialization
- Same augmentation sequence

42 is a convention (Hitchhiker's Guide), but any fixed number works.

**Q7: What's the trade-off with `focal_gamma`?**

A:
- `gamma=0`: Standard cross-entropy
- `gamma=2` (default): Focus on hard examples
- `gamma>2`: Even more focus, but may underweight easy examples too much

Higher gamma helps with class imbalance but can hurt overall accuracy if too aggressive.

**Q8: Why `ignore_index: 0` for background?**

A: Background pixels are unlabeled:
- No ground truth → don't penalize predictions
- Model learns to classify labeled regions
- Background predictions don't affect loss

### Class Definition Questions

**Q9: Why store colors as [R, G, B] lists?**

A: Easy to use in visualization:
```python
color = config["classes"]["1"]["color"]  # [255, 200, 0]
plt.plot(x, y, color=[c/255 for c in color])  # Matplotlib expects 0-1 range
```

**Q10: How is the raw-to-contiguous mapping used?**

A:
1. **Preprocessing**: Raw labels (0,1,2,...,36) → Contiguous (0,1,2,...,13)
2. **Training**: Model outputs 14 channels
3. **Inference**: Predictions (0-13) → Raw IDs for display

```python
# In prepare_dataset.py
raw_to_contig = {0: 0, 1: 1, 2: 2, ..., 36: 13}
mask = lut[mask]  # Apply mapping

# In inference
contig_to_raw = {0: 0, 1: 1, 2: 2, ..., 13: 36}
predicted_raw = contig_to_raw[predicted_class]
```

### Practical Questions

**Q11: How would you run an experiment with different hyperparameters?**

A: Options:
1. **Copy config**: `config_exp1.yaml` with changes
2. **Command-line override**: `--learning_rate 0.001`
3. **Hydra/OmegaConf**: Structured configuration management

**Q12: What happens if config.yaml is missing?**

A: `FileNotFoundError` in `load_config()`. The system can't run without configuration.

**Q13: How would you add support for a different satellite (Landsat)?**

A: Add new section:
```yaml
landsat_bands:
  - name: B1
    wavelength: 430
    resolution: 30
    description: Coastal Aerosol
  # ...

# Or make bands configurable:
satellite: sentinel2  # or "landsat8"
```
