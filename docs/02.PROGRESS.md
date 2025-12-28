# AgroVision Progress Report

**Last Updated:** December 24, 2024
**Completed By:** Student A (Data Pipeline)
**Status:** Data Pipeline Complete

---

## Project Overview

**AgroVision Crop Mapper** is an interactive crop mapping platform that uses:
- **Sentinel-2 satellite imagery** (12 spectral bands)
- **Semantic segmentation** (U-Net with transformer blocks)
- **React frontend** + **FastAPI backend** architecture

The system allows non-technical users to analyze crop distribution from satellite imagery through a map-based GUI.

---

## What Has Been Completed

### Student A — Data Pipeline (100% Complete)

The complete data preprocessing pipeline is now functional:

1. **Dataset download** via TorchGeo + azcopy
2. **Preprocessing pipeline** that converts GeoTIFFs to PyTorch-ready `.npy` files
3. **PyTorch Dataset class** for training integration
4. **Configuration system** with band definitions and class mappings
5. **Jupyter notebook** with full documentation and visualization

---

## Files Created

### Configuration

| File | Description |
|------|-------------|
| `config/config.yaml` | Central configuration: paths, bands, class definitions, colors |

### Backend Source Code

| File | Description |
|------|-------------|
| `backend/__init__.py` | Package marker |
| `backend/src/__init__.py` | Package marker |
| `backend/src/data/__init__.py` | Exports `CropDataset` |
| `backend/src/data/prepare_dataset.py` | Main preprocessing script (~400 lines) |
| `backend/src/data/dataset.py` | PyTorch Dataset + DataLoader utilities (~250 lines) |
| `backend/src/utils/__init__.py` | Exports I/O utilities |
| `backend/src/utils/io.py` | GeoTIFF loading and resampling functions (~200 lines) |

### Documentation

| File | Description |
|------|-------------|
| `CLAUDE.md` | AI assistant guidance for this repository |
| `docs/PROGRESS.md` | This file - progress tracking |

### Notebooks

| File | Description |
|------|-------------|
| `data-pipeline.ipynb` | Complete data pipeline with 9 documented sections |

---

## Data Pipeline Architecture

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE FLOW                           │
└─────────────────────────────────────────────────────────────────┘

1. DOWNLOAD (TorchGeo)
   ┌─────────────┐
   │ AgriFieldNet │ ──azcopy──► data/agrifieldnet/
   │ (Source Coop)│              ├── source/          (12 bands per tile)
   └─────────────┘              ├── train_labels/    (crop masks)
                                └── test_labels/     (field IDs only)

2. PREPROCESSING (prepare_dataset.py)
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │  Scan tiles ──► Compute stats ──► Split 80/20 ──► Normalize │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. OUTPUT (.npy files)
   data/processed/
   ├── train_images.npy      (N, 12, 256, 256) float32
   ├── train_masks.npy       (N, 256, 256) int64
   ├── val_images.npy
   ├── val_masks.npy
   ├── normalization_stats.json
   └── class_map.json

4. TRAINING (dataset.py)
   ┌────────────┐     ┌────────────┐     ┌─────────┐
   │ CropDataset │ ──► │ DataLoader │ ──► │ Model   │
   └────────────┘     └────────────┘     └─────────┘
```

### Sentinel-2 Bands Used

| Index | Band | Wavelength | Resolution | Description |
|-------|------|------------|------------|-------------|
| 0 | B01 | 443 nm | 60m | Coastal Aerosol |
| 1 | B02 | 490 nm | 10m | Blue |
| 2 | B03 | 560 nm | 10m | Green |
| 3 | B04 | 665 nm | 10m | Red |
| 4 | B05 | 705 nm | 20m | Red Edge 1 |
| 5 | B06 | 740 nm | 20m | Red Edge 2 |
| 6 | B07 | 783 nm | 20m | Canopy Structure |
| 7 | B08 | 842 nm | 10m | NIR (vegetation) |
| 8 | B8A | 865 nm | 20m | NIR Narrow |
| 9 | B09 | 945 nm | 60m | Water Vapor |
| 10 | B11 | 1610 nm | 20m | SWIR 1 (moisture) |
| 11 | B12 | 2190 nm | 20m | SWIR 2 |

All bands are resampled to 256×256 pixels using bilinear interpolation.

### Crop Classes (13 + Background)

| ID | English | Arabic | RGB Color |
|----|---------|--------|-----------|
| 0 | Background | خلفية | (0, 0, 0) |
| 1 | Wheat | قمح | (255, 200, 0) |
| 2 | Mustard | خردل | (255, 150, 0) |
| 3 | Lentil | عدس | (200, 100, 50) |
| 4 | No Crop | بور | (128, 128, 128) |
| 5 | Green Pea | بسلة خضراء | (0, 200, 0) |
| 6 | Sugarcane | قصب السكر | (100, 200, 100) |
| 8 | Garlic | ثوم | (200, 150, 200) |
| 9 | Maize | ذرة صفراء | (255, 255, 0) |
| 13 | Gram | حمص | (150, 100, 50) |
| 14 | Coriander | كزبرة | (0, 150, 100) |
| 15 | Potato | بطاطس | (200, 150, 100) |
| 16 | Berseem | برسيم | (0, 255, 150) |
| 36 | Rice | أرز | (0, 100, 255) |

---

## How to Use

### Prerequisites

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Install `azcopy` for dataset download:
   ```bash
   winget install Microsoft.AzCopy
   ```

### Option 1: Run the Notebook

Open and run `data-pipeline.ipynb` in VS Code or Jupyter. The notebook has 9 sections:

1. Import Libraries
2. Setup Data Directory
3. Download Dataset (10-30 min)
4. Verify Dataset
5. Setup DataModule
6. Run Preprocessing Pipeline (10-30 min)
7. Verify Processed Data
8. Visualize Sample Tiles
9. Test PyTorch Integration

### Option 2: Run Preprocessing Script Directly

```bash
python -m backend.src.data.prepare_dataset
```

### Using the Dataset in Training Code

```python
from backend.src.data.dataset import CropDataset, get_dataloaders

# Load datasets
train_dataset = CropDataset("data/processed", split="train")
val_dataset = CropDataset("data/processed", split="val")

# Create DataLoaders
train_loader, val_loader = get_dataloaders(
    "data/processed",
    batch_size=32,
    num_workers=0  # Use 0 on Windows/Jupyter
)

# Get a batch
batch = next(iter(train_loader))
images = batch['image']  # (B, 12, 256, 256)
masks = batch['mask']    # (B, 256, 256)

# Get class weights for imbalanced loss
weights = train_dataset.get_class_weights()
```

---

## Output Files Generated

After running the pipeline, you'll have:

```
data/
├── agrifieldnet/           # Raw downloaded data (~2.5 GB)
│   ├── source/             # 1,217 tile directories
│   ├── train_labels/       # Crop masks
│   └── test_labels/        # Test field IDs
│
├── processed/              # PyTorch-ready data
│   ├── train_images.npy    # ~3.5 GB (932, 12, 256, 256)
│   ├── train_masks.npy     # ~250 MB (932, 256, 256)
│   ├── val_images.npy      # ~900 MB (233, 12, 256, 256)
│   ├── val_masks.npy       # ~60 MB (233, 256, 256)
│   ├── normalization_stats.json
│   └── class_map.json
│
└── splits/
    ├── train_ids.csv       # 932 tile IDs
    └── val_ids.csv         # 233 tile IDs
```

---

## Known Issues & Solutions

### 1. UTF-8 Encoding (Windows)

**Problem:** `UnicodeDecodeError` when reading config/JSON files with Arabic text.

**Solution:** Always use `encoding="utf-8"`:
```python
with open(path, encoding="utf-8") as f:
    data = json.load(f)
```

### 2. Windows Multiprocessing in Jupyter

**Problem:** `OSError: [Errno 22] Invalid argument` when using DataLoader with `num_workers > 0`.

**Solution:** Use `num_workers=0` in notebooks on Windows:
```python
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

### 3. Nested Directory Structure

**Problem:** TorchGeo downloads tiles into subdirectories, not flat files.

**Solution:** The `io.py` functions handle the nested structure:
```
source/ref_agrifieldnet_..._001c1/ref_agrifieldnet_..._001c1_B02_10m.tif
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Band selection | All 12 bands | Maximize spectral information |
| Normalization | Z-score (per-band) | Standard for deep learning |
| Train/Val split | 80/20 | Industry standard |
| File format | `.npy` | Fast loading, PyTorch compatible |
| Resampling | Bilinear to 256×256 | Preserves spatial info |
| Random seed | 42 | Reproducibility |

---

## Next Steps for Team

### Student B — Model Implementation
- Location: `backend/src/models/`
- Tasks:
  - Implement `unet_baseline.py` (U-Net from scratch)
  - Implement `blocks.py` (CNN + transformer blocks)
  - Implement `unet_transformer.py` (custom architecture)
- Input shape: `(B, 12, 256, 256)`
- Output shape: `(B, 14, 256, 256)` (14 classes including background)

### Student C — Training & Evaluation
- Location: `backend/src/train/`
- Tasks:
  - Implement `train.py` (training loop)
  - Implement `evaluate.py` (validation)
  - Implement `metrics.py` (mIoU, F1, confusion matrix)
- Use `CropDataset` from `backend/src/data/dataset.py`
- Use `class_weights` for handling class imbalance

### Student D — Backend API
- Location: `backend/api/`, `backend/services/`
- Tasks:
  - Implement FastAPI endpoints (`/api/infer`, `/api/legend`)
  - Implement `inference_service.py`
  - Load trained model, run predictions

### Student E — Frontend
- Location: `frontend/src/`
- Tasks:
  - React + shadcn UI
  - Map-based AOI selection
  - Call backend API, display results

---

## Testing the Pipeline

To verify everything works:

```python
# Quick test
from backend.src.data.dataset import CropDataset

dataset = CropDataset("data/processed", split="train")
print(f"Samples: {len(dataset)}")
print(f"Channels: {dataset.num_channels}")
print(f"Classes: {dataset.num_classes}")

sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")
```

Expected output:
```
Samples: 932
Channels: 12
Classes: 14
Image shape: torch.Size([12, 256, 256])
Mask shape: torch.Size([256, 256])
```

---

## Contact

For questions about the data pipeline, refer to:
- `data-pipeline.ipynb` - Full walkthrough with visualizations
- `config/config.yaml` - Band and class definitions
- `CLAUDE.md` - Coding guidelines for AI assistants
