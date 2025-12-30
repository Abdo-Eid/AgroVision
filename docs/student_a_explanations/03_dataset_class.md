# PyTorch Dataset Class - Complete Documentation

**File**: `agrovision_core/src/agrovision_core/data/dataset.py`

**Purpose**: Provides PyTorch-compatible Dataset and DataLoader wrappers for the preprocessed `.npy` files.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Class: `CropDataset`](#3-class-cropdataset)
4. [Function: `get_dataloaders()`](#4-function-get_dataloaders)
5. [Class: `RandomFlipRotate`](#5-class-randomfliprotate)
6. [Data Flow Through the Pipeline](#6-data-flow-through-the-pipeline)
7. [Interview Questions & Answers](#7-interview-questions--answers)

---

## 1. Module Overview

### What This Module Does

This module bridges **preprocessed data** and **PyTorch training**:

```
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│  data/processed/    │       │    CropDataset      │       │    PyTorch          │
│  ├── train_images.npy │ →   │    (Dataset class)  │   →   │    Training Loop    │
│  ├── train_masks.npy  │     │                     │       │                     │
│  └── class_map.json   │     │  __getitem__(idx)   │       │  for batch in       │
│                       │     │  returns sample     │       │    dataloader:      │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `CropDataset` | PyTorch Dataset wrapping `.npy` files |
| `get_dataloaders()` | Factory function for train/val DataLoaders |
| `RandomFlipRotate` | Data augmentation transform |

---

## 2. Imports and Dependencies

```python
import json
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
```

### Import Analysis

| Import | Purpose |
|--------|---------|
| `json` | Load class_map.json, normalization_stats.json |
| `Path` | Cross-platform path handling |
| `Callable, Optional, Union` | Type hints for transforms parameter |
| `numpy` | Array operations |
| `torch` | Convert arrays to tensors |
| `DataLoader, Dataset` | PyTorch data loading infrastructure |

### PyTorch Dataset Interface

To create a PyTorch Dataset, you must implement:
```python
class MyDataset(Dataset):
    def __len__(self) -> int:
        """Return number of samples"""
        pass

    def __getitem__(self, idx: int) -> Any:
        """Return sample at index idx"""
        pass
```

---

## 3. Class: `CropDataset`

### Class Definition

```python
class CropDataset(Dataset):
    """
    PyTorch Dataset for crop classification from .npy files.

    Attributes
    ----------
    images : np.ndarray
        Array of shape (N, 12, 256, 256) containing normalized satellite imagery
    masks : np.ndarray
        Array of shape (N, 256, 256) containing crop class labels
    field_ids : np.ndarray or None
        Array of shape (N, 256, 256) containing field IDs per pixel (0 = no field)
    transforms : callable, optional
        Optional transforms to apply to (image, mask, field_ids) tuples
    class_map : dict
        Mapping of class IDs to class names and colors
    """
```

---

### 3.1 `__init__()` Method

```python
def __init__(
    self,
    data_dir: Union[str, Path],
    split: str = "train",
    transforms: Optional[Callable] = None,
    load_to_memory: bool = True,
):
```

#### Full Implementation

```python
def __init__(
    self,
    data_dir: Union[str, Path],
    split: str = "train",
    transforms: Optional[Callable] = None,
    load_to_memory: bool = True,
):
    """
    Initialize the CropDataset.

    Parameters
    ----------
    data_dir : str or Path
        Path to the processed data directory containing .npy files
    split : str
        Which split to load: 'train' or 'val'
    transforms : callable, optional
        Optional transforms to apply to each sample
    load_to_memory : bool
        If True, load entire dataset to memory for faster access.
        If False, use memory-mapped arrays (slower but lower memory).
    """
    self.data_dir = Path(data_dir)
    self.split = split
    self.transforms = transforms

    # Build file paths
    images_path = self.data_dir / f"{split}_images.npy"
    masks_path = self.data_dir / f"{split}_masks.npy"
    field_ids_path = self.data_dir / f"{split}_field_ids.npy"

    # Check if files exist
    if not images_path.exists():
        raise FileNotFoundError(
            f"Images file not found: {images_path}\n"
            "Run prepare_dataset.py first to generate the data."
        )

    # Load arrays based on memory strategy
    if load_to_memory:
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        if field_ids_path.exists():
            self.field_ids = np.load(field_ids_path)
        else:
            self.field_ids = None
    else:
        # Memory-mapped mode for large datasets
        self.images = np.load(images_path, mmap_mode="r")
        self.masks = np.load(masks_path, mmap_mode="r")
        if field_ids_path.exists():
            self.field_ids = np.load(field_ids_path, mmap_mode="r")
        else:
            self.field_ids = None

    # Load class map
    class_map_path = self.data_dir / "class_map.json"
    if class_map_path.exists():
        with open(class_map_path, encoding="utf-8") as f:
            self.class_map = json.load(f)
    else:
        self.class_map = {}

    # Build ID mappings
    classes = self.class_map.get("classes", {})
    if classes:
        raw_ids = sorted(int(k) for k in classes.keys())
        self.raw_to_contig = {raw_id: i for i, raw_id in enumerate(raw_ids)}
        self.index_to_raw = {i: raw_id for i, raw_id in enumerate(raw_ids)}
    else:
        self.raw_to_contig = {}
        self.index_to_raw = {}

    # Load normalization stats
    stats_path = self.data_dir / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            self.norm_stats = json.load(f)
    else:
        self.norm_stats = {}
```

#### Memory Loading Strategies

**`load_to_memory=True`** (Default):
```python
self.images = np.load(images_path)  # Loads entire array into RAM
```

| Aspect | Value |
|--------|-------|
| Memory usage | ~3 GB for 1000 tiles |
| Access speed | Fastest (RAM access) |
| Use case | Training with sufficient RAM |

**`load_to_memory=False`** (Memory-mapped):
```python
self.images = np.load(images_path, mmap_mode="r")  # Memory-mapped, read-only
```

| Aspect | Value |
|--------|-------|
| Memory usage | Minimal (~100 MB) |
| Access speed | Slower (disk I/O) |
| Use case | Large datasets, limited RAM |

#### Memory Map Visualization

```
load_to_memory=True:                 load_to_memory=False:
┌─────────────────────────┐          ┌─────────────────────────┐
│         RAM             │          │         RAM             │
│  ┌───────────────────┐  │          │  ┌───────────────────┐  │
│  │   images array    │  │          │  │   mmap reference  │──┼──┐
│  │   (3 GB)          │  │          │  │   (100 MB)        │  │  │
│  └───────────────────┘  │          │  └───────────────────┘  │  │
└─────────────────────────┘          └─────────────────────────┘  │
                                                                  │
┌─────────────────────────┐          ┌─────────────────────────┐  │
│         Disk            │          │         Disk            │  │
│  ┌───────────────────┐  │          │  ┌───────────────────┐  │  │
│  │ train_images.npy  │  │          │  │ train_images.npy  │←─┼──┘
│  │ (unused after     │  │          │  │ (accessed on      │  │
│  │  loading)         │  │          │  │  demand)          │  │
│  └───────────────────┘  │          │  └───────────────────┘  │
└─────────────────────────┘          └─────────────────────────┘
```

#### Interview Questions

**Q: Why store both `raw_to_contig` and `index_to_raw` mappings?**

A: Different use cases:
- `raw_to_contig`: Convert raw class IDs to training indices (preprocessing)
- `index_to_raw`: Convert model predictions back to original class IDs (inference)

**Q: Why `encoding="utf-8"` for JSON files?**

A: The class_map.json contains Arabic text (`name_ar` field). Windows default encoding can't handle non-ASCII characters.

**Q: What happens if field_ids.npy doesn't exist?**

A: `self.field_ids = None`. The `__getitem__` method handles this by returning zeros.

---

### 3.2 `__len__()` Method

```python
def __len__(self) -> int:
    """Return the number of samples in the dataset."""
    return len(self.images)
```

**Simplicity**: Just returns the first dimension of the images array.

```python
# images shape: (1000, 12, 256, 256)
len(dataset)  # Returns 1000
```

---

### 3.3 `__getitem__()` Method

```python
def __getitem__(self, idx: int) -> dict:
    """
    Get a single sample.

    Returns
    -------
    dict
        Dictionary containing:
        - 'image': torch.Tensor of shape (12, 256, 256), float32
        - 'mask': torch.Tensor of shape (256, 256), int64
        - 'field_ids': torch.Tensor of shape (256, 256), int64 (if available)
    """
    image = self.images[idx].copy()  # (12, 256, 256)
    mask = self.masks[idx].copy()    # (256, 256)

    # Get field_ids if available
    if self.field_ids is not None:
        field_ids = self.field_ids[idx].copy()
    else:
        field_ids = np.zeros_like(mask)

    # Apply transforms
    if self.transforms is not None:
        image, mask, field_ids = self.transforms(image, mask, field_ids)

    return {
        "image": torch.from_numpy(image).float(),
        "mask": torch.from_numpy(mask).long(),
        "field_ids": torch.from_numpy(field_ids).long(),
    }
```

#### Why `.copy()`?

```python
image = self.images[idx].copy()  # Creates a new array
```

**Without copy** (dangerous):
```python
image = self.images[idx]  # Returns a view, not a copy
image[0, 0, 0] = 999      # Modifies the ORIGINAL array!
```

This would corrupt the dataset since transforms might modify the array in-place.

#### Why Return a Dictionary?

```python
return {
    "image": ...,
    "mask": ...,
    "field_ids": ...,
}
```

**Benefits over tuple**:
```python
# Dictionary (clear):
batch["image"].shape
batch["mask"].shape

# Tuple (confusing):
batch[0].shape  # Is this image or mask?
batch[1].shape
```

#### Tensor Conversion

```python
torch.from_numpy(image).float()  # Convert to FloatTensor
torch.from_numpy(mask).long()    # Convert to LongTensor (int64)
```

| Method | PyTorch Type | Use Case |
|--------|--------------|----------|
| `.float()` | `torch.float32` | Model inputs, activations |
| `.long()` | `torch.int64` | Class indices for CrossEntropyLoss |

**Why `.long()` for masks?**

PyTorch's `CrossEntropyLoss` requires targets as `LongTensor`:
```python
loss = nn.CrossEntropyLoss()
loss(predictions, mask)  # mask must be LongTensor
```

---

### 3.4 `num_classes` Property

```python
@property
def num_classes(self) -> int:
    """Return the number of classes."""
    if self.class_map:
        if "num_classes" in self.class_map:
            return int(self.class_map.get("num_classes", 14))
        return len(self.index_to_raw) if self.index_to_raw else 14
    return 14
```

**Fallback chain**:
1. Check `class_map["num_classes"]`
2. Check length of ID mapping
3. Default to 14

---

### 3.5 `num_channels` Property

```python
@property
def num_channels(self) -> int:
    """Return the number of input channels (bands)."""
    return self.images.shape[1]
```

Returns 12 for Sentinel-2 (12 spectral bands).

---

### 3.6 `get_class_weights()` Method

```python
def get_class_weights(self, normalize: bool = True) -> torch.Tensor:
    """
    Compute class weights for contiguous class IDs (0..num_classes-1).
    Assumes masks are already remapped to contiguous labels by prepare_dataset.py.
    """
    num_classes = self.num_classes

    # Get counts from class_map or compute from masks
    class_counts = self.class_map.get("class_counts", None)
    if class_counts is None:
        unique, counts = np.unique(self.masks, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    else:
        class_counts = {int(k): int(v) for k, v in class_counts.items()}

    total_pixels = sum(class_counts.values())

    # Inverse frequency weighting
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls_id in range(num_classes):
        count = class_counts.get(cls_id, 1)  # Avoid division by zero
        weights[cls_id] = total_pixels / (num_classes * count)

    if normalize:
        weights = weights / weights.sum() * num_classes

    return torch.from_numpy(weights)
```

#### Why Class Weights?

**The Problem**: Class imbalance

```
Class Distribution:
Background (0): 45,000,000 pixels (45%)
Wheat (1):      12,000,000 pixels (12%)
Mustard (2):     8,000,000 pixels (8%)
...
Rice (13):         500,000 pixels (0.5%)
```

Without weighting, the model would predict "Background" for everything and still get 45% accuracy!

#### Inverse Frequency Weighting Formula

```
weight[c] = total_pixels / (num_classes × count[c])
```

| Class | Count | Weight |
|-------|-------|--------|
| Background | 45,000,000 | 0.22 |
| Wheat | 12,000,000 | 0.83 |
| Rice | 500,000 | 20.0 |

Rare classes get higher weights → more gradient signal → better learning.

#### Normalization

```python
if normalize:
    weights = weights / weights.sum() * num_classes
```

After normalization, weights sum to `num_classes` (14). This keeps the loss scale consistent regardless of class distribution.

---

### 3.7 `get_sample_weights()` Method

```python
def get_sample_weights(self, power: float = 1.0) -> torch.Tensor:
    """
    Compute per-tile sampling weights based on labeled pixel fraction.

    Tiles with more labeled pixels (non-background) get higher weights,
    making them more likely to be sampled during training.

    Parameters
    ----------
    power : float
        Exponent for weight scaling. Higher values favor tiles with more labels.
        - 1.0: Linear weighting (proportional to labeled fraction)
        - 2.0: Quadratic (strongly favor high-label tiles)
        - 0.5: Square root (moderate preference)
    """
    n_tiles = len(self.masks)
    fractions = np.zeros(n_tiles, dtype=np.float32)

    for i in range(n_tiles):
        mask = self.masks[i]
        labeled_pixels = np.sum(mask != 0)  # Non-background
        total_pixels = mask.size
        fractions[i] = labeled_pixels / total_pixels

    # Add epsilon to avoid zero weights
    weights = (fractions + 1e-6) ** power

    # Normalize to sum to len(dataset)
    weights = weights / weights.sum() * n_tiles

    return torch.from_numpy(weights).float()
```

#### Why Per-Tile Weighting?

**The Problem**: Some tiles are mostly background

```
Tile A:                      Tile B:
┌────────────────────┐       ┌────────────────────┐
│████████████████████│       │    ▓▓▓▓▓▓▓▓▓▓    │
│████████████████████│       │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│████████████████████│       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│████████████████████│       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
└────────────────────┘       └────────────────────┘
 95% background               20% background
 Low information              High information
```

Tile B has more labeled pixels → more useful for training.

#### Usage with WeightedRandomSampler

```python
from torch.utils.data import WeightedRandomSampler

weights = dataset.get_sample_weights(power=1.5)
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

### 3.8 `get_class_names()` and `get_class_colors()` Methods

```python
def get_class_names(self) -> dict[int, str]:
    """Return mapping of class ID to class name."""
    if "classes" in self.class_map:
        return {
            int(k): v["name"] for k, v in self.class_map["classes"].items()
        }
    return {}

def get_class_colors(self) -> dict[int, list[int]]:
    """Return mapping of class ID to RGB color."""
    if "classes" in self.class_map:
        return {
            int(k): v["color"] for k, v in self.class_map["classes"].items()
        }
    return {}
```

**Usage for visualization**:
```python
names = dataset.get_class_names()
colors = dataset.get_class_colors()

print(names[1])    # "Wheat"
print(colors[1])   # [255, 200, 0]
```

---

## 4. Function: `get_dataloaders()`

```python
def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    transforms: Optional[Callable] = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
```

### Full Implementation

```python
def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    transforms: Optional[Callable] = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    """
    train_dataset = CropDataset(data_dir, split="train", transforms=transforms)
    val_dataset = CropDataset(data_dir, split="val", transforms=transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,         # Randomize training order
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,       # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        # Keep validation order consistent
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,      # Keep all validation samples
    )

    return train_loader, val_loader
```

### DataLoader Parameters Explained

| Parameter | Train Value | Val Value | Why |
|-----------|-------------|-----------|-----|
| `shuffle` | `True` | `False` | Training needs randomization; validation should be reproducible |
| `drop_last` | `True` | `False` | Incomplete batches can cause issues with BatchNorm; validation uses all samples |
| `pin_memory` | `True` | `True` | Faster CPU→GPU transfer |
| `num_workers` | 4 | 4 | Parallel data loading (set to 0 on Windows/Jupyter) |

### Why `drop_last=True` for Training?

**Problem with incomplete batches**:
```
Dataset size: 1000
Batch size: 32
Full batches: 31 (992 samples)
Last batch: 8 samples

With BatchNorm, small batches have unreliable statistics.
```

**Solution**: Drop the last incomplete batch during training.

### Why `pin_memory=True`?

**Memory pinning** keeps data in non-pageable RAM, enabling faster GPU transfer:

```
pin_memory=False:               pin_memory=True:
┌───────────────────────┐       ┌───────────────────────┐
│    Pageable RAM       │       │    Pinned RAM         │
│    ┌─────────┐        │       │    ┌─────────┐        │
│    │  batch  │        │       │    │  batch  │        │
│    └────┬────┘        │       │    └────┬────┘        │
│         │             │       │         │             │
│    ┌────▼────┐        │       │         │             │
│    │  copy   │ (slow) │       │         │  (direct)   │
│    └────┬────┘        │       │         │             │
└─────────┼─────────────┘       └─────────┼─────────────┘
          │                               │
┌─────────▼─────────────┐       ┌─────────▼─────────────┐
│       GPU RAM         │       │       GPU RAM         │
│    ┌─────────┐        │       │    ┌─────────┐        │
│    │  batch  │        │       │    │  batch  │        │
│    └─────────┘        │       │    └─────────┘        │
└───────────────────────┘       └───────────────────────┘
```

### Windows/Jupyter Compatibility

```python
# On Windows/Jupyter, multiprocessing can fail
train_loader, val_loader = get_dataloaders(
    "data/processed",
    batch_size=32,
    num_workers=0  # Single-threaded loading
)
```

---

## 5. Class: `RandomFlipRotate`

```python
class RandomFlipRotate:
    """
    Random horizontal/vertical flip and 90-degree rotation augmentation.

    Applies the same transform to image, mask, and field_ids.
    """

    def __init__(self, p_flip: float = 0.5, p_rotate: float = 0.5):
        """
        Parameters
        ----------
        p_flip : float
            Probability of horizontal/vertical flip
        p_rotate : float
            Probability of 90-degree rotation
        """
        self.p_flip = p_flip
        self.p_rotate = p_rotate

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, field_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random transforms to image, mask, and field_ids."""

        # Random horizontal flip
        if np.random.random() < self.p_flip:
            image = np.flip(image, axis=2).copy()      # Flip width (axis 2 for CHW)
            mask = np.flip(mask, axis=1).copy()        # Flip width (axis 1 for HW)
            field_ids = np.flip(field_ids, axis=1).copy()

        # Random vertical flip
        if np.random.random() < self.p_flip:
            image = np.flip(image, axis=1).copy()      # Flip height
            mask = np.flip(mask, axis=0).copy()
            field_ids = np.flip(field_ids, axis=0).copy()

        # Random 90-degree rotation
        if np.random.random() < self.p_rotate:
            k = np.random.randint(1, 4)                # 1, 2, or 3 rotations
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            field_ids = np.rot90(field_ids, k, axes=(0, 1)).copy()

        return image, mask, field_ids
```

### Why These Specific Augmentations?

**Satellite imagery is rotationally invariant**:
- A wheat field looks the same whether viewed from north, south, east, or west
- Flips and 90° rotations don't change the semantic content

**Safe augmentations for satellite data**:
| Augmentation | Safe? | Why |
|--------------|-------|-----|
| Horizontal flip | ✓ | Geography unchanged |
| Vertical flip | ✓ | Geography unchanged |
| 90° rotation | ✓ | Geography unchanged |
| **Color jitter** | **✗** | Spectral signatures are meaningful! |
| **Random crop** | **✗** | Fields might be cut |

### Axis Handling

**Image**: Shape `(C, H, W)` = `(12, 256, 256)`
```
Axis 0: Channels (don't flip!)
Axis 1: Height
Axis 2: Width
```

**Mask/Field_IDs**: Shape `(H, W)` = `(256, 256)`
```
Axis 0: Height
Axis 1: Width
```

### Why `.copy()` After Operations?

```python
image = np.flip(image, axis=2).copy()
```

`np.flip()` returns a **view** (negative stride), not a new array. This can cause:
1. Issues with PyTorch tensor conversion
2. Memory layout problems
3. Unexpected behavior in downstream code

`.copy()` creates a proper contiguous array.

### Rotation Visualization

```
k=1 (90° CCW):           k=2 (180°):              k=3 (270° CCW):
┌─────┐                  ┌─────┐                  ┌─────┐
│ 1 2 │   →  ┌─────┐     │ 1 2 │   →  ┌─────┐    │ 1 2 │   →  ┌─────┐
│ 3 4 │      │ 2 4 │     │ 3 4 │      │ 4 3 │    │ 3 4 │      │ 3 1 │
└─────┘      │ 1 3 │     └─────┘      │ 2 1 │    └─────┘      │ 4 2 │
             └─────┘                  └─────┘                 └─────┘
```

### Usage Example

```python
transform = RandomFlipRotate(p_flip=0.5, p_rotate=0.5)

dataset = CropDataset(
    "data/processed",
    split="train",
    transforms=transform
)

# Each time you access a sample, transforms are applied randomly
sample1 = dataset[0]  # Might be flipped
sample2 = dataset[0]  # Might be rotated differently
```

---

## 6. Data Flow Through the Pipeline

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                      │
│                                                                              │
│  for epoch in range(epochs):                                                │
│      for batch in train_loader:  ←─────────────────────────────────┐       │
│          images = batch["image"]     # (B, 12, 256, 256)           │       │
│          masks = batch["mask"]       # (B, 256, 256)               │       │
│                                                                     │       │
│          predictions = model(images)  # Forward pass               │       │
│          loss = criterion(predictions, masks)                      │       │
│          loss.backward()             # Backward pass               │       │
│          optimizer.step()            # Update weights              │       │
└─────────────────────────────────────────────────────────────────────┼───────┘
                                                                      │
┌─────────────────────────────────────────────────────────────────────┼───────┐
│                           DataLoader                                │       │
│                                                                     │       │
│  ┌────────────────────────────────────────────────────────────┐    │       │
│  │  Batching:                                                  │    │       │
│  │    sample[0], sample[1], ..., sample[31] → batch           │    │       │
│  │                                                              │    │       │
│  │  Collation:                                                 │    │       │
│  │    Stack images: (12, 256, 256) × 32 → (32, 12, 256, 256)  │    │       │
│  │    Stack masks:  (256, 256) × 32 → (32, 256, 256)          │    │       │
│  └────────────────────────────────────────────────────────────┘    │       │
│                            ↑                                        │       │
└────────────────────────────┼────────────────────────────────────────┘       │
                             │                                                 │
┌────────────────────────────┼────────────────────────────────────────────────┐
│                      CropDataset.__getitem__(idx)                            │
│                            │                                                 │
│  ┌─────────────────────────┼────────────────────────────────────────────┐  │
│  │                         │                                             │  │
│  │  1. Load from numpy:    │                                             │  │
│  │     image = self.images[idx].copy()   # (12, 256, 256) float32       │  │
│  │     mask = self.masks[idx].copy()     # (256, 256) int64             │  │
│  │     field_ids = self.field_ids[idx].copy()                           │  │
│  │                         │                                             │  │
│  │  2. Apply transforms:   │                                             │  │
│  │     if self.transforms: │                                             │  │
│  │         image, mask, field_ids = self.transforms(...)                │  │
│  │                         │                                             │  │
│  │  3. Convert to tensors: │                                             │  │
│  │     return {            │                                             │  │
│  │         "image": torch.from_numpy(image).float(),                    │  │
│  │         "mask": torch.from_numpy(mask).long(),                       │  │
│  │         "field_ids": torch.from_numpy(field_ids).long(),             │  │
│  │     }                   │                                             │  │
│  └─────────────────────────┼────────────────────────────────────────────┘  │
│                            │                                                 │
└────────────────────────────┼─────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────────────────────┐
│                   RandomFlipRotate Transform (optional)                       │
│                            │                                                  │
│  ┌─────────────────────────┼────────────────────────────────────────────┐   │
│  │                         │                                             │   │
│  │  Original:              │    After random transforms:                │   │
│  │  ┌───────────────┐      │    ┌───────────────┐                       │   │
│  │  │    🌾🌾🌾     │      │    │     🌾🌽      │                       │   │
│  │  │    🌽🌽       │   →  │    │   🌽🌽🌾🌾    │ (rotated 90°)        │   │
│  │  └───────────────┘      │    └───────────────┘                       │   │
│  │                         │                                             │   │
│  └─────────────────────────┼────────────────────────────────────────────┘   │
│                            │                                                  │
└────────────────────────────┼──────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────────────┐
│                      .npy Files (Preprocessed Data)                           │
│                            │                                                  │
│  data/processed/           │                                                  │
│  ├── train_images.npy ─────┘  Shape: (N_train, 12, 256, 256)                 │
│  ├── train_masks.npy          Shape: (N_train, 256, 256)                     │
│  ├── train_field_ids.npy      Shape: (N_train, 256, 256)                     │
│  ├── val_images.npy           Shape: (N_val, 12, 256, 256)                   │
│  ├── val_masks.npy            Shape: (N_val, 256, 256)                       │
│  └── val_field_ids.npy        Shape: (N_val, 256, 256)                       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Interview Questions & Answers

### Dataset Design Questions

**Q1: Why inherit from `torch.utils.data.Dataset` instead of using a custom class?**

A: PyTorch's DataLoader expects Dataset interface:
- `__len__()` for knowing dataset size
- `__getitem__()` for random access
- Enables automatic batching, shuffling, multiprocessing

**Q2: Why return a dictionary from `__getitem__` instead of a tuple?**

A: Self-documenting code:
```python
# Dictionary (clear)
loss = criterion(output, batch["mask"])

# Tuple (confusing)
loss = criterion(output, batch[1])  # What is batch[1]?
```

**Q3: What would happen without `.copy()` in `__getitem__`?**

A: The transform might modify the original array in `self.images`, corrupting the dataset. Next access to the same index would return corrupted data.

### Performance Questions

**Q4: When would you use `load_to_memory=False`?**

A: When dataset doesn't fit in RAM:
- 10,000+ tiles
- Limited RAM (<8 GB)
- Testing/debugging only a few samples

Trade-off: 10-100× slower iteration.

**Q5: Why `pin_memory=True`?**

A: Enables asynchronous CPU→GPU transfer. Data is copied to pinned (non-pageable) memory, allowing CUDA to transfer directly without intermediate copy.

**Q6: What's the impact of `num_workers`?**

A:
- `num_workers=0`: Single-threaded (safe but slow)
- `num_workers=4`: 4 worker processes loading data in parallel
- Higher values: Diminishing returns, more RAM usage

### Augmentation Questions

**Q7: Why not use torchvision transforms?**

A: Our data has 3 components (image, mask, field_ids) that must transform together. Standard torchvision transforms only handle images.

**Q8: Would color augmentation (brightness, contrast) be safe?**

A: **No!** Satellite bands have physical meaning:
- Band values represent actual reflectance
- Changing brightness would simulate different atmospheric conditions
- This could confuse the model about spectral signatures

**Q9: Why only 90° rotations instead of arbitrary angles?**

A:
- 90° rotations are lossless (no interpolation)
- Arbitrary rotations need interpolation → blurry edges
- Agricultural fields are often rectangular, aligning with cardinal directions

### Class Weighting Questions

**Q10: What's the difference between class weights and sample weights?**

A:
- **Class weights** (`get_class_weights()`): Per-class loss scaling (used in CrossEntropyLoss)
- **Sample weights** (`get_sample_weights()`): Per-tile sampling probability (used in WeightedRandomSampler)

Both address imbalance, but at different levels.

**Q11: Why normalize class weights to sum to `num_classes`?**

A: Keeps loss scale consistent. Without normalization:
- Unbalanced dataset: loss ~0.5
- Balanced dataset: loss ~2.0

Normalization ensures comparable loss values across datasets.

### Edge Cases

**Q12: What happens if a tile has all background (class 0)?**

A: The tile is still included. The model learns to predict background where appropriate. Filtering out such tiles would reduce training data.

**Q13: What if masks contain unexpected class IDs?**

A: If raw IDs weren't properly remapped:
```python
mask = lut[mask]  # Unknown ID → maps to 0 (background)
```

This silently corrupts labels. Solution: Validate class IDs during preprocessing.

**Q14: What happens with empty transforms (transforms=None)?**

A: No augmentation applied. Useful for:
- Validation (should be deterministic)
- Testing (want reproducible results)
- Debugging (isolate augmentation effects)
