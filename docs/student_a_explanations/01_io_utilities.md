# I/O Utilities Module - Complete Documentation

**File**: `agrovision_core/src/agrovision_core/utils/io.py`

**Purpose**: Provides utilities for loading GeoTIFF satellite imagery files and resampling bands to a consistent resolution.

*A GeoTIFF is a TIFF image file with extra georeferencing information embedded in its header*

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Function: `_require_rasterio()`](#3-function-_require_rasterio)
4. [Function: `load_band_tiff()`](#4-function-load_band_tiff)
5. [Function: `load_label_tiff()`](#5-function-load_label_tiff)
6. [Function: `resample_to_target_size()`](#6-function-resample_to_target_size)
7. [Function: `get_tile_ids_from_source()`](#7-function-get_tile_ids_from_source)
8. [Function: `get_band_filepath()`](#8-function-get_band_filepath)
9. [Function: `get_label_filepath()`](#9-function-get_label_filepath)
10. [Function: `load_config()`](#10-function-load_config)
11. [Function: `ensure_dir()`](#11-function-ensure_dir)
12. [Function: `write_json()`](#12-function-write_json)
13. [Function: `resolve_path()`](#13-function-resolve_path)
14. [Interview Questions & Answers](#14-interview-questions--answers)

---

## 1. Module Overview

### What This Module Does

This module is the **foundation of the data pipeline**. It handles:

| Responsibility                   | Functions                                     |
| -------------------------------- | --------------------------------------------- |
| Loading satellite bands          | `load_band_tiff()`                            |
| Loading label masks              | `load_label_tiff()`                           |
| Resampling different resolutions | `resample_to_target_size()`                   |
| File path construction           | `get_band_filepath()`, `get_label_filepath()` |
| Tile discovery                   | `get_tile_ids_from_source()`                  |
| Configuration management         | `load_config()`, `write_json()`               |
| Directory utilities              | `ensure_dir()`, `resolve_path()`              |

### Why This Module Exists

Sentinel-2 satellite data comes in **GeoTIFF format** with bands at **different native resolutions**:

| Resolution | Bands                        | Pixel Size       |
| ---------- | ---------------------------- | ---------------- |
| 10m        | B02, B03, B04, B08           | 256×256 natively |
| 20m        | B05, B06, B07, B8A, B11, B12 | 128×128 natively |
| 60m        | B01, B09                     | ~43×43 natively  |

**Problem**: We need all 12 bands at the **same 256×256 resolution** to stack them into a single tensor.

**Solution**: This module provides resampling functions to upscale lower-resolution bands.

---

## 2. Imports and Dependencies

```python
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
```

### Line-by-Line Explanation

| Line                            | Import               | Purpose                            |
| ------------------------------- | -------------------- | ---------------------------------- |
| `import json`                   | Standard library     | Read/write JSON files for metadata |
| `from pathlib import Path`      | Standard library     | Cross-platform path handling       |
| `from typing import Any, Union` | Standard library     | Type hints for function signatures |
| `import numpy as np`            | Third-party          | Array operations for image data    |
| `import yaml`                   | Third-party (PyYAML) | Read YAML configuration files      |

### Why These Imports?

**Q: Why `pathlib.Path` instead of `os.path`?**

| Aspect      | `os.path`            | `pathlib.Path`       |
| ----------- | -------------------- | -------------------- |
| Syntax      | `os.path.join(a, b)` | `a / b`              |
| Type        | Returns strings      | Returns Path objects |
| Methods     | Separate functions   | Object methods       |
| Readability | More verbose         | More Pythonic        |

```python
# os.path approach (older)
import os
filepath = os.path.join(source_dir, tile_dir, filename)
exists = os.path.exists(filepath)

# pathlib approach (modern, used in this module)
from pathlib import Path
filepath = source_dir / tile_dir / filename
exists = filepath.exists()
```

**Q: Why are `rasterio` and `scipy` not imported at the top?**

These are **optional dependencies** that may not be installed. They're imported **lazily** inside functions:
- `rasterio` - Only needed for GeoTIFF I/O (heavy dependency)
- `scipy` - Only needed for resampling (alternative to rasterio resampling)

This allows the module to be imported even if these packages aren't installed, as long as those specific functions aren't called.

---

## 3. Function: `_require_rasterio()`

```python
def _require_rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rasterio is required for GeoTIFF I/O. Install it to use "
            "load_band_tiff/load_label_tiff."
        ) from exc
    return rasterio, Resampling
```

### Line-by-Line Breakdown

| Line | Code                                    | What It Does                                  |
| ---- | --------------------------------------- | --------------------------------------------- |
| 1    | `def _require_rasterio():`              | Define helper function (underscore = private) |
| 2    | `try:`                                  | Start exception handling block                |
| 3    | `import rasterio`                       | Attempt to import rasterio library            |
| 4    | `from rasterio.enums import Resampling` | Import resampling enum types                  |
| 5    | `except ModuleNotFoundError as exc:`    | Catch if rasterio isn't installed             |
| 6-8  | `raise ModuleNotFoundError(...)`        | Re-raise with helpful error message           |
| 9    | `return rasterio, Resampling`           | Return both imports for caller to use         |

### Why This Pattern?

**Lazy Loading Pattern Benefits:**

```
                    Without Lazy Loading          With Lazy Loading
                    ─────────────────────         ─────────────────
Import time         Fails immediately if          Succeeds (no rasterio needed)
                    rasterio missing

When rasterio       At import time                Only when load_band_tiff()
is actually needed                                is called

Error clarity       Generic ImportError           Custom message explaining
                                                  exactly what to install
```

### Interview Questions

**Q: Why use `from exc` in the raise statement?**

```python
raise ModuleNotFoundError(...) from exc
```

This creates an **exception chain**. The original exception (`exc`) is preserved as the `__cause__` of the new exception. This helps debugging by showing both:
1. The user-friendly message we wrote
2. The original technical error

**Output without `from exc`:**
```
ModuleNotFoundError: rasterio is required for GeoTIFF I/O...
```

**Output with `from exc`:**
```
ModuleNotFoundError: No module named 'rasterio'

The above exception was the direct cause of the following exception:

ModuleNotFoundError: rasterio is required for GeoTIFF I/O...
```

**Q: Why return a tuple `(rasterio, Resampling)` instead of just importing globally?**

1. **Encapsulation**: Keeps the import local to where it's needed
2. **Testing**: Easier to mock for unit tests
3. **Optional dependency**: Module works without rasterio for non-GeoTIFF operations

---

## 4. Function: `load_band_tiff()`

```python
def load_band_tiff(
    filepath: Union[str, Path],
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
```

### Full Code with Annotations

```python
def load_band_tiff(
    filepath: Union[str, Path],        # Accept both string and Path
    target_size: tuple[int, int] = (256, 256),  # Default to 256×256
) -> np.ndarray:                       # Return numpy array
    """
    Load a single-band GeoTIFF and resample to target size.
    """
    filepath = Path(filepath)          # Convert string to Path if needed

    rasterio, Resampling = _require_rasterio()  # Lazy import

    with rasterio.open(filepath) as src:  # Open GeoTIFF file
        # Check if resampling is needed
        if src.height == target_size[0] and src.width == target_size[1]:
            # No resampling needed - already correct size
            data = src.read(1).astype(np.float32)
        else:
            # Resample to target size using bilinear interpolation
            data = src.read(
                1,                          # Read band 1 (GeoTIFFs are 1-indexed)
                out_shape=target_size,      # Output dimensions
                resampling=Resampling.bilinear,  # Interpolation method
            ).astype(np.float32)

    return data
```

### Parameter Deep Dive

| Parameter     | Type               | Default      | Purpose                  |
| ------------- | ------------------ | ------------ | ------------------------ |
| `filepath`    | `Union[str, Path]` | Required     | Path to the GeoTIFF file |
| `target_size` | `tuple[int, int]`  | `(256, 256)` | Output (height, width)   |

### Why `Union[str, Path]`?

```python
# Both of these work:
load_band_tiff("data/band.tif")           # String path
load_band_tiff(Path("data/band.tif"))     # Path object

# The function normalizes internally:
filepath = Path(filepath)  # String -> Path, Path -> Path (no-op)
```

### Resampling Methods Explained

| Method     | When to Use             | Mathematical Approach      |
| ---------- | ----------------------- | -------------------------- |
| `nearest`  | Labels/masks            | Take nearest pixel value   |
| `bilinear` | Continuous data (bands) | Linear interpolation in 2D |
| `cubic`    | High-quality resize     | Cubic polynomial fitting   |
| `lanczos`  | Maximum quality         | Sinc function windowed     |

**Why bilinear for bands?**

```
Original 128×128 (20m band)      After bilinear to 256×256
┌─────────────────────┐          ┌─────────────────────────────┐
│ A │ B │             │          │ A │ (A+B)/2 │ B │           │
├───┼───┤             │    →     ├───┼─────────┼───┤           │
│ C │ D │             │          │(A+C)/2│ avg │(B+D)/2│       │
│   │   │             │          ├───┼─────────┼───┤           │
└─────────────────────┘          │ C │ (C+D)/2 │ D │           │
                                 └─────────────────────────────┘
```

Bilinear creates smooth transitions between pixels, which is appropriate for continuous reflectance values.

### Context Manager (`with` statement)

```python
with rasterio.open(filepath) as src:
    data = src.read(1)
# File automatically closed here, even if exception occurs
```

**Why use `with`?**

| Without `with`                   | With `with`        |
| -------------------------------- | ------------------ |
| Must manually call `src.close()` | Automatic cleanup  |
| Risk of resource leaks           | Guaranteed cleanup |
| More verbose                     | Cleaner code       |
| Exception handling complex       | Exception-safe     |

### Input/Output Example

**Input**: 20m resolution band file (128×128 pixels)
```
Shape: (128, 128)
Values: [0, 10000] (Sentinel-2 reflectance * 10000)
dtype: uint16
```

**Output**: Resampled array
```
Shape: (256, 256)
Values: [0.0, 10000.0] (same range, now float)
dtype: float32
```

### What If Questions

**Q: What if the file doesn't exist?**
```python
>>> load_band_tiff("nonexistent.tif")
rasterio.errors.RasterioIOError: nonexistent.tif: No such file or directory
```

**Q: What if we used `nearest` instead of `bilinear`?**

```
Bilinear (smooth):           Nearest (blocky):
┌───────────────┐            ┌───────────────┐
│░░░▒▒▒▓▓▓████│            │░░░░▒▒▒▒▓▓▓▓████│
│░░░▒▒▒▓▓▓████│            │░░░░▒▒▒▒▓▓▓▓████│
└───────────────┘            └───────────────┘

Nearest creates "staircase" artifacts at edges.
For satellite bands (continuous values), bilinear is better.
```

**Q: What if `target_size` matches the source size?**

The optimization check on line 55-57 skips resampling entirely:
```python
if src.height == target_size[0] and src.width == target_size[1]:
    data = src.read(1).astype(np.float32)  # Direct read, no resampling
```

This saves computation for 10m bands that are already 256×256.

---

## 5. Function: `load_label_tiff()`

```python
def load_label_tiff(
    filepath: Union[str, Path],
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
```

### Full Code with Annotations

```python
def load_label_tiff(
    filepath: Union[str, Path],
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Load a label mask GeoTIFF and resample to target size.
    Uses nearest-neighbor resampling to preserve discrete class values.
    """
    filepath = Path(filepath)

    rasterio, Resampling = _require_rasterio()

    with rasterio.open(filepath) as src:
        if src.height == target_size[0] and src.width == target_size[1]:
            data = src.read(1).astype(np.int64)
        else:
            # Use nearest neighbor for labels to preserve discrete values
            data = src.read(
                1,
                out_shape=target_size,
                resampling=Resampling.nearest,  # CRITICAL: nearest, not bilinear
            ).astype(np.int64)

    return data
```

### Key Difference from `load_band_tiff()`

| Aspect       | `load_band_tiff()`     | `load_label_tiff()` |
| ------------ | ---------------------- | ------------------- |
| Resampling   | `bilinear`             | `nearest`           |
| Output dtype | `float32`              | `int64`             |
| Use case     | Continuous reflectance | Discrete class IDs  |

### Why Nearest-Neighbor for Labels?

**The Problem with Bilinear for Labels:**

```
Original labels (2×2):     Bilinear interpolation (4×4):
┌─────┬─────┐              ┌───┬───┬───┬───┐
│  1  │  2  │              │ 1 │1.5│1.5│ 2 │  ← 1.5 is NOT a valid class!
├─────┼─────┤      →       ├───┼───┼───┼───┤
│  3  │  4  │              │ 2 │2.5│2.5│ 3 │
└─────┴─────┘              ├───┼───┼───┼───┤
                           │ 2 │2.5│3.5│ 3 │
                           ├───┼───┼───┼───┤
                           │ 3 │3.5│3.5│ 4 │
                           └───┴───┴───┴───┘
```

Bilinear creates **fractional values** between classes, which are meaningless!

**Nearest-neighbor preserves discrete values:**

```
Original labels (2×2):     Nearest-neighbor (4×4):
┌─────┬─────┐              ┌───┬───┬───┬───┐
│  1  │  2  │              │ 1 │ 1 │ 2 │ 2 │  ← All valid classes!
├─────┼─────┤      →       ├───┼───┼───┼───┤
│  3  │  4  │              │ 1 │ 1 │ 2 │ 2 │
└─────┴─────┘              ├───┼───┼───┼───┤
                           │ 3 │ 3 │ 4 │ 4 │
                           ├───┼───┼───┼───┤
                           │ 3 │ 3 │ 4 │ 4 │
                           └───┴───┴───┴───┘
```

### Why `int64` for Labels?

| dtype       | Range           | Why not?                             |
| ----------- | --------------- | ------------------------------------ |
| `int8`      | -128 to 127     | Could work but limited               |
| `int16`     | -32768 to 32767 | Could work                           |
| `int32`     | ~±2 billion     | Sufficient                           |
| **`int64`** | ~±9 quintillion | **PyTorch default for `LongTensor`** |

PyTorch's `CrossEntropyLoss` expects targets as `LongTensor` (int64). Using int64 from the start avoids conversion.

### Interview Questions

**Q: Why not use `uint8` since we only have 14 classes (0-13)?**

While `uint8` (0-255) would work for storage, there are issues:
1. PyTorch expects `int64` for class indices
2. Negative values might appear during processing (errors)
3. Memory savings are minimal (256×256 = 64KB per tile vs 512KB)
4. Conversion overhead on every batch

**Q: What happens if you accidentally use bilinear for labels?**

```python
# WRONG:
labels = load_band_tiff("labels.tif")  # Uses bilinear!

# Result:
>>> np.unique(labels)
array([0., 0.5, 1., 1.5, 2., 2.5, 3., ...])  # Fractional values!

# Training with CrossEntropyLoss:
>>> loss = nn.CrossEntropyLoss()(predictions, labels.long())
# The .long() truncates 1.5 → 1, 2.5 → 2, etc.
# This SILENTLY corrupts your labels!
```

---

## 6. Function: `resample_to_target_size()`

```python
def resample_to_target_size(
    data: np.ndarray,
    target_size: tuple[int, int],
    method: str = "bilinear",
) -> np.ndarray:
```

### Full Code with Annotations

```python
def resample_to_target_size(
    data: np.ndarray,
    target_size: tuple[int, int],
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resample a 2D array to target size.
    """
    from scipy.ndimage import zoom  # Lazy import

    if data.shape == target_size:
        return data  # Early return if no resampling needed

    # Calculate zoom factors
    zoom_factors = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])

    if method == "nearest":
        return zoom(data, zoom_factors, order=0)  # order=0 is nearest
    else:  # bilinear
        return zoom(data, zoom_factors, order=1)  # order=1 is bilinear
```

### When Is This Function Used?

This function provides **rasterio-free resampling** using scipy. It's an alternative when:
1. Data is already in numpy arrays (not on disk)
2. Rasterio isn't installed
3. Simple array resizing is needed

### Zoom Order Parameter

| Order | Method           | Use Case                  |
| ----- | ---------------- | ------------------------- |
| 0     | Nearest-neighbor | Labels, categorical data  |
| 1     | Bilinear         | Continuous data (default) |
| 2     | Quadratic        | Smooth images             |
| 3     | Cubic            | High-quality resize       |

### Example Calculation

```python
# 20m band at 128×128 → 256×256
data = np.random.rand(128, 128)
target_size = (256, 256)

zoom_factors = (256/128, 256/128)  # = (2.0, 2.0)

# Each dimension is scaled by 2x
result = zoom(data, (2.0, 2.0), order=1)
print(result.shape)  # (256, 256)
```

### Difference from rasterio Resampling

| Aspect           | `scipy.ndimage.zoom`  | `rasterio` resampling  |
| ---------------- | --------------------- | ---------------------- |
| Input            | numpy array           | GeoTIFF file on disk   |
| Geospatial aware | No                    | Yes (preserves CRS)    |
| Memory           | Array must fit in RAM | Can stream large files |
| Dependencies     | scipy (common)        | rasterio (specialized) |

---

## 7. Function: `get_tile_ids_from_source()`

```python
def get_tile_ids_from_source(source_dir: Union[str, Path]) -> list[str]:
```

### Full Code with Annotations

```python
def get_tile_ids_from_source(source_dir: Union[str, Path]) -> list[str]:
    """
    Extract unique tile IDs from the source directory.

    The data is organized in subdirectories:
    source/ref_agrifieldnet_competition_v1_source_{tile_id}/
    """
    source_dir = Path(source_dir)
    tile_ids = set()  # Use set to avoid duplicates

    # Each tile has its own subdirectory
    for tile_dir in source_dir.iterdir():
        if tile_dir.is_dir() and tile_dir.name.startswith("ref_agrifieldnet"):
            # Extract tile_id from directory name
            # Pattern: ref_agrifieldnet_competition_v1_source_{tile_id}
            parts = tile_dir.name.split("_")
            if len(parts) >= 5:
                tile_id = parts[5]  # tile_id is after "source"
                tile_ids.add(tile_id)

    return sorted(tile_ids)
```

### Directory Structure Explanation

```
data/agrifieldnet/source/
├── ref_agrifieldnet_competition_v1_source_001c1/
│   ├── ref_agrifieldnet_competition_v1_source_001c1_B01_10m.tif
│   ├── ref_agrifieldnet_competition_v1_source_001c1_B02_10m.tif
│   └── ... (12 bands)
├── ref_agrifieldnet_competition_v1_source_002a5/
│   └── ...
└── ref_agrifieldnet_competition_v1_source_003b7/
    └── ...
```

### String Splitting Logic

```python
name = "ref_agrifieldnet_competition_v1_source_001c1"
parts = name.split("_")
# parts = ["ref", "agrifieldnet", "competition", "v1", "source", "001c1"]
# Index:     0         1              2          3       4        5

tile_id = parts[5]  # "001c1"
```

### Why Use a Set?

```python
tile_ids = set()  # Automatically handles duplicates

# If we used a list:
tile_ids = []
for tile_dir in source_dir.iterdir():
    tile_id = extract_id(tile_dir)
    if tile_id not in tile_ids:  # Extra check needed
        tile_ids.append(tile_id)
```

Sets provide O(1) duplicate checking vs O(n) for lists.

### Why Return Sorted?

```python
return sorted(tile_ids)
```

**Benefits of sorting:**
1. **Reproducibility**: Same order every run
2. **Debugging**: Easier to find specific tiles
3. **Deterministic splits**: Train/val split will be consistent

### Interview Questions

**Q: What if the directory structure changes?**

The function is **tightly coupled** to AgriFieldNet's naming convention. If the structure changed:
- Option 1: Update the parsing logic
- Option 2: Use regex for more flexible matching
- Option 3: Accept a pattern parameter

```python
# More flexible version with regex:
import re

def get_tile_ids_flexible(source_dir, pattern=r"source_(\w+)$"):
    tile_ids = set()
    for tile_dir in Path(source_dir).iterdir():
        match = re.search(pattern, tile_dir.name)
        if match:
            tile_ids.add(match.group(1))
    return sorted(tile_ids)
```

**Q: Why `iterdir()` instead of `glob()`?**

```python
# iterdir() - all items, then filter
for item in source_dir.iterdir():
    if item.is_dir() and item.name.startswith("ref"):
        ...

# glob() - pattern matching built-in
for item in source_dir.glob("ref_agrifieldnet*"):
    ...
```

Both work; `iterdir()` with explicit checks is more readable for this case.

---

## 8. Function: `get_band_filepath()`

```python
def get_band_filepath(
    source_dir: Union[str, Path],
    tile_id: str,
    band_name: str,
) -> Path:
```

### Full Code with Annotations

```python
def get_band_filepath(
    source_dir: Union[str, Path],
    tile_id: str,
    band_name: str,
) -> Path:
    """
    Construct the filepath for a specific band of a tile.

    Data is organized as:
    source/ref_agrifieldnet_competition_v1_source_{tile_id}/{filename}.tif
    """
    source_dir = Path(source_dir)

    # Subdirectory: ref_agrifieldnet_competition_v1_source_{tile_id}
    tile_dir = f"ref_agrifieldnet_competition_v1_source_{tile_id}"

    # Filename: ref_agrifieldnet_competition_v1_source_{tile_id}_{band}_10m.tif
    filename = f"ref_agrifieldnet_competition_v1_source_{tile_id}_{band_name}_10m.tif"

    return source_dir / tile_dir / filename
```

### Path Construction Example

```python
>>> get_band_filepath("data/agrifieldnet/source", "001c1", "B04")
Path('data/agrifieldnet/source/ref_agrifieldnet_competition_v1_source_001c1/ref_agrifieldnet_competition_v1_source_001c1_B04_10m.tif')
```

### Why This Pattern?

The AgriFieldNet dataset uses this specific naming convention from the Radiant MLHub. The function:
1. **Encapsulates** the naming logic in one place
2. **Prevents errors** from manually constructing paths
3. **Makes refactoring easy** if the convention changes

### Note About `_10m.tif`

All bands are stored with `_10m.tif` suffix regardless of their native resolution. This is because they've been pre-resampled during dataset creation. The actual native resolutions vary (10m, 20m, 60m), but files are named consistently.

---

## 9. Function: `get_label_filepath()`

```python
def get_label_filepath(
    labels_dir: Union[str, Path],
    tile_id: str,
    label_type: str = "raster",
) -> Path:
```

### Full Code with Annotations

```python
def get_label_filepath(
    labels_dir: Union[str, Path],
    tile_id: str,
    label_type: str = "raster",
) -> Path:
    """
    Construct the filepath for a label file.

    Parameters
    ----------
    label_type : str
        Type of label: 'raster' for crop labels, 'field_ids' for field IDs
    """
    labels_dir = Path(labels_dir)

    if label_type == "field_ids":
        filename = f"ref_agrifieldnet_competition_v1_labels_train_{tile_id}_field_ids.tif"
    else:  # raster labels
        filename = f"ref_agrifieldnet_competition_v1_labels_train_{tile_id}.tif"

    return labels_dir / filename
```

### Two Types of Labels

| Label Type    | Filename Pattern                    | Content                     |
| ------------- | ----------------------------------- | --------------------------- |
| `"raster"`    | `..._train_{tile_id}.tif`           | Crop class per pixel (0-36) |
| `"field_ids"` | `..._train_{tile_id}_field_ids.tif` | Unique field ID per pixel   |

### Why Field IDs?

The AgriFieldNet challenge evaluates predictions **per field**, not per pixel. Each agricultural field has a unique ID, and the model should predict a single class for each field.

```
Crop Labels:                   Field IDs:
┌─────────────────────┐        ┌─────────────────────┐
│ 1 │ 1 │ 2 │ 2 │ 2 │        │ 101 │ 101 │ 102 │ 102 │ 102 │
│ 1 │ 1 │ 2 │ 2 │ 2 │        │ 101 │ 101 │ 102 │ 102 │ 102 │
│ 3 │ 3 │ 3 │ 2 │ 2 │        │ 103 │ 103 │ 103 │ 102 │ 102 │
└─────────────────────┘        └─────────────────────┘

Field 101 → Wheat (class 1)
Field 102 → Mustard (class 2)
Field 103 → Lentil (class 3)
```

---

## 10. Function: `load_config()`

```python
def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file into a dict."""
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
```

### Why `encoding="utf-8"`?

The config file contains **Arabic text** for crop names:
```yaml
classes:
  1:
    name: Wheat
    name_ar: قمح    # Arabic text
```

Without explicit encoding on Windows:
```python
# Windows default encoding (cp1252) can't decode Arabic
>>> open("config.yaml").read()
UnicodeDecodeError: 'charmap' codec can't decode byte...
```

### Why `yaml.safe_load()` instead of `yaml.load()`?

| Function           | Security                                  | Features               |
| ------------------ | ----------------------------------------- | ---------------------- |
| `yaml.load()`      | **UNSAFE** - can execute arbitrary Python | Full YAML spec         |
| `yaml.safe_load()` | Safe - only basic types                   | Limited but sufficient |

```yaml
# Malicious YAML that yaml.load() would execute:
!!python/object/apply:os.system ['rm -rf /']
```

`yaml.safe_load()` prevents code execution attacks.

---

## 11. Function: `ensure_dir()`

```python
def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

### Parameter Explanation

| Parameter       | Effect                                  |
| --------------- | --------------------------------------- |
| `parents=True`  | Create parent directories if needed     |
| `exist_ok=True` | Don't error if directory already exists |

### Example

```python
>>> ensure_dir("data/processed/train")
# Creates: data/ → data/processed/ → data/processed/train/
# Returns: Path('data/processed/train')
```

### Why Return the Path?

Allows chaining:
```python
output_file = ensure_dir("data/processed") / "train_images.npy"
np.save(output_file, images)
```

---

## 12. Function: `write_json()`

```python
def write_json(path: Union[str, Path], obj: Any) -> None:
    """Write a JSON file with UTF-8 encoding."""
    path = Path(path)
    ensure_dir(path.parent)  # Create parent directories
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
```

### Key Parameters

| Parameter            | Value    | Purpose                              |
| -------------------- | -------- | ------------------------------------ |
| `encoding="utf-8"`   | Always   | Support Arabic text                  |
| `indent=2`           | 2 spaces | Human-readable formatting            |
| `ensure_ascii=False` | False    | Write Unicode directly (not escaped) |

### Output Comparison

**With `ensure_ascii=True` (default):**
```json
{
  "name_ar": "\u0642\u0645\u062d"
}
```

**With `ensure_ascii=False`:**
```json
{
  "name_ar": "قمح"
}
```

---

## 13. Function: `resolve_path()`

```python
def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / path
```

### Path Resolution Logic

```
                        ┌─────────────────┐
                        │   Input Path    │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Is it absolute?       │
                    │   (/home/user/... or    │
                    │    C:\Users\...)        │
                    └────────────┬────────────┘
                           Yes   │   No
                    ┌────────────┴────────────┐
                    ▼                         ▼
              Return as-is          ┌─────────────────┐
                                    │  Does it exist  │
                                    │  relative to    │
                                    │  current dir?   │
                                    └────────┬────────┘
                                       Yes   │   No
                                    ┌────────┴────────┐
                                    ▼                 ▼
                              Return as-is    Prepend repo_root
```

### Finding Repository Root

```python
repo_root = Path(__file__).resolve().parents[3]
```

| Step          | Value                                                      |
| ------------- | ---------------------------------------------------------- |
| `__file__`    | `io.py`                                                    |
| `.resolve()`  | `/project/agrovision_core/src/agrovision_core/utils/io.py` |
| `.parents[0]` | `.../utils/`                                               |
| `.parents[1]` | `.../agrovision_core/`                                     |
| `.parents[2]` | `.../src/`                                                 |
| `.parents[3]` | `/project/` (repo root)                                    |

---

## 14. Interview Questions & Answers

### General Questions

**Q1: Why is this module called `io.py` instead of something more specific?**

A: The name `io` (input/output) reflects its primary purpose: reading/writing data files. It follows Python conventions where `io` modules handle data transfer. The underscore-prefixed location (`utils/io.py`) indicates it's a utility module, not a main interface.

**Q2: What happens if rasterio isn't installed?**

A: The module can still be imported. Only when `load_band_tiff()` or `load_label_tiff()` is called will it raise a `ModuleNotFoundError` with a helpful message explaining what to install.

**Q3: How would you add support for a new file format (e.g., NetCDF)?**

A:
1. Create a new function `load_netcdf_band()`
2. Add lazy import for `netCDF4` library
3. Follow the same pattern: load, optionally resample, return numpy array
4. Update `_require_*()` helper pattern

### Resampling Questions

**Q4: What's the computational complexity of bilinear resampling?**

A: O(H × W × 4) where H×W is the output size. Each output pixel requires reading 4 input pixels and computing 3 interpolations.

**Q5: Would cubic resampling be better for satellite imagery?**

A: Cubic produces slightly sharper results but:
1. Computational cost is higher (16 input pixels per output)
2. Can introduce ringing artifacts at sharp edges
3. For 2× upscaling (128→256), bilinear is sufficient
4. The model learns to handle any consistent resampling

**Q6: What if we didn't resample at all?**

A: Two options:
1. **Use smallest resolution for all**: Downsample 10m bands to 128×128 (loses detail)
2. **Multi-scale input**: Process bands at native resolutions (complex architecture)

Upsampling lower-res bands is the standard approach as it preserves maximum information.

### File Handling Questions

**Q7: Why use context managers (`with`) for file operations?**

A: Context managers ensure proper cleanup:
1. Files are closed even if exceptions occur
2. No resource leaks (file handles are limited)
3. Clear scope of file access
4. PEP 8 recommended practice

**Q8: What if two processes try to write the same JSON file simultaneously?**

A: Race condition! Solutions:
1. File locking (`fcntl` on Unix, `msvcrt` on Windows)
2. Write to temp file, then atomic rename
3. Use a database instead of files
4. Application-level coordination

For this offline pipeline, concurrent writes aren't expected.

### Architecture Questions

**Q9: Why separate `get_band_filepath()` from `load_band_tiff()`?**

A: **Single Responsibility Principle**:
- `get_band_filepath()`: Knows dataset naming conventions
- `load_band_tiff()`: Knows how to read/resample GeoTIFFs

This separation allows:
- Testing path construction without file I/O
- Reusing path logic for existence checks
- Changing file format without affecting path logic

**Q10: How would you modify this module for a different satellite (e.g., Landsat)?**

A: Create parallel functions:
```python
def get_landsat_band_filepath(source_dir, scene_id, band_name):
    # Landsat naming convention: LC08_L1TP_039035_20200101_..._B4.TIF
    ...

def load_landsat_band_tiff(filepath, target_size=(256, 256)):
    # Same logic as load_band_tiff() - satellites have similar data
    return load_band_tiff(filepath, target_size)
```

Or make `get_band_filepath()` configurable:
```python
def get_band_filepath(source_dir, tile_id, band_name, dataset="agrifieldnet"):
    patterns = {
        "agrifieldnet": "ref_agrifieldnet_...",
        "landsat": "LC08_L1TP_...",
    }
    ...
```
