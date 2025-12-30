# Module Initialization Files - Complete Documentation

**Files**:
- `agrovision_core/src/agrovision_core/data/__init__.py`
- `agrovision_core/src/agrovision_core/utils/__init__.py`

**Purpose**: Python package initialization and public API definition.

---

## Table of Contents

1. [What Are `__init__.py` Files?](#1-what-are-__init__py-files)
2. [Data Module: `data/__init__.py`](#2-data-module-data__init__py)
3. [Utils Module: `utils/__init__.py`](#3-utils-module-utils__init__py)
4. [Import Patterns](#4-import-patterns)
5. [Interview Questions & Answers](#5-interview-questions--answers)

---

## 1. What Are `__init__.py` Files?

### Purpose

`__init__.py` files serve multiple purposes in Python:

| Purpose | Explanation |
|---------|-------------|
| **Package marker** | Tells Python this directory is a package |
| **Initialization** | Code runs when package is imported |
| **Public API** | Defines what `from package import *` exports |
| **Convenience imports** | Shortens import paths for users |

### With vs Without `__init__.py`

```
project/
├── mypackage/
│   ├── __init__.py       ← Makes it a package
│   └── module.py
└── notapackage/
    └── module.py         ← No __init__.py = not a package
```

```python
# With __init__.py
from mypackage import module  # Works!

# Without __init__.py (Python 3.3+ namespace packages)
import notapackage.module     # Works, but different behavior
from notapackage import module  # Might work, less predictable
```

### Package Hierarchy

```
agrovision_core/
└── src/
    └── agrovision_core/
        ├── __init__.py           ← Root package
        ├── data/
        │   ├── __init__.py       ← Data subpackage
        │   ├── dataset.py
        │   └── prepare_dataset.py
        └── utils/
            ├── __init__.py       ← Utils subpackage
            └── io.py
```

---

## 2. Data Module: `data/__init__.py`

### File Content

```python
# Data pipeline module
from .dataset import CropDataset

__all__ = ["CropDataset"]
```

### Line-by-Line Explanation

#### Line 1: Comment

```python
# Data pipeline module
```

Simple documentation. Could be a docstring instead:
```python
"""Data pipeline module for AgroVision crop classification."""
```

#### Line 2: Relative Import

```python
from .dataset import CropDataset
```

| Part | Meaning |
|------|---------|
| `from` | Start an import statement |
| `.` | Current package (`data/`) |
| `.dataset` | Module `dataset.py` in current package |
| `import CropDataset` | Import the `CropDataset` class |

**Relative vs Absolute imports**:
```python
# Relative (preferred inside packages)
from .dataset import CropDataset

# Absolute (works but more verbose)
from agrovision_core.data.dataset import CropDataset
```

#### Line 3: `__all__` Definition

```python
__all__ = ["CropDataset"]
```

This controls what `from data import *` exports:
```python
from agrovision_core.data import *
# Only imports CropDataset, not other internals
```

### Usage Examples

**Before `__init__.py`**:
```python
# Must use full path
from agrovision_core.data.dataset import CropDataset
```

**After `__init__.py`**:
```python
# Shorter, cleaner import
from agrovision_core.data import CropDataset
```

### What's NOT Exported

The `__init__.py` only exports `CropDataset`. Other items in `dataset.py` are not in the public API:

| Item | Exported? | Reason |
|------|-----------|--------|
| `CropDataset` | ✓ Yes | In `__all__` |
| `get_dataloaders` | ✗ No | Not imported into `__init__.py` |
| `RandomFlipRotate` | ✗ No | Not imported into `__init__.py` |

**To export more items**:
```python
from .dataset import CropDataset, get_dataloaders, RandomFlipRotate

__all__ = ["CropDataset", "get_dataloaders", "RandomFlipRotate"]
```

---

## 3. Utils Module: `utils/__init__.py`

### File Content

```python
"""Utility functions for AgroVision."""

from .io import (
    ensure_dir,
    load_band_tiff,
    load_config,
    load_label_tiff,
    resample_to_target_size,
    resolve_path,
    write_json,
)

__all__ = [
    "ensure_dir",
    "load_band_tiff",
    "load_config",
    "load_label_tiff",
    "resample_to_target_size",
    "resolve_path",
    "write_json",
]
```

### Line-by-Line Explanation

#### Line 1: Module Docstring

```python
"""Utility functions for AgroVision."""
```

Documents the module. Shows up in:
```python
>>> import agrovision_core.utils
>>> help(agrovision_core.utils)
Utility functions for AgroVision.
```

#### Lines 3-11: Multiple Imports

```python
from .io import (
    ensure_dir,
    load_band_tiff,
    load_config,
    load_label_tiff,
    resample_to_target_size,
    resolve_path,
    write_json,
)
```

**Why parentheses?**
- Allows multi-line imports without backslashes
- More readable for many imports
- Easier to add/remove items

**Alternative (not recommended)**:
```python
from .io import ensure_dir, load_band_tiff, load_config, load_label_tiff, \
    resample_to_target_size, resolve_path, write_json
```

#### Lines 13-21: `__all__` List

```python
__all__ = [
    "ensure_dir",
    "load_band_tiff",
    ...
]
```

**Why match imports?**
- Documents the public API
- `from utils import *` works correctly
- IDE autocompletion shows these items

### What's NOT Exported

Some items from `io.py` are intentionally private:

| Item | Exported? | Reason |
|------|-----------|--------|
| `load_band_tiff` | ✓ Yes | Public function |
| `_require_rasterio` | ✗ No | Private helper (underscore prefix) |
| `get_tile_ids_from_source` | ✗ No | Not imported (internal use only) |
| `get_band_filepath` | ✗ No | Not imported (internal use only) |

---

## 4. Import Patterns

### How Imports Work

When you write:
```python
from agrovision_core.data import CropDataset
```

Python does:
1. Find `agrovision_core/` package
2. Execute `agrovision_core/__init__.py`
3. Find `agrovision_core/data/` subpackage
4. Execute `agrovision_core/data/__init__.py`
5. That `__init__.py` imports from `.dataset`
6. `CropDataset` is now available

### Import Tree

```
from agrovision_core.data import CropDataset

agrovision_core/
    __init__.py           ← Executed (may be empty)
        ↓
    data/
        __init__.py       ← Executed
            ├── from .dataset import CropDataset
            └── CropDataset now in data namespace
                ↓
        dataset.py        ← CropDataset class defined here
```

### User Import Options

After proper `__init__.py` setup:

```python
# Option 1: From package (recommended)
from agrovision_core.data import CropDataset

# Option 2: Import package, access attribute
from agrovision_core import data
dataset = data.CropDataset(...)

# Option 3: Full path (still works)
from agrovision_core.data.dataset import CropDataset

# Option 4: Star import (generally discouraged)
from agrovision_core.data import *
```

### Cross-Module Imports

Within the project, modules import from each other:

```python
# In prepare_dataset.py
from ..utils.io import (
    get_band_filepath,
    get_label_filepath,
    get_tile_ids_from_source,
    load_band_tiff,
    load_label_tiff,
)
```

| Part | Meaning |
|------|---------|
| `..` | Parent package (`agrovision_core/`) |
| `..utils` | Sibling package (`utils/`) |
| `..utils.io` | Module `io.py` in `utils/` |

---

## 5. Interview Questions & Answers

### Basic Questions

**Q1: What happens if `__init__.py` is missing?**

A: In Python 3.3+, the directory becomes a **namespace package**:
- Can still import modules: `import mypackage.module`
- No initialization code runs
- `from mypackage import *` doesn't work
- Some tools/IDEs don't recognize it as a package

**Q2: What is `__all__` for?**

A: Controls `from package import *`:
```python
__all__ = ["public_func"]

# When someone does:
from mypackage import *
# Only public_func is imported, not _private_func
```

Also documents the public API for developers and tools.

**Q3: Why use relative imports (`.module`) vs absolute (`mypackage.module`)?**

A: Relative imports:
- Shorter, more readable
- Rename-safe (package rename doesn't break internal imports)
- Clear that it's internal

Absolute imports:
- Clearer for someone reading the code
- Required at the top level (outside packages)

### Design Questions

**Q4: Why only export `CropDataset` from `data/`?**

A: **Minimal public API**:
- `CropDataset` is the main interface users need
- `get_dataloaders` and `RandomFlipRotate` are secondary
- Keep the API simple; users can still import directly if needed:
  ```python
  from agrovision_core.data.dataset import get_dataloaders
  ```

**Q5: Why export so many functions from `utils/`?**

A: These are **utility functions** meant for reuse:
- `load_band_tiff`: Used in preprocessing and inference
- `load_config`: Used throughout the project
- `write_json`: Used for saving outputs

Making them accessible reduces import boilerplate.

**Q6: Why is `_require_rasterio` not exported?**

A: **Private helper** pattern:
- Underscore prefix (`_`) = "don't use externally"
- Implementation detail of `load_band_tiff`
- Not useful outside `io.py`

Even though Python doesn't enforce privacy, the underscore signals intent.

### Technical Questions

**Q7: What order do `__init__.py` files execute?**

A: **Parent before child**:
```python
from agrovision_core.data import CropDataset
# Execution order:
# 1. agrovision_core/__init__.py
# 2. agrovision_core/data/__init__.py
```

**Q8: Can `__init__.py` import from sibling packages?**

A: Yes, using relative imports:
```python
# In data/__init__.py
from ..utils.io import load_config  # Import from sibling utils/
```

But be careful of **circular imports**!

**Q9: What's the difference between:**
```python
from .io import load_config
from . import io
```

A:
| Import | Result |
|--------|--------|
| `from .io import load_config` | `load_config` is directly available |
| `from . import io` | `io` module is available, access as `io.load_config` |

### Edge Cases

**Q10: What if `__init__.py` has a bug?**

A: The entire package import fails:
```python
# If data/__init__.py has: 1/0

from agrovision_core.data import CropDataset
# ZeroDivisionError: division by zero
```

Keep `__init__.py` simple to avoid this!

**Q11: Can `__init__.py` be empty?**

A: Yes! An empty `__init__.py` just marks the directory as a package:
```python
# Empty __init__.py

from agrovision_core.data.dataset import CropDataset  # Still works!
```

But you lose the convenience of:
```python
from agrovision_core.data import CropDataset
```

**Q12: Why not put all code in `__init__.py`?**

A: **Separation of concerns**:
- `__init__.py`: Package setup and exports
- Individual modules: Actual implementation

Benefits:
- Cleaner organization
- Smaller files are easier to maintain
- Can import specific modules without loading everything

### Best Practices

**Summary of `__init__.py` best practices**:

| Practice | Why |
|----------|-----|
| Keep it simple | Avoid import errors |
| Use `__all__` | Document public API |
| Import main classes | Enable short imports |
| Don't put logic here | Separate concerns |
| Use relative imports | Package rename-safe |
| Match `__all__` to imports | Consistency |
