# AgroVision Project Restructure Report: Extracting Shared Python Code into `agrovision_core`

## Summary

We changed the repo structure to make shared Python code (models, training, utilities) **importable from anywhere** in the project—especially from `notebooks/`—without `sys.path` hacks.

We achieved this by extracting shared code into a standalone, installable Python library package called **`agrovision_core`**, managed with **uv**, and installing it in **editable mode** so development remains fast.

---

## Problem Statement

### What was hard before

* The reusable code lived under:

  ```
  backend/src/models
  backend/src/train
  backend/src/utils
  ```

* Notebooks run from `notebooks/` and therefore **Python’s import system could not “see”** `backend/src/` by default.

* This caused friction such as:

  * Import errors in notebooks (`ModuleNotFoundError`)
  * Needing per-notebook `sys.path.append(...)`
  * Imports working in one context but failing in another (backend vs notebooks vs scripts)
  * Confusing ownership: code used by training/notebooks was “inside backend”

### Root cause

The shared code was **not packaged/installed** as a Python module, so it was not globally importable within the environment.

---

## Goal

Create a clean, reusable shared library so all parts of the project can do:

```python
from agrovision_core.models ...
from agrovision_core.train ...
from agrovision_core.utils ...
```

…and have it work consistently from:

* `notebooks/`
* `backend/`
* training/evaluation scripts
* any future tools

---

## Final Solution

### High-level decision

We created a new top-level Python library package:

```
agrovision_core/
  src/agrovision_core/
    models/
    train/
    utils/
```

Then we installed it into the project environment using uv in **editable mode**, so changes are immediately reflected without reinstalling.

---

## Before vs After

### Before (simplified)

```
agrovision/
  backend/
    src/
      models/
      train/
      utils/
  notebooks/
  config/
  data/
  frontend/
```

### After (simplified)

```
agrovision/
  agrovision_core/
    src/agrovision_core/
      models/
      train/
      utils/
      __init__.py
    pyproject.toml
    README.md
    py.typed
  backend/
    src/
      __init__.py
  notebooks/
  config/
  data/
  frontend/
  pyproject.toml        # root project config (now depends on agrovision_core)
  uv.lock
```

---

## Implementation Steps (Start → Finish)

### 1) Create the shared library package using uv

From repo root:

```bash
uv init --lib agrovision_core
```

This generated a proper Python package skeleton with:

* `agrovision_core/pyproject.toml`
* `agrovision_core/src/agrovision_core/`
* `agrovision_core/README.md`

---

### 2) Move shared code into the new package

Moved these directories:

* `backend/src/models` → `agrovision_core/src/agrovision_core/models`
* `backend/src/train`  → `agrovision_core/src/agrovision_core/train`
* `backend/src/utils`  → `agrovision_core/src/agrovision_core/utils`

Result matches:

```
agrovision_core/src/agrovision_core/
  models/
  train/
  utils/
  __init__.py
```

---

### 3) Add the package to the project environment (editable)

From repo root:

```bash
uv add --editable ./agrovision_core
```

What this does:

* Adds the local library as a dependency in the **root `pyproject.toml`**
* Updates **`uv.lock`**
* Installs the library into the environment in **editable** mode

---

### 4) Verify the install (import test)

From repo root:

```bash
uv run python -c "import agrovision_core; print('OK:', agrovision_core.__file__)"
```

Expected output (example from Windows):

```
OK: D:\trying\AgroVision\agrovision_core\src\agrovision_core\__init__.py
```

This confirms:

* The import works
* It is loading from the **source path**
* Editable install is effective

---

### 5) Confirm root dependency was added automatically

uv updated the root `pyproject.toml` automatically to include the local editable dependency (and updated `uv.lock`).

No manual editing was required.

---

### 6) Validate notebooks can import without hacks

Opened a notebook from:

```
notebooks/
```

Tested:

```python
import agrovision_core
```

Result:

* ✅ Works from inside notebooks without `sys.path` modifications

---

## Required Code Changes (Imports)

### Update imports across the repo

Any existing imports like:

```python
from models...
from train...
from utils...
```

should be changed to:

```python
from agrovision_core.models...
from agrovision_core.train...
from agrovision_core.utils...
```

This ensures consistency across:

* notebooks
* backend runtime
* CLI scripts
* tests

---

## Why This Approach

### Benefits

* **One canonical shared library** (`agrovision_core`) used everywhere
* **No path hacks** (`sys.path.append`)
* **Editable install** = instant feedback loop during development
* Clear separation of concerns:

  * `agrovision_core` = reusable ML + training + utilities
  * `backend` = FastAPI service consuming the library
  * `notebooks` = experiments consuming the library
* Easier testing and packaging later

### Tradeoffs

* Requires adjusting imports to use the package namespace
* Need to ensure backend runs under the same environment where `agrovision_core` is installed

---

## Operating Rules Going Forward

### Recommended usage

From anywhere, prefer:

```bash
uv run python ...
uv run jupyter lab
uv run <script>
```

This ensures the uv-managed environment is used and local editable dependencies are available.

### Where to put reusable code

* ✅ Put shared code in: `agrovision_core/src/agrovision_core/...`
* ❌ Avoid placing reusable code back into `backend/src/...` unless it is truly backend-only

---

## Troubleshooting

### Notebook can’t import `agrovision_core`

Checklist:

1. Confirm dependency exists:

```bash
uv run python -c "import agrovision_core; print(agrovision_core.__file__)"
```

2. Confirm you’re running Jupyter via uv:

```bash
uv run jupyter lab
```

3. Confirm the notebook kernel points to the same environment used by uv.

---

## Final State Confirmation

✅ `agrovision_core` is a real Python library
✅ Installed as local editable dependency via `uv add --editable`
✅ Imports work from repo root and from `notebooks/`
✅ Root `pyproject.toml` updated automatically
✅ Verified by import-path output pointing into `agrovision_core/src/...`

---

## Appendix: Current Shared Package Layout (as created)

```
agrovision_core/
  src/agrovision_core/
    data/
    models/
    train/
    utils/
    __init__.py
    py.typed
  pyproject.toml
  README.md
```
