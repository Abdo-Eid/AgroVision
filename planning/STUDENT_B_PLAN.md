# Student B Task Plan (Model Implementation) — Two-Phase Delivery

Goal: deliver clean, reusable segmentation models (**baseline U-Net first**, then a **distinct architecture with attention/transformer-style components**) that plug into training and backend inference.

Source references:
- docs/01.PRD.md (project scope)
- README.md (current folder ownership + architecture)
- docs/02.PROGRESS.md (Student A notes, legacy paths)
- docs/04.UNET_TRAINING_GUIDE.md (label semantics, ignore_index)

Important alignment note:
- README.md says model code lives in `agrovision_core/src/agrovision_core/models/`
- docs/02.PROGRESS.md mentions `backend/src/models/` (legacy not active now).

---

## Checklist

### 1) Repo alignment and interfaces

- [x] Confirm model folder: agrovision_core/src/agrovision_core/models/

  ✅ **Confirmed.**  
  Shared ML code was extracted from `backend/src/` into a standalone installable package.  
  The **active** model path is now: `agrovision_core/src/agrovision_core/models/`

- [x] Confirm expected input channels (C=12) and num classes (K+1 if using ignore_index=0)

  ✅ **Confirmed.**  
  - **Input channels:** `C = 12` (Sentinel-2 chips contain 12 spectral bands).
  - **Num classes:** depends on how labels are handled:

  **Option A (recommended):** use `ignore_index=0`
  - `0` = ignore / background (excluded from loss)
  - valid classes = `1..K`
  - model outputs = `K + 1` channels (including index 0)

  **Option B:** remap labels to `0..K-1` and do not use `ignore_index`
  - valid classes = `0..K-1`
  - model outputs = `K` channels

  ✅ Default recommendation for this project:
  - **Use `ignore_index=0`**
  - **Set `num_classes = K + 1`**

- [x] Confirm how labels are remapped (contiguous IDs) and where mapping lives (config or dataset)

  ✅ **Confirmed (best practice alignment).**  
  Label remapping must happen in the **data layer**, not inside the model.

  **Source of truth options:** `data/processed/class_map.json` (generated during dataset prep)

  **Expected implementation:**
  - `agrovision_core/data/prepare_dataset.py` produces/validates remapped labels
  - `agrovision_core/data/dataset.py` applies remapping consistently at runtime

  ✅ Student B assumption:
  - model receives labels that are already **contiguous**
  - `ignore_index` behavior is enforced in **loss / training config**, not model code

- [x] Agree on model constructor signature (in_channels, num_classes, base_channels, etc.)

  ✅ **Confirmed (standardized interface).**  
  All models in `agrovision_core/models/` should share a consistent constructor:

  ```python
  def __init__(
      self,
      in_channels: int,
      num_classes: int,
      base_channels: int = 64,
      depth: int = 4,
      dropout: float = 0.0,
      **kwargs
  ):
      ...
  ```

  **Minimum required args:**

  * `in_channels` (C=12)
  * `num_classes` (K+1)

---

# Phase 1 — Baseline U-Net (Finish First ✅ so Training can start)

> **Objective:** deliver a simple, from-scratch U-Net baseline that Student C can immediately plug into training + evaluation.

### Phase 1 Checklist

### 2) Baseline U-Net (from scratch)

* [x] Implement Conv-BN-ReLU blocks (2x conv per stage)
* [x] Implement downsampling (maxpool or stride conv) and upsampling (transpose conv or upsample+conv)
* [x] Add skip connections with channel alignment
* [x] Output logits shape: `(B, num_classes, H, W)`
* [x] Keep file I/O out of model code
* [x] Make sure it's a **simple** implementation from scratch (no fancy stuff)

### 3) Reusable blocks module (baseline-only subset)

> Keep this minimal for Phase 1. Add only what baseline U-Net needs.

* [x] CNN blocks: `ConvBlock`, `DownBlock`, `UpBlock`
* [x] Ensure blocks are **modular** and reusable later

### 5) Sanity tests (dummy tensors) — baseline only

* [x] Create a small test snippet in the model files or a minimal script:

  * [x] Input: `torch.randn(B, 12, 256, 256)`
  * [x] Output: `(B, num_classes, 256, 256)`
* [x] Test the baseline model forward pass

### 6) Documentation for handoff — baseline only

* [x] Brief docstring or README notes for the baseline:

  * [x] Expected input/output shapes
  * [x] Constructor args / config knobs


## Phase 1 Acceptance Criteria (deliverables)

* [x] `unet_baseline.py` implemented and imports cleanly
* [x] `blocks.py` implemented (CNN blocks required for baseline)
* [x] Dummy shape tests pass for baseline model
* [x] Training can instantiate baseline with:

  * `in_channels=12`
  * `num_classes=K+1` (ignore_index=0 in loss)

✅ Once Phase 1 is complete, **Student C can start training + evaluation immediately**.

### Phase 1 user-run steps (verify baseline locally)

1) make sure you Installed the package in editable mode (from repo root) if done skip this step:

```bash
uv add --editable ./agrovision_core
```

2) Run the baseline smoke test:

```bash
uv run python -m agrovision_core.models.unet_baseline
```

3) Expected output shows matching shapes, e.g.:

```text
input: torch.Size([2, 12, 256, 256]) output: torch.Size([2, K+1, 256, 256])
```
✅ **Confirmed** input: torch.Size([2, 12, 256, 256]) output: torch.Size([2, 14, 256, 256]) 

---

# Phase 2 — New Architecture (Attention / Transformer-style) ✅ After baseline training is running

> **Objective:** deliver a distinct architecture that is NOT just a U-Net variant, includes attention/transformer-style components, and remains compatible with the same training pipeline.

### Phase 2 Checklist

### 3) Reusable blocks module (advanced blocks)

* [ ] Optional advanced blocks (only if needed by the alternative architecture):

  * [ ] Attention or transformer-style block
  * [ ] MLP / feedforward block with residuals and LayerNorm
* [ ] Keep blocks modular so the alternative architecture can compose them

### 4) Alternative architecture

* [ ] Propose a distinct segmentation model (**not a U-Net variant**)
* [ ] Define core building blocks and data flow clearly
* [ ] Ensure output logits shape: `(B, num_classes, H, W)`
* [ ] Make key hyperparameters configurable
* [ ] Verify parameter count is reasonable for training on lab GPU

### 5) Sanity tests (dummy tensors) — alternative architecture

* [ ] Create a test snippet:

  * [ ] Input: `torch.randn(B, 12, 256, 256)`
  * [ ] Output: `(B, num_classes, 256, 256)`
* [ ] Test both baseline and alternative architecture variants

### 6) Documentation for handoff — alternative architecture

* [ ] Brief docstring / README notes:

  * [ ] Architectural choices (why attention/transformer location)
  * [ ] Expected input/output shapes
  * [ ] Config knobs


## Phase 2 Acceptance Criteria (deliverables)

* [ ] Advanced blocks added to `blocks.py` (attention/transformer + MLP block)
* [ ] Alternative model file implemented and runs forward pass
* [ ] Dummy shape tests pass for alternative model
* [ ] Training pipeline can instantiate both models using the same constructor interface

---

## Risks / questions to resolve early

* Are labels remapped to contiguous IDs? If not, confirm num_classes strategy (max ID + 1).
* Should model include class 0 channel (recommended for ignore_index=0)?
* Final place for model code: `agrovision_core` vs `backend/src` (avoid duplication).

---

## Suggested order of work (two-phase)

### Phase 1 (enable training ASAP)

1. Confirm folder + constructor signature with team
2. Implement CNN blocks and baseline U-Net
3. Run dummy shape test
4. Hand off baseline to Student C for training/evaluation

### Phase 2 (new architecture)

1. Implement transformer/attention blocks in `blocks.py`
2. Propose + implement alternative segmentation architecture
3. Run dummy shape tests for new architecture
4. Share architecture notes + config knobs with Student C and Student D
