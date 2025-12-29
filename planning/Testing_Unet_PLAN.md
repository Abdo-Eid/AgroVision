- **Goal:** Create a notebook that (1) loads the saved model `unet_baseline_best_model.pth`,
  (2) runs the model on one training sample and visualizes results,
  (3) downloads a Sentinel-2 tile over Egypt, preprocesses it to exactly match the
  AgriFieldNet training pipeline (bands, resolution, normalization, shape),
  runs the model, and visualizes results.

- **Chosen Sentinel-2 option:** **Microsoft Planetary Computer STAC**
  (public, signed assets via `planetary_computer`).

---

## Checklist (high-level) 🔧

### 0. Lock model input contract (AgriFieldNet compatibility) 🔒
- [x] Bands (12, fixed order):
      `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`
- [x] All bands resampled to a **10 m grid** using **bilinear interpolation**
- [x] Tile size fixed to **256×256 pixels**
- [ ] Confirm raw value scaling before normalization
      (e.g. Sentinel-2 reflectance `0–10000` → float)
- [x] Normalization strictly from `normalization_stats.json`
- [x] No extra indices or channels unless explicitly confirmed

---

### 1. Notebook setup 🧪
- [x] Create notebook `notebooks/test_unet_and_sentinel2.ipynb`

---

### 2. Inspect dataset & model artifacts 🔍

- [x] Inspect `normalization_stats.json` and `class_map.json`
- [x] Load one sample from `train_images.npy`
      - verify shape `(C, 256, 256)`
      - verify band order
      - check min / max values before normalization
- [x] Load one sample from `train_masks.npy`
      - verify shape `(256, 256)`
      - confirm integer class IDs
- [x] Confirm model input channels by opening
      `agrovision_core/models/unet_baseline.py`
      (`in_channels` used during training)

---

### 3. Load model & run on one training sample ▶️
- [x] Import model architecture
      `agrovision_core.models.unet_baseline.UNet`
- [x] Load weights from `unet_baseline_best_model.pth`
      - load from `checkpoint["model_state"]` (not raw state_dict)
- [x] Take a single training image
      - apply identical normalization
      - add batch dimension
      - run inference
- [x] Visualize:
      - RGB composite (B04, B03, B02)
      - ground-truth mask
      - predicted mask (argmax)
      - probability maps for selected classes
      - class colormap uses `contig_to_raw`

---

### 4. Sentinel-2 (Egypt) fetch & preprocess 🌍
**Scope: extract ONE random 256×256 tile inside Egypt (not full coverage)**  
**Download method (no credentials): Microsoft Planetary Computer STAC**

- [x] Randomly sample **one coordinate inside Egypt**
      (lat, lon within Egypt bounds)
- [x] Define a short date window around the chosen date
- [x] STAC search for **a single Sentinel-2 L2A scene**
      intersecting the chosen coordinate
- [x] Select one scene (best cloud score)
- [x] Retrieve COG URLs for bands:
      `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12`
- [x] Read band data using windowed access
      (no full-scene download)
- [x] Reproject / resample **all bands to a common 10 m grid**
      using **bilinear interpolation**
- [x] Crop **exactly one 256×256 pixel tile**
      centered on the chosen coordinate
- [x] Stack bands in the exact training order
- [x] Apply the same scaling and `normalization_stats.json`


### 5. Run model on Sentinel-2 Egypt tile 🧪

- [x] Assert input tensor shape `(1, C, 256, 256)`
- [x] Run model inference
- [x] Visualize:
      - RGB composite
      - predicted segmentation mask
      - optional class-probability maps
      - overlay prediction on RGB

---

### 6. Notebook finishing touches 📚
- [x] Add markdown cells explaining each major step
- [x] Include sanity checks for:
      - shapes
      - band order
      - value ranges before/after normalization
- [x] Save generated figures under `outputs/`
- [x] Add a short “Results & Notes” section

---

## Findings & Issues (brief)

- **Checkpoint format:** `unet_baseline_best_model.pth` is a training checkpoint; load via `checkpoint["model_state"]` (not a raw state_dict).
- **Colormap mapping:** Class IDs in `class_map.json` are raw IDs (non-contiguous). Use `contig_to_raw` to build the colormap; direct `classes[str(idx)]` fails (missing `7`).
- **Memmap warning:** `np.load(..., mmap_mode="r")` returns non-writable arrays; use `sample_img.copy()` before `torch.from_numpy`.
- **STAC source switch:** AWS EarthSearch searches returned no items for random Egypt points; Planetary Computer STAC worked (signed assets via `planetary_computer`).
- **STAC object type:** With `pystac_client`, items are `pystac.Item` objects; access assets via `item.assets[band].href`, not dict `.get(...)`.
- **Rasterio warning:** `windows.from_bounds(..., height/width=...)` is deprecated; remove those args.
- **Raw scaling still to confirm:** Sentinel-2 reflectance scale vs normalization stats (likely `0-10000` -> `0-100` using `S2_SCALE = 0.01`), needs explicit confirmation.

