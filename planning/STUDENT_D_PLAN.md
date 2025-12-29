# Student D Task Plan (Backend Inference + FastAPI)

Goal: deliver a FastAPI backend that loads the trained model once, runs inference on an AOI request, and returns overlay + stats + exports for the frontend.

Sources:
- docs/01.PRD.md (MVP features + acceptance criteria)
- README.md (backend folder ownership + API contract)
- docs/04.UNET_TRAINING_GUIDE.md (ignore_index and class semantics)
- docs/02.PROGRESS.md (label list + model output shape expectations)
- notebooks/test_unet_and_sentinel2.ipynb (model loading + normalization + colormap usage)
- planning/STUDENT_B_PLAN.md (model interface) and planning/STUDENT_C_PLAN.md (checkpoint location)

Owner scope: `backend/main.py`, `backend/api/*`, `backend/services/*`

---

## Checklist

### 1) Repo alignment and API contract

- [x] Create `backend/main.py` FastAPI app entry point
- [x] Create `backend/api/routes.py` and `backend/api/schemas.py`
- [ ] Confirm expected request/response schema with Student E (frontend)
- [x] Publish API schema doc for frontend in `docs/API_SCHEMA.md`
- [x] Wire backend imports from `agrovision_core` (no legacy `backend/src` usage)
- [x] Load config from `config/config.yaml` as single source of truth

### 2) Model loading and caching

- [x] Load model at startup (or first request) and cache globally
- [x] Support `model_path` from config; handle both:
  - [x] `outputs/runs/best_model.pth` (training output, not here now)
  - [x] `outputs/models/unet_baseline_best_model.pth` (notebook path)
- [x] Respect `device` from config (cuda/cpu)
- [x] Ensure output logits shape is `(B, num_classes, H, W)`
- [x] Checkpoint loading must accept `checkpoint["model_state"]` or raw state_dict
- [x] Read `in_channels` and `num_classes` from checkpoint when present

### 3) Inference service (core backend logic)

- [x] Implement `backend/services/inference_service.py`
- [x] Inputs: AOI or tile list (mocked first; real later)
- [ ] Steps:
  - [x] Convert AOI to tile(s) (or accept fixed demo tiles)
  - [x] Load imagery chips (initially from demo or cached tiles)
  - [x] Normalize using `normalization_stats.json`
  - [x] Run model inference and argmax to mask
  - [x] Compute per-class stats: pixel_count, percent, mean_confidence
  - [x] Render overlay PNG (RGB + mask + alpha)
- [ ] Output:
  - [x] `overlay_png` (base64 or file URL)
  - [x] `stats_table` (class_name, class_id, pixel_count, percent, mean_confidence)
  - [ ] `raw_mask` (optional, for debugging)
  - [x] `latency_ms` (optional)

### 3a) Dataset artifacts and visualization (from Student B notebook)

- [x] Use `data/processed/normalization_stats.json` for z-score normalization
- [x] Use `data/processed/class_map.json` for class names and colors
- [x] Build colormap using `contig_to_raw` mapping (not raw class IDs)
- [x] RGB composite uses bands `B04, B03, B02` (red, green, blue)
- [x] Apply percentile stretch (2-98%) for display-friendly RGB
- [x] Ignore class `0` in stats (unlabeled)

### 4) API endpoints (MVP)

- [x] `POST /api/infer`
  - [x] Accept viewport payload from frontend map:
    - [x] `center: { lat, lng }`, `zoom`, `tileCount`
    - [x] Optional `bbox` support for future
  - [x] Optional `options` (e.g., `includeConfidence`)
  - [x] Enforce tile limit from config and return clear error if exceeded
  - [x] Return overlay + stats + legend in one response
- [x] `GET /api/legend`
  - [x] Return class names + colors from config/class_map
- [ ] `POST /api/export` (optional)
  - [ ] Export CSV summary and overlay PNG to `outputs/exports/`

### 5) Error handling and UX-friendly responses

- [x] Clear error messages for missing model, missing data, or bad AOI
- [x] Structured error JSON (code + message)
- [x] Time inference and include latency in response (optional)

### 6) Mock-first workflow (to unblock frontend)

- [x] Provide a mock response path in `POST /api/infer` (config flag)
- [x] Mock overlay + stats for frontend integration tests
- [x] Switch to real inference when checkpoint exists

---

## API schema doc checklist (frontend handoff)

- [x] Document request body for `/api/infer` matching frontend `runInference`:
  - [x] `viewport.center.lat`, `viewport.center.lng`
  - [x] `viewport.zoom`, `viewport.tileCount`
  - [x] `options.includeConfidence` (boolean)
- [x] Document response shape expected by frontend:
  - [x] `overlayImage` (base64 or URL)
  - [x] `legend[]` entries: `{ id, name, color }`
  - [x] `stats[]` rows: `{ id, name, pixels, percent, confidence }`
  - [x] `meta.runtimeMs`, `meta.isMock`
- [ ] Document `/api/export` request/response (format + viewport)

---

## Acceptance Criteria (done when)

- [ ] `uv run uvicorn backend.main:app --reload` starts server
- [ ] `POST /api/infer` returns overlay + stats in < 10 seconds for demo AOI
- [ ] `GET /api/legend` returns class names + colors
- [ ] API responses match frontend expectations and PRD (CSV + PNG export ready)

---

## Run steps (local demo)

1) Backend (from repo root):
   - `uv run uvicorn backend.main:app --reload`
2) Frontend (new terminal):
   - `cd frontend`
   - `npm install`
   - `npm run dev`
3) Open the frontend URL and click **Run Analysis**.

Notes:
- Keep `backend.mock_inference: true` for demo mode, or set to `false` once a checkpoint exists.
- Backend uses `data/processed` demo tiles; make sure `class_map.json` + `normalization_stats.json` exist.

---

## Notes and risks

- The backend must ignore class 0 in stats (unlabeled pixels).
- Model checkpoint format should match training output (likely `checkpoint["model_state"]`).
- If AOI->tile logic is not ready, use fixed demo tiles to keep UI usable.
- Ensure Windows file paths and UTF-8 JSON handling are robust.

---

## Suggested order of work

1) Define API schema + mock response (unblocks frontend)
2) Implement model loader + inference service
3) Add stats + overlay rendering
4) Add export endpoints
5) Add tile-limit checks + error states

---

# notes, findings and problems
- Mock inference uses demo tiles from `data/processed` (config: `backend.demo_split`, `backend.demo_tile_index`).
- RGB overlay de-normalizes using `normalization_stats.json` for display; AOI imagery fetch/tiling is still mocked.
- `POST /api/export` is not implemented yet.
- Switch to real model by setting `backend.mock_inference: false` and ensuring the checkpoint path exists.
