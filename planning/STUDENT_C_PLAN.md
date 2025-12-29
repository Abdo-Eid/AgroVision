# Student C Task Plan (Training + Evaluation)

Goal: implement training and evaluation for AgroVision segmentation models so results are reproducible, comparable, and ready for backend inference.

Sources:
- docs/01.PRD.md (MVP goals + metrics)
- README.md (repo structure + ownership)
- docs/04.UNET_TRAINING_GUIDE.md (ignore_index=0 rules)
- docs/05.Project_Restructure_Report.md (agrovision_core layout)
- docs/06.findings.md (class weights not provided)
- planning/STUDENT_B_PLAN.md (model interfaces and assumptions)

Owner scope: `agrovision_core/src/agrovision_core/train/*`

---

## Checklist

### 1) Repo alignment and interfaces

- [x] Confirm training entry points live in `agrovision_core/src/agrovision_core/train/`
- [x] Confirm imports use `agrovision_core.*` (no legacy `backend/src` paths)
- [x] Confirm model constructor interface matches Student B
- [x] Confirm config file location `config/config.yaml` is used as source of truth

### 2) Data + label handling (critical)

- [x] Confirm dataset returns masks with `0 = ignore` and `1..K = classes`
- [x] Set `IGNORE_INDEX = 0` and use `CrossEntropyLoss(ignore_index=0)`
- [x] Ensure metrics exclude pixels where `mask == 0`
- [x] use weights: compute from masks or `data/processed/class_map.json`
- [x] `num_classes = K + 1` num_classes strategy: labels are remapped to contiguous IDs

### 3) Simple Training loop (baseline ready)

- [x] Implement `train.py` with:
  - [x] device selection (cuda from config)
  - [x] optimizer (configurable)
  - [x] checkpointing to `outputs/runs/`
  - [x] resume from checkpoint (optional but recommended)
  - [x] clear logging (loss, mIoU, IoU per class, F1)
- [x] Support baseline U-Net first (`unet_baseline.py`)
- [x] Accept config args: `in_channels`, `num_classes`, `base_channels`, `depth`, `dropout`

### 4) Evaluation pipeline

- [x] Implement `evaluate.py` to:
  - [x] load best checkpoint
  - [x] run on validation split
  - [x] compute mIoU, per-class IoU, macro F1
  - [x] export metrics JSON/CSV to `outputs/runs/`
- [x] Add confusion matrix or per-class table (optional but useful)

### 5) Metrics implementation

- [x] Create or validate `metrics.py` with ignore_index handling
- [x] Verify metrics on a tiny dummy batch (unit sanity check)

### 6) Exploration notebook (visual sanity checks)

- [ ] Create a notebook to visualize:
  - [ ] input image (RGB composite)
  - [ ] ground-truth mask
  - [ ] baseline prediction mask
- [ ] Save under `notebooks/` with a clear name (e.g., `explore_baseline_predictions.ipynb`)


---

## Acceptance Criteria (done when)

- [ ] `uv run python -m agrovision_core.train.train` trains baseline successfully
- [ ] `uv run python -m agrovision_core.train.evaluate` produces metrics files
- [ ] Best checkpoint saved at `outputs/runs/best_model.pth`
- [ ] Metrics ignore `mask == 0` and are reproducible

---

## Notes and risks

- The dataset is highly unlabeled; failing to ignore class `0` will collapse training.
- If class IDs are not remapped, confirm `num_classes` and mapping strategy with Student A.
- Training should be offline only; backend only consumes the checkpoint.

---

## Colab training (note)

Purpose: quick, reproducible training run in Colab without relying on legacy `backend/src` paths.

Assumptions:
- You will clone the repo inside Colab.
- You will run commands from the repo root.
- The training entry point is `agrovision_core.train.train`.

How to run training in Colab (from the cloned repo root):

1) `!git clone https://github.com/Abdo-Eid/AgroVision`
2) `!pip install -U uv`
3) `%cd AgroVision`
4) `from google.colab import drive; drive.mount('/content/drive')`
   - Put the processed data folder on Drive (e.g., `/content/drive/MyDrive/AgroVision/data`) so training can read it.
   - To unzip a Drive archive into the repo root:
     - `!unzip -q "/content/drive/MyDrive/data.zip" -d .`
5) `!uv add --editable ./agrovision_core`
6) `!uv run python -m agrovision_core.train.train`

Notes:
- If you already have `uv`, you can skip step 1.
- Use `--config` to point to a custom YAML if needed.
- Mount Drive only if you want to read data from or save outputs to `/content/drive`.
