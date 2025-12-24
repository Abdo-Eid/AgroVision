# PRD — Interactive Crop Mapping Platform (Satellite → Crop Map GUI)
## 1) Product summary

**Product name:** AgroVision Crop Mapper

**Problem:** Non-technical users need fast, understandable **crop-type maps** from satellite imagery for monitoring, reporting, and planning—without handling bands/GeoTIFFs or GIS tooling.

**Solution:** A **map-first GUI** that lets users pan/zoom to an Area of Interest (AOI) and returns:

- satellite RGB preview
- predicted crop-type overlay (semantic segmentation)
- per-class area + confidence summary
- exportable results (PNG + CSV)

**Primary dataset:** AgriFieldNet (Sentinel-2 multi-band chips, field IDs, crop labels).

- **sources:**
    
    https://zindi.africa/competitions/agrifieldnet-india-challenge
    
    https://source.coop/radiantearth/agrifieldnet-competition
    
    https://github.com/radiantearth/agrifieldnet_india_competition
    
    https://github.com/radiantearth/model_ecaas_agrifieldnet_gold
    

**Core constraint:** Final deliverable must be a **GUI app**, not just notebooks. **No TensorFlow.**

### Using the Dataset for Egyptian Crops (Label Adaptation)

Although the AgriFieldNet dataset is geographically sourced from India, the **crop labels are agronomically generic** and largely overlap with crops cultivated in Egypt.

### Label relevance to Egypt

- **Directly relevant / common in Egypt**
    - **Wheat (قمح)**
    - **Rice (أرز)**
    - **Maize (ذرة صفراء)**
    - **Sugarcane (قصب السكر)**
    - **Potato (بطاطس)**
    - **Berseem / Egyptian clover (برسيم)**
- **Less common / region-specific**
    - **Mustard (خردل)**
    - **Lentil (عدس)**
    - **Green pea (بسلة خضراء)**
    - **Gram / Chickpea (حمص)**
    - **Coriander (كزبرة)**
    - **Garlic (ثوم)**
- **Universal**
    - No Crop / Fallow (exists in all agricultural regions)

## why not using an Egyptian dataset?

Your **safe, honest answer**:

- Public, pixel-level labeled crop datasets for Egypt are **extremely limited**
- The project goal is **methodology + deployment**, not national yield estimation
- The approach is **transferable** to Egyptian data when labels become available

This framing is **academically correct** and **industry-aligned**.

## 2) Goals and success criteria

### Goals

- Make inference usable by non-technical users with **2–3 clicks**.
- Provide **interpretable outputs** (legend, area stats, confidence, exports).
- Deliver a stable demo that runs on a laptop or lab PC.

### Success criteria (MVP)

- User can select AOI → run inference → view overlay + stats in **< 10 seconds** for an AOI within the allowed size (local demo).
- Exports: **overlay PNG**, **CSV** with class areas (% and optionally hectares if scale/geo metadata is available).
- Model reaches acceptable quality on validation (e.g., **mIoU ≥ 0.45** baseline; adjust target based on compute).

## 3) Non-goals (MVP)

- No “bring your own satellite data”.
- No continuous real-time monitoring pipeline.
- No interactive boundary digitization/editing tools (view + summarize only).

## 4) Personas — *Needs, motivation, real example*

### **Agri Analyst (Primary)**

Needs quick crop inventory summaries for a region.

→ To efficiently generate reliable crop statistics that support agricultural monitoring, policy formulation, and seasonal reporting without complex GIS workflows.

→ For example, producing a district-level report showing **60% wheat, 25% rice, and 15% fallow land** to support a government agricultural bulletin.

### **Field Operations Manager**

Wants a crop map with accurate area estimates for planning inputs and logistics.

→ To make informed operational decisions on seeds, fertilizers, irrigation, and workforce allocation based on actual crop distribution.

→ For example, adjusting fertilizer procurement after identifying **1,200 hectares of rice cultivation** within the selected area.

### **Instructor/TA (Evaluator)**

Wants a complete end-to-end pipeline with a functional GUI and reproducible results.

→ To verify that the project applies deep learning correctly across data preparation, modeling, inference, and deployment stages.

→ For example, cloning the repository, running the GUI, and obtaining the same predictions and metrics described in the project report.

## 5) User stories

- As an **agricultural analyst**, I can **pan/zoom a map** to define a region of interest and **run crop mapping** to analyze crop distribution in that area.
- As an **agricultural analyst**, I can **toggle overlay transparency, zoom, and pan** to visually verify predictions against the satellite image.
- As an **agricultural analyst**, I can view a **legend and per-class area breakdown** to understand crop composition at a glance.
- As an **agricultural analyst**, I can **export results** (CSV summary and visualization PNG) to include in reports and share with stakeholders.
- As an **Instructor/TA**, I can **run the app end-to-end from the repository** using documented README steps to verify reproducibility.

### Notes & Clarifications

- **ArcGIS-like region selection (zoom controls size)**
    
    Users freely pan/zoom, but prediction is enabled only when the visible map region maps to a **bounded number of tiles** (e.g., **1–9 tiles**). This keeps runtime predictable while preserving a GIS-like experience.
    
- **Region size and content**
    
    The user-selected region may include **multiple agricultural fields** and also **non-agricultural pixels** (roads/buildings/bare land). The system outputs a **pixel-wise semantic map** across the entire selected region.
    
- **Output in mixed land-use regions**
    
    If “non-crop/background” exists in the label space, the model may predict it and the stats panel will include it; otherwise, non-agricultural pixels are treated as **background/ignored** and excluded from crop-only summaries (configurable as “Crop-only stats” toggle).
    
- **Overlay transparency toggle**
    
    The transparency slider adjusts the alpha of the prediction mask over the RGB image to help users compare boundaries and confirm spatial alignment.
    

## 6) Scope and features

### MVP features (must-have)

**A. Region selection**

- Map UI with pan/zoom and a visible bounding box (the current viewport).
- Display tile-count estimate for the current view (e.g., “6 tiles”).

**B. One-click inference**

- “Run Analysis” button triggers backend inference for all tiles in the selected region.
- Button disabled + message when region is out of bounds (too large / too many tiles).
- Shows progress + clear error messages (model missing, data missing, etc.).

**C. Visualization**

- RGB preview (true color composite).
- Predicted mask overlay with:
    - legend (class → color)
    - opacity slider
    - optional click probe: predicted class + confidence at pixel

**D. Analytics panel**

- Table: Class → Area (pixels + %, optional hectares) + mean confidence.
- Summary card: dominant crop + coverage.

**E. Export**

- Download:
    - overlay PNG (RGB + mask)
    - CSV summary

**F. Reproducibility**

- `README` with install + run + demo steps.
- Config file for model path, device, tile limit, caching.

### Nice-to-have (bonus)

- Frontend in HTML/CSS/JS + backend REST API.
- Field-instance summary using `field_ids` (per-field crop label + area).

## 7) UX requirements (user-friendly by design)

- Hide remote sensing jargon by default; expose technical details only via an **optional advanced/academic view**.
    - **Academic / Debug mode (optional toggle):** A collapsible section under **Run Analysis** (e.g., “Show Technical Details”). Displays intermediate steps such as: (RGB composite used for visualization, Individual input bands or band groups (optional thumbnails), Normalized input preview, Raw probability map (per-class heatmap), Tile count and inference time)
- Inputs are human: “Zoom”, “Run Analysis”, “Download”.
- Default to a **Demo region** so the app always works in grading.
- Clear legend with crop names.
- Actionable error states (“Zoom in: selected region is too large (18 tiles). Max allowed: 9.”).

## 8) User flow (MVP)

1. Open app → map loads with a “Demo region” ready
2. User pans/zooms to desired area (tile count updates)
3. Click **Run Analysis** (enabled only if within limits)
4. Results:
    - RGB image + overlay
    - stats table
5. Click **Download Report** / **Download Overlay**

---

## 9) Functional requirements (acceptance criteria)

### FR-1 Region selection + constraints

- Must compute tile-count for the visible viewport and enforce max tiles (configurable).
- Must provide user feedback when out of bounds.

### FR-2 Inference execution

- Must load model once (cached) and run per-tile inference.

### FR-3 Overlay rendering

- Overlay aligns with RGB preview.
- Opacity slider updates overlay alpha in real time.

### FR-4 Stats computation

- Computes per-class pixel counts and percentages.
- Optional hectares if scale/metadata available; otherwise show “relative area (pixels)”.

### FR-5 Export

- CSV contains: class_name, class_id, pixel_count, percent, mean_confidence.
- PNG exports match the on-screen visualization.


## Evaluation metrics

- mIoU, macro F1, per-class IoU
- Confusion matrix
- Inference latency per tile + per viewport


**Storage**

- Local disk caching (per region / per tile)
 