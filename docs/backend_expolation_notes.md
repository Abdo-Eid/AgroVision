Below is the **clean, coherent, updated version of the file**, with consistent terminology, clear flow, and the **fixed viewport model as the latest/authoritative approach**.

I did **not** change the technical meaning—only structure, ordering, and clarity.

---

# Frontend → Backend Inference Contract (AgriFieldNet-style Model)

## 1. What the frontend sends (authoritative: fixed viewport)

The frontend **does NOT send images**.
It sends **geographic view information** describing *what the user is currently looking at*.

The frontend always shows RGB for visualization only.

### Frontend payload (fixed viewport)

```json
{
  "viewport": {
    "bounds": {
      "minLat": 30.55,
      "minLon": 30.75,
      "maxLat": 31.25,
      "maxLon": 31.55
    },
    "center": { "lat": 30.90, "lng": 31.10 },
    "zoom": 12,
    "tileCount": 6
  },
  "options": {
    "includeConfidence": true,
    "returnOverlayPng": true
  }
}
```

### Meaning of fields

* **`viewport.bounds`**
  The geographic bounding box currently visible on screen.
  This is the **AOI** for inference.

* **`zoom`**
  UI zoom level (Mapbox / Leaflet style).
  Used only to help the backend choose a reasonable output resolution.

* **`tileCount`**
  Optional hint about how many internal 256×256 tiles are expected.

* **`returnOverlayPng`**
  If `true`, backend returns a raster segmentation overlay.

* **`includeConfidence`**
  If `true`, backend includes confidence information (raster or summary).

---

## 2. AOI and BBox definition

The backend treats the AOI as a **bounding box** derived directly from the viewport:

```
AOI (bbox) = [minLon, minLat, maxLon, maxLat]
```

Example:

```json
"bbox": [30.75, 30.55, 31.55, 31.25]
```

This bbox is used for:

* imagery fetching
* clipping
* overlay alignment

No polygon drawing is required.

---

## 3. What the backend fetches (critical rule)

Even though the frontend shows **RGB**, the backend must fetch **the imagery type the model was trained on**.

Example:

* Model trained on **Sentinel-2 multispectral**
  → backend fetches Sentinel-2 bands (e.g. B2, B3, B4, B8)

Frontend RGB is **only for user interaction** and has no role in inference.

---

## 4. Core backend processing pipeline

### Step-by-step overview

1. **Fetch imagery** covering the viewport bbox
2. **Clip** imagery to exactly the viewport bounds
3. **Resample** to the training resolution (e.g. 10 m / pixel)
4. **Tile** into 256×256 patches
5. **Run inference** on each tile
6. **Stitch** all tile predictions into one mask
7. **Return overlay** aligned to the viewport

---

## 5. What is a “tile” (256×256)?

A **tile** is the fixed-size input unit expected by the model.

### Tile characteristics

* **Size:** `256 × 256` pixels
* **Channels:**

  * RGB model → 3 channels
  * Multispectral model → 4/6/10+ channels
* **Ground coverage depends on resolution**

Example at **10 m / pixel**:

* Tile width = `256 × 10 m = 2560 m` (~2.56 km)
* Tile area ≈ `2.56 km × 2.56 km`

### How tiles look

**Input tile**

* Satellite image patch (fields, roads, canals)

**Mask tile**

* Binary or instance segmentation of fields

```
Input tile (RGB)             Mask tile
┌───────────────┐            ┌───────────────┐
│ fields + roads│            │ 1111000011110 │
│ green/brown   │            │ 1111000011110 │
│ textures      │            │ 0000111100000 │
└───────────────┘            └───────────────┘
```

---

## 6. Detailed explanation of pipeline steps

### (a) Clip to viewport (AOI)

* Imagery may cover a larger region
* Backend crops it to exactly `viewport.bounds`

### (b) Resample to training resolution

* Model expects a specific ground resolution (e.g. 10 m/pixel)
* All bands are resampled so **1 pixel = same ground size as training**

### (c) Tile into 256×256 patches

* Viewport is usually larger than one tile
* Raster is split into many overlapping or non-overlapping tiles

### (d) Run inference

* Each tile is passed through the trained model
* Output: one segmentation mask per tile

### (e) Stitch tiles back together

* Tile masks are placed back into their original positions
* Overlaps are blended or center pixels are preferred
* Result: **one seamless mask covering the viewport**

---

## 7. Pipeline diagram

```
Viewport bounds
┌───────────────────────────────────────┐
│        fetched satellite raster        │
└───────────────────────────────────────┘
                 │
                 │ clip to viewport
                 ▼
┌───────────────────────────────────────┐
│     clipped raster (viewport area)    │
└───────────────────────────────────────┘
                 │
                 │ resample to training resolution
                 ▼
┌───────────────────────────────────────┐
│  raster at correct pixel scale        │
└───────────────────────────────────────┘
                 │
                 │ tile into 256×256
                 ▼
┌───────┐ ┌───────┐ ┌───────┐
│tile 1 │ │tile 2 │ │tile 3 │  ...
└───────┘ └───────┘ └───────┘
    │         │         │
    │ infer   │ infer   │ infer
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│mask 1 │ │mask 2 │ │mask 3 │  ...
└───────┘ └───────┘ └───────┘
                 │
                 │ stitch masks
                 ▼
┌───────────────────────────────────────┐
│ final segmentation mask (viewport)    │
└───────────────────────────────────────┘
```

---

## 8. Returning the overlay to the frontend

### Raster overlay (PNG – recommended for now)

Backend returns:

* **Transparent PNG mask**
* **Metadata for alignment**

  * `bbox` (same as viewport bounds)
  * `width`, `height` (pixels)

Frontend renders it using:

* Leaflet `ImageOverlay`, or
* Mapbox `raster` source/layer

### Important clarification

**Mapbox raster layers are map-anchored**

* They move with pan, zoom, rotate, pitch
* They are rendered in geographic coordinates
* They do **not** stay fixed on screen

Screen-fixed visuals require HTML/DOM overlays, not raster layers.

---

## 9. Practical alignment rule (most important)

* `zoom` helps choose output resolution
* **True alignment is guaranteed only by:**

  * correct `bbox`
  * correct image width/height
    *(or by returning GeoJSON instead of PNG)*