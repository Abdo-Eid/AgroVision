# Tile Count Calculation (Documentation)

## Purpose
We limit inference by counting how many map tiles intersect the current viewport. This keeps runtime predictable and makes the tile limit independent of screen size.

## Coordinate system
We use Web Mercator (EPSG:3857), which is the standard for slippy map tiles (OpenStreetMap, MapLibre, etc.).

## Tile scheme
At zoom level `z`, the world is split into:

- `n = 2^z` tiles per axis
- total tiles worldwide: `n × n`

Each tile is indexed by integer `(x, y)` in `[0, n-1]`.

## Lat/Lon to tile index
Given latitude `lat` and longitude `lon` in degrees:

```
 n = 2^z
 x = floor((lon + 180) / 360 * n)
 y = floor((1 - ln(tan(lat * pi/180) + sec(lat * pi/180)) / pi) / 2 * n)
```

Clamp `lat` to valid Mercator range ±85.05112878.

## Viewport to tile count
Let the viewport bounds be `(minLon, minLat, maxLon, maxLat)`.

Compute tile indices for each corner:

```
 xMin = tileX(minLon, z)
 xMax = tileX(maxLon, z)
 yMin = tileY(maxLat, z)
 yMax = tileY(minLat, z)
```

Then tile count is:

```
 tileCount = (xMax - xMin + 1) * (yMax - yMin + 1)
```

## Why we use it
- Stable proxy for compute cost (tiles ≈ inference units).
- Enforces a max tile limit (e.g., 9) to keep processing under target time.
- Consistent across frontend (MapLibre) and backend (FastAPI inference).

## Implementation notes
- Frontend can compute tile count from map bounds and send it with requests.
- Backend should recompute tile count to validate input.
- This calculation depends only on viewport + zoom, not on model or dataset.

## Sentinel-2 pixel size note
- Sentinel-2 RGB bands (B2/B3/B4) are 10 m per pixel.
- Other Sentinel-2 bands are 20 m or 60 m per pixel, so only RGB is 10 m.
- A 256x256 chip at 10 m covers roughly 2.56 km x 2.56 km on the ground.

## Dataset-based tile count (Sentinel-2, 10 m)
Use the dataset chip size instead of generic web tiles:

- `chipMeters = 256 * 10 = 2560 m`
- Project viewport bounds to meters (e.g., EPSG:3857/Web Mercator).
- Compute viewport width/height in meters, then:

```
viewportWidth_m = |x_east - x_west|
viewportHeight_m = |y_north - y_south|

tilesX = ceil(viewportWidth_m / chipMeters)
tilesY = ceil(viewportHeight_m / chipMeters)
tileCount = tilesX * tilesY
```

This matches the AgriFieldNet chip footprint (256x256 at 10 m) and aligns
frontend estimates with backend inference workload.

## What the frontend sends to the backend (viewport inference)
The frontend should send a payload that fully describes the current viewport
and the dataset-based tile estimate. This lets the backend select the chips
that intersect the viewport and run inference only on that region.

Recommended request body (example):

```
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

Backend responsibilities:
- Recompute tileCount from viewport bounds (do not trust client).
- Query chips intersecting the bounds and run inference on those tiles.
- Return overlay image (URL or base64) and stats table for the viewport.