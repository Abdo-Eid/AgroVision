# AgroVision Backend API Schema

Base URL: `http://localhost:8000`

## POST /api/infer

Runs inference for the current viewport. The backend enforces a tile limit from
`config/config.yaml`.

Request body:

```json
{
  "viewport": {
    "center": { "lat": 30.90, "lng": 31.10 },
    "zoom": 12,
    "tileCount": 6,
    "bounds": {
      "minLat": 30.55,
      "minLon": 30.75,
      "maxLat": 31.25,
      "maxLon": 31.55
    }
  },
  "options": {
    "includeConfidence": true
  }
}
```

Response body:

```json
{
  "overlayImage": "data:image/png;base64,...",
  "legend": [
    { "id": 1, "name": "Wheat", "color": "#FFC800" }
  ],
  "stats": [
    {
      "id": 1,
      "name": "Wheat",
      "pixels": 45210,
      "percent": 38.7,
      "confidence": 0.92
    }
  ],
  "meta": {
    "runtimeMs": 1120,
    "overlayBounds": {
      "minLat": 30.55,
      "minLon": 30.75,
      "maxLat": 31.25,
      "maxLon": 31.55
    }
  }
}
```

Notes:
- `overlayImage` is a PNG data URI (base64).
- `legend.color` is a CSS hex string.
- `stats` excludes raw class `0` (unlabeled/background).
 - `meta.overlayBounds` is optional; when present, it should be used for map alignment.

Error response:

```json
{ "code": "tile_limit_exceeded", "message": "Tile limit exceeded: 12 > 9. Zoom in to continue." }
```

## GET /api/legend

Returns class metadata for the legend.

Response body:

```json
[
  { "id": 0, "name": "Background", "color": "#000000" },
  { "id": 1, "name": "Wheat", "color": "#FFC800" }
]
```

## POST /api/export (optional)

Not implemented yet. Intended to export overlay PNG or CSV summaries to
`outputs/exports/`.
