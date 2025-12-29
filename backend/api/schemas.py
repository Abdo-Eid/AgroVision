from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ViewportCenter(BaseModel):
    lat: float
    lng: float


class ViewportBounds(BaseModel):
    minLat: float
    minLon: float
    maxLat: float
    maxLon: float


class Viewport(BaseModel):
    center: ViewportCenter
    zoom: int
    tileCount: int
    bounds: ViewportBounds | None = None


class InferOptions(BaseModel):
    includeConfidence: bool = False


class InferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    viewport: Viewport
    options: InferOptions | None = None


class LegendEntry(BaseModel):
    id: int
    name: str
    color: str


class StatsRow(BaseModel):
    id: int
    name: str
    pixels: int
    percent: float
    confidence: float


class InferMeta(BaseModel):
    runtimeMs: int = Field(..., ge=0)
    overlayBounds: ViewportBounds | None = None


class InferResponse(BaseModel):
    overlayImage: str | None = None
    legend: list[LegendEntry]
    stats: list[StatsRow]
    meta: InferMeta


class ErrorResponse(BaseModel):
    code: str
    message: str
