from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ViewportCenter(BaseModel):
    lat: float = Field(30.598302903519112, description="Center latitude.")
    lng: float = Field(30.995760839950435, description="Center longitude.")


class ViewportBounds(BaseModel):
    minLat: float = Field(30.58924359580533)
    minLon: float = Field(30.989642634080695)
    maxLat: float = Field(30.607361364246543)
    maxLon: float = Field(31.001879045815627)


class Viewport(BaseModel):
    center: ViewportCenter
    zoom: int = Field(14, ge=0)
    tileCount: int = Field(1, ge=1)
    bounds: ViewportBounds | None = None


class InferOptions(BaseModel):
    includeConfidence: bool = True


class InferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    viewport: Viewport
    options: InferOptions | None = Field(
        default_factory=InferOptions,
        description="Inference options.",
    )


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
    maskImage: str | None = None
    legend: list[LegendEntry]
    stats: list[StatsRow]
    meta: InferMeta


class ErrorResponse(BaseModel):
    code: str
    message: str
