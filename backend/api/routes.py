import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.schemas import ErrorResponse, InferRequest, InferResponse, LegendEntry
from backend.services.inference_service import InferenceError, InferenceService

router = APIRouter()
service = InferenceService()
logger = logging.getLogger("agrovision.backend.routes")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(code=code, message=message).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/api/legend", response_model=list[LegendEntry])
def get_legend():
    try:
        logger.info("GET /api/legend")
        return service.get_legend()
    except InferenceError as exc:
        logger.warning("Legend error: %s | %s", exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)


@router.post("/api/infer", response_model=InferResponse)
def infer(request: InferRequest):
    try:
        logger.info(
            "POST /api/infer | zoom=%s tileCount=%s",
            request.viewport.zoom,
            request.viewport.tileCount,
        )
        include_confidence = bool(
            request.options.includeConfidence if request.options else False
        )
        result = service.run_inference(
            tile_count=request.viewport.tileCount,
            include_confidence=include_confidence,
            bounds=request.viewport.bounds.model_dump() if request.viewport.bounds else None,
            center=request.viewport.center.model_dump(),
        )
    except InferenceError as exc:
        logger.warning("Inference error: %s | %s", exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)

    logger.info("Inference complete | runtimeMs=%s", result.runtime_ms)
    return InferResponse(
        overlayImage=result.overlay_image,
        maskImage=result.mask_image,
        legend=result.legend,
        stats=result.stats,
        meta={"runtimeMs": result.runtime_ms, "overlayBounds": result.overlay_bounds},
    )
