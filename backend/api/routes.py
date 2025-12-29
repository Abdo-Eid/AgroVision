from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.api.schemas import ErrorResponse, InferRequest, InferResponse, LegendEntry
from backend.services.inference_service import InferenceError, InferenceService

router = APIRouter()
service = InferenceService()


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(code=code, message=message).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/api/legend", response_model=list[LegendEntry])
def get_legend():
    try:
        return service.get_legend()
    except InferenceError as exc:
        return _error_response(exc.status_code, exc.code, exc.message)


@router.post("/api/infer", response_model=InferResponse)
def infer(request: InferRequest):
    try:
        include_confidence = bool(
            request.options.includeConfidence if request.options else False
        )
        result = service.run_inference(
            tile_count=request.viewport.tileCount,
            include_confidence=include_confidence,
        )
    except InferenceError as exc:
        return _error_response(exc.status_code, exc.code, exc.message)

    return InferResponse(
        overlayImage=result.overlay_image,
        legend=result.legend,
        stats=result.stats,
        meta={"runtimeMs": result.runtime_ms, "isMock": result.is_mock},
    )
