import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_api_key, get_model_registry
from app.schemas.detection import DetectionResponse, TextDetectionRequest
from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

router = APIRouter()


@router.post("/detection/upload", response_model=DetectionResponse)
async def upload_image(
    file: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large, max 10MB")

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_image(image_bytes)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=result.allergens,
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
    )


@router.post("/detection/text", response_model=DetectionResponse)
async def classify_text(
    request: TextDetectionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_text(request.text)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=result.allergens,
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
    )


@router.get("/detection/{detection_id}")
async def get_detection_result(detection_id: str):
    raise HTTPException(status_code=501, detail="Result storage not implemented")
