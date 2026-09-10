import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_model_registry
from app.schemas.detection import DetectionResponse, TextDetectionRequest
from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/detect/upload", response_model=DetectionResponse)
async def upload_image(
    file: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_model_registry),
):
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_image(image_bytes)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=result.allergens,
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
    )


@router.post("/detect/text", response_model=DetectionResponse)
async def classify_text(
    request: TextDetectionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
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
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=result.allergens,
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
    )


@router.get("/detect/{detection_id}")
async def get_detection_result(detection_id: str):
    raise HTTPException(status_code=501, detail="Result storage not implemented")
