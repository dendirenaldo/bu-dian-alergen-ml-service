import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import get_api_key, get_model_registry
from app.schemas.detection import (
    AllergenResultSchema,
    DetectionResponse,
    TextDetectionRequest,
)
from app.services.detection_pipeline import DetectionPipeline
from app.services.model_registry import ModelRegistry


def _to_allergen_schemas(allergens) -> list:
    """Konversi dataclass pipeline -> schema API (pydantic v2 ketat)."""
    out = []
    for a in allergens or []:
        if isinstance(a, dict):
            out.append(AllergenResultSchema(**a))
        else:
            out.append(AllergenResultSchema(
                name=getattr(a, "name", ""),
                confidence=float(getattr(a, "confidence", 0.0)),
                severity=getattr(a, "severity", ""),
            ))
    return out


logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

router = APIRouter()

ModelChoice = Literal["bilstm", "bert", "ensemble"]


def _check_ready(registry: ModelRegistry, model: str) -> None:
    # Early-return per model: BERT-only tidak boleh menuntut BiLSTM.
    if model == "bert":
        if not registry.is_ready("bert"):
            raise HTTPException(status_code=503, detail="Model BERT belum dimuat")
        return
    if model == "ensemble" and not registry.is_ready("ensemble"):
        raise HTTPException(
            status_code=503,
            detail="Ensemble butuh BiLSTM+BERT; salah satu belum dimuat",
        )
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")


@router.post("/detection/upload", response_model=DetectionResponse)
async def upload_image(
    file: UploadFile | None = File(default=None),
    # Alias kompatibilitas: backend lama mengirim field 'image'.
    image: UploadFile | None = File(default=None),
    model: ModelChoice = Query(default="bilstm", description="Model: bilstm|bert|ensemble"),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str | None = Depends(get_api_key),
):
    _check_ready(registry, model)

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Field 'file' (atau 'image') wajib diisi")

    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large, max 10MB")

    # Validasi magic-byte: pastikan benar-benar image (bukan sekadar content-type).
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=400, detail="File gambar rusak/tidak valid")
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_image(image_bytes, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Input tidak valid")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e) or "Models not loaded")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=_to_allergen_schemas(result.allergens),
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
        model_name=result.model_name,
        model_version=result.model_version,
        scores=result.scores,
    )


@router.post("/detection/text", response_model=DetectionResponse)
async def classify_text(
    request: TextDetectionRequest,
    model: ModelChoice = Query(default="bilstm", description="Model: bilstm|bert|ensemble"),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    _check_ready(registry, model)

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_text(request.text, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Input tidak valid")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e) or "Models not loaded")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=_to_allergen_schemas(result.allergens),
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
        model_name=result.model_name,
        model_version=result.model_version,
        scores=result.scores,
    )


@router.get("/detection/{detection_id}")
async def get_detection_result(detection_id: str):
    raise HTTPException(status_code=501, detail="Result storage not implemented")
