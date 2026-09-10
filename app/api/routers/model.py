import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key, get_model_registry
from app.config import settings
from app.schemas.model import ModelStatusResponse, TrainingStartRequest, TrainingStatusResponse
from app.services.model_registry import ModelRegistry
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)

router = APIRouter()

_training_service: Optional[TrainingService] = None


def _get_training_service(registry: ModelRegistry) -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService(registry)
    return _training_service


def _validate_data_path(data_path: str) -> str:
    """Validate and resolve data_path to prevent path traversal."""
    abs_path = os.path.abspath(data_path)
    allowed_dirs = [os.path.abspath(d) for d in settings.ALLOWED_DATA_DIRS.split(",")]
    if not any(abs_path.startswith(d) for d in allowed_dirs):
        raise HTTPException(
            status_code=403,
            detail="Data path is outside allowed directories",
        )
    if ".." in data_path:
        raise HTTPException(status_code=400, detail="Path must not contain '..'")
    return abs_path


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    return ModelStatusResponse(
        loaded=registry.is_loaded,
        model_dir=settings.MODEL_DIR,
    )


@router.get("/model/metrics")
async def model_metrics(
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    return {
        "loaded": registry.is_loaded,
        "has_bi_lstm": registry.bi_lstm_model is not None,
        "has_word2vec": registry.word2vec_model is not None,
        "has_tokenizer": registry.tokenizer is not None,
        "has_label_encoder": registry.label_encoder is not None,
    }


@router.post("/model/train", response_model=TrainingStatusResponse)
async def start_training(
    request: TrainingStartRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    _validate_data_path(request.data_path)
    service = _get_training_service(registry)

    try:
        service.start_training(
            data_path=request.data_path,
            text_col=request.text_col,
            label_col=request.label_col,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    status = service.status
    return TrainingStatusResponse(
        status=status.status,
        progress=status.progress,
        epoch=status.epoch,
        total_epochs=status.total_epochs,
        loss=status.loss,
        accuracy=status.accuracy,
        val_loss=status.val_loss,
        val_accuracy=status.val_accuracy,
        message=status.message,
        started_at=status.started_at,
        finished_at=status.finished_at,
        error=status.error,
    )


@router.get("/model/train/status", response_model=TrainingStatusResponse)
async def training_status(
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    service = _get_training_service(registry)
    status = service.status
    return TrainingStatusResponse(
        status=status.status,
        progress=status.progress,
        epoch=status.epoch,
        total_epochs=status.total_epochs,
        loss=status.loss,
        accuracy=status.accuracy,
        val_loss=status.val_loss,
        val_accuracy=status.val_accuracy,
        message=status.message,
        started_at=status.started_at,
        finished_at=status.finished_at,
        error=status.error,
    )
