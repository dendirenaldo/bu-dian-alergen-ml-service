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
    # FIX: commonpath (bukan startswith) agar /data_evil tidak lolos untuk /data.
    if not any(
        os.path.commonpath([abs_path, d]) == d for d in allowed_dirs
    ):
        raise HTTPException(
            status_code=403,
            detail="Data path is outside allowed directories",
        )
    return abs_path


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    bert_ok = registry.is_ready("bert")
    return ModelStatusResponse(
        loaded=registry.is_loaded,
        model_dir=settings.MODEL_DIR,
        models=[
            {"name": "bilstm", "loaded": registry.is_loaded,
             "version": (registry.metadata or {}).get("git_sha", "")},
            {"name": "bert", "loaded": bert_ok,
             "version": (registry.bert_snapshot()[3] or "")},
        ],
    )


@router.get("/model/list")
async def model_list(
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    """Daftar model dual-model + kesiapan ensemble."""
    _, _, _, bert_name = registry.bert_snapshot()
    return {
        "models": [
            {"name": "bilstm", "loaded": registry.is_loaded, "type": "word2vec-bilstm"},
            {"name": "bert", "loaded": registry.is_ready("bert"),
             "type": bert_name or settings.BERT_MODEL_NAME},
            {"name": "ensemble", "loaded": registry.is_ready("ensemble"),
             "type": f"ensemble({settings.ENSEMBLE_STRATEGY})"},
        ],
        "default_model": settings.DEFAULT_MODEL,
        "thresholds": {
            "bilstm": registry.get_threshold("bilstm", settings.DEFAULT_THRESHOLD),
            "bert": registry.get_threshold("bert", settings.BERT_THRESHOLD),
        },
    }


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
        "has_bert": registry.is_ready("bert"),
        "has_ensemble": registry.is_ready("ensemble"),
        "thresholds": {
            "bilstm": registry.get_threshold("bilstm", settings.DEFAULT_THRESHOLD),
            "bert": registry.get_threshold("bert", settings.BERT_THRESHOLD),
        },
    }


@router.post("/model/train", response_model=TrainingStatusResponse)
async def start_training(
    request: TrainingStartRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    # DINONAKTIFKAN: pipeline retrain API melanggar kontrak leakage-safe
    # (split acak, Sastrawi, min_count divergen, tanpa frozen holdout).
    # Training resmi hanya via repo ml-training + HPC. Lihat docs/bert-hpc.md.
    raise HTTPException(
        status_code=503,
        detail="Retraining via API dinonaktifkan: melanggar kontrak leakage-safe. "
        "Gunakan pipeline resmi (repo ml-training).",
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
