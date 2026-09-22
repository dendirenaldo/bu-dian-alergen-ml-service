import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_api_key, get_model_registry
from app.config import settings
from app.schemas.model import ModelStatusResponse
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.post("/model/train")
async def start_training(
    _api_key: str = Depends(get_api_key),
):
    """DINONAKTIFKAN permanen.

    Pipeline retrain API melanggar kontrak leakage-safe (split acak, Sastrawi,
    min_count divergen, tanpa frozen holdout). Training resmi hanya via repo
    ml-training (BiLSTM) dan artifacts/bert (HPC). Lihat docs/bert-hpc.md.
    Endpoint dipertahankan agar klien lama mendapat pesan jelas (503),
    bukan 404/422.
    """
    raise HTTPException(
        status_code=503,
        detail="Retraining via API dinonaktifkan: melanggar kontrak leakage-safe. "
        "Gunakan pipeline resmi (repo ml-training).",
    )
