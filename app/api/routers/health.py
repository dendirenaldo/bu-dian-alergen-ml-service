from fastapi import APIRouter, Depends

from app.api.deps import get_model_registry
from app.config import settings
from app.services.model_registry import ModelRegistry

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(registry: ModelRegistry = Depends(get_model_registry)):
    default_model = (settings.DEFAULT_MODEL or "bert").lower()
    per_model = {
        "bilstm": registry.is_loaded,
        "bert": registry.is_ready("bert"),
        "ensemble": registry.is_ready("ensemble"),
    }
    default_ready = bool(per_model.get(default_model, False))
    return {
        "status": "ready" if default_ready else "degraded",
        "default_model": default_model,
        "default_ready": default_ready,
        "models": per_model,
        "models_loaded": registry.is_loaded,
    }
