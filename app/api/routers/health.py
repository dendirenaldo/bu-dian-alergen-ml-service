from fastapi import APIRouter, Depends

from app.api.deps import get_model_registry
from app.services.model_registry import ModelRegistry

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(registry: ModelRegistry = Depends(get_model_registry)):
    return {"status": "ready" if registry.is_loaded else "not_ready", "models_loaded": registry.is_loaded}
