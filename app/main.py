import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exceptions import (
    ModelNotLoadedError,
    PipelineError,
    TrainingInProgressError,
    generic_error_handler,
    model_not_loaded_handler,
    pipeline_error_handler,
    training_in_progress_handler,
)
from app.api.routers import detection, health, model
from app.config import settings
from app.services.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    try:
        registry.load_models(settings.MODEL_DIR)
    except Exception as e:
        logger.error(f"Gagal load model saat startup: {e}")
    # FIX: simpan di app.state agar tidak bergantung singleton implisit
    # (lebih rapi untuk testing & multi-worker).
    app.state.registry = registry
    if not registry.is_loaded:
        logger.warning(
            f"ML model belum lengkap di {settings.MODEL_DIR}. "
            "Service tetap jalan (health OK), tapi /detection akan 503."
        )
    yield


app = FastAPI(
    title="Bu Dian Allergen Detection API",
    description="ML Service for food allergen detection using OCR + BiLSTM",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(TrainingInProgressError, training_in_progress_handler)
app.add_exception_handler(PipelineError, pipeline_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["health"])
app.include_router(detection.router, prefix=settings.API_V1_PREFIX, tags=["detection"])
app.include_router(model.router, prefix=settings.API_V1_PREFIX, tags=["model"])


@app.get("/")
def root():
    return {"message": "Bu Dian Allergen Detection ML Service"}
