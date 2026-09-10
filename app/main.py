import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import detection, health, model
from app.config import settings
from app.services.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Bu Dian Allergen Detection API",
    description="ML Service for food allergen detection using OCR + BiLSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["health"])
app.include_router(detection.router, prefix=settings.API_V1_PREFIX, tags=["detection"])
app.include_router(model.router, prefix=settings.API_V1_PREFIX, tags=["model"])


@app.on_event("startup")
async def startup():
    registry = ModelRegistry()
    registry.load_models(settings.MODEL_DIR)


@app.get("/")
def root():
    return {"message": "Bu Dian Allergen Detection ML Service"}
