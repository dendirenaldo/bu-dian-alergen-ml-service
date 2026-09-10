import logging
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    pass


class TrainingInProgressError(Exception):
    pass


class PipelineError(Exception):
    pass


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    logger.warning(f"Model not loaded: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "ML models not loaded yet", "code": "model_not_loaded"},
    )


async def training_in_progress_handler(request: Request, exc: TrainingInProgressError):
    logger.warning(f"Training in progress: {exc}")
    return JSONResponse(
        status_code=409,
        content={"detail": "Training already in progress", "code": "training_in_progress"},
    )


async def pipeline_error_handler(request: Request, exc: PipelineError):
    logger.error(f"Pipeline error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Detection pipeline error", "code": "pipeline_error"},
    )


async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "internal_error"},
    )
