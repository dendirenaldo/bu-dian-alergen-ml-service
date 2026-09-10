from fastapi import Depends, Header, HTTPException

from app.config import settings
from app.services.model_registry import ModelRegistry


def get_model_registry() -> ModelRegistry:
    """Dependency to get the singleton ModelRegistry."""
    return ModelRegistry()


async def get_api_key(x_api_key: str = Header(...)) -> str:
    """Validate API key from request header."""
    if not settings.ML_API_KEY:
        return x_api_key
    if x_api_key != settings.ML_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
