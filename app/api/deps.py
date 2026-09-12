from fastapi import Depends, Header, HTTPException

from app.config import settings
from app.services.model_registry import ModelRegistry


def get_model_registry() -> ModelRegistry:
    """Dependency to get the singleton ModelRegistry."""
    return ModelRegistry()


async def get_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """Validate API key from request header.

    Jika ML_API_KEY kosong (dev), header boleh absen.
    Jika ML_API_KEY diset, header wajib cocok.
    """
    if not settings.ML_API_KEY:
        return x_api_key
    if not x_api_key or x_api_key != settings.ML_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
