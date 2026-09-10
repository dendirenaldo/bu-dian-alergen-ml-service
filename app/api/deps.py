from fastapi import Depends

from app.services.model_registry import ModelRegistry


def get_model_registry() -> ModelRegistry:
    """Dependency to get the singleton ModelRegistry."""
    return ModelRegistry()
