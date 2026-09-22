from typing import Optional

from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_dir: str
    models: Optional[list] = None
