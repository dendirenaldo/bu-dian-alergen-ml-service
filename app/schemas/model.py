from typing import Optional

from pydantic import BaseModel


class TrainingStatusResponse(BaseModel):
    status: str
    progress: float
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    message: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_dir: str


class TrainingStartRequest(BaseModel):
    data_path: str
    text_col: str = "text"
    label_col: str = "label"
