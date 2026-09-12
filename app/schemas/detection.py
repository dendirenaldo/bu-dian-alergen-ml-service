from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TextDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class AllergenResultSchema(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str


class DetectionResponse(BaseModel):
    result: Literal["safe", "unsafe"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    ocr_text: Optional[str] = None
    allergens: List[AllergenResultSchema] = []
    processing_time_ms: int = Field(..., ge=0)
    detection_method: Literal["image_ocr", "text_input"]
