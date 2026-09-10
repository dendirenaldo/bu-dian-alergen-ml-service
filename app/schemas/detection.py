from typing import List, Optional

from pydantic import BaseModel


class TextDetectionRequest(BaseModel):
    text: str


class AllergenResultSchema(BaseModel):
    name: str
    confidence: float
    severity: str


class DetectionResponse(BaseModel):
    result: str
    confidence_score: float
    ocr_text: Optional[str] = None
    allergens: List[AllergenResultSchema] = []
    processing_time_ms: int
    detection_method: str
