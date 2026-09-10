from typing import Any, Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None
