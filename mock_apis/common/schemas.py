"""Response envelopes shared by both APIs so agents can parse every reply the same way."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Successful reply: `{"success": true, "message": "...", "data": {...}}`."""

    success: bool = True
    message: str
    data: T


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """Failed reply: `{"success": false, "error": {"code", "message", "details"}}`."""

    success: bool = False
    error: ErrorBody


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
    409: {"model": ErrorResponse, "description": "Conflict with current state"},
    422: {"model": ErrorResponse, "description": "Invalid request"},
}
