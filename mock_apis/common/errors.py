"""Domain error type plus FastAPI handlers that render every failure in one envelope."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Business or validation failure with an HTTP status and a machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> JSONResponse:
    body = {"success": False, "error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status_code, content=body)


def _validation_details(exc: RequestValidationError) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(part) for part in err["loc"] if part not in ("body", "query"))
            or "request",
            "message": str(err["msg"]),
        }
        for err in exc.errors()
    ]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422, "VALIDATION_ERROR", "The request is invalid.", _validation_details(exc)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return error_response(500, "INTERNAL_ERROR", "Something went wrong. Please try again.")
