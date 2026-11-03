"""HTTP client for the Flight API. Every method returns a `ToolResult`; nothing here raises.

Responsibilities: timeouts, retry of safe calls, mapping transport / HTTP / envelope failures
into one error vocabulary, and validating that `data` has the shape the agent relies on.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from travel_agent.config import FlightAPISettings
from travel_agent.flight_agent.schemas import BookingData, SearchData, ToolError, ToolResult

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


class FlightAPIClient:
    def __init__(self, settings: FlightAPISettings | None = None, http: httpx.Client | None = None) -> None:
        self._settings = settings or FlightAPISettings()
        self._http = http or httpx.Client(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(self._settings.read_timeout_s, connect=self._settings.connect_timeout_s),
        )

    # ------------------------------------------------------------------ public operations

    def search_flights(self, params: dict[str, Any]) -> ToolResult:
        return self._call("check_flight_availability", "GET", "/flights/search", SearchData, params=params)

    def create_booking(self, body: dict[str, Any], idempotency_key: str) -> ToolResult:
        return self._call(
            "book_flight", "POST", "/bookings", BookingData, json=body,
            headers={"Idempotency-Key": idempotency_key}, mutating=True,
        )

    def get_booking(self, booking_id: str) -> ToolResult:
        return self._call("get_booking_status", "GET", f"/bookings/{booking_id}", BookingData)

    def cancel_booking(self, booking_id: str) -> ToolResult:
        return self._call("cancel_flight_booking", "POST", f"/bookings/{booking_id}/cancel", BookingData, mutating=True)

    # ----------------------------------------------------------------------- internals

    def _call(
        self,
        tool: str,
        method: str,
        path: str,
        model: type[BaseModel],
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        mutating: bool = False,
    ) -> ToolResult:
        try:
            response = self._http.request(method, API_PREFIX + path, params=params, json=json, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return _failure(tool, "CONNECTION_ERROR", "The flight service could not be reached.", retryable=True)
        except httpx.TimeoutException:
            return _failure(
                tool, "TIMEOUT", "The flight service took too long to answer.", retryable=True, unknown=mutating
            )
        except httpx.HTTPError as exc:
            logger.warning("%s transport error: %s", tool, exc)
            return _failure(
                tool, "CONNECTION_ERROR", "The connection to the flight service was interrupted.",
                retryable=True, unknown=mutating,
            )
        return self._parse(tool, response, model, mutating)

    def _parse(self, tool: str, response: httpx.Response, model: type[BaseModel], mutating: bool) -> ToolResult:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        if status >= 500:
            return _failure(
                tool, "SERVER_ERROR", "The flight service had an internal problem.",
                status=status, retryable=True, unknown=mutating,
            )
        if not isinstance(body, dict):
            return _failure(tool, "INVALID_RESPONSE", "The flight service returned an unreadable response.",
                            status=status, unknown=mutating)
        if status >= 400 or body.get("success") is not True:
            error = body.get("error") if isinstance(body.get("error"), dict) else {}
            return ToolResult(
                tool=tool, ok=False, http_status=status,
                error=ToolError(
                    code=str(error.get("code") or "HTTP_ERROR"),
                    message=str(error.get("message") or f"The flight service returned HTTP {status}."),
                    details=error.get("details"),
                ),
            )
        try:
            data = model.model_validate(body.get("data")).model_dump(exclude_none=True)
        except ValidationError as exc:
            logger.warning("%s returned unexpected data: %s", tool, exc.error_count())
            return _failure(tool, "INVALID_RESPONSE", "The flight service returned incomplete data.",
                            status=status, unknown=mutating)
        return ToolResult(tool=tool, ok=True, http_status=status, message=body.get("message"), data=data)


def _failure(
    tool: str, code: str, message: str, *, status: int | None = None, retryable: bool = False, unknown: bool = False
) -> ToolResult:
    return ToolResult(
        tool=tool, ok=False, http_status=status,
        error=ToolError(code=code, message=message, retryable=retryable, outcome_unknown=unknown),
    )


def backoff(settings: FlightAPISettings, attempt: int) -> None:
    """Sleep before retry number `attempt` (1-based)."""
    if settings.retry_backoff_s > 0:
        time.sleep(settings.retry_backoff_s * attempt)
