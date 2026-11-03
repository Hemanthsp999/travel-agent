"""LangChain tools over the Flight API. Each returns a compact `ToolResult` dict and never raises."""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any, Callable, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from travel_agent.config import FlightAPISettings
from travel_agent.flight_agent.api_client import FlightAPIClient, backoff
from travel_agent.flight_agent.schemas import ToolError, ToolResult

logger = logging.getLogger(__name__)

BookingId = Annotated[str, Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9-]+$")]
MAX_PASSENGERS = 9


class AvailabilityArgs(BaseModel):
    origin: str = Field(min_length=2, max_length=60)
    destination: str = Field(min_length=2, max_length=60)
    departure_date: date
    passengers: int = Field(ge=1, le=MAX_PASSENGERS)


class PassengerArgs(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=120)
    gender: Literal["MALE", "FEMALE", "OTHER"]


class BookFlightArgs(BaseModel):
    flight_id: str = Field(min_length=3, max_length=20, pattern=r"^[A-Za-z0-9-]+$")
    departure_date: date
    passengers: list[PassengerArgs] = Field(min_length=1, max_length=MAX_PASSENGERS)
    contact_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    contact_phone: str = Field(pattern=r"^\+?[0-9]{10,15}$")
    idempotency_key: str = Field(min_length=8, max_length=128)


class BookingIdArgs(BaseModel):
    booking_id: BookingId


def build_flight_tools(client: FlightAPIClient, settings: FlightAPISettings | None = None) -> dict[str, BaseTool]:
    """Create the four flight tools bound to `client`.

    The Flight API has no flight-details, booking-list, modify, check-in or refund endpoints, so no
    tools exist for them; the agent says so instead of improvising.
    """
    settings = settings or FlightAPISettings()

    def with_retry(call: Callable[[], ToolResult]) -> ToolResult:
        """Retry transient failures; only used for calls that are safe to repeat."""
        result = call()
        attempt = 1
        while not result.ok and result.error and result.error.retryable and attempt < settings.max_attempts:
            backoff(settings, attempt)
            attempt += 1
            result = call()
        return result.model_copy(update={"attempts": attempt})

    def check_flight_availability(**kwargs: Any) -> dict[str, Any]:
        args = AvailabilityArgs(**kwargs)
        query = args.model_dump(mode="json")
        return with_retry(lambda: client.search_flights(query)).model_dump(mode="json")

    def book_flight(**kwargs: Any) -> dict[str, Any]:
        args = BookFlightArgs(**kwargs)
        body = {
            "flight_id": args.flight_id,
            "departure_date": args.departure_date.isoformat(),
            "passengers": [p.model_dump() for p in args.passengers],
            "contact": {"email": args.contact_email, "phone": args.contact_phone},
        }
        # The idempotency key makes retrying a possibly-applied booking safe.
        return with_retry(lambda: client.create_booking(body, args.idempotency_key)).model_dump(mode="json")

    def get_booking_status(booking_id: str) -> dict[str, Any]:
        return with_retry(lambda: client.get_booking(booking_id)).model_dump(mode="json")

    def cancel_flight_booking(booking_id: str) -> dict[str, Any]:
        # Not retried: a repeated cancel would report "already cancelled" and hide the real outcome.
        return client.cancel_booking(booking_id).model_dump(mode="json")

    specs: list[tuple[Callable[..., dict[str, Any]], type[BaseModel], str]] = [
        (check_flight_availability, AvailabilityArgs, "Search flights with free seats between two cities on a date."),
        (book_flight, BookFlightArgs, "Book a flight. Only call after the user explicitly confirmed."),
        (get_booking_status, BookingIdArgs, "Get the current status and details of a booking by booking_id."),
        (cancel_flight_booking, BookingIdArgs, "Cancel a booking. Only call after the user explicitly confirmed."),
    ]
    return {
        fn.__name__: StructuredTool.from_function(func=fn, name=fn.__name__, description=desc, args_schema=schema)
        for fn, schema, desc in specs
    }


def run_tool(tools: dict[str, BaseTool], name: str, args: dict[str, Any]) -> ToolResult:
    """Invoke a tool by name; invalid arguments or unexpected failures become a `ToolResult` error."""
    tool = tools.get(name)
    if tool is None:
        return _client_error(name, "UNKNOWN_TOOL", f"Tool '{name}' is not available.")
    try:
        return ToolResult.model_validate(tool.invoke(args))
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        return _client_error(name, "INVALID_ARGUMENTS", "Some details are not valid.", details=fields)
    except Exception as exc:  # never let a tool crash the graph
        logger.exception("Tool %s failed", name)
        return _client_error(name, "TOOL_ERROR", f"Tool failed unexpectedly: {type(exc).__name__}")


def _client_error(tool: str, code: str, message: str, details: Any | None = None) -> ToolResult:
    return ToolResult(tool=tool, ok=False, error=ToolError(code=code, message=message, details=details))
