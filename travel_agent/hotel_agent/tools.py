"""LangChain tools over the Hotel API. Each returns a compact `ToolResult` dict and never raises."""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any, Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError

from travel_agent.config import HotelAPISettings
from travel_agent.hotel_agent.api_client import HotelAPIClient, backoff
from travel_agent.hotel_agent.schemas import ToolError, ToolResult

logger = logging.getLogger(__name__)

HotelId = Annotated[str, Field(min_length=3, max_length=20, pattern=r"^[A-Za-z0-9-]+$")]
BookingId = Annotated[str, Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9-]+$")]


class AvailabilityArgs(BaseModel):
    city: str = Field(min_length=2, max_length=60)
    check_in: date
    check_out: date
    guests: int = Field(1, ge=1, le=20)
    rooms: int = Field(1, ge=1, le=5)
    max_price_per_night: int | None = Field(None, gt=0)
    min_rating: float | None = Field(None, ge=0, le=5)


class HotelIdArgs(BaseModel):
    hotel_id: HotelId


class BookHotelArgs(BaseModel):
    hotel_id: HotelId
    check_in: date
    check_out: date
    rooms: int = Field(1, ge=1, le=5)
    guests: int = Field(1, ge=1, le=20)
    guest_name: str = Field(min_length=2, max_length=100)
    guest_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    guest_phone: str = Field(pattern=r"^\+?[0-9]{10,15}$")
    idempotency_key: str = Field(min_length=8, max_length=128)


class BookingIdArgs(BaseModel):
    booking_id: BookingId


def build_hotel_tools(client: HotelAPIClient, settings: HotelAPISettings | None = None) -> dict[str, BaseTool]:
    """Create the five hotel tools bound to `client`."""
    settings = settings or HotelAPISettings()

    def with_retry(call: Callable[[], ToolResult]) -> ToolResult:
        """Retry transient failures; only used for calls that are safe to repeat."""
        result = call()
        attempt = 1
        while not result.ok and result.error and result.error.retryable and attempt < settings.max_attempts:
            backoff(settings, attempt)
            attempt += 1
            result = call()
        return result.model_copy(update={"attempts": attempt})

    def check_hotel_availability(**kwargs: Any) -> dict[str, Any]:
        args = AvailabilityArgs(**kwargs)
        query = args.model_dump(mode="json", exclude_none=True)
        return with_retry(lambda: client.search_hotels(query)).model_dump(mode="json")

    def get_hotel_details(hotel_id: str) -> dict[str, Any]:
        return with_retry(lambda: client.get_hotel(hotel_id)).model_dump(mode="json")

    def book_hotel(**kwargs: Any) -> dict[str, Any]:
        args = BookHotelArgs(**kwargs)
        body = {
            "hotel_id": args.hotel_id,
            "check_in": args.check_in.isoformat(),
            "check_out": args.check_out.isoformat(),
            "rooms": args.rooms,
            "guests": args.guests,
            "guest": {"name": args.guest_name, "email": args.guest_email, "phone": args.guest_phone},
        }
        # The idempotency key makes retrying a possibly-applied booking safe.
        return with_retry(lambda: client.create_booking(body, args.idempotency_key)).model_dump(mode="json")

    def get_booking_status(booking_id: str) -> dict[str, Any]:
        return with_retry(lambda: client.get_booking(booking_id)).model_dump(mode="json")

    def cancel_hotel_booking(booking_id: str) -> dict[str, Any]:
        # Not retried: a repeated cancel would report "already cancelled" and hide the real outcome.
        return client.cancel_booking(booking_id).model_dump(mode="json")

    specs: list[tuple[Callable[..., dict[str, Any]], type[BaseModel], str]] = [
        (check_hotel_availability, AvailabilityArgs, "Search hotels with free rooms in a city for given dates."),
        (get_hotel_details, HotelIdArgs, "Get details (rating, price, amenities) of one hotel by hotel_id."),
        (book_hotel, BookHotelArgs, "Book a hotel. Only call after the user explicitly confirmed."),
        (get_booking_status, BookingIdArgs, "Get the current status of a booking by booking_id."),
        (cancel_hotel_booking, BookingIdArgs, "Cancel a booking. Only call after the user explicitly confirmed."),
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
