"""Request/response contracts: intents, orchestrator task I/O, tool results and API data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    CHECK_AVAILABILITY = "CHECK_AVAILABILITY"
    BOOK_HOTEL = "BOOK_HOTEL"
    CANCEL_BOOKING = "CANCEL_BOOKING"
    GET_BOOKING_STATUS = "GET_BOOKING_STATUS"
    GENERAL_HOTEL_QUERY = "GENERAL_HOTEL_QUERY"
    UNKNOWN = "UNKNOWN"


class GeneralTopic(str, Enum):
    HOTEL_DETAILS = "HOTEL_DETAILS"
    COMPARE_HOTELS = "COMPARE_HOTELS"
    SHOW_MORE = "SHOW_MORE"
    BUDGET_SEARCH = "BUDGET_SEARCH"
    ROOM_AVAILABILITY = "ROOM_AVAILABILITY"
    HELP = "HELP"
    UNSUPPORTED = "UNSUPPORTED"
    NONE = "NONE"


class TaskStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    NEEDS_CONFIRMATION = "needs_confirmation"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    DECLINED = "declined"


# ------------------------------------------------------------- orchestrator contract


class HotelTask(BaseModel):
    """Structured task an orchestrator (or the CLI) sends to the hotel agent."""

    message: str = Field(min_length=1, description="User text, or the user's reply to a pending question")
    thread_id: str = Field("default", description="Conversation key; the same id continues the same dialogue")
    intent: Intent | None = Field(None, description="Optional pre-classified intent (skips classification)")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional pre-extracted parameters")


class HotelTaskResult(BaseModel):
    """What the agent returns: user-facing text plus machine-readable status and API-derived data."""

    thread_id: str
    status: TaskStatus
    intent: Intent | None = None
    message: str = Field(description="Text to show the user")
    question: str | None = Field(None, description="Set when status is needs_input / needs_confirmation")
    missing_params: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict, description="Facts taken from API responses only")
    error: dict[str, Any] | None = None
    nodes_run: list[str] = Field(default_factory=list, description="Graph nodes executed in this call")


# ------------------------------------------------------------------ LLM output schema


class IntentClassification(BaseModel):
    """Constrained JSON the local model must produce (dates and numbers are parsed by code)."""

    intent: Literal[
        "CHECK_AVAILABILITY", "BOOK_HOTEL", "CANCEL_BOOKING", "GET_BOOKING_STATUS", "GENERAL_HOTEL_QUERY", "UNKNOWN"
    ]
    confidence: float = Field(ge=0, le=1)
    general_topic: Literal[
        "HOTEL_DETAILS", "COMPARE_HOTELS", "SHOW_MORE", "BUDGET_SEARCH", "ROOM_AVAILABILITY", "HELP", "UNSUPPORTED", "NONE"
    ]
    city: str | None
    hotel_name: str | None
    guest_name: str | None


# ------------------------------------------------------------------- tool results


class ToolError(BaseModel):
    code: str
    message: str
    details: Any | None = None
    retryable: bool = False
    outcome_unknown: bool = Field(False, description="True if a write may or may not have been applied")


class ToolResult(BaseModel):
    """Uniform return value of every API tool; tools never raise."""

    tool: str
    ok: bool
    http_status: int | None = None
    message: str | None = None
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    attempts: int = 1


# ------------------------------------------- validated shapes of Hotel API `data` payloads


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HotelData(_ApiModel):
    """Search hit or hotel details. Only `hotel_id` and `name` are required; the rest is optional."""

    hotel_id: str
    name: str
    city: str | None = None
    locality: str | None = None
    address: str | None = None
    rating: float | None = None
    star_category: int | None = None
    price_per_night: int | None = None
    currency: str | None = None
    total_rooms: int | None = None
    max_guests_per_room: int | None = None
    amenities: list[str] | None = None
    rooms_available: int | None = None
    nights: int | None = None
    rooms_requested: int | None = None
    estimated_total_price: int | None = None


class SearchData(_ApiModel):
    city: str | None = None
    check_in: str | None = None
    check_out: str | None = None
    nights: int | None = None
    guests: int | None = None
    rooms: int | None = None
    hotels: list[HotelData]


class BookingHotelData(_ApiModel):
    hotel_id: str
    name: str | None = None
    city: str | None = None
    locality: str | None = None


class GuestData(_ApiModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None


class BookingData(_ApiModel):
    booking_id: str
    status: str
    hotel: BookingHotelData | None = None
    check_in: str | None = None
    check_out: str | None = None
    nights: int | None = None
    rooms: int | None = None
    guests: int | None = None
    price_per_night: int | None = None
    total_price: int | None = None
    currency: str | None = None
    guest: GuestData | None = None
    created_at: str | None = None
    cancelled_at: str | None = None
