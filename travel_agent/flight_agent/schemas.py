"""Contracts of the flight agent: intents, orchestrator task I/O, tool results and API data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    CHECK_FLIGHT_AVAILABILITY = "CHECK_FLIGHT_AVAILABILITY"
    BOOK_FLIGHT = "BOOK_FLIGHT"
    CANCEL_FLIGHT = "CANCEL_FLIGHT"
    GET_BOOKING_STATUS = "GET_BOOKING_STATUS"
    GENERAL_FLIGHT_QUERY = "GENERAL_FLIGHT_QUERY"
    UNKNOWN = "UNKNOWN"


class GeneralTopic(str, Enum):
    FLIGHT_DETAILS = "FLIGHT_DETAILS"
    COMPARE_FLIGHTS = "COMPARE_FLIGHTS"
    SHOW_MORE = "SHOW_MORE"
    HELP = "HELP"
    UNSUPPORTED = "UNSUPPORTED"
    NONE = "NONE"


class ExecutionStatus(str, Enum):
    """Status vocabulary shared with the orchestrator."""

    SUCCESS = "SUCCESS"  # the request was fully handled and the API data proves it
    FAILED = "FAILED"  # nothing was changed (or the user declined / the request is unsupported)
    PENDING = "PENDING"  # an action is prepared and waits for the user's confirmation
    REQUIRES_USER_INPUT = "REQUIRES_USER_INPUT"  # parameters are missing
    UNKNOWN = "UNKNOWN"  # an API write may or may not have been applied; do not assume either


# ------------------------------------------------------------- orchestrator contract


class FlightTask(BaseModel):
    """Structured task an orchestrator (or the CLI) sends to the flight agent."""

    message: str = Field(min_length=1, description="User text, or the user's reply to a pending question")
    thread_id: str = Field("default", description="Conversation key; the same id continues the same dialogue")
    intent: Intent | None = Field(None, description="Optional pre-classified intent (skips classification)")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional pre-extracted parameters")


class FlightTaskResult(BaseModel):
    """Structured answer: user-facing text plus machine-readable status and API-derived data."""

    thread_id: str
    intent: Intent | None = None
    status: ExecutionStatus
    data: dict[str, Any] = Field(default_factory=dict, description="Facts taken from API responses only")
    error: dict[str, Any] | None = None
    requires_user_input: bool = False
    missing_parameters: list[str] = Field(default_factory=list)
    message: str = Field(description="Text to show the user")
    question: str | None = Field(None, description="The pending question when requires_user_input is true")
    outcome: str = Field("", description="Fine-grained result kind, e.g. booking_confirmed")
    nodes_run: list[str] = Field(default_factory=list, description="Graph nodes executed in this call")

    def contract(self) -> dict[str, Any]:
        """The six fields the orchestrator contract requires."""
        return {
            "intent": self.intent.value if self.intent else None,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "requires_user_input": self.requires_user_input,
            "missing_parameters": self.missing_parameters,
        }


# ------------------------------------------------------------------ LLM output schema


class IntentClassification(BaseModel):
    """Constrained JSON the local model must produce (dates, numbers and IDs are parsed by code)."""

    intent: Literal[
        "CHECK_FLIGHT_AVAILABILITY", "BOOK_FLIGHT", "CANCEL_FLIGHT", "GET_BOOKING_STATUS",
        "GENERAL_FLIGHT_QUERY", "UNKNOWN",
    ]
    confidence: float = Field(ge=0, le=1)
    general_topic: Literal["FLIGHT_DETAILS", "COMPARE_FLIGHTS", "SHOW_MORE", "HELP", "UNSUPPORTED", "NONE"]
    origin: str | None
    destination: str | None


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


# ------------------------------------------- validated shapes of Flight API `data` payloads


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AirportData(_ApiModel):
    city: str
    code: str


class FlightData(_ApiModel):
    """One flight from a search. Everything the agent shows comes from these fields."""

    flight_id: str
    airline: str
    flight_number: str
    origin: AirportData
    destination: AirportData
    departure_date: str
    departure_time: str
    arrival_time: str
    price_per_passenger: int
    currency: str
    airline_code: str | None = None
    aircraft: str | None = None
    duration_minutes: int | None = None
    total_price: int | None = None
    seats_available: int | None = None


class SearchData(_ApiModel):
    origin: str | None = None
    destination: str | None = None
    departure_date: str | None = None
    passengers: int | None = None
    count: int | None = None
    flights: list[FlightData]


class BookingFlightData(_ApiModel):
    flight_id: str
    airline: str | None = None
    flight_number: str | None = None
    origin: AirportData | None = None
    destination: AirportData | None = None
    departure_time: str | None = None
    arrival_time: str | None = None


class PassengerData(_ApiModel):
    name: str
    age: int | None = None
    gender: str | None = None


class ContactData(_ApiModel):
    email: str | None = None
    phone: str | None = None


class BookingData(_ApiModel):
    booking_id: str
    status: str
    flight: BookingFlightData | None = None
    departure_date: str | None = None
    num_passengers: int | None = None
    passengers: list[PassengerData] | None = None
    price_per_passenger: int | None = None
    total_price: int | None = None
    currency: str | None = None
    contact: ContactData | None = None
    created_at: str | None = None
    cancelled_at: str | None = None
