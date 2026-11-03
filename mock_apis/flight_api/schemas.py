"""Request and response schemas for the flight API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mock_apis.flight_api.config import MAX_PASSENGERS_PER_BOOKING

Email = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
Phone = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\+?[0-9]{10,15}$")]

# ----------------------------------------------------------------------------- requests


class FlightSearchParams(BaseModel):
    """Query parameters for `GET /api/v1/flights/search` (one-way)."""

    model_config = ConfigDict(extra="forbid")

    origin: str = Field(min_length=2, max_length=60, description="Departure city or IATA code, e.g. Bangalore or BLR")
    destination: str = Field(min_length=2, max_length=60, description="Destination city or IATA code")
    departure_date: date = Field(description="Departure date (YYYY-MM-DD)")
    passengers: int = Field(1, ge=1, le=MAX_PASSENGERS_PER_BOOKING)

    @model_validator(mode="after")
    def _different_airports(self) -> FlightSearchParams:
        if self.origin.strip().lower() == self.destination.strip().lower():
            raise ValueError("origin and destination must be different")
        return self


class PassengerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=120)
    gender: Literal["MALE", "FEMALE", "OTHER"]


class ContactIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: Email
    phone: Phone


class BookingCreate(BaseModel):
    """Body for `POST /api/v1/bookings`. Fares are always computed server-side."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "flight_id": "FL-1001",
                "departure_date": "2026-12-10",
                "passengers": [
                    {"name": "Asha Rao", "age": 31, "gender": "FEMALE"},
                    {"name": "Vikram Rao", "age": 34, "gender": "MALE"},
                ],
                "contact": {"email": "asha@example.com", "phone": "+919876543210"},
            }
        },
    )

    flight_id: str = Field(min_length=3, max_length=20, description="`flight_id` from the search results")
    departure_date: date
    passengers: list[PassengerIn] = Field(min_length=1, max_length=MAX_PASSENGERS_PER_BOOKING)
    contact: ContactIn


# ---------------------------------------------------------------------------- responses


class Airport(BaseModel):
    city: str
    code: str


class FlightOption(BaseModel):
    flight_id: str
    airline: str
    airline_code: str
    flight_number: str
    aircraft: str
    origin: Airport
    destination: Airport
    departure_date: date
    departure_time: datetime = Field(description="Local time at origin")
    arrival_time: datetime = Field(description="Local time at destination (all times are IST)")
    duration_minutes: int
    price_per_passenger: int
    total_price: int = Field(description="Price per passenger x requested passengers")
    currency: str
    seats_available: int


class FlightSearchResult(BaseModel):
    origin: str
    destination: str
    departure_date: date
    passengers: int
    count: int
    flights: list[FlightOption]


class FlightSummary(BaseModel):
    flight_id: str
    airline: str
    flight_number: str
    origin: Airport
    destination: Airport
    departure_time: datetime
    arrival_time: datetime


class PassengerOut(BaseModel):
    name: str
    age: int
    gender: str


class ContactOut(BaseModel):
    email: str
    phone: str


class BookingOut(BaseModel):
    booking_id: str
    status: str = Field(description="CONFIRMED or CANCELLED")
    flight: FlightSummary
    departure_date: date
    num_passengers: int
    passengers: list[PassengerOut]
    price_per_passenger: int
    total_price: int
    currency: str
    contact: ContactOut
    created_at: str
    cancelled_at: str | None = None


class HealthOut(BaseModel):
    service: str
    status: str
