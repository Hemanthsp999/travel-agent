"""Request and response schemas for the hotel API."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mock_apis.hotel_api.config import (
    MAX_GUESTS_PER_BOOKING,
    MAX_ROOMS_PER_BOOKING,
    MAX_STAY_NIGHTS,
)

Email = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
Phone = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\+?[0-9]{10,15}$")]


def _check_stay(check_in: date, check_out: date) -> None:
    nights = (check_out - check_in).days
    if nights < 1:
        raise ValueError("check_out must be after check_in")
    if nights > MAX_STAY_NIGHTS:
        raise ValueError(f"stay cannot exceed {MAX_STAY_NIGHTS} nights")


# ----------------------------------------------------------------------------- requests


class HotelSearchParams(BaseModel):
    """Query parameters for `GET /api/v1/hotels/search`."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=2, max_length=60, description="City name, e.g. Bangalore")
    check_in: date = Field(description="Check-in date (YYYY-MM-DD)")
    check_out: date = Field(description="Check-out date (YYYY-MM-DD), after check_in")
    guests: int = Field(1, ge=1, le=MAX_GUESTS_PER_BOOKING)
    rooms: int = Field(1, ge=1, le=MAX_ROOMS_PER_BOOKING)
    min_rating: float | None = Field(None, ge=0, le=5, description="Optional minimum rating")
    max_price_per_night: int | None = Field(None, gt=0, description="Optional price ceiling (INR)")

    @model_validator(mode="after")
    def _validate_dates(self) -> HotelSearchParams:
        _check_stay(self.check_in, self.check_out)
        return self


class GuestIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=100)
    email: Email
    phone: Phone


class BookingCreate(BaseModel):
    """Body for `POST /api/v1/bookings`. Prices are always computed server-side."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "hotel_id": "HT-1001",
                "check_in": "2026-12-10",
                "check_out": "2026-12-12",
                "rooms": 1,
                "guests": 2,
                "guest": {"name": "Asha Rao", "email": "asha@example.com", "phone": "+919876543210"},
            }
        },
    )

    hotel_id: str = Field(min_length=3, max_length=20)
    check_in: date
    check_out: date
    rooms: int = Field(1, ge=1, le=MAX_ROOMS_PER_BOOKING)
    guests: int = Field(1, ge=1, le=MAX_GUESTS_PER_BOOKING)
    guest: GuestIn

    @model_validator(mode="after")
    def _validate_dates(self) -> BookingCreate:
        _check_stay(self.check_in, self.check_out)
        return self


# ---------------------------------------------------------------------------- responses


class HotelSummary(BaseModel):
    hotel_id: str
    name: str
    city: str
    locality: str
    rating: float


class HotelDetails(HotelSummary):
    address: str
    star_category: int
    price_per_night: int
    currency: str
    total_rooms: int
    max_guests_per_room: int
    amenities: list[str]


class HotelAvailability(HotelDetails):
    rooms_available: int = Field(description="Rooms free on every night of the stay")
    nights: int
    rooms_requested: int
    estimated_total_price: int


class HotelSearchResult(BaseModel):
    city: str
    check_in: date
    check_out: date
    nights: int
    guests: int
    rooms: int
    count: int
    hotels: list[HotelAvailability]


class GuestOut(BaseModel):
    name: str
    email: str
    phone: str


class BookingOut(BaseModel):
    booking_id: str
    status: str = Field(description="CONFIRMED or CANCELLED")
    hotel: HotelSummary
    check_in: date
    check_out: date
    nights: int
    rooms: int
    guests: int
    price_per_night: int
    total_price: int
    currency: str
    guest: GuestOut
    created_at: str
    cancelled_at: str | None = None


class HealthOut(BaseModel):
    service: str
    status: str
