"""HTTP routes for the flight API. Handlers only translate HTTP to service calls."""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response

from mock_apis.common.schemas import ERROR_RESPONSES, ApiResponse
from mock_apis.flight_api import service
from mock_apis.flight_api.config import SERVICE_NAME, FlightSettings
from mock_apis.flight_api.database import get_conn, get_settings
from mock_apis.flight_api.schemas import (
    BookingCreate,
    BookingOut,
    FlightSearchParams,
    FlightSearchResult,
    HealthOut,
)

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Settings = Annotated[FlightSettings, Depends(get_settings)]

health_router = APIRouter(tags=["System"])
router = APIRouter(prefix="/api/v1", responses=ERROR_RESPONSES)


@health_router.get("/health", response_model=ApiResponse[HealthOut], summary="Liveness check")
def health() -> ApiResponse[HealthOut]:
    return ApiResponse(message="Service is healthy.", data=HealthOut(service=SERVICE_NAME, status="ok"))


@router.get(
    "/flights/search",
    response_model=ApiResponse[FlightSearchResult],
    tags=["Availability"],
    summary="Check flight availability",
    description="One-way search. Returns flights on the route with enough free seats for all passengers.",
)
def search_flights(
    params: Annotated[FlightSearchParams, Query()], conn: Conn, settings: Settings
) -> ApiResponse[FlightSearchResult]:
    result = service.search_flights(conn, settings, params)
    message = (
        f"Found {result.count} flight(s) from {result.origin} to {result.destination}."
        if result.count
        else f"No flights from {result.origin} to {result.destination} have availability on this date."
    )
    return ApiResponse(message=message, data=result)


@router.post(
    "/bookings",
    response_model=ApiResponse[BookingOut],
    status_code=201,
    tags=["Bookings"],
    summary="Book a flight",
    description=(
        "Reserves one seat per passenger and prices the booking on the server. "
        "Send an `Idempotency-Key` header to make retries safe: the same key with the same body "
        "returns the original booking (HTTP 200) instead of booking twice."
    ),
)
def create_booking(
    body: BookingCreate,
    response: Response,
    conn: Conn,
    settings: Settings,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", min_length=8, max_length=128)] = None,
) -> ApiResponse[BookingOut]:
    booking, replayed = service.create_booking(conn, settings, body, idempotency_key)
    if replayed:
        response.status_code = 200
        return ApiResponse(message="Booking already created for this Idempotency-Key.", data=booking)
    return ApiResponse(message="Flight booked successfully.", data=booking)


@router.get(
    "/bookings/{booking_id}",
    response_model=ApiResponse[BookingOut],
    tags=["Bookings"],
    summary="Get a booking",
)
def get_booking(booking_id: str, conn: Conn) -> ApiResponse[BookingOut]:
    return ApiResponse(message="Booking found.", data=service.get_booking(conn, booking_id))


@router.post(
    "/bookings/{booking_id}/cancel",
    response_model=ApiResponse[BookingOut],
    tags=["Bookings"],
    summary="Cancel a flight booking",
    description="Cancels the booking and restores its seats. A second cancel returns 409 and changes nothing.",
)
def cancel_booking(booking_id: str, conn: Conn) -> ApiResponse[BookingOut]:
    return ApiResponse(message="Booking cancelled successfully.", data=service.cancel_booking(conn, booking_id))
