"""HTTP routes for the hotel API. Handlers only translate HTTP to service calls."""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response

from mock_apis.common.schemas import ERROR_RESPONSES, ApiResponse
from mock_apis.hotel_api import service
from mock_apis.hotel_api.config import HotelSettings, SERVICE_NAME
from mock_apis.hotel_api.database import get_conn, get_settings
from mock_apis.hotel_api.schemas import (
    BookingCreate,
    BookingOut,
    HealthOut,
    HotelDetails,
    HotelSearchParams,
    HotelSearchResult,
)

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Settings = Annotated[HotelSettings, Depends(get_settings)]

health_router = APIRouter(tags=["System"])
router = APIRouter(prefix="/api/v1", responses=ERROR_RESPONSES)


@health_router.get("/health", response_model=ApiResponse[HealthOut], summary="Liveness check")
def health() -> ApiResponse[HealthOut]:
    return ApiResponse(message="Service is healthy.", data=HealthOut(service=SERVICE_NAME, status="ok"))


@router.get(
    "/hotels/search",
    response_model=ApiResponse[HotelSearchResult],
    tags=["Availability"],
    summary="Check hotel availability",
    description="Returns hotels in a city with enough rooms free on every night of the stay.",
)
def search_hotels(
    params: Annotated[HotelSearchParams, Query()], conn: Conn, settings: Settings
) -> ApiResponse[HotelSearchResult]:
    result = service.search_hotels(conn, settings, params)
    message = (
        f"Found {result.count} hotel(s) in {result.city}."
        if result.count
        else f"No hotels in {result.city} have availability for these dates."
    )
    return ApiResponse(message=message, data=result)


@router.get(
    "/hotels/{hotel_id}",
    response_model=ApiResponse[HotelDetails],
    tags=["Availability"],
    summary="Get hotel details",
)
def get_hotel(hotel_id: str, conn: Conn) -> ApiResponse[HotelDetails]:
    return ApiResponse(message="Hotel found.", data=service.get_hotel(conn, hotel_id))


@router.post(
    "/bookings",
    response_model=ApiResponse[BookingOut],
    status_code=201,
    tags=["Bookings"],
    summary="Book a hotel",
    description=(
        "Reserves rooms for every night of the stay and prices the booking on the server. "
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
    return ApiResponse(message="Hotel booked successfully.", data=booking)


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
    summary="Cancel a hotel booking",
    description="Cancels the booking and restores its rooms. A second cancel returns 409 and changes nothing.",
)
def cancel_booking(booking_id: str, conn: Conn) -> ApiResponse[BookingOut]:
    return ApiResponse(message="Booking cancelled successfully.", data=service.cancel_booking(conn, booking_id))
