"""Hotel business logic: validation, pricing, availability and transactional booking/cancellation."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone

from mock_apis.common.db import write_transaction
from mock_apis.common.errors import APIError
from mock_apis.common.ids import new_booking_id, request_fingerprint
from mock_apis.hotel_api import repository as repo
from mock_apis.hotel_api.config import HotelSettings
from mock_apis.hotel_api.models import STATUS_CANCELLED, STATUS_CONFIRMED, Booking, Hotel
from mock_apis.hotel_api.schemas import (
    BookingCreate,
    BookingOut,
    GuestOut,
    HotelAvailability,
    HotelDetails,
    HotelSearchParams,
    HotelSearchResult,
    HotelSummary,
)


def _today() -> date:
    return date.today()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_window(check_in: date, check_out: date, settings: HotelSettings) -> None:
    today = _today()
    if check_in < today:
        raise APIError(422, "INVALID_DATE", "check_in cannot be in the past.", {"today": today.isoformat()})
    last_night = check_out - timedelta(days=1)
    last_bookable = today + timedelta(days=settings.inventory_horizon_days)
    if last_night > last_bookable:
        raise APIError(
            422,
            "INVALID_DATE",
            f"Bookings are open only up to {last_bookable.isoformat()}.",
            {"last_bookable_night": last_bookable.isoformat()},
        )


def _details(hotel: Hotel) -> HotelDetails:
    return HotelDetails(**asdict(hotel))


def _summary(hotel: Hotel) -> HotelSummary:
    return HotelSummary(
        hotel_id=hotel.hotel_id,
        name=hotel.name,
        city=hotel.city,
        locality=hotel.locality,
        rating=hotel.rating,
    )


def _booking_out(booking: Booking, hotel: Hotel) -> BookingOut:
    return BookingOut(
        booking_id=booking.booking_id,
        status=booking.status,
        hotel=_summary(hotel),
        check_in=booking.check_in,
        check_out=booking.check_out,
        nights=booking.nights,
        rooms=booking.rooms,
        guests=booking.guests,
        price_per_night=booking.price_per_night,
        total_price=booking.total_price,
        currency=booking.currency,
        guest=GuestOut(name=booking.guest_name, email=booking.guest_email, phone=booking.guest_phone),
        created_at=booking.created_at,
        cancelled_at=booking.cancelled_at,
    )


# ------------------------------------------------------------------------------- queries


def search_hotels(
    conn: sqlite3.Connection, settings: HotelSettings, params: HotelSearchParams
) -> HotelSearchResult:
    _validate_window(params.check_in, params.check_out, settings)
    nights = (params.check_out - params.check_in).days
    found = repo.search_available(
        conn,
        city=params.city.strip(),
        check_in=params.check_in.isoformat(),
        check_out=params.check_out.isoformat(),
        nights=nights,
        rooms=params.rooms,
        guests=params.guests,
        min_rating=params.min_rating,
        max_price=params.max_price_per_night,
    )
    hotels = [
        HotelAvailability(
            **_details(hotel).model_dump(),
            rooms_available=rooms_left,
            nights=nights,
            rooms_requested=params.rooms,
            estimated_total_price=hotel.price_per_night * nights * params.rooms,
        )
        for hotel, rooms_left in found
    ]
    return HotelSearchResult(
        city=params.city.strip(),
        check_in=params.check_in,
        check_out=params.check_out,
        nights=nights,
        guests=params.guests,
        rooms=params.rooms,
        count=len(hotels),
        hotels=hotels,
    )


def get_hotel(conn: sqlite3.Connection, hotel_id: str) -> HotelDetails:
    hotel = repo.get_hotel(conn, hotel_id)
    if hotel is None:
        raise APIError(404, "HOTEL_NOT_FOUND", f"Hotel '{hotel_id}' does not exist.")
    return _details(hotel)


def get_booking(conn: sqlite3.Connection, booking_id: str) -> BookingOut:
    booking = _require_booking(conn, booking_id)
    return _booking_out(booking, _require_hotel(conn, booking.hotel_id))


# ------------------------------------------------------------------------------ commands


def create_booking(
    conn: sqlite3.Connection,
    settings: HotelSettings,
    request: BookingCreate,
    idempotency_key: str | None = None,
) -> tuple[BookingOut, bool]:
    """Create a booking atomically. Returns (booking, replayed) where replayed means an
    earlier request with the same idempotency key was returned instead of booking again."""
    fingerprint = request_fingerprint(request.model_dump(mode="json"))
    nights = (request.check_out - request.check_in).days
    check_in, check_out = request.check_in.isoformat(), request.check_out.isoformat()

    with write_transaction(conn):
        if idempotency_key:
            existing = repo.get_booking_by_key(conn, idempotency_key)
            if existing:
                if existing.request_hash != fingerprint:
                    raise APIError(
                        409,
                        "IDEMPOTENCY_KEY_CONFLICT",
                        "This Idempotency-Key was already used with a different request.",
                    )
                return _booking_out(existing, _require_hotel(conn, existing.hotel_id)), True

        _validate_window(request.check_in, request.check_out, settings)
        hotel = repo.get_hotel(conn, request.hotel_id)
        if hotel is None:
            raise APIError(404, "HOTEL_NOT_FOUND", f"Hotel '{request.hotel_id}' does not exist.")

        capacity = hotel.max_guests_per_room * request.rooms
        if request.guests > capacity:
            raise APIError(
                422,
                "CAPACITY_EXCEEDED",
                f"{request.rooms} room(s) at {hotel.name} host at most {capacity} guests.",
                {"max_guests": capacity, "requested_guests": request.guests},
            )

        days, rooms_left = repo.stay_availability(conn, hotel.hotel_id, check_in, check_out)
        if days != nights or rooms_left < request.rooms:
            raise APIError(
                409,
                "ROOMS_UNAVAILABLE",
                f"Not enough rooms at {hotel.name} for the whole stay.",
                {"rooms_requested": request.rooms, "rooms_available": rooms_left if days == nights else 0},
            )
        if repo.decrement_inventory(conn, hotel.hotel_id, check_in, check_out, request.rooms) != nights:
            raise APIError(409, "ROOMS_UNAVAILABLE", f"Not enough rooms at {hotel.name} for the whole stay.")

        booking = Booking(
            booking_id=new_booking_id("HB"),
            hotel_id=hotel.hotel_id,
            guest_name=request.guest.name,
            guest_email=request.guest.email,
            guest_phone=request.guest.phone,
            check_in=check_in,
            check_out=check_out,
            nights=nights,
            rooms=request.rooms,
            guests=request.guests,
            price_per_night=hotel.price_per_night,
            total_price=hotel.price_per_night * nights * request.rooms,
            currency=hotel.currency,
            status=STATUS_CONFIRMED,
            idempotency_key=idempotency_key or None,
            request_hash=fingerprint,
            created_at=_now_iso(),
            cancelled_at=None,
        )
        repo.insert_booking(conn, booking)
    return _booking_out(booking, hotel), False


def cancel_booking(conn: sqlite3.Connection, booking_id: str) -> BookingOut:
    """Cancel a booking and give its rooms back exactly once."""
    with write_transaction(conn):
        booking = _require_booking(conn, booking_id)
        if booking.status == STATUS_CANCELLED:
            raise APIError(
                409,
                "BOOKING_ALREADY_CANCELLED",
                f"Booking '{booking_id}' is already cancelled.",
                {"cancelled_at": booking.cancelled_at},
            )
        cancelled_at = _now_iso()
        if not repo.mark_cancelled(conn, booking_id, cancelled_at):
            raise APIError(409, "BOOKING_ALREADY_CANCELLED", f"Booking '{booking_id}' is already cancelled.")
        repo.increment_inventory(conn, booking.hotel_id, booking.check_in, booking.check_out, booking.rooms)
        hotel = _require_hotel(conn, booking.hotel_id)
    cancelled = replace(booking, status=STATUS_CANCELLED, cancelled_at=cancelled_at)
    return _booking_out(cancelled, hotel)


# ------------------------------------------------------------------------------- helpers


def _require_booking(conn: sqlite3.Connection, booking_id: str) -> Booking:
    booking = repo.get_booking(conn, booking_id)
    if booking is None:
        raise APIError(404, "BOOKING_NOT_FOUND", f"Booking '{booking_id}' does not exist.")
    return booking


def _require_hotel(conn: sqlite3.Connection, hotel_id: str) -> Hotel:
    hotel = repo.get_hotel(conn, hotel_id)
    if hotel is None:
        raise APIError(404, "HOTEL_NOT_FOUND", f"Hotel '{hotel_id}' does not exist.")
    return hotel
