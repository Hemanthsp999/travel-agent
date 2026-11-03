"""Flight business logic: validation, pricing, search and transactional booking/cancellation."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

from mock_apis.common.db import write_transaction
from mock_apis.common.errors import APIError
from mock_apis.common.ids import new_booking_id, request_fingerprint
from mock_apis.flight_api import repository as repo
from mock_apis.flight_api.config import FlightSettings
from mock_apis.flight_api.models import (
    STATUS_CANCELLED,
    STATUS_CONFIRMED,
    Booking,
    Flight,
    Passenger,
)
from mock_apis.flight_api.schemas import (
    Airport,
    BookingCreate,
    BookingOut,
    ContactOut,
    FlightOption,
    FlightSearchParams,
    FlightSearchResult,
    FlightSummary,
    PassengerOut,
)


def _today() -> date:
    return date.today()


def _now() -> datetime:
    return datetime.now()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _departure(flight: Flight, flight_date: date) -> datetime:
    return datetime.combine(flight_date, time.fromisoformat(flight.departure_time))


def _arrival(flight: Flight, flight_date: date) -> datetime:
    return _departure(flight, flight_date) + timedelta(minutes=flight.duration_minutes)


def _validate_date(flight_date: date, settings: FlightSettings) -> None:
    today = _today()
    if flight_date < today:
        raise APIError(422, "INVALID_DATE", "departure_date cannot be in the past.", {"today": today.isoformat()})
    last_bookable = today + timedelta(days=settings.inventory_horizon_days)
    if flight_date > last_bookable:
        raise APIError(
            422,
            "INVALID_DATE",
            f"Flights can be booked only up to {last_bookable.isoformat()}.",
            {"last_bookable_date": last_bookable.isoformat()},
        )


def _has_departed(flight: Flight, flight_date: date) -> bool:
    return _departure(flight, flight_date) <= _now()


def _origin(flight: Flight) -> Airport:
    return Airport(city=flight.origin_city, code=flight.origin_code)


def _destination(flight: Flight) -> Airport:
    return Airport(city=flight.destination_city, code=flight.destination_code)


def _booking_out(booking: Booking, flight: Flight, passengers: list[Passenger]) -> BookingOut:
    flight_date = date.fromisoformat(booking.departure_date)
    return BookingOut(
        booking_id=booking.booking_id,
        status=booking.status,
        flight=FlightSummary(
            flight_id=flight.flight_id,
            airline=flight.airline,
            flight_number=flight.flight_number,
            origin=_origin(flight),
            destination=_destination(flight),
            departure_time=_departure(flight, flight_date),
            arrival_time=_arrival(flight, flight_date),
        ),
        departure_date=flight_date,
        num_passengers=booking.num_passengers,
        passengers=[PassengerOut(name=p.name, age=p.age, gender=p.gender) for p in passengers],
        price_per_passenger=booking.price_per_passenger,
        total_price=booking.total_price,
        currency=booking.currency,
        contact=ContactOut(email=booking.contact_email, phone=booking.contact_phone),
        created_at=booking.created_at,
        cancelled_at=booking.cancelled_at,
    )


# ------------------------------------------------------------------------------- queries


def search_flights(
    conn: sqlite3.Connection, settings: FlightSettings, params: FlightSearchParams
) -> FlightSearchResult:
    _validate_date(params.departure_date, settings)
    found = repo.search_flights(
        conn,
        origin=params.origin.strip(),
        destination=params.destination.strip(),
        flight_date=params.departure_date.isoformat(),
        passengers=params.passengers,
    )
    options = [
        FlightOption(
            flight_id=flight.flight_id,
            airline=flight.airline,
            airline_code=flight.airline_code,
            flight_number=flight.flight_number,
            aircraft=flight.aircraft,
            origin=_origin(flight),
            destination=_destination(flight),
            departure_date=params.departure_date,
            departure_time=_departure(flight, params.departure_date),
            arrival_time=_arrival(flight, params.departure_date),
            duration_minutes=flight.duration_minutes,
            price_per_passenger=flight.price,
            total_price=flight.price * params.passengers,
            currency=flight.currency,
            seats_available=seats,
        )
        for flight, seats in found
        if not _has_departed(flight, params.departure_date)
    ]
    return FlightSearchResult(
        origin=params.origin.strip(),
        destination=params.destination.strip(),
        departure_date=params.departure_date,
        passengers=params.passengers,
        count=len(options),
        flights=options,
    )


def get_booking(conn: sqlite3.Connection, booking_id: str) -> BookingOut:
    booking = _require_booking(conn, booking_id)
    return _booking_out(
        booking, _require_flight(conn, booking.flight_id), repo.get_passengers(conn, booking_id)
    )


# ------------------------------------------------------------------------------ commands


def create_booking(
    conn: sqlite3.Connection,
    settings: FlightSettings,
    request: BookingCreate,
    idempotency_key: str | None = None,
) -> tuple[BookingOut, bool]:
    """Create a booking atomically. Returns (booking, replayed) where replayed means an
    earlier request with the same idempotency key was returned instead of booking again."""
    fingerprint = request_fingerprint(request.model_dump(mode="json"))
    seats_needed = len(request.passengers)
    flight_date = request.departure_date.isoformat()

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
                return (
                    _booking_out(
                        existing,
                        _require_flight(conn, existing.flight_id),
                        repo.get_passengers(conn, existing.booking_id),
                    ),
                    True,
                )

        _validate_date(request.departure_date, settings)
        flight = repo.get_flight(conn, request.flight_id)
        if flight is None:
            raise APIError(404, "FLIGHT_NOT_FOUND", f"Flight '{request.flight_id}' does not exist.")
        if _has_departed(flight, request.departure_date):
            raise APIError(
                422, "FLIGHT_DEPARTED", f"Flight {flight.flight_number} has already departed on that date."
            )

        free = repo.seats_available(conn, flight.flight_id, flight_date) or 0
        if free < seats_needed or not repo.decrement_seats(conn, flight.flight_id, flight_date, seats_needed):
            raise APIError(
                409,
                "SEATS_UNAVAILABLE",
                f"Only {free} seat(s) left on {flight.flight_number} for {flight_date}.",
                {"seats_requested": seats_needed, "seats_available": free},
            )

        passengers = [Passenger(name=p.name, age=p.age, gender=p.gender) for p in request.passengers]
        booking = Booking(
            booking_id=new_booking_id("FB"),
            flight_id=flight.flight_id,
            departure_date=flight_date,
            num_passengers=seats_needed,
            price_per_passenger=flight.price,
            total_price=flight.price * seats_needed,
            currency=flight.currency,
            contact_email=request.contact.email,
            contact_phone=request.contact.phone,
            status=STATUS_CONFIRMED,
            idempotency_key=idempotency_key or None,
            request_hash=fingerprint,
            created_at=_now_iso(),
            cancelled_at=None,
        )
        repo.insert_booking(conn, booking, passengers)
    return _booking_out(booking, flight, passengers), False


def cancel_booking(conn: sqlite3.Connection, booking_id: str) -> BookingOut:
    """Cancel a booking and give its seats back exactly once."""
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
        repo.increment_seats(conn, booking.flight_id, booking.departure_date, booking.num_passengers)
        flight = _require_flight(conn, booking.flight_id)
        passengers = repo.get_passengers(conn, booking_id)
    cancelled = replace(booking, status=STATUS_CANCELLED, cancelled_at=cancelled_at)
    return _booking_out(cancelled, flight, passengers)


# ------------------------------------------------------------------------------- helpers


def _require_booking(conn: sqlite3.Connection, booking_id: str) -> Booking:
    booking = repo.get_booking(conn, booking_id)
    if booking is None:
        raise APIError(404, "BOOKING_NOT_FOUND", f"Booking '{booking_id}' does not exist.")
    return booking


def _require_flight(conn: sqlite3.Connection, flight_id: str) -> Flight:
    flight = repo.get_flight(conn, flight_id)
    if flight is None:
        raise APIError(404, "FLIGHT_NOT_FOUND", f"Flight '{flight_id}' does not exist.")
    return flight
