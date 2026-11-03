"""Shared helpers for flight API tests."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from mock_apis.common.db import connect
from mock_apis.flight_api.config import FlightSettings

TINY_FLIGHT = "FL-9001"
TINY_SEATS = 3


def day(offset: int) -> date:
    return date.today() + timedelta(days=offset)


def passenger(name: str = "Asha Rao", age: int = 31, gender: str = "FEMALE") -> dict:
    return {"name": name, "age": age, "gender": gender}


def booking_payload(
    flight_id: str = "FL-1001", offset: int = 30, passengers: int = 1, **overrides: Any
) -> dict:
    payload = {
        "flight_id": flight_id,
        "departure_date": day(offset).isoformat(),
        "passengers": [passenger(f"Traveller {i}", 20 + i) for i in range(passengers)],
        "contact": {"email": "asha@example.com", "phone": "+919876543210"},
    }
    payload.update(overrides)
    return payload


def seats_on(settings: FlightSettings, flight_id: str, offset: int = 30) -> int:
    """Seats free on a date, read straight from the database."""
    conn = connect(settings.db_path)
    try:
        row = conn.execute(
            "SELECT seats_available FROM flight_inventory WHERE flight_id = ? AND flight_date = ?",
            (flight_id, day(offset).isoformat()),
        ).fetchone()
    finally:
        conn.close()
    return row["seats_available"]
