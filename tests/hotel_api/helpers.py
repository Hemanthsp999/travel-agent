"""Shared helpers for hotel API tests."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from mock_apis.common.db import connect
from mock_apis.hotel_api.config import HotelSettings

TINY_HOTEL = "HT-9001"
TINY_ROOMS = 2


def day(offset: int) -> date:
    return date.today() + timedelta(days=offset)


def booking_payload(hotel_id: str = "HT-1001", start: int = 30, nights: int = 2, **overrides: Any) -> dict:
    payload = {
        "hotel_id": hotel_id,
        "check_in": day(start).isoformat(),
        "check_out": day(start + nights).isoformat(),
        "rooms": 1,
        "guests": 2,
        "guest": {"name": "Asha Rao", "email": "asha@example.com", "phone": "+919876543210"},
    }
    payload.update(overrides)
    return payload


def rooms_by_date(settings: HotelSettings, hotel_id: str, start: int, nights: int) -> list[int]:
    """Rooms free per night, read straight from the database."""
    conn = connect(settings.db_path)
    try:
        rows = conn.execute(
            """SELECT rooms_available FROM room_inventory
               WHERE hotel_id = ? AND stay_date >= ? AND stay_date < ? ORDER BY stay_date""",
            (hotel_id, day(start).isoformat(), day(start + nights).isoformat()),
        ).fetchall()
    finally:
        conn.close()
    return [r["rooms_available"] for r in rows]
