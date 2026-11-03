"""Database models: table definitions and typed row objects for the hotel database."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS hotels (
    hotel_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    city                TEXT NOT NULL COLLATE NOCASE,
    locality            TEXT NOT NULL,
    address             TEXT NOT NULL,
    rating              REAL NOT NULL CHECK (rating BETWEEN 0 AND 5),
    star_category       INTEGER NOT NULL CHECK (star_category BETWEEN 1 AND 5),
    price_per_night     INTEGER NOT NULL CHECK (price_per_night > 0),
    currency            TEXT NOT NULL DEFAULT 'INR',
    total_rooms         INTEGER NOT NULL CHECK (total_rooms > 0),
    max_guests_per_room INTEGER NOT NULL CHECK (max_guests_per_room > 0),
    amenities           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hotels_city ON hotels (city);

CREATE TABLE IF NOT EXISTS room_inventory (
    hotel_id        TEXT NOT NULL REFERENCES hotels (hotel_id),
    stay_date       TEXT NOT NULL,
    rooms_available INTEGER NOT NULL CHECK (rooms_available >= 0),
    PRIMARY KEY (hotel_id, stay_date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bookings (
    booking_id      TEXT PRIMARY KEY,
    hotel_id        TEXT NOT NULL REFERENCES hotels (hotel_id),
    guest_name      TEXT NOT NULL,
    guest_email     TEXT NOT NULL,
    guest_phone     TEXT NOT NULL,
    check_in        TEXT NOT NULL,
    check_out       TEXT NOT NULL,
    nights          INTEGER NOT NULL,
    rooms           INTEGER NOT NULL,
    guests          INTEGER NOT NULL,
    price_per_night INTEGER NOT NULL,
    total_price     INTEGER NOT NULL,
    currency        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('CONFIRMED', 'CANCELLED')),
    idempotency_key TEXT UNIQUE,
    request_hash    TEXT,
    created_at      TEXT NOT NULL,
    cancelled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_bookings_hotel ON bookings (hotel_id);
"""

STATUS_CONFIRMED = "CONFIRMED"
STATUS_CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Hotel:
    hotel_id: str
    name: str
    city: str
    locality: str
    address: str
    rating: float
    star_category: int
    price_per_night: int
    currency: str
    total_rooms: int
    max_guests_per_room: int
    amenities: list[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Hotel:
        return cls(
            hotel_id=row["hotel_id"],
            name=row["name"],
            city=row["city"],
            locality=row["locality"],
            address=row["address"],
            rating=row["rating"],
            star_category=row["star_category"],
            price_per_night=row["price_per_night"],
            currency=row["currency"],
            total_rooms=row["total_rooms"],
            max_guests_per_room=row["max_guests_per_room"],
            amenities=json.loads(row["amenities"]),
        )


@dataclass(frozen=True)
class Booking:
    booking_id: str
    hotel_id: str
    guest_name: str
    guest_email: str
    guest_phone: str
    check_in: str
    check_out: str
    nights: int
    rooms: int
    guests: int
    price_per_night: int
    total_price: int
    currency: str
    status: str
    idempotency_key: str | None
    request_hash: str | None
    created_at: str
    cancelled_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Booking:
        return cls(**{name: row[name] for name in cls.__dataclass_fields__})
