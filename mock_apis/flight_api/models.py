"""Database models: table definitions and typed row objects for the flight database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    flight_id        TEXT PRIMARY KEY,
    airline          TEXT NOT NULL,
    airline_code     TEXT NOT NULL,
    flight_number    TEXT NOT NULL,
    origin_city      TEXT NOT NULL COLLATE NOCASE,
    origin_code      TEXT NOT NULL COLLATE NOCASE,
    destination_city TEXT NOT NULL COLLATE NOCASE,
    destination_code TEXT NOT NULL COLLATE NOCASE,
    departure_time   TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL CHECK (duration_minutes > 0),
    price            INTEGER NOT NULL CHECK (price > 0),
    currency         TEXT NOT NULL DEFAULT 'INR',
    total_seats      INTEGER NOT NULL CHECK (total_seats > 0),
    aircraft         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flights_route ON flights (origin_city, destination_city);

CREATE TABLE IF NOT EXISTS flight_inventory (
    flight_id       TEXT NOT NULL REFERENCES flights (flight_id),
    flight_date     TEXT NOT NULL,
    seats_available INTEGER NOT NULL CHECK (seats_available >= 0),
    PRIMARY KEY (flight_id, flight_date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bookings (
    booking_id           TEXT PRIMARY KEY,
    flight_id            TEXT NOT NULL REFERENCES flights (flight_id),
    departure_date       TEXT NOT NULL,
    num_passengers       INTEGER NOT NULL,
    price_per_passenger  INTEGER NOT NULL,
    total_price          INTEGER NOT NULL,
    currency             TEXT NOT NULL,
    contact_email        TEXT NOT NULL,
    contact_phone        TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('CONFIRMED', 'CANCELLED')),
    idempotency_key      TEXT UNIQUE,
    request_hash         TEXT,
    created_at           TEXT NOT NULL,
    cancelled_at         TEXT
);

CREATE TABLE IF NOT EXISTS passengers (
    passenger_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id   TEXT NOT NULL REFERENCES bookings (booking_id),
    name         TEXT NOT NULL,
    age          INTEGER NOT NULL,
    gender       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passengers_booking ON passengers (booking_id);
"""

STATUS_CONFIRMED = "CONFIRMED"
STATUS_CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Flight:
    flight_id: str
    airline: str
    airline_code: str
    flight_number: str
    origin_city: str
    origin_code: str
    destination_city: str
    destination_code: str
    departure_time: str
    duration_minutes: int
    price: int
    currency: str
    total_seats: int
    aircraft: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Flight:
        return cls(**{name: row[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Booking:
    booking_id: str
    flight_id: str
    departure_date: str
    num_passengers: int
    price_per_passenger: int
    total_price: int
    currency: str
    contact_email: str
    contact_phone: str
    status: str
    idempotency_key: str | None
    request_hash: str | None
    created_at: str
    cancelled_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Booking:
        return cls(**{name: row[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Passenger:
    name: str
    age: int
    gender: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Passenger:
        return cls(name=row["name"], age=row["age"], gender=row["gender"])
