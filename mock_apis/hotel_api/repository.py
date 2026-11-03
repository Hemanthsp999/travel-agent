"""SQL statements for the hotel database. No business rules live here."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict

from mock_apis.hotel_api.models import Booking, Hotel


def get_hotel(conn: sqlite3.Connection, hotel_id: str) -> Hotel | None:
    row = conn.execute("SELECT * FROM hotels WHERE hotel_id = ?", (hotel_id,)).fetchone()
    return Hotel.from_row(row) if row else None


def search_available(
    conn: sqlite3.Connection,
    *,
    city: str,
    check_in: str,
    check_out: str,
    nights: int,
    rooms: int,
    guests: int,
    min_rating: float | None,
    max_price: int | None,
) -> list[tuple[Hotel, int]]:
    """Hotels with `rooms` free on every night in [check_in, check_out), with the tightest night's stock."""
    rows = conn.execute(
        """
        SELECT h.*, MIN(i.rooms_available) AS rooms_left, COUNT(*) AS days
        FROM hotels h
        JOIN room_inventory i
          ON i.hotel_id = h.hotel_id AND i.stay_date >= :ci AND i.stay_date < :co
        WHERE h.city = :city
          AND h.max_guests_per_room * :rooms >= :guests
          AND (:min_rating IS NULL OR h.rating >= :min_rating)
          AND (:max_price IS NULL OR h.price_per_night <= :max_price)
        GROUP BY h.hotel_id
        HAVING days = :nights AND rooms_left >= :rooms
        ORDER BY h.price_per_night, h.rating DESC
        """,
        {
            "ci": check_in,
            "co": check_out,
            "city": city,
            "rooms": rooms,
            "guests": guests,
            "min_rating": min_rating,
            "max_price": max_price,
            "nights": nights,
        },
    ).fetchall()
    return [(Hotel.from_row(r), r["rooms_left"]) for r in rows]


def stay_availability(
    conn: sqlite3.Connection, hotel_id: str, check_in: str, check_out: str
) -> tuple[int, int]:
    """Return (number of inventory days found, minimum rooms free across them)."""
    row = conn.execute(
        """SELECT COUNT(*) AS days, COALESCE(MIN(rooms_available), 0) AS rooms_left
           FROM room_inventory WHERE hotel_id = ? AND stay_date >= ? AND stay_date < ?""",
        (hotel_id, check_in, check_out),
    ).fetchone()
    return row["days"], row["rooms_left"]


def decrement_inventory(
    conn: sqlite3.Connection, hotel_id: str, check_in: str, check_out: str, rooms: int
) -> int:
    """Take `rooms` from each night; the guard means a night without stock is simply not updated."""
    cur = conn.execute(
        """UPDATE room_inventory SET rooms_available = rooms_available - :rooms
           WHERE hotel_id = :h AND stay_date >= :ci AND stay_date < :co AND rooms_available >= :rooms""",
        {"rooms": rooms, "h": hotel_id, "ci": check_in, "co": check_out},
    )
    return cur.rowcount


def increment_inventory(
    conn: sqlite3.Connection, hotel_id: str, check_in: str, check_out: str, rooms: int
) -> None:
    """Give rooms back, never exceeding the hotel's total room count."""
    conn.execute(
        """UPDATE room_inventory
           SET rooms_available = MIN(rooms_available + :rooms,
                                     (SELECT total_rooms FROM hotels WHERE hotel_id = :h))
           WHERE hotel_id = :h AND stay_date >= :ci AND stay_date < :co""",
        {"rooms": rooms, "h": hotel_id, "ci": check_in, "co": check_out},
    )


def insert_booking(conn: sqlite3.Connection, booking: Booking) -> None:
    data = asdict(booking)
    columns = ", ".join(data)
    placeholders = ", ".join(f":{name}" for name in data)
    conn.execute(f"INSERT INTO bookings ({columns}) VALUES ({placeholders})", data)


def get_booking(conn: sqlite3.Connection, booking_id: str) -> Booking | None:
    row = conn.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
    return Booking.from_row(row) if row else None


def get_booking_by_key(conn: sqlite3.Connection, idempotency_key: str) -> Booking | None:
    row = conn.execute(
        "SELECT * FROM bookings WHERE idempotency_key = ?", (idempotency_key,)
    ).fetchone()
    return Booking.from_row(row) if row else None


def mark_cancelled(conn: sqlite3.Connection, booking_id: str, cancelled_at: str) -> bool:
    """Flip CONFIRMED -> CANCELLED. Returns False if the booking was not CONFIRMED."""
    cur = conn.execute(
        """UPDATE bookings SET status = 'CANCELLED', cancelled_at = ?
           WHERE booking_id = ? AND status = 'CONFIRMED'""",
        (cancelled_at, booking_id),
    )
    return cur.rowcount == 1
