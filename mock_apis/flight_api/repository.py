"""SQL statements for the flight database. No business rules live here."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict

from mock_apis.flight_api.models import Booking, Flight, Passenger


def get_flight(conn: sqlite3.Connection, flight_id: str) -> Flight | None:
    row = conn.execute("SELECT * FROM flights WHERE flight_id = ?", (flight_id,)).fetchone()
    return Flight.from_row(row) if row else None


def search_flights(
    conn: sqlite3.Connection, *, origin: str, destination: str, flight_date: str, passengers: int
) -> list[tuple[Flight, int]]:
    """Flights on the route with at least `passengers` seats free on `flight_date`, earliest first."""
    rows = conn.execute(
        """
        SELECT f.*, i.seats_available
        FROM flights f
        JOIN flight_inventory i ON i.flight_id = f.flight_id AND i.flight_date = :date
        WHERE (f.origin_city = :origin OR f.origin_code = :origin)
          AND (f.destination_city = :destination OR f.destination_code = :destination)
          AND i.seats_available >= :passengers
        ORDER BY f.departure_time, f.price
        """,
        {"origin": origin, "destination": destination, "date": flight_date, "passengers": passengers},
    ).fetchall()
    return [(Flight.from_row(r), r["seats_available"]) for r in rows]


def seats_available(conn: sqlite3.Connection, flight_id: str, flight_date: str) -> int | None:
    """Seats free on a flight date, or None if no inventory exists for it."""
    row = conn.execute(
        "SELECT seats_available FROM flight_inventory WHERE flight_id = ? AND flight_date = ?",
        (flight_id, flight_date),
    ).fetchone()
    return row["seats_available"] if row else None


def decrement_seats(conn: sqlite3.Connection, flight_id: str, flight_date: str, seats: int) -> bool:
    """Take seats only if enough remain; returns False when the guard rejected the update."""
    cur = conn.execute(
        """UPDATE flight_inventory SET seats_available = seats_available - :seats
           WHERE flight_id = :f AND flight_date = :d AND seats_available >= :seats""",
        {"seats": seats, "f": flight_id, "d": flight_date},
    )
    return cur.rowcount == 1


def increment_seats(conn: sqlite3.Connection, flight_id: str, flight_date: str, seats: int) -> None:
    """Give seats back, never exceeding the aircraft's total seat count."""
    conn.execute(
        """UPDATE flight_inventory
           SET seats_available = MIN(seats_available + :seats,
                                     (SELECT total_seats FROM flights WHERE flight_id = :f))
           WHERE flight_id = :f AND flight_date = :d""",
        {"seats": seats, "f": flight_id, "d": flight_date},
    )


def insert_booking(conn: sqlite3.Connection, booking: Booking, passengers: list[Passenger]) -> None:
    data = asdict(booking)
    columns = ", ".join(data)
    placeholders = ", ".join(f":{name}" for name in data)
    conn.execute(f"INSERT INTO bookings ({columns}) VALUES ({placeholders})", data)
    conn.executemany(
        "INSERT INTO passengers (booking_id, name, age, gender) VALUES (?, ?, ?, ?)",
        [(booking.booking_id, p.name, p.age, p.gender) for p in passengers],
    )


def get_booking(conn: sqlite3.Connection, booking_id: str) -> Booking | None:
    row = conn.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
    return Booking.from_row(row) if row else None


def get_booking_by_key(conn: sqlite3.Connection, idempotency_key: str) -> Booking | None:
    row = conn.execute(
        "SELECT * FROM bookings WHERE idempotency_key = ?", (idempotency_key,)
    ).fetchone()
    return Booking.from_row(row) if row else None


def get_passengers(conn: sqlite3.Connection, booking_id: str) -> list[Passenger]:
    rows = conn.execute(
        "SELECT name, age, gender FROM passengers WHERE booking_id = ? ORDER BY passenger_id",
        (booking_id,),
    ).fetchall()
    return [Passenger.from_row(r) for r in rows]


def mark_cancelled(conn: sqlite3.Connection, booking_id: str, cancelled_at: str) -> bool:
    """Flip CONFIRMED -> CANCELLED. Returns False if the booking was not CONFIRMED."""
    cur = conn.execute(
        """UPDATE bookings SET status = 'CANCELLED', cancelled_at = ?
           WHERE booking_id = ? AND status = 'CONFIRMED'""",
        (cancelled_at, booking_id),
    )
    return cur.rowcount == 1
