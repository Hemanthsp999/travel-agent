"""Seed data and database initialisation for the flight API.

Run `python -m mock_apis.flight_api.seed [--reset]` to create and seed the database by hand.
Seeding is idempotent: existing flights and seat inventory are never overwritten, and
inventory is topped up to cover today plus the configured horizon.
Flights are daily schedules; seats are tracked separately for every date.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, timedelta

from mock_apis.common.db import connect, write_transaction
from mock_apis.flight_api.config import FlightSettings, load_settings
from mock_apis.flight_api.database import create_schema

CITY_CODES = {"Bangalore": "BLR", "Mumbai": "BOM", "Delhi": "DEL", "Hyderabad": "HYD", "Chennai": "MAA"}

# (flight_id, airline, code, number, origin, destination, departs, duration min, fare INR, seats, aircraft)
FLIGHTS: list[tuple] = [
    ("FL-1001", "IndiGo", "6E", "6E-501", "Bangalore", "Mumbai", "06:15", 105, 4800, 180, "Airbus A320neo"),
    ("FL-1002", "Air India", "AI", "AI-502", "Bangalore", "Mumbai", "11:30", 110, 6200, 162, "Airbus A320"),
    ("FL-1003", "Akasa Air", "QP", "QP-1303", "Bangalore", "Mumbai", "19:45", 105, 5300, 189, "Boeing 737 MAX"),
    ("FL-1004", "IndiGo", "6E", "6E-2102", "Bangalore", "Delhi", "07:00", 170, 7200, 186, "Airbus A321neo"),
    ("FL-1005", "Air India", "AI", "AI-805", "Bangalore", "Delhi", "13:20", 175, 8900, 162, "Airbus A320neo"),
    ("FL-1006", "SpiceJet", "SG", "SG-8161", "Bangalore", "Delhi", "21:10", 175, 6400, 189, "Boeing 737-800"),
    ("FL-1007", "IndiGo", "6E", "6E-6112", "Delhi", "Mumbai", "08:30", 130, 5900, 186, "Airbus A321neo"),
    ("FL-1008", "Air India", "AI", "AI-687", "Delhi", "Mumbai", "15:45", 125, 7500, 162, "Airbus A320"),
    ("FL-1009", "Akasa Air", "QP", "QP-1411", "Delhi", "Mumbai", "20:30", 130, 5100, 189, "Boeing 737 MAX"),
    ("FL-1010", "IndiGo", "6E", "6E-5203", "Mumbai", "Chennai", "09:10", 120, 5400, 186, "Airbus A321neo"),
    ("FL-1011", "SpiceJet", "SG", "SG-3012", "Mumbai", "Chennai", "17:25", 125, 4700, 189, "Boeing 737-800"),
    ("FL-1012", "IndiGo", "6E", "6E-7156", "Hyderabad", "Bangalore", "06:45", 80, 3600, 186, "Airbus A320neo"),
    ("FL-1013", "Air India", "AI", "AI-544", "Hyderabad", "Bangalore", "18:00", 85, 4900, 162, "Airbus A320"),
    ("FL-1014", "IndiGo", "6E", "6E-6807", "Chennai", "Delhi", "05:50", 170, 7000, 186, "Airbus A321neo"),
    ("FL-1015", "Air India", "AI", "AI-439", "Chennai", "Delhi", "16:15", 175, 8200, 162, "Airbus A320neo"),
    ("FL-1016", "IndiGo", "6E", "6E-502", "Mumbai", "Bangalore", "14:00", 105, 4900, 180, "Airbus A320neo"),
    ("FL-1017", "Akasa Air", "QP", "QP-1202", "Delhi", "Bangalore", "10:15", 170, 6800, 189, "Boeing 737 MAX"),
    ("FL-1018", "Air India", "AI", "AI-688", "Mumbai", "Delhi", "22:05", 130, 6900, 162, "Airbus A320"),
]


def seed_flights(conn: sqlite3.Connection) -> None:
    rows = [
        (fid, airline, code, number, origin, CITY_CODES[origin], dest, CITY_CODES[dest], departs, mins, fare, "INR", seats, aircraft)
        for fid, airline, code, number, origin, dest, departs, mins, fare, seats, aircraft in FLIGHTS
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO flights
           (flight_id, airline, airline_code, flight_number, origin_city, origin_code,
            destination_city, destination_code, departure_time, duration_minutes, price,
            currency, total_seats, aircraft)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def ensure_inventory(conn: sqlite3.Connection, horizon_days: int, today: date | None = None) -> None:
    """Create missing per-date seat rows from today through `today + horizon_days`."""
    start = today or date.today()
    days = [(start + timedelta(days=i)).isoformat() for i in range(horizon_days + 1)]
    flights = conn.execute("SELECT flight_id, total_seats FROM flights").fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO flight_inventory (flight_id, flight_date, seats_available) VALUES (?, ?, ?)",
        [(f["flight_id"], day, f["total_seats"]) for f in flights for day in days],
    )


def initialize_database(settings: FlightSettings, reset: bool = False) -> None:
    """Create the schema, seed dummy flights and extend inventory. Safe to call on every start."""
    if reset and settings.db_path.exists():
        for suffix in ("", "-wal", "-shm"):
            settings.db_path.with_name(settings.db_path.name + suffix).unlink(missing_ok=True)
    conn = connect(settings.db_path)
    try:
        create_schema(conn)
        with write_transaction(conn):
            seed_flights(conn)
            ensure_inventory(conn, settings.inventory_horizon_days)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise and seed the flight mock database.")
    parser.add_argument("--db-path", help="SQLite file (default: data/flight.db or $FLIGHT_DB_PATH)")
    parser.add_argument("--reset", action="store_true", help="Delete the database before seeding")
    args = parser.parse_args()
    settings = load_settings(args.db_path)
    initialize_database(settings, reset=args.reset)
    print(f"Flight database ready at {settings.db_path} ({len(FLIGHTS)} flights)")


if __name__ == "__main__":
    main()
