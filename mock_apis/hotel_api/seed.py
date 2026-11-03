"""Seed data and database initialisation for the hotel API.

Run `python -m mock_apis.hotel_api.seed [--reset]` to create and seed the database by hand.
Seeding is idempotent: existing hotels and inventory are never overwritten, and room
inventory is topped up to cover today plus the configured horizon.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, timedelta

from mock_apis.common.db import connect, write_transaction
from mock_apis.hotel_api.config import HotelSettings, load_settings
from mock_apis.hotel_api.database import create_schema

# (hotel_id, name, city, locality, address, rating, stars, price/night INR, total rooms, guests/room, amenities)
HOTELS: list[tuple] = [
    ("HT-1001", "Lakeview Grand Bengaluru", "Bangalore", "MG Road", "12 MG Road, Bengaluru 560001", 4.5, 5, 9500, 40, 3, ["WiFi", "Pool", "Spa", "Gym", "Restaurant"]),
    ("HT-1002", "Indiranagar Urban Stay", "Bangalore", "Indiranagar", "88 100 Feet Road, Indiranagar, Bengaluru 560038", 4.1, 4, 5200, 30, 3, ["WiFi", "Gym", "Breakfast", "Bar"]),
    ("HT-1003", "Whitefield Business Inn", "Bangalore", "Whitefield", "5 ITPL Main Road, Whitefield, Bengaluru 560066", 3.9, 3, 3400, 25, 2, ["WiFi", "Breakfast", "Parking"]),
    ("HT-1004", "Marine Drive Residency", "Mumbai", "Churchgate", "21 Marine Drive, Mumbai 400020", 4.4, 5, 12500, 35, 3, ["WiFi", "Pool", "Sea View", "Spa", "Restaurant"]),
    ("HT-1005", "Bandra Bay Suites", "Mumbai", "Bandra West", "14 Linking Road, Bandra West, Mumbai 400050", 4.2, 4, 8200, 28, 3, ["WiFi", "Gym", "Rooftop Bar", "Breakfast"]),
    ("HT-1006", "Andheri Airport Comfort", "Mumbai", "Andheri East", "3 Sahar Road, Andheri East, Mumbai 400099", 3.8, 3, 3900, 45, 2, ["WiFi", "Airport Shuttle", "Breakfast"]),
    ("HT-1007", "Connaught Heritage Hotel", "Delhi", "Connaught Place", "7 Barakhamba Road, New Delhi 110001", 4.3, 5, 10800, 38, 3, ["WiFi", "Pool", "Spa", "Restaurant", "Bar"]),
    ("HT-1008", "Aerocity Express Inn", "Delhi", "Aerocity", "Plot 4, Aerocity, New Delhi 110037", 4.0, 4, 5600, 50, 2, ["WiFi", "Airport Shuttle", "Gym", "Breakfast"]),
    ("HT-1009", "Hitech City Towers", "Hyderabad", "HITEC City", "9 Madhapur Main Road, Hyderabad 500081", 4.4, 5, 7800, 42, 3, ["WiFi", "Pool", "Gym", "Restaurant", "Business Centre"]),
    ("HT-1010", "Charminar Palace Stay", "Hyderabad", "Old City", "2 Charminar Road, Hyderabad 500002", 4.0, 3, 3200, 22, 3, ["WiFi", "Breakfast", "Heritage Tours"]),
    ("HT-1011", "Marina Sands Hotel", "Chennai", "Marina Beach Road", "30 Kamarajar Salai, Chennai 600004", 4.2, 4, 6100, 30, 3, ["WiFi", "Pool", "Sea View", "Restaurant"]),
    ("HT-1012", "T Nagar Central Inn", "Chennai", "T Nagar", "17 Usman Road, T Nagar, Chennai 600017", 3.7, 3, 2900, 20, 2, ["WiFi", "Breakfast", "Parking"]),
    ("HT-1013", "Koregaon Park Boutique", "Pune", "Koregaon Park", "6 North Main Road, Koregaon Park, Pune 411001", 4.5, 4, 6800, 18, 2, ["WiFi", "Garden", "Cafe", "Spa"]),
    ("HT-1014", "Hinjewadi Tech Suites", "Pune", "Hinjewadi", "11 Phase 1, Hinjewadi, Pune 411057", 3.9, 3, 3600, 32, 3, ["WiFi", "Gym", "Breakfast", "Parking"]),
]


def seed_hotels(conn: sqlite3.Connection) -> None:
    rows = [
        (hid, name, city, locality, address, rating, stars, price, "INR", rooms, cap, json.dumps(amenities))
        for hid, name, city, locality, address, rating, stars, price, rooms, cap, amenities in HOTELS
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO hotels
           (hotel_id, name, city, locality, address, rating, star_category, price_per_night,
            currency, total_rooms, max_guests_per_room, amenities)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def ensure_inventory(conn: sqlite3.Connection, horizon_days: int, today: date | None = None) -> None:
    """Create missing per-date inventory rows from today through `today + horizon_days`."""
    start = today or date.today()
    days = [(start + timedelta(days=i)).isoformat() for i in range(horizon_days + 1)]
    hotels = conn.execute("SELECT hotel_id, total_rooms FROM hotels").fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO room_inventory (hotel_id, stay_date, rooms_available) VALUES (?, ?, ?)",
        [(h["hotel_id"], day, h["total_rooms"]) for h in hotels for day in days],
    )


def initialize_database(settings: HotelSettings, reset: bool = False) -> None:
    """Create the schema, seed dummy hotels and extend inventory. Safe to call on every start."""
    if reset and settings.db_path.exists():
        for suffix in ("", "-wal", "-shm"):
            settings.db_path.with_name(settings.db_path.name + suffix).unlink(missing_ok=True)
    conn = connect(settings.db_path)
    try:
        create_schema(conn)
        with write_transaction(conn):
            seed_hotels(conn)
            ensure_inventory(conn, settings.inventory_horizon_days)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise and seed the hotel mock database.")
    parser.add_argument("--db-path", help="SQLite file (default: data/hotel.db or $HOTEL_DB_PATH)")
    parser.add_argument("--reset", action="store_true", help="Delete the database before seeding")
    args = parser.parse_args()
    settings = load_settings(args.db_path)
    initialize_database(settings, reset=args.reset)
    print(f"Hotel database ready at {settings.db_path} ({len(HOTELS)} hotels)")


if __name__ == "__main__":
    main()
