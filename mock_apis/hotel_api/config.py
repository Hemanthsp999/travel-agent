"""Hotel API settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "hotel.db"

SERVICE_NAME = "hotel-api"
DEFAULT_PORT = 8001

MAX_STAY_NIGHTS = 30
MAX_ROOMS_PER_BOOKING = 5
MAX_GUESTS_PER_BOOKING = 20


@dataclass(frozen=True)
class HotelSettings:
    db_path: Path
    inventory_horizon_days: int = 365


def load_settings(db_path: str | Path | None = None, horizon_days: int | None = None) -> HotelSettings:
    """Build settings from arguments, then `HOTEL_DB_PATH` / `HOTEL_HORIZON_DAYS`, then defaults."""
    path = Path(db_path or os.getenv("HOTEL_DB_PATH") or DEFAULT_DB_PATH)
    horizon = horizon_days or int(os.getenv("HOTEL_HORIZON_DAYS", "365"))
    return HotelSettings(db_path=path, inventory_horizon_days=horizon)
