"""Flight API settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "flight.db"

SERVICE_NAME = "flight-api"
DEFAULT_PORT = 8002

MAX_PASSENGERS_PER_BOOKING = 9


@dataclass(frozen=True)
class FlightSettings:
    db_path: Path
    inventory_horizon_days: int = 365


def load_settings(db_path: str | Path | None = None, horizon_days: int | None = None) -> FlightSettings:
    """Build settings from arguments, then `FLIGHT_DB_PATH` / `FLIGHT_HORIZON_DAYS`, then defaults."""
    path = Path(db_path or os.getenv("FLIGHT_DB_PATH") or DEFAULT_DB_PATH)
    horizon = horizon_days or int(os.getenv("FLIGHT_HORIZON_DAYS", "365"))
    return FlightSettings(db_path=path, inventory_horizon_days=horizon)
