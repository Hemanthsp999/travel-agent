"""Fixtures for flight API tests. Every test gets its own SQLite file."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_apis.common.db import connect, write_transaction
from mock_apis.flight_api.app import create_app
from mock_apis.flight_api.config import FlightSettings, load_settings
from mock_apis.flight_api.seed import ensure_inventory
from tests.flight_api.helpers import TINY_FLIGHT, TINY_SEATS


@pytest.fixture
def settings(tmp_path: Path) -> FlightSettings:
    return load_settings(tmp_path / "flight.db", horizon_days=120)


@pytest.fixture
def client(settings: FlightSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


@pytest.fixture
def tiny_flight(client: TestClient, settings: FlightSettings) -> str:
    """A 3-seat flight Testville -> Nowhereburg so tests can sell it out quickly."""
    conn = connect(settings.db_path)
    try:
        with write_transaction(conn):
            conn.execute(
                """INSERT INTO flights VALUES (?, 'Test Air', 'TA', 'TA-1', 'Testville', 'TST',
                   'Nowhereburg', 'NWB', '23:30', 60, 2000, 'INR', ?, 'Cessna')""",
                (TINY_FLIGHT, TINY_SEATS),
            )
            ensure_inventory(conn, settings.inventory_horizon_days)
    finally:
        conn.close()
    return TINY_FLIGHT
