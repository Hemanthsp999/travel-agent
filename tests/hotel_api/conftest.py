"""Fixtures for hotel API tests. Every test gets its own SQLite file."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_apis.common.db import connect, write_transaction
from mock_apis.hotel_api.app import create_app
from mock_apis.hotel_api.config import HotelSettings, load_settings
from mock_apis.hotel_api.seed import ensure_inventory
from tests.hotel_api.helpers import TINY_HOTEL, TINY_ROOMS


@pytest.fixture
def settings(tmp_path: Path) -> HotelSettings:
    return load_settings(tmp_path / "hotel.db", horizon_days=120)


@pytest.fixture
def client(settings: HotelSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


@pytest.fixture
def tiny_hotel(client: TestClient, settings: HotelSettings) -> str:
    """A 2-room hotel in 'Testville' so tests can sell it out quickly."""
    conn = connect(settings.db_path)
    try:
        with write_transaction(conn):
            conn.execute(
                """INSERT INTO hotels VALUES (?, 'Tiny Test Inn', 'Testville', 'Centre', '1 Test Street',
                   4.0, 3, 1000, 'INR', ?, 2, ?)""",
                (TINY_HOTEL, TINY_ROOMS, json.dumps(["WiFi"])),
            )
            ensure_inventory(conn, settings.inventory_horizon_days)
    finally:
        conn.close()
    return TINY_HOTEL
