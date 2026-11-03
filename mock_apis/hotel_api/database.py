"""Database access helpers: connection dependency and schema creation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Request

from mock_apis.common.db import connect
from mock_apis.hotel_api.config import HotelSettings
from mock_apis.hotel_api.models import SCHEMA


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def get_settings(request: Request) -> HotelSettings:
    return request.app.state.settings


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one short-lived connection per request."""
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()
