"""SQLite helpers: connection factory and an explicit write-transaction context manager."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BUSY_TIMEOUT_MS = 10_000


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection in manual-transaction mode with WAL and foreign keys enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each connection serves a single request at a time; FastAPI may hop threads between
    # a dependency and its endpoint, so the same-thread check is disabled.
    conn = sqlite3.connect(
        path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside BEGIN IMMEDIATE so concurrent writers are serialised.

    The write lock is taken before any availability check, which prevents two
    requests from reading the same stock and both deciding to book it.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
