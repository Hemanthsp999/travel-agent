"""Checkpointer factory: SQLite when available (survives restarts), otherwise in-memory."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


def create_checkpointer(db_path: Path | None = None) -> BaseCheckpointSaver:
    """Return a SQLite checkpointer for `db_path`, or an in-memory one if `db_path` is None."""
    if db_path is None:
        return InMemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite missing; using in-memory checkpoints")
        return InMemorySaver()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
