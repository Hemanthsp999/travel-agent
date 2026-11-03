"""Identifier and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def new_booking_id(prefix: str) -> str:
    """Return an ID like `HB-3F9A1C07D2` (10 random hex chars)."""
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def request_fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of a request body, used to detect idempotency-key reuse with new data."""
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
