"""Parameter handling: merging extracted values, required-field rules and question ordering."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from travel_agent.hotel_agent.parsing import Extracted
from travel_agent.hotel_agent.schemas import Intent

# Selectors and filters belong to one request; stay-defining values carry over between requests.
TASK_SCOPED = ("hotel_id", "hotel_name", "booking_id", "max_price_per_night", "min_rating")
DIRECT_KEYS = (
    "booking_id", "hotel_id", "guest_email", "guest_phone", "guest_name",
    "guests", "rooms", "max_price_per_night", "min_rating",
)
GUEST_FIELDS = ("guest_name", "guest_email", "guest_phone")
# Order in which missing values are asked.
ASK_ORDER = ("city", "check_in", "check_out", "hotel", "guests", "rooms", *GUEST_FIELDS, "booking_id")

_CHECKOUT_WORDS = re.compile(r"\b(?:check-?out|leave|leaving|depart|departing|till|until)\b")
_CHECKIN_WORDS = re.compile(r"\b(?:check-?in|arrive|arriving|arrival)\b")


def merge_extracted(params: dict[str, Any], found: Extracted, text: str, awaiting: str | None) -> dict[str, Any]:
    """Return `params` updated with rule-extracted values (newest message wins)."""
    merged = dict(params)
    for key in DIRECT_KEYS:
        if found.params.get(key) is not None:
            merged[key] = found.params[key]
    _assign_dates(merged, found, text.lower(), awaiting)
    return merged


def _assign_dates(params: dict[str, Any], found: Extracted, lowered: str, awaiting: str | None) -> None:
    dates = found.dates
    slot: str | None = None
    if len(dates) >= 2:
        params["check_in"], params["check_out"] = dates[0].isoformat(), dates[1].isoformat()
    elif len(dates) == 1:
        has_in, has_out = bool(params.get("check_in")), bool(params.get("check_out"))
        says_out, says_in = bool(_CHECKOUT_WORDS.search(lowered)), bool(_CHECKIN_WORDS.search(lowered))
        if awaiting == "check_out" or (says_out and not says_in):
            slot = "check_out"
        elif awaiting == "check_in" or says_in:
            slot = "check_in"
        elif has_in and not has_out:
            slot = "check_out"
        else:
            slot = "check_in"
        params[slot] = dates[0].isoformat()
        if slot == "check_in" and has_out:
            keep = (awaiting == "check_in" or says_in) and date.fromisoformat(params["check_out"]) > dates[0]
            if not keep:  # old check-out no longer fits; ask again rather than guess
                params.pop("check_out")
    if found.nights and params.get("check_in") and slot != "check_out" and len(dates) < 2:
        start = date.fromisoformat(params["check_in"])
        params["check_out"] = (start + timedelta(days=found.nights)).isoformat()


def missing_for(intent: Intent, params: dict[str, Any]) -> list[str]:
    """Required values still absent for `intent`, in the order they should be asked."""
    need: set[str] = set()
    if intent is Intent.CHECK_AVAILABILITY:
        need = {"city", "check_in", "check_out"}
    elif intent is Intent.BOOK_HOTEL:
        need = {"check_in", "check_out", "guests", "rooms", *GUEST_FIELDS}
        if not params.get("hotel_id") and not params.get("hotel_name"):
            need.add("hotel")
        if not params.get("hotel_id") and not params.get("city"):
            need.add("city")
    elif intent in (Intent.CANCEL_BOOKING, Intent.GET_BOOKING_STATUS):
        need = {"booking_id"}
    present = dict(params)
    present["hotel"] = params.get("hotel_id") or params.get("hotel_name")
    return [key for key in ASK_ORDER if key in need and not present.get(key)]


def ask_group(missing: list[str]) -> list[str]:
    """Which missing values to ask in one question (one natural question at a time)."""
    if not missing:
        return []
    first = missing[0]
    for group in (("check_in", "check_out"), ("guests", "rooms"), GUEST_FIELDS):
        if first in group:
            return [k for k in missing if k in group]
    return [first]
