"""Turn API / tool failures into a recovery plan: either ask the user to fix a value, or explain."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FIELD_LABELS = {
    "origin": "departure city", "destination": "destination city", "departure_date": "travel date",
    "passenger_count": "number of passengers", "selected_flight": "flight", "booking_id": "booking ID",
    "contact_email": "email address", "contact_phone": "phone number",
}
ACTION_WORDS = {
    "search": "check availability", "precheck": "check availability", "book": "complete the booking",
    "lookup": "look up the booking", "cancel": "cancel the booking", "verify": "verify the cancellation",
}
UNREACHABLE = {"CONNECTION_ERROR", "TIMEOUT", "SERVER_ERROR"}
_PASSENGER_FIELD = re.compile(r"^passengers\.(\d+)\.(name|age|gender)$")
_DIRECT = {
    "origin": "origin", "destination": "destination", "departure_date": "departure_date",
    "passengers": "passenger_count", "flight_id": "selected_flight", "booking_id": "booking_id",
    "contact.email": "contact_email", "contact.phone": "contact_phone",
}


@dataclass
class ErrorPlan:
    """What to do about a failed tool call."""

    kind: str = "error"  # outcome kind when we simply explain
    message: str = ""
    recover: bool = False  # True: clear `clear` values and ask again
    clear: list[str] = field(default_factory=list)
    note: str | None = None
    code: str = ""


def clear_keys_from_details(details: Any) -> list[str]:
    """Values the API rejected, as clearable keys (passenger fields look like `passenger_2_age`)."""
    found: list[str] = []
    for item in details or []:
        raw = str(item.get("field") if isinstance(item, dict) else item)
        if match := _PASSENGER_FIELD.match(raw):
            key = f"passenger_{int(match.group(1)) + 1}_{match.group(2)}"
        else:
            key = _DIRECT.get(raw, "")
        if key and key not in found:
            found.append(key)
    return found


def _label(key: str) -> str:
    if key in FIELD_LABELS:
        return FIELD_LABELS[key]
    if match := re.fullmatch(r"passenger_(\d+)_(name|age|gender)", key):
        return f"{match.group(2)} of passenger {match.group(1)}"
    return key


def plan_for_error(stage: str, error: dict[str, Any]) -> ErrorPlan:
    code, api_message = error.get("code", ""), error.get("message", "")
    action = ACTION_WORDS.get(stage, "complete that")
    plan = ErrorPlan(code=code)

    if code in UNREACHABLE:
        plan.message = (
            f"I'm having trouble reaching the flight system, so I couldn't {action}. "
            "Nothing was changed. Please try again in a moment."
        )
    elif code == "INVALID_DATE":
        plan.recover, plan.clear = True, ["departure_date"]
        plan.note = f"That date doesn't work: {api_message}"
    elif code in {"VALIDATION_ERROR", "INVALID_ARGUMENTS"}:
        keys = clear_keys_from_details(error.get("details"))
        if keys:
            plan.recover, plan.clear = True, keys
            plan.note = f"I need a valid {', '.join(_label(k) for k in keys)} (the flight system rejected the one I had)."
        else:
            plan.message = f"The flight system rejected the request: {api_message}"
    elif code == "FLIGHT_NOT_FOUND":
        plan.recover, plan.clear = True, ["selected_flight"]
        plan.note = "I couldn't find that flight in the system."
    elif code == "FLIGHT_DEPARTED":
        plan.recover, plan.clear = True, ["selected_flight"]
        plan.note = "That flight has already departed."
    elif code == "BOOKING_NOT_FOUND":
        plan.recover, plan.clear = True, ["booking_id"]
        plan.note = "I couldn't find a booking with that ID."
    elif code == "SEATS_UNAVAILABLE":
        plan.kind = "booking_failed"
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        left = details.get("seats_available")
        extra = f" Only {left} seat(s) are left." if left is not None else ""
        plan.message = f"That flight doesn't have enough seats any more.{extra} Nothing was booked. Want me to check other flights?"
    elif code == "BOOKING_ALREADY_CANCELLED":
        plan.kind = "already_cancelled"
    elif code == "IDEMPOTENCY_KEY_CONFLICT":
        plan.kind = "booking_failed"
        plan.message = "A technical conflict stopped this booking. Nothing was booked, so please ask me to book again."
    elif code == "INVALID_RESPONSE":
        plan.message = (
            f"The flight system sent back something I couldn't understand, so I couldn't {action}. "
            "Please try again."
        )
    else:
        plan.message = f"I couldn't {action}: {api_message or 'the flight system reported a problem'}."
    if stage == "book" and plan.kind == "error":
        plan.kind = "booking_failed"
    if stage == "cancel" and plan.kind == "error":
        plan.kind = "cancel_failed"
    return plan
