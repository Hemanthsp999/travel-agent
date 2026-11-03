"""Turn API / tool failures into a recovery plan: either ask the user to fix a value, or explain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FIELD_TO_PARAM = {
    "city": "city", "check_in": "check_in", "check_out": "check_out", "guests": "guests", "rooms": "rooms",
    "max_price_per_night": "max_price_per_night", "min_rating": "min_rating", "hotel_id": "hotel_id",
    "guest.name": "guest_name", "guest.email": "guest_email", "guest.phone": "guest_phone",
    "guest_name": "guest_name", "guest_email": "guest_email", "guest_phone": "guest_phone",
    "booking_id": "booking_id", "request": "",
}
FIELD_LABELS = {
    "city": "city", "check_in": "check-in date", "check_out": "check-out date", "guests": "guest count",
    "rooms": "room count", "guest_name": "guest name", "guest_email": "email address",
    "guest_phone": "phone number", "hotel_id": "hotel", "booking_id": "booking ID",
    "max_price_per_night": "budget", "min_rating": "rating filter",
}
ACTION_WORDS = {
    "search": "check availability", "precheck": "check availability", "hotel_lookup": "look up the hotel",
    "details": "look up the hotel", "book": "complete the booking", "booking_lookup": "look up the booking",
    "status": "look up the booking", "cancel": "cancel the booking", "cancel_verify": "verify the cancellation",
}
UNREACHABLE = {"CONNECTION_ERROR", "TIMEOUT", "SERVER_ERROR"}


@dataclass
class ErrorPlan:
    """What to do about a failed tool call."""

    kind: str = "error"  # outcome kind when we simply explain
    message: str = ""
    recover: bool = False  # True: clear `clear` params and ask again
    clear: list[str] = field(default_factory=list)
    note: str | None = None
    code: str = ""


def _fields_from_details(details: Any) -> list[str]:
    """Parameter names named in an API validation error (`details`) or a client-side field list."""
    found: list[str] = []
    for item in details or []:
        raw = item.get("field") if isinstance(item, dict) else item
        param = FIELD_TO_PARAM.get(str(raw), "")
        if param and param not in found:
            found.append(param)
    return found


def plan_for_error(stage: str, error: dict[str, Any]) -> ErrorPlan:
    code, api_message = error.get("code", ""), error.get("message", "")
    action = ACTION_WORDS.get(stage, "complete that")
    plan = ErrorPlan(code=code)

    if code in UNREACHABLE:
        plan.message = (
            f"I'm having trouble reaching the hotel system, so I couldn't {action}. "
            "Nothing was changed. Please try again in a moment."
        )
    elif code == "INVALID_DATE":
        plan.recover, plan.clear = True, ["check_in", "check_out"]
        plan.note = f"That date range doesn't work: {api_message}"
    elif code in {"VALIDATION_ERROR", "INVALID_ARGUMENTS"}:
        names = _fields_from_details(error.get("details"))
        if names:
            plan.recover, plan.clear = True, names
            labels = ", ".join(FIELD_LABELS.get(n, n) for n in names)
            plan.note = f"I need a valid {labels} (the hotel system rejected the one I had)."
        else:
            plan.message = f"The hotel system rejected the request: {api_message}"
    elif code == "HOTEL_NOT_FOUND":
        plan.recover, plan.clear = True, ["hotel_id", "hotel_name"]
        plan.note = "I couldn't find that hotel in the system."
    elif code == "BOOKING_NOT_FOUND":
        plan.recover, plan.clear = True, ["booking_id"]
        plan.note = "I couldn't find a booking with that ID."
    elif code == "CAPACITY_EXCEEDED":
        plan.recover, plan.clear = True, ["guests", "rooms"]
        plan.note = api_message
    elif code == "ROOMS_UNAVAILABLE":
        plan.kind = "booking_failed"
        plan.message = (
            f"{api_message} Nothing was booked. Want me to check other dates or other hotels?"
        )
    elif code == "BOOKING_ALREADY_CANCELLED":
        plan.kind = "already_cancelled"
    elif code == "IDEMPOTENCY_KEY_CONFLICT":
        plan.kind = "booking_failed"
        plan.message = "A technical conflict stopped this booking. Nothing was booked, so please ask me to book again."
    elif code == "INVALID_RESPONSE":
        plan.message = (
            f"The hotel system sent back something I couldn't understand, so I couldn't {action}. "
            "Please try again."
        )
    else:
        plan.message = f"I couldn't {action}: {api_message or 'the hotel system reported a problem'}."
    if stage in {"book"} and plan.kind == "error":
        plan.kind = "booking_failed"
    if stage in {"cancel"} and plan.kind == "error":
        plan.kind = "cancel_failed"
    return plan
