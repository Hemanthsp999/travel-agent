"""Parameter handling: merging extracted values, required-field rules and question ordering."""

from __future__ import annotations

import re
from typing import Any

from travel_agent.flight_agent.parsing import Extracted, canonical_key, merge_passengers
from travel_agent.flight_agent.schemas import Intent
from travel_agent.flight_agent.state import FlightAgentState

MAX_PASSENGERS = 9
PASSENGER_FIELDS = ("name", "age", "gender")
_PASSENGER_KEY = re.compile(r"^passenger_(\d+)_(name|age|gender)$")
# Values that belong to one request; trip and traveller details carry over within a conversation.
TASK_SCOPED = {"booking_id": None, "selected_flight": None, "sort_hint": ""}


def apply_extracted(state: FlightAgentState, found: Extracted) -> dict[str, Any]:
    """Partial state update from rule-extracted values (newest message wins)."""
    update: dict[str, Any] = {}
    if found.origin:
        update["origin"] = found.origin
    if found.destination:
        update["destination"] = found.destination
    if found.departure_date:
        update["departure_date"] = found.departure_date.isoformat()
    if found.booking_id:
        update["booking_id"] = found.booking_id

    count = state.get("passenger_count")
    details = [dict(p) for p in state.get("passenger_details") or []]
    if found.passenger_count is not None:
        if count is not None and found.passenger_count != count:
            details = []  # a different party size: re-collect rather than guess who is dropped or added
        count = found.passenger_count
    complete = [p for p in found.passengers if all(k in p for k in PASSENGER_FIELDS)]
    if count is None and complete:
        count = len(complete)
    if found.passengers:
        details = merge_passengers(details, found.passengers, count)
    if count and len(details) > count:
        details = details[:count]
    if count != state.get("passenger_count"):
        update["passenger_count"] = count
    if details != (state.get("passenger_details") or []):
        update["passenger_details"] = details

    contact = dict(state.get("contact") or {})
    if found.email:
        contact["email"] = found.email
    if found.phone:
        contact["phone"] = found.phone
    if contact != (state.get("contact") or {}):
        update["contact"] = contact
    return update


def missing_for(intent: Intent, state: FlightAgentState) -> list[str]:
    """Required values still absent for `intent`, in the order they should be asked."""
    need: list[str] = []
    if intent in (Intent.CHECK_FLIGHT_AVAILABILITY, Intent.BOOK_FLIGHT):
        need += [k for k in ("origin", "destination", "departure_date", "passenger_count") if not state.get(k)]
    if intent is Intent.BOOK_FLIGHT:
        if not state.get("selected_flight"):
            need.append("flight")
        count = state.get("passenger_count")
        details = state.get("passenger_details") or []
        for i in range(count or 0):
            person = details[i] if i < len(details) else {}
            need += [f"passenger_{i + 1}_{k}" for k in PASSENGER_FIELDS if k not in person]
        contact = state.get("contact") or {}
        need += [f"contact_{k}" for k in ("email", "phone") if not contact.get(k)]
    elif intent in (Intent.CANCEL_FLIGHT, Intent.GET_BOOKING_STATUS) and not state.get("booking_id"):
        need.append("booking_id")
    return need


def ask_group(missing: list[str]) -> list[str]:
    """Which missing values to ask in one question (one natural question at a time)."""
    if not missing:
        return []
    first = missing[0]
    if match := _PASSENGER_KEY.match(first):
        return [k for k in missing if k.startswith(f"passenger_{match.group(1)}_")]
    if first.startswith("contact_"):
        return [k for k in missing if k.startswith("contact_")]
    return [first]


def awaiting_for(group: list[str]) -> str:
    """Value recorded in `awaiting` so a bare reply can be interpreted correctly."""
    first = group[0]
    if _PASSENGER_KEY.match(first):
        return "passenger_details"
    if first.startswith("contact_"):
        return "contact_details"
    return first


def clear_values(state: FlightAgentState, keys: list[str]) -> dict[str, Any]:
    """Partial state update that forgets the named (rejected) values."""
    update: dict[str, Any] = {}
    details = [dict(p) for p in state.get("passenger_details") or []]
    contact = dict(state.get("contact") or {})
    for key in keys:
        if key in ("origin", "destination", "departure_date", "booking_id"):
            update[key] = None
        elif key == "passenger_count":
            update["passenger_count"], details = None, []
        elif key == "selected_flight":
            update["selected_flight"], update["verified_key"] = None, None
        elif key in ("contact_email", "contact_phone"):
            contact.pop(key.split("_", 1)[1], None)
        elif match := _PASSENGER_KEY.match(key):
            index = int(match.group(1)) - 1
            if index < len(details):
                details[index].pop(match.group(2), None)
    update["passenger_details"], update["contact"] = details, contact
    return update


def trip_key(state: FlightAgentState, flight_id: str) -> str:
    """Identity of a verified (route, date, party size, flight) combination."""
    return "|".join((
        canonical_key(state.get("origin")), canonical_key(state.get("destination")),
        str(state.get("departure_date")), str(state.get("passenger_count")), flight_id,
    ))
