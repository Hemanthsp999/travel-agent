"""Validate node: sanity-check values, fill gaps from context, and route to the right execution path."""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_agent.flight_agent.nodes.deps import AgentDeps, compact_flight, current_query, make_outcome, same_query
from travel_agent.flight_agent.params import MAX_PASSENGERS, ask_group, clear_values, missing_for, trip_key
from travel_agent.flight_agent.parsing import canonical_key
from travel_agent.flight_agent.responses import booking_prompt, fmt_date
from travel_agent.flight_agent.schemas import Intent
from travel_agent.flight_agent.state import FlightAgentState


def gave_up(state: FlightAgentState, group: list[str], max_attempts: int) -> bool:
    """True once the same question has been asked `max_attempts` times without an answer."""
    return (state.get("ask_counts") or {}).get(group[0], 0) >= max_attempts


def booking_signature(payload: dict[str, Any]) -> str:
    names = ",".join(sorted(p["name"].lower() for p in payload["passengers"]))
    return f"{payload['flight_id']}|{payload['departure_date']}|{names}|{payload['contact_email']}"


def make_validate(deps: AgentDeps):
    max_attempts = deps.settings.agent.max_ask_attempts

    def sanity(view: dict[str, Any], note: str | None) -> tuple[dict[str, Any], str | None]:
        """Drop values that cannot be right and say why. Returns (changes, note)."""
        changes: dict[str, Any] = {}
        when = view.get("departure_date")
        if when and date.fromisoformat(when) < deps.today():
            note = f"{fmt_date(when)} is already in the past."
            changes.update(clear_values(view, ["departure_date"]))  # type: ignore[arg-type]
        count = view.get("passenger_count")
        if count is not None and not 1 <= count <= MAX_PASSENGERS:
            note = f"I can book 1 to {MAX_PASSENGERS} passengers at a time."
            changes.update(clear_values(view, ["passenger_count"]))  # type: ignore[arg-type]
        origin, destination = view.get("origin"), view.get("destination")
        if origin and destination and canonical_key(origin) == canonical_key(destination):
            note = "The departure and destination cities have to be different."
            changes.update(clear_values(view, ["destination"]))  # type: ignore[arg-type]
        return changes, note

    def validate(state: FlightAgentState) -> dict[str, Any]:
        intent = Intent(state["identified_intent"])
        changes, note = sanity(dict(state), state.get("validation_note"))
        view: FlightAgentState = {**state, **changes}  # type: ignore[typeddict-item]

        if intent in (Intent.CANCEL_FLIGHT, Intent.GET_BOOKING_STATUS) and not view.get("booking_id"):
            known = state.get("known_bookings") or []
            if len(known) == 1:
                changes["booking_id"] = view["booking_id"] = known[0]
            elif len(known) > 1:
                note = note or f"I know of these bookings from our chat: {', '.join(known)}."

        missing = missing_for(intent, view)
        update: dict[str, Any] = {**changes, "identified_intent": intent.value, "missing_parameters": missing, "validation_note": note}
        query = current_query(view)
        selected = view.get("selected_flight")

        if intent is Intent.BOOK_FLIGHT and query:
            if selected and view.get("verified_key") != trip_key(view, selected["flight_id"]):
                return {**update, "next": "check_seats"}  # confirm the flight really has seats before collecting details
            last = state.get("last_search") or {}
            if not selected and "flight" in missing and not (same_query(last.get("query"), query) and last.get("flights")):
                return {**update, "invite_booking": True, "next": "search_flights"}

        if not missing:
            if intent is Intent.CHECK_FLIGHT_AVAILABILITY:
                return {**update, "next": "search_flights"}
            if intent is Intent.BOOK_FLIGHT:
                return {**update, **_prepare_booking(view, deps, note), "next": "confirm"}
            return {**update, "next": "lookup_booking"}

        group = ask_group(missing)
        if gave_up(state, group, max_attempts):
            return {**update, "next": "respond", "outcome": make_outcome("gave_up", missing=group)}
        return {**update, "next": "ask_user"}

    return validate


def _prepare_booking(view: FlightAgentState, deps: AgentDeps, note: str | None) -> dict[str, Any]:
    """Build the pending booking: exact payload, price quote from API data, idempotency key and prompt."""
    flight = view["selected_flight"]
    count = view["passenger_count"]
    contact = view["contact"]
    payload = {
        "flight_id": flight["flight_id"],
        "departure_date": view["departure_date"],
        "passengers": [{"name": p["name"], "age": p["age"], "gender": p["gender"]} for p in view["passenger_details"][:count]],
        "contact_email": contact["email"],
        "contact_phone": contact["phone"],
    }
    signature = booking_signature(payload)
    existing = (view.get("booking_signatures") or {}).get(signature)
    warning = f"you already booked this exact flight for these passengers ({existing}); a second booking will be a new one." if existing else None
    total = flight.get("total_price")
    if total is None:
        total = flight["price_per_passenger"] * count
    quote = {"flight": compact_flight(flight), "total_price": total, "passengers": count}
    pending = {
        "type": "book", "payload": payload, "quote": quote, "idempotency_key": deps.new_key(),
        "prompt": booking_prompt(payload, quote, warning), "intent": Intent.BOOK_FLIGHT.value, "signature": signature,
    }
    return {"pending_action": pending, "validation_note": None}
