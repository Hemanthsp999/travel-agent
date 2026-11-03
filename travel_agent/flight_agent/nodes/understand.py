"""Understand node: intent, trip details and flight references for the newest user message.

Rules extract every exact value (dates, numbers, IDs, contact details, passengers). The LLM only labels
the intent and copies city names, and each copied name must appear in the user's text.
"""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.intent import (
    Classification,
    ClassifyContext,
    classify,
    mentions_booking,
    resolve_flight,
)
from travel_agent.flight_agent.nodes.deps import AgentDeps, current_query, make_outcome, same_query, shown_flights
from travel_agent.flight_agent.params import apply_extracted, trip_key
from travel_agent.flight_agent.parsing import Extracted, canonical_key, extract, is_abort, is_retry, normalize_place
from travel_agent.flight_agent.schemas import GeneralTopic, Intent
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.memory.short_term import last_user_text, recent_dialogue

_HINT_KEYS = {"origin", "destination", "departure_date", "passenger_count", "booking_id", "passenger_details", "contact"}
_MAX_REPLY_WORDS = 14


def _covers(awaiting: str | None, found: Extracted, hit: dict[str, Any] | None) -> bool:
    """Did the user's text supply the value we asked for?"""
    return bool({
        "origin": found.origin, "destination": found.destination, "departure_date": found.departure_date,
        "passenger_count": found.passenger_count, "booking_id": found.booking_id,
        "flight": hit or found.flight_id or found.flight_number,
        "passenger_details": found.passengers, "contact_details": found.email or found.phone,
        "confirmation": found.has_values,
    }.get(awaiting or ""))


def _fallback_intent(state: FlightAgentState, found: Extracted, hit: dict[str, Any] | None) -> Intent:
    """When nothing was classified, continue the current or previous task if the text fits it."""
    if state.get("identified_intent"):
        return Intent(state["identified_intent"])
    last = state.get("last_intent")
    searching = (Intent.CHECK_FLIGHT_AVAILABILITY.value, Intent.BOOK_FLIGHT.value)
    if hit is not None and last in searching:
        return Intent.BOOK_FLIGHT
    if (found.has_values or hit) and last in searching:
        return Intent(last)
    if found.booking_id and last in (Intent.CANCEL_FLIGHT.value, Intent.GET_BOOKING_STATUS.value):
        return Intent(last)
    return Intent.UNKNOWN


def make_understand(deps: AgentDeps):
    cfg = deps.settings.agent

    def understand(state: FlightAgentState) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        prev_intent = state.get("identified_intent") or None
        awaiting = state.get("awaiting")
        task = state.get("task") or {}

        if (prev_intent or awaiting) and is_abort(text):
            return {"outcome": make_outcome("aborted"), "next": "respond", "awaiting": None, "pending_action": None}

        retry = state.get("retryable_action")
        if retry and not prev_intent and is_retry(text):
            return {
                "identified_intent": Intent.BOOK_FLIGHT.value, "pending_action": retry,
                "user_confirmation_status": "confirmed", "retryable_action": None, "next": "book_flight",
                "awaiting": None,
            }

        found = extract(
            text, deps.today(), awaiting,
            have_origin=bool(state.get("origin")), have_destination=bool(state.get("destination")),
            allow_passengers=prev_intent == Intent.BOOK_FLIGHT.value or mentions_booking(text),
        )
        update = apply_extracted(state, found)
        hinted = None if prev_intent else task.get("intent")
        if not prev_intent:
            for key, value in (task.get("params") or {}).items():
                if key in _HINT_KEYS and value not in (None, "", []):
                    update[key] = value

        shown = shown_flights(state)
        hit, ambiguous = resolve_flight(text, shown, allow_bare_number=awaiting == "flight")
        answered = bool(prev_intent) and len(text.split()) <= _MAX_REPLY_WORDS and _covers(awaiting, found, hit)

        if hinted:
            cls = Classification(intent=Intent(hinted), confidence=1.0, source="task")
        elif answered:
            cls = Classification(intent=Intent(prev_intent), source="context")
        else:
            ctx = ClassifyContext(
                awaiting, prev_intent, [f"{f['airline']} {f['flight_number']}" for f in shown],
                recent_dialogue(state["messages"][:-1], 2),
            )
            cls = classify(deps.llm if cfg.use_llm else None, text, ctx, cfg.llm_confidence_threshold)

        intent, topic = cls.intent, cls.topic
        if intent is Intent.UNKNOWN:
            intent = _fallback_intent(state, found, hit)
        if cls.source == "context":
            topic = GeneralTopic(state.get("general_topic") or "NONE")
        elif intent is Intent.GENERAL_FLIGHT_QUERY and topic is GeneralTopic.NONE:
            topic = GeneralTopic.HELP

        # Cities copied by the model are used only when the rules found none (and they are grounded).
        if cls.origin and "origin" not in update and not found.origin:
            update["origin"] = normalize_place(cls.origin)
        if cls.destination and "destination" not in update and not found.destination:
            update["destination"] = normalize_place(cls.destination)

        view: FlightAgentState = {**state, **update}  # type: ignore[typeddict-item]
        note: str | None = state.get("validation_note")
        if found.round_trip or found.extra_dates:
            when = f" on {found.departure_date.strftime('%d %b %Y').lstrip('0')}" if found.departure_date else ""
            note = f"I can only search one-way flights, so I'll look at the first date{when}."

        selected = state.get("selected_flight")
        route_changed = any(
            k in update and canonical_key(update[k]) != canonical_key(state.get(k)) for k in ("origin", "destination")
        )
        date_changed = "departure_date" in update and update["departure_date"] != state.get("departure_date")
        if (route_changed or date_changed) and not hit and not found.flight_id and selected:
            # Flights run daily: on a date change keep the choice and let the seat check re-verify it.
            selected = {"flight_id": selected["flight_id"]} if not route_changed else None
        wants_flight = intent is Intent.BOOK_FLIGHT or (
            intent is Intent.GENERAL_FLIGHT_QUERY and topic is GeneralTopic.FLIGHT_DETAILS
        )
        if wants_flight:
            if hit is not None:
                selected = hit
                query = current_query(view)
                same = same_query((state.get("last_search") or {}).get("query"), query)
                if same:
                    update["verified_key"] = trip_key(view, hit["flight_id"])
            elif ambiguous:
                note = "More than one flight matches: " + ", ".join(f"{f['airline']} {f['flight_number']}" for f in ambiguous) + "."
            elif found.flight_id:
                if not selected or selected.get("flight_id") != found.flight_id:
                    selected = {"flight_id": found.flight_id}
            elif found.flight_number and intent is Intent.BOOK_FLIGHT:
                note = (
                    f"I can only book flights that I've shown you from a search, and I don't have {found.flight_number} "
                    "in the current results."
                )

        changed_intent = bool(prev_intent) and intent.value != prev_intent
        update.update({
            "identified_intent": intent.value,
            "general_topic": topic.value,
            "selected_flight": selected,
            "validation_note": note,
            "user_confirmation_status": None,
        })
        if found.sort_hint:
            update["sort_hint"] = found.sort_hint
        if changed_intent:
            update.update(pending_action=None, awaiting=None)

        if intent is Intent.UNKNOWN:
            update.update(outcome=make_outcome("unrecognized"), next="respond")
        elif intent is Intent.GENERAL_FLIGHT_QUERY:
            update["next"] = "general_query"
        else:
            update["next"] = "validate"
        return update

    return understand
