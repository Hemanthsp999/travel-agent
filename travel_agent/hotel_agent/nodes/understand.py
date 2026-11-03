"""Understand node: intent, parameters and hotel references for the newest user message.

Rules extract every exact value (dates, numbers, IDs, contact details). The LLM only labels the
intent and copies city / hotel / guest names, and each copied name must appear in the user's text.
"""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.intent import (
    Classification,
    ClassifyContext,
    classify,
    normalize,
    resolve_hotel,
    rule_city,
)
from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome, shown_hotels
from travel_agent.hotel_agent.parsing import Extracted, extract_params, is_abort, is_retry
from travel_agent.hotel_agent.params import merge_extracted
from travel_agent.hotel_agent.schemas import GeneralTopic, Intent
from travel_agent.hotel_agent.state import HotelAgentState
from travel_agent.memory.short_term import last_user_text, recent_dialogue

_HINT_KEYS = {"city", "check_in", "check_out", "guests", "rooms", "hotel_id", "hotel_name", "booking_id",
              "guest_name", "guest_email", "guest_phone", "max_price_per_night", "min_rating"}
_MAX_REPLY_WORDS = 12


def _covers(awaiting: str | None, found: Extracted, hit: dict[str, Any] | None) -> bool:
    """Did the user's text supply the value we asked for?"""
    if awaiting in ("check_in", "check_out"):
        return bool(found.dates or found.nights)
    if awaiting == "hotel":
        return hit is not None or "hotel_id" in found.params
    if awaiting == "confirmation":
        return bool(found.params or found.dates or found.nights)
    return awaiting is not None and awaiting in found.params


def _fallback_intent(
    state: HotelAgentState, found: Extracted, hit: dict[str, Any] | None, city: str | None
) -> Intent:
    """When nothing was classified, continue the current or previous task if the text fits it."""
    if state.get("intent"):
        return Intent(state["intent"])
    last = state.get("last_intent")
    has_values = bool(found.params or found.dates or found.nights or city)
    if hit is not None and last in (Intent.CHECK_AVAILABILITY.value, Intent.BOOK_HOTEL.value):
        return Intent.BOOK_HOTEL
    if has_values and last in (Intent.CHECK_AVAILABILITY.value, Intent.BOOK_HOTEL.value):
        return Intent(last)
    return Intent.UNKNOWN


def make_understand(deps: AgentDeps):
    cfg = deps.settings.agent

    def understand(state: HotelAgentState) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        prev_intent = state.get("intent") or None
        awaiting = state.get("awaiting")
        task = state.get("task") or {}

        if (prev_intent or awaiting) and is_abort(text):
            return {"outcome": make_outcome("aborted"), "next": "respond", "awaiting": None, "pending_action": None}

        retry = state.get("retryable_action")
        if retry and not prev_intent and is_retry(text):
            return {
                "intent": Intent.BOOK_HOTEL.value, "pending_action": retry, "confirmation": "confirmed",
                "retryable_action": None, "next": "select_tool", "awaiting": None,
            }

        old_params = dict(state.get("params") or {})
        found = extract_params(text, deps.today(), awaiting)
        params = merge_extracted(old_params, found, text, awaiting)
        if not prev_intent:
            hints = {k: v for k, v in (task.get("params") or {}).items() if k in _HINT_KEYS and v not in (None, "")}
            params.update(hints)

        shown = shown_hotels(state)
        hit, ambiguous = resolve_hotel(text, shown, allow_bare_number=awaiting == "hotel")
        hinted = None if prev_intent else task.get("intent")
        answered = bool(prev_intent) and len(text.split()) <= _MAX_REPLY_WORDS and _covers(awaiting, found, hit)

        if hinted:
            cls = Classification(intent=Intent(hinted), confidence=1.0, source="task")
        elif answered:
            cls = Classification(intent=Intent(prev_intent), source="context")
            cls.city = rule_city(text, known_only=True)
        else:
            ctx = ClassifyContext(awaiting, prev_intent, [h["name"] for h in shown], recent_dialogue(state["messages"][:-1], 2))
            cls = classify(deps.llm if cfg.use_llm else None, text, ctx, cfg.llm_confidence_threshold)

        intent, topic = cls.intent, cls.topic
        if intent is Intent.GENERAL_HOTEL_QUERY and topic in (GeneralTopic.BUDGET_SEARCH, GeneralTopic.ROOM_AVAILABILITY):
            intent = Intent.CHECK_AVAILABILITY
        if intent is Intent.UNKNOWN:
            intent = _fallback_intent(state, found, hit, cls.city or rule_city(text, known_only=True))
            if intent is not Intent.UNKNOWN and not cls.city:
                cls.city = rule_city(text, known_only=True)
        if cls.source == "context":
            topic = GeneralTopic(state.get("general_topic") or "NONE")

        # A new city invalidates earlier hotel choices and the list the user was shown.
        city_changed = bool(cls.city) and normalize(cls.city) != normalize(params.get("city", ""))
        if cls.city:
            params["city"] = cls.city
        if city_changed:
            hit, ambiguous = None, []
            if "hotel_id" not in found.params:
                params.pop("hotel_id", None)

        selected = state.get("selected_hotel")
        note: str | None = state.get("validation_note")
        if intent is Intent.BOOK_HOTEL or (intent is Intent.GENERAL_HOTEL_QUERY and topic is GeneralTopic.HOTEL_DETAILS):
            if hit is not None:
                params["hotel_id"], selected = hit["hotel_id"], hit
                if hit.get("city") and not params.get("city"):
                    params["city"] = hit["city"]
            elif ambiguous:
                names = ", ".join(h["name"] for h in ambiguous)
                note = f"More than one hotel matches: {names}."
            elif cls.hotel_name and not params.get("hotel_id"):
                params["hotel_name"] = cls.hotel_name
            if intent is Intent.BOOK_HOTEL and not params.get("hotel_id") and not params.get("hotel_name") and selected:
                params["hotel_id"] = selected["hotel_id"]
        elif intent is Intent.CHECK_AVAILABILITY and (cls.hotel_name or "hotel_id" in found.params):
            if cls.hotel_name:
                params["hotel_name"] = cls.hotel_name
        if selected and selected.get("hotel_id") != params.get("hotel_id"):
            selected = None

        if cls.guest_name and "guest_name" not in found.params:
            params["guest_name"] = cls.guest_name

        changed_intent = bool(prev_intent) and intent.value != prev_intent
        update: dict[str, Any] = {
            "intent": intent.value,
            "general_topic": topic.value,
            "params": params,
            "selected_hotel": selected,
            "validation_note": note,
            "confirmation": None,
        }
        if changed_intent:
            update.update(pending_action=None, awaiting=None)

        if intent is Intent.UNKNOWN:
            message = "I'm not sure I caught that. I can check hotel availability, give hotel details, book, look up a booking, or cancel one. What would you like?"
            update.update(outcome=make_outcome("no_context", message=message), next="respond")
        elif intent is Intent.GENERAL_HOTEL_QUERY:
            update["next"] = "general_query"
        else:
            update["next"] = "validate"
        return update

    return understand
