"""Validate node: sanity-check dates, fill gaps from context, and decide whether to ask or act."""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.hotel_agent.params import ask_group, missing_for
from travel_agent.hotel_agent.responses import fmt_date
from travel_agent.hotel_agent.schemas import Intent
from travel_agent.hotel_agent.state import HotelAgentState


def gave_up(state: HotelAgentState, group: list[str], max_attempts: int) -> bool:
    """True once the same question has been asked `max_attempts` times without an answer."""
    return (state.get("ask_counts") or {}).get(group[0], 0) >= max_attempts


def make_validate(deps: AgentDeps):
    max_attempts = deps.settings.agent.max_ask_attempts

    def validate(state: HotelAgentState) -> dict[str, Any]:
        intent = Intent(state["intent"])
        params = dict(state.get("params") or {})
        note = state.get("validation_note")
        today = deps.today()
        invite = bool(state.get("invite_booking"))

        check_in = date.fromisoformat(params["check_in"]) if params.get("check_in") else None
        check_out = date.fromisoformat(params["check_out"]) if params.get("check_out") else None
        if check_in and check_in < today:
            note = f"{fmt_date(check_in)} is already in the past."
            params.pop("check_in"), params.pop("check_out", None)
        elif check_in and check_out and check_out <= check_in:
            note = "The check-out date has to be after the check-in date."
            params.pop("check_out")

        if intent in (Intent.CANCEL_BOOKING, Intent.GET_BOOKING_STATUS) and not params.get("booking_id"):
            known = state.get("known_bookings") or []
            if len(known) == 1:
                params["booking_id"] = known[0]
            elif len(known) > 1:
                note = note or f"I know of these bookings from our chat: {', '.join(known)}."

        missing = missing_for(intent, params)
        if intent is Intent.BOOK_HOTEL and "hotel" in missing and params.get("city") and params.get("check_in") and params.get("check_out"):
            intent, invite = Intent.CHECK_AVAILABILITY, True
            missing = missing_for(intent, params)

        update: dict[str, Any] = {
            "intent": intent.value, "params": params, "missing_params": missing,
            "validation_note": note, "invite_booking": invite,
        }
        if not missing:
            return {**update, "next": "select_tool"}
        group = ask_group(missing)
        if gave_up(state, group, max_attempts):
            return {**update, "next": "respond", "outcome": make_outcome("gave_up", missing=group)}
        return {**update, "next": "ask_user"}

    return validate
