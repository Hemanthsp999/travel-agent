"""Respond node: build the final reply from the outcome and API data (no model-written facts)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.hotel_agent.responses import STATUS_BY_KIND, render
from travel_agent.hotel_agent.schemas import Intent, TaskStatus
from travel_agent.hotel_agent.state import HotelAgentState

_TRACKED = {
    Intent.CHECK_AVAILABILITY.value, Intent.BOOK_HOTEL.value,
    Intent.CANCEL_BOOKING.value, Intent.GET_BOOKING_STATUS.value,
}


def make_respond(deps: AgentDeps):
    max_shown = deps.settings.agent.max_hotels_shown

    def respond(state: HotelAgentState) -> dict[str, Any]:
        outcome = state.get("outcome") or make_outcome(
            "error", message="Sorry, something went wrong on my side. Could you say that again?"
        )
        text = render(outcome, max_shown)
        status = STATUS_BY_KIND.get(outcome["kind"], TaskStatus.FAILED)
        error = outcome.get("error")
        if status in (TaskStatus.FAILED, TaskStatus.UNVERIFIED) and not error:
            error = {"code": outcome["kind"].upper(), "message": text.splitlines()[0]}
        intent = state.get("intent") or None
        result = {
            "status": status.value, "intent": intent or None, "message": text, "question": None,
            "missing_params": [], "data": outcome.get("data") or {}, "error": error,
        }
        return {
            "messages": [AIMessage(content=text)],
            "result": result,
            "last_intent": intent if intent in _TRACKED else state.get("last_intent", ""),
            "awaiting": None, "pending_action": None, "confirmation": None, "force_stage": None, "next": "end",
        }

    return respond
