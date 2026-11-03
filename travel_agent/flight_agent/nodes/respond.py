"""Respond node: build the final reply and the structured result from the outcome (no model-written facts)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from travel_agent.flight_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.flight_agent.responses import ERROR_CODE_BY_KIND, STATUS_BY_KIND, render
from travel_agent.flight_agent.schemas import ExecutionStatus, Intent
from travel_agent.flight_agent.state import FlightAgentState

_TRACKED = {
    Intent.CHECK_FLIGHT_AVAILABILITY.value, Intent.BOOK_FLIGHT.value,
    Intent.CANCEL_FLIGHT.value, Intent.GET_BOOKING_STATUS.value,
}


def make_respond(deps: AgentDeps):
    max_shown = deps.settings.agent.max_flights_shown

    def respond(state: FlightAgentState) -> dict[str, Any]:
        outcome = state.get("outcome") or make_outcome(
            "error", message="Sorry, something went wrong on my side. Could you say that again?"
        )
        kind = outcome["kind"]
        text = render(outcome, max_shown)
        status = STATUS_BY_KIND.get(kind, ExecutionStatus.FAILED)
        error = outcome.get("error")
        if status in (ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN):
            error = error or {"code": ERROR_CODE_BY_KIND.get(kind, "AGENT_ERROR"), "message": text.splitlines()[0]}
        else:
            error = None
        intent = state.get("identified_intent") or None
        needs_input = status is ExecutionStatus.REQUIRES_USER_INPUT
        result = {
            "status": status.value, "intent": intent, "message": text, "question": None,
            "missing_parameters": list(outcome.get("missing", [])) if needs_input else [],
            "requires_user_input": needs_input, "data": outcome.get("data") or {}, "error": error, "outcome": kind,
        }
        return {
            "messages": [AIMessage(content=text)],
            "result": result,
            "final_response": text,
            "error_information": error,
            "last_intent": intent if intent in _TRACKED else state.get("last_intent", ""),
            "awaiting": None, "pending_action": None, "user_confirmation_status": None, "missing_parameters": [],
            "next": "end",
        }

    return respond
