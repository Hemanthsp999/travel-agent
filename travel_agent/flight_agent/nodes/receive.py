"""Entry node: reset per-request fields and trim old messages."""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.nodes.deps import AgentDeps
from travel_agent.flight_agent.params import TASK_SCOPED
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.memory.short_term import last_user_text, prune_messages


def make_receive(deps: AgentDeps):
    keep = deps.settings.agent.history_keep_messages

    def receive(state: FlightAgentState) -> dict[str, Any]:
        return {
            **TASK_SCOPED,
            "user_request": last_user_text(state["messages"]),
            "identified_intent": "",
            "general_topic": "NONE",
            "missing_parameters": [],
            "awaiting": None,
            "ask_counts": {},
            "validation_note": None,
            "pending_action": None,
            "user_confirmation_status": None,
            "invite_booking": False,
            "booking_details": None,
            "cancellation_details": None,
            "error_information": None,
            "final_response": "",
            "next": "understand",
            "outcome": {},
            "result": {},
            "messages": prune_messages(state["messages"], keep),
        }

    return receive
