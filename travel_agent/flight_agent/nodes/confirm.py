"""Confirm node: show the exact action and wait for an explicit yes before anything irreversible."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from travel_agent.flight_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.flight_agent.parsing import classify_confirmation, extract
from travel_agent.flight_agent.state import FlightAgentState


def make_confirm(deps: AgentDeps):
    def confirm(state: FlightAgentState) -> dict[str, Any]:
        pending = state["pending_action"]
        summary = pending.get("quote") or {"booking_id": pending.get("booking_id")}
        answer = str(interrupt({
            "kind": "confirmation", "question": pending["prompt"], "action": pending["type"],
            "intent": pending["intent"], "missing_params": [], "data": summary,
        }))
        found = extract(answer, deps.today(), "confirmation", allow_passengers=False)
        verdict = classify_confirmation(answer, pending["type"], found.has_values)
        update: dict[str, Any] = {
            "messages": [AIMessage(content=pending["prompt"]), HumanMessage(content=answer)],
            "user_confirmation_status": verdict, "awaiting": "confirmation",
        }
        if verdict == "confirmed":
            return {**update, "next": "book_flight" if pending["type"] == "book" else "cancel_flight"}
        if verdict == "declined":
            outcome = make_outcome("declined", action=pending["type"])
            return {**update, "pending_action": None, "outcome": outcome, "next": "respond"}
        return {**update, "pending_action": None, "next": "understand"}

    return confirm
