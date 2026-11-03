"""Confirm node: show the exact action and wait for an explicit yes before anything irreversible."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.hotel_agent.parsing import classify_confirmation, extract_params
from travel_agent.hotel_agent.state import HotelAgentState


def make_confirm(deps: AgentDeps):
    def confirm(state: HotelAgentState) -> dict[str, Any]:
        pending = state["pending_action"]
        summary = pending.get("quote") or {"booking_id": pending.get("booking_id")}
        answer = str(interrupt({
            "kind": "confirmation", "question": pending["prompt"], "action": pending["type"],
            "intent": pending["intent"], "missing_params": [], "data": summary,
        }))
        found = extract_params(answer, deps.today(), "confirmation")
        verdict = classify_confirmation(answer, pending["type"], bool(found.params or found.dates or found.nights))
        update: dict[str, Any] = {
            "messages": [AIMessage(content=pending["prompt"]), HumanMessage(content=answer)],
            "confirmation": verdict, "awaiting": "confirmation",
        }
        if verdict == "confirmed":
            return {**update, "next": "select_tool"}
        if verdict == "declined":
            outcome = make_outcome("declined", action=pending["type"])
            return {**update, "pending_action": None, "outcome": outcome, "next": "respond"}
        return {**update, "pending_action": None, "next": "understand"}

    return confirm
