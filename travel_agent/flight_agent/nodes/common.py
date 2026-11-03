"""Shared failure handling and search execution for the API-calling nodes."""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.errors import UNREACHABLE, plan_for_error
from travel_agent.flight_agent.nodes.deps import AgentDeps, current_query, make_outcome
from travel_agent.flight_agent.params import clear_values
from travel_agent.flight_agent.schemas import ToolResult
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.flight_agent.tools import run_tool

RETRY_HINT = ' Say "retry" and I\'ll try again.'


def search_tool(deps: AgentDeps, state: FlightAgentState) -> ToolResult:
    """Call `check_flight_availability` for the current trip details."""
    query = current_query(state)
    if query is None:
        raise ValueError("search requested without complete trip details")
    return run_tool(deps.tools, "check_flight_availability", query)


def remember_search(state: FlightAgentState, data: dict[str, Any], max_shown: int) -> dict[str, Any]:
    """`last_search` record built from API data only (the query is echoed from the request)."""
    query = current_query(state) or {}
    flights = data["flights"]
    return {"query": query, "flights": flights, "shown_ids": [f["flight_id"] for f in flights[:max_shown]]}


def failure_update(state: FlightAgentState, stage: str, result: ToolResult) -> dict[str, Any]:
    """Recovery for a failed API call: ask the user to fix a rejected value, or explain what happened."""
    error = result.error.model_dump() if result.error else {"code": "UNKNOWN", "message": ""}
    pending = state.get("pending_action")
    plan = plan_for_error(stage, error)

    if plan.kind == "already_cancelled":
        booking_id = (pending or {}).get("booking_id") or state.get("booking_id") or ""
        return {
            "outcome": make_outcome("already_cancelled", error=error, booking_id=booking_id),
            "pending_action": None, "next": "respond",
        }
    if plan.recover:
        return {
            **clear_values(state, plan.clear),
            "validation_note": plan.note, "pending_action": None, "next": "validate",
        }
    message = plan.message
    update: dict[str, Any] = {"pending_action": None, "next": "respond"}
    if stage == "book" and plan.code in UNREACHABLE:
        update["retryable_action"] = pending
        message += RETRY_HINT
    update["outcome"] = make_outcome(plan.kind, error=error, message=message)
    return update
