"""Search nodes: availability search (answer or invite a booking) and pre-booking seat verification."""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.nodes.common import failure_update, remember_search, search_tool
from travel_agent.flight_agent.nodes.deps import AgentDeps, compact_flight, current_query, make_outcome
from travel_agent.flight_agent.params import trip_key
from travel_agent.flight_agent.state import FlightAgentState


def make_search_flights(deps: AgentDeps):
    max_shown = deps.settings.agent.max_flights_shown

    def search_flights(state: FlightAgentState) -> dict[str, Any]:
        result = search_tool(deps, state)
        if not result.ok:
            return {**failure_update(state, "search", result), "api_execution_status": "FAILED"}
        search = remember_search(state, result.data or {"flights": []}, max_shown)
        query, flights = search["query"], search["flights"]
        update: dict[str, Any] = {"last_search": search, "api_execution_status": "SUCCESS", "missing_parameters": []}
        if not flights:
            return {**update, "outcome": make_outcome("no_results", {"query": query, "count": 0, "flights": []}, query=query),
                    "next": "respond", "validation_note": None}
        if state.get("invite_booking"):
            return {**update, "missing_parameters": ["flight"], "next": "validate"}  # list is shown by ask_user
        data = {"query": query, "count": len(flights), "flights": [compact_flight(f) for f in flights]}
        outcome = make_outcome(
            "search_results", data, query=query, flights=flights, shown_ids=search["shown_ids"],
            sort_hint=state.get("sort_hint", ""), note=state.get("validation_note"),
        )
        return {**update, "outcome": outcome, "next": "respond", "validation_note": None}

    return search_flights


def make_check_seats(deps: AgentDeps):
    max_shown = deps.settings.agent.max_flights_shown

    def check_seats(state: FlightAgentState) -> dict[str, Any]:
        selected = state["selected_flight"]
        result = search_tool(deps, state)
        if not result.ok:
            return {**failure_update(state, "precheck", result), "api_execution_status": "FAILED"}
        search = remember_search(state, result.data or {"flights": []}, max_shown)
        match = next((f for f in search["flights"] if f["flight_id"] == selected["flight_id"]), None)
        update: dict[str, Any] = {"last_search": search, "api_execution_status": "SUCCESS"}
        if match is None:
            query = current_query(state) or {}
            label = selected.get("flight_number") and f"{selected.get('airline', '')} {selected['flight_number']}".strip() or selected["flight_id"]
            outcome = make_outcome(
                "booking_unavailable", {"query": query, "flight_id": selected["flight_id"]}, query=query,
                flight_label=label, alternatives=search["flights"][:max_shown], passengers=query.get("passengers"),
                error={"code": "FLIGHT_UNAVAILABLE", "message": f"{selected['flight_id']} has no seats for this search."},
            )
            return {**update, "selected_flight": None, "verified_key": None, "outcome": outcome, "next": "respond"}
        return {**update, "selected_flight": match, "verified_key": trip_key(state, match["flight_id"]), "next": "validate"}

    return check_seats
