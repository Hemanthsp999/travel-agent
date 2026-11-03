"""General-query node: help, unsupported requests, "show all", comparisons and flight details.

Answers come only from the latest search results held in state (API data); no API call is needed.
"""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.nodes.deps import AgentDeps, compact_flight, make_outcome, shown_flights
from travel_agent.flight_agent.schemas import GeneralTopic
from travel_agent.flight_agent.state import FlightAgentState

_NO_RESULTS = (
    "I don't have any flight results from this chat to refer to yet. "
    "Tell me where you're flying from and to, the date and the number of passengers, and I'll search."
)
_TRIP_FIELDS = ["origin", "destination", "departure_date", "passenger_count"]


def _no_context(message: str = _NO_RESULTS) -> dict[str, Any]:
    return {"outcome": make_outcome("no_context", message=message, missing=_TRIP_FIELDS), "next": "respond"}


def make_general_query(deps: AgentDeps):
    def general_query(state: FlightAgentState) -> dict[str, Any]:
        topic = GeneralTopic(state.get("general_topic") or "NONE")
        search = state.get("last_search") or {}
        flights = search.get("flights") or []
        shown = shown_flights(state)
        query = search.get("query") or {}
        base = {"missing_parameters": [], "pending_action": None}

        if topic is GeneralTopic.HELP:
            return {**base, "outcome": make_outcome("help"), "next": "respond"}
        if topic is GeneralTopic.SHOW_MORE:
            if not flights:
                return {**base, **_no_context()}
            data = {"query": query, "count": len(flights), "flights": [compact_flight(f) for f in flights]}
            return {**base, "outcome": make_outcome("show_all", data, query=query, flights=flights), "next": "respond"}
        if topic is GeneralTopic.COMPARE_FLIGHTS:
            if len(shown) < 2:
                return {**base, **_no_context("I need at least two flights from a search to compare. Tell me a route and date first.")}
            data = {"query": query, "flights": [compact_flight(f) for f in shown]}
            outcome = make_outcome("comparison", data, flights=shown, passengers=query.get("passengers"))
            return {**base, "outcome": outcome, "next": "respond"}
        if topic is GeneralTopic.FLIGHT_DETAILS:
            return {**base, **_details(state, shown, flights, query)}
        return {**base, "outcome": make_outcome("unsupported"), "next": "respond"}

    return general_query


def _details(state: FlightAgentState, shown: list[dict], flights: list[dict], query: dict) -> dict[str, Any]:
    chosen = state.get("selected_flight")
    if chosen and "airline" not in chosen:  # only an ID was given: look it up in the results we hold
        chosen = next((f for f in flights if f["flight_id"] == chosen["flight_id"]), None)
        if chosen is None:
            return _no_context("I only have details for flights from a search in this chat. Tell me the route and date first.")
    if chosen is None and len(shown) == 1:
        chosen = shown[0]
    if chosen is None:
        if not shown:
            return _no_context()
        return {"missing_parameters": ["flight"], "next": "ask_user"}
    outcome = make_outcome(
        "flight_details", {"flight": compact_flight(chosen)}, flight=chosen, passengers=query.get("passengers"),
    )
    return {"missing_parameters": [], "outcome": outcome, "next": "respond"}
