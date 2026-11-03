"""Conditional-edge table: every node writes `state["next"]`; this maps it to a node name."""

from __future__ import annotations

from travel_agent.flight_agent.state import FlightAgentState

ROUTES: dict[str, tuple[str, ...]] = {
    "understand": ("validate", "general_query", "book_flight", "respond"),
    "validate": ("ask_user", "search_flights", "check_seats", "confirm", "lookup_booking", "respond"),
    "general_query": ("ask_user", "respond"),
    "search_flights": ("validate", "respond"),
    "check_seats": ("validate", "respond"),
    "lookup_booking": ("confirm", "validate", "respond"),
    "confirm": ("book_flight", "cancel_flight", "understand", "respond"),
    "book_flight": ("validate", "respond"),
    "cancel_flight": ("validate", "respond"),
}


def next_step(state: FlightAgentState) -> str:
    """Name of the node to run next (set by the node that just finished)."""
    return state["next"]
