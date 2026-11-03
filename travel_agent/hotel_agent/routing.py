"""Conditional-edge helpers: every node writes `next`; edges only map it to a node name."""

from __future__ import annotations

from travel_agent.hotel_agent.state import HotelAgentState

# node -> the set of nodes it may route to
ROUTES: dict[str, tuple[str, ...]] = {
    "understand": ("validate", "general_query", "select_tool", "respond"),
    "validate": ("ask_user", "select_tool", "respond"),
    "general_query": ("select_tool", "ask_user", "respond"),
    "process_result": ("respond", "confirm", "validate", "general_query", "select_tool"),
    "confirm": ("select_tool", "respond", "understand"),
}


def next_step(state: HotelAgentState) -> str:
    """Name of the node chosen by the previous node."""
    return state["next"]
