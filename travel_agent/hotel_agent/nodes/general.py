"""General-query node: help, hotel details, comparison, 'show all' and unsupported requests."""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome, shown_hotels
from travel_agent.hotel_agent.nodes.validate import gave_up
from travel_agent.hotel_agent.schemas import GeneralTopic
from travel_agent.hotel_agent.state import HotelAgentState

NO_SEARCH = (
    "I don't have a hotel to work with yet. Tell me a hotel ID like HT-1001, or search a city and dates "
    "first and I'll show you options."
)


def make_general_query(deps: AgentDeps):
    max_attempts = deps.settings.agent.max_ask_attempts

    def general_query(state: HotelAgentState) -> dict[str, Any]:
        topic = GeneralTopic(state.get("general_topic") or "NONE")
        params = dict(state.get("params") or {})
        search = state.get("last_search") or {}
        hotels, shown = search.get("hotels", []), shown_hotels(state)

        def reply(kind: str, data: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
            return {"outcome": make_outcome(kind, data, **fields), "next": "respond"}

        if topic is GeneralTopic.HELP:
            return reply("help")
        if topic is GeneralTopic.SHOW_MORE:
            if not hotels:
                return reply("no_context", message=NO_SEARCH)
            everything = {**search, "shown_ids": [h["hotel_id"] for h in hotels]}
            data = {"query": search["query"], "hotels": hotels}
            return {**reply("show_all", data, query=search["query"], hotels=hotels), "last_search": everything}
        if topic is GeneralTopic.COMPARE_HOTELS:
            if len(shown) < 2:
                return reply("no_context", message="I need at least two hotels from a search to compare. Tell me a city and dates first.")
            return reply("comparison", {"hotels": shown}, hotels=shown)
        if topic is GeneralTopic.HOTEL_DETAILS:
            hotel_id = params.get("hotel_id") or (state.get("selected_hotel") or {}).get("hotel_id")
            if not hotel_id and len(shown) == 1:
                hotel_id = shown[0]["hotel_id"]
            if hotel_id:
                return {"params": {**params, "hotel_id": hotel_id}, "next": "select_tool"}
            if not shown:
                return reply("no_context", message=NO_SEARCH)
            if gave_up(state, ["hotel"], max_attempts):
                return reply("gave_up", missing=["hotel"])
            return {"missing_params": ["hotel"], "next": "ask_user"}
        return reply("unsupported")

    return general_query
