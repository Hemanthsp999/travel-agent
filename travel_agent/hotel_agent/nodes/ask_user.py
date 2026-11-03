"""Ask-user node: ask one natural question and pause the graph until the user answers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from travel_agent.hotel_agent.nodes.deps import AgentDeps
from travel_agent.hotel_agent.params import ask_group
from travel_agent.hotel_agent.responses import ask_text
from travel_agent.hotel_agent.state import HotelAgentState


def make_ask_user(deps: AgentDeps):
    def ask_user(state: HotelAgentState) -> dict[str, Any]:
        missing = state.get("missing_params") or []
        group = ask_group(missing)
        question = ask_text(
            group, state.get("params") or {}, state.get("validation_note"),
            (state.get("ask_counts") or {}).get(group[0], 0), state.get("intent", ""),
        )
        answer = interrupt({
            "kind": "input", "question": question, "missing_params": missing,
            "intent": state.get("intent"), "data": {},
        })
        counts = dict(state.get("ask_counts") or {})
        counts[group[0]] = counts.get(group[0], 0) + 1
        return {
            "messages": [AIMessage(content=question), HumanMessage(content=str(answer))],
            "awaiting": group[0],
            "ask_counts": counts,
            "validation_note": None,
        }

    return ask_user
