"""Entry node: reset per-request fields and trim old messages."""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.nodes.deps import AgentDeps
from travel_agent.hotel_agent.params import TASK_SCOPED
from travel_agent.hotel_agent.state import HotelAgentState
from travel_agent.memory.short_term import prune_messages


def make_receive(deps: AgentDeps):
    keep = deps.settings.agent.history_keep_messages

    def receive(state: HotelAgentState) -> dict[str, Any]:
        params = {k: v for k, v in (state.get("params") or {}).items() if k not in TASK_SCOPED}
        return {
            "params": params,
            "intent": "",
            "general_topic": "NONE",
            "missing_params": [],
            "awaiting": None,
            "ask_counts": {},
            "validation_note": None,
            "pending_action": None,
            "confirmation": None,
            "tool_call": None,
            "api_result": None,
            "stage": "",
            "force_stage": None,
            "invite_booking": False,
            "next": "understand",
            "outcome": {},
            "result": {},
            "messages": prune_messages(state["messages"], keep),
        }

    return receive
