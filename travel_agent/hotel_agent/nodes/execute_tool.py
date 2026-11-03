"""Execute-tool node: run the selected API tool and store its compact result."""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.nodes.deps import AgentDeps
from travel_agent.hotel_agent.state import HotelAgentState
from travel_agent.hotel_agent.tools import run_tool


def make_execute_tool(deps: AgentDeps):
    def execute_tool(state: HotelAgentState) -> dict[str, Any]:
        call = state["tool_call"]
        result = run_tool(deps.tools, call["name"], call["args"])
        return {"api_result": result.model_dump(mode="json")}

    return execute_tool
