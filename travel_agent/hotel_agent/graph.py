"""Compile the hotel agent graph and save its workflow image on every build."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from travel_agent.graph.visualize import save_graph_artifacts
from travel_agent.hotel_agent.nodes.ask_user import make_ask_user
from travel_agent.hotel_agent.nodes.confirm import make_confirm
from travel_agent.hotel_agent.nodes.deps import AgentDeps
from travel_agent.hotel_agent.nodes.execute_tool import make_execute_tool
from travel_agent.hotel_agent.nodes.general import make_general_query
from travel_agent.hotel_agent.nodes.process_result import make_process_result
from travel_agent.hotel_agent.nodes.receive import make_receive
from travel_agent.hotel_agent.nodes.respond import make_respond
from travel_agent.hotel_agent.nodes.select_tool import make_select_tool
from travel_agent.hotel_agent.nodes.understand import make_understand
from travel_agent.hotel_agent.nodes.validate import make_validate
from travel_agent.hotel_agent.routing import ROUTES, next_step
from travel_agent.hotel_agent.state import HotelAgentState

GRAPH_NAME = "hotel_agent_graph"


def build_hotel_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None, save_artifacts: bool = True
) -> CompiledStateGraph:
    """Build and compile the graph; the workflow image is (re)written every time."""
    builder = StateGraph(HotelAgentState)
    nodes = {
        "receive": make_receive(deps),
        "understand": make_understand(deps),
        "validate": make_validate(deps),
        "ask_user": make_ask_user(deps),
        "general_query": make_general_query(deps),
        "select_tool": make_select_tool(deps),
        "execute_tool": make_execute_tool(deps),
        "process_result": make_process_result(deps),
        "confirm": make_confirm(deps),
        "respond": make_respond(deps),
    }
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "receive")
    builder.add_edge("receive", "understand")
    for source, targets in ROUTES.items():
        builder.add_conditional_edges(source, next_step, {t: t for t in targets})
    builder.add_edge("ask_user", "understand")
    builder.add_edge("select_tool", "execute_tool")
    builder.add_edge("execute_tool", "process_result")
    builder.add_edge("respond", END)

    compiled = builder.compile(checkpointer=checkpointer)
    if save_artifacts:
        settings = deps.settings.graph
        save_graph_artifacts(compiled, GRAPH_NAME, settings.artifacts_dir, settings.remote_png)
    return compiled
