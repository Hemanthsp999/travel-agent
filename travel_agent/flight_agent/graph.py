"""Compile the flight agent graph and save its workflow image on every build."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from travel_agent.flight_agent.nodes.ask_user import make_ask_user
from travel_agent.flight_agent.nodes.book_flight import make_book_flight
from travel_agent.flight_agent.nodes.booking_ops import make_cancel_flight, make_lookup_booking
from travel_agent.flight_agent.nodes.confirm import make_confirm
from travel_agent.flight_agent.nodes.deps import AgentDeps
from travel_agent.flight_agent.nodes.general_query import make_general_query
from travel_agent.flight_agent.nodes.receive import make_receive
from travel_agent.flight_agent.nodes.respond import make_respond
from travel_agent.flight_agent.nodes.search import make_check_seats, make_search_flights
from travel_agent.flight_agent.nodes.understand import make_understand
from travel_agent.flight_agent.nodes.validate import make_validate
from travel_agent.flight_agent.routing import ROUTES, next_step
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.graph.visualize import save_graph_artifacts

GRAPH_NAME = "flight_agent_graph"


def build_flight_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None, save_artifacts: bool = True
) -> CompiledStateGraph:
    """Build and compile the graph; the workflow image is (re)written every time."""
    builder = StateGraph(FlightAgentState)
    nodes = {
        "receive": make_receive(deps),
        "understand": make_understand(deps),
        "validate": make_validate(deps),
        "ask_user": make_ask_user(deps),
        "general_query": make_general_query(deps),
        "search_flights": make_search_flights(deps),
        "check_seats": make_check_seats(deps),
        "lookup_booking": make_lookup_booking(deps),
        "confirm": make_confirm(deps),
        "book_flight": make_book_flight(deps),
        "cancel_flight": make_cancel_flight(deps),
        "respond": make_respond(deps),
    }
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "receive")
    builder.add_edge("receive", "understand")
    for source, targets in ROUTES.items():
        builder.add_conditional_edges(source, next_step, {t: t for t in targets})
    builder.add_edge("ask_user", "understand")
    builder.add_edge("respond", END)

    compiled = builder.compile(checkpointer=checkpointer)
    if save_artifacts:
        settings = deps.settings.graph
        save_graph_artifacts(compiled, GRAPH_NAME, settings.artifacts_dir, settings.remote_png)
    return compiled
