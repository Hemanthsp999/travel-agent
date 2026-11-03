"""Graph structure and the workflow image written on every build."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.flight_agent.conftest import FakeLLM, FlakyHTTP
from travel_agent.config import Settings
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.api_client import FlightAPIClient
from travel_agent.flight_agent.graph import GRAPH_NAME
from travel_agent.flight_agent.routing import ROUTES

NODES = {
    "receive", "understand", "validate", "ask_user", "general_query", "search_flights", "check_seats",
    "lookup_booking", "confirm", "book_flight", "cancel_flight", "respond",
}


@pytest.fixture
def built(api, agent_settings: Settings) -> FlightAgent:
    return FlightAgent(
        agent_settings, llm=None, client=FlightAPIClient(agent_settings.flight_api, http=FlakyHTTP(api)),  # type: ignore[arg-type]
        load_model=False, save_artifacts=True,
    )


def test_every_route_target_is_a_node() -> None:
    for source, targets in ROUTES.items():
        assert source in NODES
        assert set(targets) <= NODES | {"ask_user"}, (source, targets)


def test_graph_contains_all_nodes(built: FlightAgent) -> None:
    drawn = set(built.graph.get_graph().nodes) - {"__start__", "__end__"}
    assert drawn == NODES


def test_building_the_graph_writes_non_empty_artifacts(built: FlightAgent, agent_settings: Settings) -> None:
    out = agent_settings.graph.artifacts_dir
    for suffix in ("mmd", "png", "txt"):
        path = Path(out) / f"{GRAPH_NAME}.{suffix}"
        assert path.exists() and path.stat().st_size > 0, path
    source = (Path(out) / f"{GRAPH_NAME}.mmd").read_text()
    assert all(node in source for node in NODES)
    assert (Path(out) / f"{GRAPH_NAME}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_artifact_failures_do_not_break_the_agent(api, agent_settings: Settings, tmp_path: Path) -> None:
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("x")
    settings = replace(agent_settings, graph=replace(agent_settings.graph, artifacts_dir=blocked / "sub"))
    agent = FlightAgent(
        settings, llm=FakeLLM(None), client=FlightAPIClient(settings.flight_api, http=FlakyHTTP(api)),  # type: ignore[arg-type]
        load_model=False, save_artifacts=True,
    )
    assert agent.chat("hello", "t").message
