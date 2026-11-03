"""Workflow image: written on every build, non-empty, and shows every node."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from travel_agent.hotel_agent.agent import HotelAgent
from travel_agent.hotel_agent.graph import GRAPH_NAME

NODES = [
    "receive", "understand", "validate", "ask_user", "general_query", "select_tool", "execute_tool",
    "process_result", "confirm", "respond",
]


def test_every_build_writes_all_artifacts(agent_settings):
    out = agent_settings.graph.artifacts_dir
    HotelAgent(agent_settings, load_model=False)  # save_artifacts defaults to True
    for suffix in ("mmd", "png", "txt"):
        path = out / f"{GRAPH_NAME}.{suffix}"
        assert path.exists() and path.stat().st_size > 0, path


def test_artifacts_mention_every_node(agent_settings):
    HotelAgent(agent_settings, load_model=False)
    out = agent_settings.graph.artifacts_dir
    mermaid, ascii_art = (out / f"{GRAPH_NAME}.mmd").read_text(), (out / f"{GRAPH_NAME}.txt").read_text()
    for node in NODES:
        assert node in mermaid and node in ascii_art


def test_png_is_a_real_image(agent_settings):
    HotelAgent(agent_settings, load_model=False)
    with Image.open(agent_settings.graph.artifacts_dir / f"{GRAPH_NAME}.png") as image:
        assert image.format == "PNG" and image.width > 300 and image.height > 300


def test_artifacts_are_refreshed_on_rebuild(agent_settings):
    HotelAgent(agent_settings, load_model=False)
    png = agent_settings.graph.artifacts_dir / f"{GRAPH_NAME}.png"
    png.write_bytes(b"stale")
    HotelAgent(agent_settings, load_model=False)
    assert png.read_bytes() != b"stale"


def test_unwritable_artifacts_dir_does_not_break_the_agent(agent_settings, tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    settings = replace(agent_settings, graph=replace(agent_settings.graph, artifacts_dir=blocker / "sub"))
    agent = HotelAgent(settings, load_model=False)
    assert agent.chat("hello", "t").message
