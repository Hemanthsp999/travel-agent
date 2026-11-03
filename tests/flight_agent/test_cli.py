"""The interactive loop: never crashes, one thread, friendly exits."""

from __future__ import annotations

from collections.abc import Callable

from tests.flight_agent.test_agent_flows import D
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.cli import GOODBYE, GREETING, run_chat


def run(agent: FlightAgent, inputs: list) -> list[str]:
    """Feed `inputs` (strings, or exceptions to raise) to the loop and return everything it printed."""
    queue = list(inputs)
    printed: list[str] = []

    def read(_prompt: str) -> str:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    run_chat(agent, "cli-test", read=read, write=printed.append)
    return printed


def test_greeting_then_goodbye(agent: FlightAgent) -> None:
    out = run(agent, ["quit"])
    assert out == [f"Agent: {GREETING}", f"Agent: {GOODBYE}"]


def test_empty_input_is_skipped_and_the_conversation_continues(agent: FlightAgent) -> None:
    out = run(agent, ["", "   ", f"flights from Bangalore to Mumbai on {D} for 1 passenger", "EXIT"])
    assert len(out) == 3 and "6E-501" in out[1] and out[2].endswith(GOODBYE)


def test_follow_up_answers_use_the_same_thread(agent: FlightAgent) -> None:
    out = run(agent, ["flights from Bangalore to Mumbai for 2 passengers", D, "bye"])
    assert "What date" in out[1] and "6E-501" in out[2]


def test_eof_and_ctrl_c_end_politely(agent: FlightAgent) -> None:
    assert run(agent, [EOFError()])[-1].endswith(GOODBYE)
    assert run(agent, [KeyboardInterrupt()])[-1].endswith(GOODBYE)


def test_a_crashing_turn_keeps_the_loop_alive(agent: FlightAgent, monkeypatch) -> None:
    calls = {"n": 0}
    original = agent.chat

    def flaky(message: str, thread_id: str = "default"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(message, thread_id)

    monkeypatch.setattr(agent, "chat", flaky)
    out = run(agent, ["hello", "hello", "quit"])
    assert "something went wrong" in out[1] and "boom" not in out[1]
    assert out[2].startswith("Agent:") and out[-1].endswith(GOODBYE)


def test_interrupting_a_request_returns_to_the_prompt(agent: FlightAgent, monkeypatch) -> None:
    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent, "chat", interrupted)
    out = run(agent, ["hello", "quit"])
    assert "Stopped that request" in out[1] and out[-1].endswith(GOODBYE)


def test_progress_mode_prints_node_names(agent: FlightAgent) -> None:
    printed: list[str] = []
    queue = [f"flights from Bangalore to Mumbai on {D} for 1 passenger", "quit"]
    run_chat(agent, "p", read=lambda _p: queue.pop(0), write=printed.append, show_progress=True)
    assert any("search_flights" in line for line in printed)
