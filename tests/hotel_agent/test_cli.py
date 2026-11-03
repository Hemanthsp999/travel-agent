"""CLI loop: never crashes, exits cleanly, ignores empty input."""

from __future__ import annotations

from tests.hotel_agent.conftest import day
from travel_agent.hotel_agent.cli import GOODBYE, run_chat


def drive(agent, lines, **kwargs):
    feed = iter(lines)
    out: list[str] = []

    def read(_prompt: str) -> str:
        try:
            item = next(feed)
        except StopIteration:
            raise EOFError
        if isinstance(item, BaseException):
            raise item
        return item

    run_chat(agent, "cli-test", read=read, write=out.append, **kwargs)
    return out


def test_quit_says_goodbye(agent):
    assert drive(agent, ["quit"])[-1] == f"Agent: {GOODBYE}"


def test_ctrl_c_and_eof_exit_cleanly(agent):
    assert GOODBYE in drive(agent, [KeyboardInterrupt()])[-1]
    assert GOODBYE in drive(agent, [])[-1]


def test_empty_input_is_ignored_and_conversation_continues(agent):
    out = drive(agent, ["", "   ", f"hotels in Pune {day(20)} to {day(22)}", "exit"])
    assert sum(line.startswith("Agent:") for line in out) == 3  # greeting, answer, goodbye
    assert "Pune" in out[1]


def test_same_thread_across_turns(agent):
    out = drive(agent, ["I need a hotel", "Pune", f"{day(20)} to {day(22)}", "quit"])
    assert "Pune" in out[3]


def test_internal_failure_does_not_end_the_loop(agent, monkeypatch):
    calls = {"n": 0}
    original = agent.chat

    def flaky(message, thread_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(message, thread_id)

    monkeypatch.setattr(agent, "chat", flaky)
    out = drive(agent, ["hello", "hello", "quit"])
    assert any("something went wrong" in line for line in out) and "boom" not in "".join(out)
    assert any("hotel assistant" in line for line in out[2:])


def test_progress_mode_shows_node_names(agent):
    out = drive(agent, ["hello", "quit"], show_progress=True)
    assert any(line.strip().startswith("... understand") for line in out)
