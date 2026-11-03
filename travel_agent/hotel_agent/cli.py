"""Interactive chat loop for the hotel agent. Run: `python -m travel_agent.hotel_agent`."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Callable
from dataclasses import replace

from travel_agent.config import get_settings
from travel_agent.hotel_agent.agent import HotelAgent
from travel_agent.hotel_agent.schemas import HotelTaskResult, TaskStatus
from travel_agent.memory.checkpointer import create_checkpointer

EXIT_WORDS = {"quit", "exit", "bye", "goodbye", ":q"}
GREETING = (
    "Hi! I'm your hotel assistant. I can check availability, find hotels within a budget, "
    "book a room, check a booking or cancel one. What can I help you with?"
)
GOODBYE = "Goodbye! Have a wonderful trip."


def run_chat(
    agent: HotelAgent,
    thread_id: str,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    show_progress: bool = False,
) -> None:
    """Endless loop: read a line, run the graph on the same thread, print the reply."""
    write(f"Agent: {GREETING}")
    while True:
        try:
            line = read("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            write(f"\nAgent: {GOODBYE}")
            return
        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            write(f"Agent: {GOODBYE}")
            return
        try:
            result = _turn(agent, line, thread_id, write if show_progress else None)
        except KeyboardInterrupt:
            write("\nAgent: Stopped that request. What would you like to do next?")
            continue
        except Exception:  # a failed turn must never end the conversation
            logging.getLogger(__name__).exception("Turn failed")
            write("Agent: Sorry, something went wrong on my side. Please try again.")
            continue
        write(f"Agent: {result.message}")
        if result.status is TaskStatus.UNVERIFIED:
            write("       (Say 'retry' to check again.)")


def _turn(agent: HotelAgent, line: str, thread_id: str, progress: Callable[[str], None] | None) -> HotelTaskResult:
    if progress is None:
        return agent.chat(line, thread_id)
    steps: list[str] = []
    try:
        for node in agent.stream_chat(line, thread_id):
            steps.append(node)
            progress(f"  ... {node}")
        return agent.last_result(thread_id)
    except Exception:
        logging.getLogger(__name__).exception("Streaming failed after %s", steps)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the hotel booking agent.")
    parser.add_argument("--thread", default=None, help="Conversation id; reuse it to resume a saved conversation.")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based understanding only (no model load).")
    parser.add_argument("--verbose", action="store_true", help="Show which graph node is running.")
    parser.add_argument("--debug", action="store_true", help="Show log messages.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.debug else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if args.no_llm:
        settings = replace(settings, agent=replace(settings.agent, use_llm=False))
    agent = HotelAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db))
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    run_chat(agent, thread_id, show_progress=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
