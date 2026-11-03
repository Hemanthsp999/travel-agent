"""Interactive command line for the hotel and flight supervisor."""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Callable
from dataclasses import replace

from travel_agent.config import get_settings
from travel_agent.flight_agent import FlightAgent
from travel_agent.hotel_agent import HotelAgent
from travel_agent.memory.checkpointer import create_checkpointer
from travel_agent.orchestrator.graph import Orchestrator

EXIT_WORDS = {"quit", "exit", "bye", "goodbye", ":q"}
GREETING = (
    "Hi! I can help you search, book, or cancel hotels and flights. "
    "Tell me what you're planning, and I'll ask for any details I need."
)
GOODBYE = "Goodbye! Have a wonderful trip."


def run_chat(
    agent: Orchestrator,
    thread_id: str,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    """Keep the same checkpointed conversation active until the user exits."""
    write(f"Assistant: {GREETING}")
    while True:
        try:
            line = read("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            write(f"\nAssistant: {GOODBYE}")
            return
        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            write(f"Assistant: {GOODBYE}")
            return
        try:
            result = agent.chat(line, thread_id)
            write(f"Assistant: {result.get('final_response') or 'Sorry, I could not complete that request.'}")
        except KeyboardInterrupt:
            write("\nAssistant: I stopped that request. What would you like to do next?")
        except Exception:
            logging.getLogger(__name__).exception("Orchestrator turn failed")
            write("Assistant: Sorry, something went wrong. Could you try that again?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the travel booking orchestrator.")
    parser.add_argument("--thread", help="Conversation id to resume; a new id is created by default.")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based understanding only.")
    parser.add_argument("--debug", action="store_true", help="Show error logs.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.debug else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    if args.no_llm:
        settings = replace(settings, agent=replace(settings.agent, use_llm=False))
    # Each graph owns its saver but uses the configured persistent checkpoint database.
    hotel = HotelAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db))
    flight = FlightAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db))
    supervisor = Orchestrator(hotel, flight, checkpointer=create_checkpointer(settings.checkpoint_db))
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    print(f"Conversation thread: {thread_id}")
    run_chat(supervisor, thread_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
