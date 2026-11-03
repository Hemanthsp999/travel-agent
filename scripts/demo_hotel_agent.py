"""Scripted multi-turn demo of the hotel agent with the real model and the running Hotel API.

Start the API first:  scripts/run_mocks.sh --only hotel
Run:                  python scripts/demo_hotel_agent.py [--no-llm]
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

from travel_agent.config import get_settings  # noqa: E402
from travel_agent.hotel_agent import HotelAgent  # noqa: E402
from travel_agent.hotel_agent.schemas import TaskStatus  # noqa: E402


def main() -> None:
    settings = get_settings()
    if "--no-llm" in sys.argv:
        settings = replace(settings, agent=replace(settings.agent, use_llm=False))
    agent = HotelAgent(settings)
    d = lambda n: (date.today() + timedelta(days=n)).isoformat()  # noqa: E731
    script = [
        ("availability, details missing", ["I need a hotel", "Pune", f"{d(30)} to {d(32)}"]),
        ("budget search", [f"Show me hotels in Bangalore from {d(30)} to {d(32)} under 6000 for 2 guests"]),
        ("book with confirmation", ["book the second one", "1 room", "Asha Rao, asha@example.com, 9876543210", "yes"]),
        ("status", ["what's my booking status?"]),
        ("general query", [f"hotels in Mumbai {d(30)} to {d(32)}", "compare them", "tell me about the cheapest one"]),
        ("cancel with confirmation", ["cancel my last booking"]),
        ("out of scope / no hallucination", ["what is the weather in Goa and do you have a pool at HT-1001?"]),
    ]
    booking_id = ""
    for title, turns in script:
        print(f"\n=== {title} ===")
        thread = "demo" if title != "out of scope / no hallucination" else "demo-oos"
        for line in turns:
            if title == "status":
                line = f"status of {booking_id}"
            if title.startswith("cancel"):
                line = f"cancel {booking_id}"
            started = time.perf_counter()
            result = agent.chat(line, thread)
            print(f"You:   {line}\nAgent: {result.message}\n       [{result.status.value} | {time.perf_counter() - started:.1f}s]")
            if result.data.get("booking"):
                booking_id = result.data["booking"]["booking_id"]
            if title.startswith("cancel") and result.status is TaskStatus.NEEDS_CONFIRMATION:
                result = agent.chat("yes", thread)
                print(f"You:   yes\nAgent: {result.message}\n       [{result.status.value}]")


if __name__ == "__main__":
    main()
