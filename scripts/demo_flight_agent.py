"""Scripted multi-turn demo of the flight agent with the real model and the running Flight API.

Start the API first:  scripts/run_mocks.sh --only flight
Run:                  python scripts/demo_flight_agent.py [--no-llm]
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from travel_agent.config import get_settings  # noqa: E402
from travel_agent.flight_agent import FlightAgent  # noqa: E402
from travel_agent.flight_agent.schemas import ExecutionStatus  # noqa: E402


def main() -> None:
    settings = get_settings()
    if "--no-llm" in sys.argv:
        settings = replace(settings, agent=replace(settings.agent, use_llm=False))
    agent = FlightAgent(settings)
    when = (date.today() + timedelta(days=8)).strftime("%d %b").lstrip("0")
    later = (date.today() + timedelta(days=9)).strftime("%d %b").lstrip("0")
    script = [
        ("availability, details missing", ["I need a flight", "Bangalore", "Mumbai", when, "2 passengers"]),
        ("other route and details", [f"any flights from Delhi to Mumbai on {later} for 1 person?", "tell me about the first one", "compare them"]),
        ("book with confirmation", [
            f"book a flight from Bangalore to Mumbai on {when} for 2 passengers", "the second one",
            "Asha Rao, 31, female", "Ravi Rao, 35, male", "asha@example.com +919876543210", "yes",
        ]),
        ("status", ["status of {booking_id}"]),
        ("cancel with confirmation", ["cancel {booking_id}", "yes"]),
        ("unsupported / no hallucination", ["what is the baggage allowance and can I get a refund?", "find me a hotel in Goa"]),
        ("unserved route", [f"flights from Delhi to Chennai on {when} for 1 person"]),
    ]
    booking_id = ""
    for title, turns in script:
        print(f"\n=== {title} ===")
        thread = "demo-" + title.split()[0]
        for line in turns:
            line = line.replace("{booking_id}", booking_id)
            started = time.perf_counter()
            result = agent.chat(line, thread)
            print(f"You:   {line}\nAgent: {result.message}\n       [{result.status.value} | {time.perf_counter() - started:.1f}s]")
            if (booking := result.data.get("booking")) and result.status is ExecutionStatus.SUCCESS:
                booking_id = booking["booking_id"]


if __name__ == "__main__":
    main()
