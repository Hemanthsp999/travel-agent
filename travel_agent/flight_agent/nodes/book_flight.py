"""Booking node: create the booking with an idempotency key and verify the API's answer before reporting success."""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.nodes.common import RETRY_HINT, failure_update
from travel_agent.flight_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.flight_agent.tools import run_tool

_UNKNOWN_MESSAGE = (
    "I sent your booking but didn't get a clear answer from the flight system, so I can't tell whether it went "
    "through. I haven't assumed either way. Retrying is safe because it reuses the same request key, so it "
    "cannot create a duplicate."
)


def booking_problems(booking: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Differences between what we asked for and what the API says was booked."""
    problems: list[str] = []
    if booking.get("status") != "CONFIRMED":
        problems.append(f"status is {booking.get('status')}")
    if (booking.get("flight") or {}).get("flight_id") != payload["flight_id"]:
        problems.append("a different flight")
    if booking.get("departure_date") and booking["departure_date"] != payload["departure_date"]:
        problems.append("a different date")
    count = booking.get("num_passengers") or len(booking.get("passengers") or [])
    if count and count != len(payload["passengers"]):
        problems.append("a different number of passengers")
    return problems


def make_book_flight(deps: AgentDeps):
    def book_flight(state: FlightAgentState) -> dict[str, Any]:
        pending = state["pending_action"]
        payload = pending["payload"]
        result = run_tool(deps.tools, "book_flight", {**payload, "idempotency_key": pending["idempotency_key"]})

        if result.ok:
            booking = result.data or {}
            problems = booking_problems(booking, payload)
            if problems:
                message = (
                    f"The flight system answered, but its record doesn't match my request ({'; '.join(problems)}). "
                    f"I can't confirm the booking is correct. Please check booking {booking.get('booking_id', '(no ID returned)')}."
                )
                outcome = make_outcome("booking_unverified", {"booking": booking}, message=message,
                                       error={"code": "BOOKING_MISMATCH", "message": message})
                return {"outcome": outcome, "pending_action": None, "api_execution_status": "UNKNOWN", "next": "respond"}
            known = [*(state.get("known_bookings") or [])]
            if booking["booking_id"] not in known:
                known.append(booking["booking_id"])
            return {
                "outcome": make_outcome("booking_success", {"booking": booking}, booking=booking),
                "booking_details": booking, "known_bookings": known,
                "booking_signatures": {**(state.get("booking_signatures") or {}), pending["signature"]: booking["booking_id"]},
                "pending_action": None, "retryable_action": None, "api_execution_status": "SUCCESS",
                "selected_flight": None, "verified_key": None, "origin": None, "destination": None, "departure_date": None,
                "next": "respond",
            }

        if result.error and result.error.outcome_unknown:
            error = result.error.model_dump()
            message = _UNKNOWN_MESSAGE + RETRY_HINT
            return {
                "outcome": make_outcome("booking_unverified", error=error, message=message),
                "retryable_action": pending, "pending_action": None, "api_execution_status": "UNKNOWN", "next": "respond",
            }
        return {**failure_update(state, "book", result), "api_execution_status": "FAILED"}

    return book_flight
