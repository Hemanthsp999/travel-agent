"""Booking lookup and cancellation nodes."""

from __future__ import annotations

from typing import Any

from travel_agent.flight_agent.nodes.common import failure_update
from travel_agent.flight_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.flight_agent.responses import cancel_prompt
from travel_agent.flight_agent.schemas import Intent
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.flight_agent.tools import run_tool


def _remember(state: FlightAgentState, booking: dict[str, Any]) -> list[str]:
    known = list(state.get("known_bookings") or [])
    if booking["booking_id"] not in known:
        known.append(booking["booking_id"])
    return known


def make_lookup_booking(deps: AgentDeps):
    def lookup_booking(state: FlightAgentState) -> dict[str, Any]:
        result = run_tool(deps.tools, "get_booking_status", {"booking_id": state["booking_id"]})
        if not result.ok:
            return {**failure_update(state, "lookup", result), "api_execution_status": "FAILED"}
        booking = result.data or {}
        update: dict[str, Any] = {"known_bookings": _remember(state, booking), "api_execution_status": "SUCCESS",
                                  "missing_parameters": []}
        if state.get("identified_intent") == Intent.CANCEL_FLIGHT.value:
            if booking["status"] == "CANCELLED":
                outcome = make_outcome("already_cancelled", {"booking": booking}, booking_id=booking["booking_id"])
                return {**update, "outcome": outcome, "next": "respond"}
            pending = {
                "type": "cancel", "booking_id": booking["booking_id"], "booking": booking,
                "quote": {"booking_id": booking["booking_id"], "status": booking["status"],
                          "total_price": booking.get("total_price"), "currency": booking.get("currency")},
                "prompt": cancel_prompt(booking), "intent": Intent.CANCEL_FLIGHT.value,
            }
            return {**update, "pending_action": pending, "next": "confirm"}
        outcome = make_outcome("booking_status", {"booking": booking}, booking=booking)
        return {**update, "booking_details": booking, "outcome": outcome, "next": "respond"}

    return lookup_booking


def make_cancel_flight(deps: AgentDeps):
    def cancel_flight(state: FlightAgentState) -> dict[str, Any]:
        pending = state["pending_action"]
        booking_id = pending["booking_id"]
        result = run_tool(deps.tools, "cancel_flight_booking", {"booking_id": booking_id})

        if result.ok:
            booking = result.data or {}
            if booking.get("status") == "CANCELLED":
                return _cancelled(state, booking, verified=False)
            message = f"The flight system replied, but booking {booking_id} shows status {booking.get('status')}. I can't confirm the cancellation."
            return {"outcome": make_outcome("cancel_unverified", {"booking": booking}, message=message,
                                            error={"code": "CANCELLATION_MISMATCH", "message": message}),
                    "pending_action": None, "api_execution_status": "UNKNOWN", "next": "respond"}

        if result.error and result.error.outcome_unknown:
            return _reconcile(deps, state, booking_id, result.error.model_dump())
        return {**failure_update(state, "cancel", result), "api_execution_status": "FAILED"}

    return cancel_flight


def _cancelled(state: FlightAgentState, booking: dict[str, Any], verified: bool) -> dict[str, Any]:
    signatures = {k: v for k, v in (state.get("booking_signatures") or {}).items() if v != booking["booking_id"]}
    return {
        "outcome": make_outcome("cancel_success", {"booking": booking}, booking=booking, verified_by_lookup=verified),
        "cancellation_details": booking, "known_bookings": _remember(state, booking), "booking_signatures": signatures,
        "pending_action": None, "api_execution_status": "SUCCESS", "next": "respond",
    }


def _reconcile(deps: AgentDeps, state: FlightAgentState, booking_id: str, error: dict[str, Any]) -> dict[str, Any]:
    """The cancel reply was lost: read the booking back instead of guessing."""
    look = run_tool(deps.tools, "get_booking_status", {"booking_id": booking_id})
    if look.ok and (look.data or {}).get("status") == "CANCELLED":
        return _cancelled(state, look.data or {}, verified=True)
    if look.ok:
        message = (
            f"The cancellation did not go through: booking {booking_id} is still {look.data.get('status')}. "
            "Nothing was changed. Do you want me to try again?"
        )
        return {"outcome": make_outcome("cancel_failed", {"booking": look.data}, error=error, message=message),
                "pending_action": None, "api_execution_status": "FAILED", "next": "respond"}
    message = (
        f"I asked to cancel booking {booking_id} but couldn't confirm whether it worked, and I can't reach the flight "
        'system to check. I haven\'t assumed either way. Ask me for the status of this booking in a moment.'
    )
    return {"outcome": make_outcome("cancel_unverified", error=error, message=message),
            "pending_action": None, "api_execution_status": "UNKNOWN", "next": "respond"}
