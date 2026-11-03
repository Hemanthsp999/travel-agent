"""The orchestrator-facing contract: `intent, status, data, error, requires_user_input, missing_parameters`."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from tests.flight_agent.conftest import CONTACT_LINE, PASSENGER_LINE, say
from tests.flight_agent.test_agent_flows import D, SECOND_PASSENGER, TRIP, reach_confirmation
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.schemas import ExecutionStatus as S, FlightTask, FlightTaskResult, Intent

CONTRACT_KEYS = {"intent", "status", "data", "error", "requires_user_input", "missing_parameters"}
VOCABULARY = {"SUCCESS", "FAILED", "PENDING", "REQUIRES_USER_INPUT", "UNKNOWN"}


def collect(agent: FlightAgent) -> list[FlightTaskResult]:
    """One conversation that touches every status and intent."""
    search = f"flights from Bangalore to Mumbai on {D} for 2 passengers"
    messages = [
        "hello", "I need a flight", search, "tell me about the first one", "find me a hotel in Goa", "asdf qwer",
        TRIP, "the second one", PASSENGER_LINE, SECOND_PASSENGER, CONTACT_LINE, "yes",
    ]
    results = [agent.chat(m, "contract") for m in messages]
    booking_id = results[-1].data["booking"]["booking_id"]
    for m in (f"status of {booking_id}", f"cancel {booking_id}", "yes", f"status of {booking_id}", "status of FB-NOPE00"):
        results.append(agent.chat(m, "contract"))
    return results


def test_every_result_honours_the_contract(agent: FlightAgent) -> None:
    results = collect(agent)
    assert {r.status for r in results} >= {S.SUCCESS, S.PENDING, S.REQUIRES_USER_INPUT, S.FAILED}
    for result in results:
        contract = result.contract()
        assert set(contract) == CONTRACT_KEYS
        assert contract["status"] in VOCABULARY
        json.dumps(contract)  # plain JSON for the orchestrator
        assert result.message.strip()


def test_input_flags_match_the_status(agent: FlightAgent) -> None:
    for result in collect(agent):
        waiting = result.status in (S.PENDING, S.REQUIRES_USER_INPUT)
        assert result.requires_user_input is waiting
        if result.status is S.REQUIRES_USER_INPUT:
            assert result.missing_parameters and result.question
        if result.status is S.PENDING:
            assert result.question and "yes" in result.question.lower()
        if not waiting:
            assert result.missing_parameters == [] and result.question is None


def test_failures_and_unknowns_carry_an_error(agent: FlightAgent) -> None:
    for result in collect(agent):
        if result.status in (S.FAILED, S.UNKNOWN):
            assert result.error and result.error.get("code") and result.error.get("message")
        if result.status is S.SUCCESS:
            assert result.error is None


def test_the_five_intents_are_reported(agent: FlightAgent) -> None:
    seen = {r.intent for r in collect(agent)}
    assert {
        Intent.CHECK_FLIGHT_AVAILABILITY, Intent.BOOK_FLIGHT, Intent.CANCEL_FLIGHT,
        Intent.GET_BOOKING_STATUS, Intent.GENERAL_FLIGHT_QUERY,
    } <= seen


def test_structured_tasks_can_preselect_the_intent_and_params(agent: FlightAgent) -> None:
    task = FlightTask(
        message="search", thread_id="task", intent=Intent.CHECK_FLIGHT_AVAILABILITY,
        params={"origin": "Bangalore", "destination": "Mumbai", "passenger_count": 1},
    )
    asked = agent.run_task(task)
    assert asked.status is S.REQUIRES_USER_INPUT and asked.missing_parameters == ["departure_date"]
    done = agent.run_task(FlightTask(message=D, thread_id="task"))
    assert done.status is S.SUCCESS and done.data["count"] == 3


def test_booking_data_is_taken_from_the_api(agent: FlightAgent, api) -> None:
    reach_confirmation(agent)
    done = say(agent, "yes")
    stored = api.get(f"/api/v1/bookings/{done.data['booking']['booking_id']}").json()["data"]
    for key in ("booking_id", "status", "total_price", "departure_date"):
        assert done.data["booking"][key] == stored[key]


def test_orchestrator_can_resume_a_paused_thread_with_run_task(agent: FlightAgent) -> None:
    first = agent.run_task(FlightTask(message="flights from Bangalore to Mumbai for 1 passenger", thread_id="o"))
    assert first.requires_user_input
    second = agent.run_task(FlightTask(message=D, thread_id="o"))
    assert second.status is S.SUCCESS


def test_an_empty_task_message_is_rejected_by_the_schema() -> None:
    with pytest.raises(Exception):
        FlightTask(message="")
