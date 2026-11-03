"""Failure handling: outages, timeouts, lost replies, and never claiming what the API did not confirm."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from tests.flight_agent.conftest import api_booking, say
from tests.flight_agent.test_agent_flows import D, SECOND_PASSENGER, TRIP, reach_confirmation, writes
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.schemas import ExecutionStatus as S

SEARCH = f"flights from Bangalore to Mumbai on {D} for 2 passengers"


def all_bookings(api: TestClient) -> list[dict]:
    """Bookings that exist in the API store (read straight from the database file)."""
    from mock_apis.common.db import connect

    conn = connect(api.app.state.settings.db_path)  # type: ignore[attr-defined]
    try:
        return [dict(row) for row in conn.execute("SELECT booking_id, status FROM bookings")]
    finally:
        conn.close()


@pytest.mark.parametrize("failure", ["connect", "timeout", "500", "garbage"])
def test_search_failures_never_produce_flights(make_agent: Callable[..., FlightAgent], failure: str) -> None:
    agent = make_agent(default=failure)
    result = say(agent, SEARCH)
    assert result.status is S.FAILED and result.error is not None
    assert not result.data.get("flights")
    assert "FL-10" not in result.message


def test_safe_search_calls_are_retried(make_agent: Callable[..., FlightAgent]) -> None:
    agent = make_agent(script=["timeout"])
    result = say(agent, SEARCH)
    assert result.status is S.SUCCESS and result.data["count"] == 3
    assert len(agent.http.requests) == 2  # type: ignore[attr-defined]


def test_search_gives_up_after_the_configured_attempts(make_agent: Callable[..., FlightAgent]) -> None:
    agent = make_agent(default="500")
    say(agent, SEARCH)
    assert len(agent.http.requests) == agent.settings.flight_api.max_attempts  # type: ignore[attr-defined]


def test_booking_with_a_lost_reply_is_reconciled_through_the_idempotent_retry(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    agent.http.script[:] = ["send_timeout"]  # type: ignore[attr-defined]  # first POST applied, reply lost
    done = say(agent, "yes")
    assert done.status is S.SUCCESS
    assert len(all_bookings(api)) == 1
    assert api_booking(api, done.data["booking"]["booking_id"])["status"] == "CONFIRMED"


def test_booking_with_every_reply_lost_is_unknown_and_a_retry_never_duplicates(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    agent.http.script[:] = ["send_timeout"] * 5  # type: ignore[attr-defined]
    unknown = say(agent, "yes")
    assert unknown.status is S.UNKNOWN and unknown.error is not None
    assert "booked" not in unknown.message.lower().replace("not booked", "").replace("isn't booked", "") or "?" in unknown.message
    assert len(all_bookings(api)) == 1  # the server did apply it exactly once

    agent.http.script[:] = []  # type: ignore[attr-defined]
    retried = say(agent, "retry")
    assert retried.status is S.SUCCESS
    assert len(all_bookings(api)) == 1
    assert retried.data["booking"]["booking_id"] == all_bookings(api)[0]["booking_id"]


def test_booking_when_the_api_is_down_fails_cleanly_and_can_be_retried(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    agent.http.script[:] = ["connect"] * 5  # type: ignore[attr-defined]
    failed = say(agent, "yes")
    assert failed.status in (S.FAILED, S.UNKNOWN) and failed.error is not None
    assert all_bookings(api) == []
    agent.http.script[:] = []  # type: ignore[attr-defined]
    retried = say(agent, "try again")
    assert retried.status is S.SUCCESS and len(all_bookings(api)) == 1


def test_a_server_error_on_booking_is_never_reported_as_success(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    agent.http.script[:] = ["500"] * 5  # type: ignore[attr-defined]
    result = say(agent, "yes")
    assert result.status is not S.SUCCESS and "booking_id" not in result.data.get("booking", {})
    assert all_bookings(api) == []


def test_cancel_with_a_lost_reply_is_verified_by_looking_the_booking_up(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    booked = say(agent, "yes")
    booking_id = booked.data["booking"]["booking_id"]
    say(agent, f"cancel booking {booking_id}")
    agent.http.script[:] = ["send_timeout"]  # type: ignore[attr-defined]
    result = say(agent, "yes")
    assert result.status is S.SUCCESS and result.data["booking"]["status"] == "CANCELLED"
    assert api_booking(api, booking_id)["status"] == "CANCELLED"


def test_cancel_whose_outcome_cannot_be_verified_is_unknown(
    make_agent: Callable[..., FlightAgent], api: TestClient
) -> None:
    agent = make_agent()
    reach_confirmation(agent)
    booking_id = say(agent, "yes").data["booking"]["booking_id"]
    say(agent, f"cancel booking {booking_id}")
    agent.http.script[:] = ["send_timeout", "timeout", "timeout", "timeout"]  # type: ignore[attr-defined]
    result = say(agent, "yes")
    assert result.status is S.UNKNOWN and result.outcome == "cancel_unverified" and result.error is not None


def test_status_lookup_failure_does_not_invent_a_status(make_agent: Callable[..., FlightAgent]) -> None:
    agent = make_agent(default="500")
    result = say(agent, "status of FB-ABC123")
    assert result.status is S.FAILED and "CONFIRMED" not in result.message and not result.data.get("booking")


def test_unexpected_node_crash_becomes_a_failed_result_not_an_exception(
    make_agent: Callable[..., FlightAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    agent = make_agent()
    monkeypatch.setattr(agent.graph, "stream", boom)
    result = say(agent, SEARCH)
    assert result.status is S.FAILED and result.error["code"] == "AGENT_ERROR"
    assert "kaboom" not in result.message
    assert not result.requires_user_input
