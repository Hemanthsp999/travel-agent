"""The four API tools: validation, idempotency and error mapping, with no LLM involved."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.flight_agent.conftest import FlakyHTTP, book_via_api, day
from travel_agent.config import FlightAPISettings
from travel_agent.flight_agent.api_client import FlightAPIClient
from travel_agent.flight_agent.tools import build_flight_tools, run_tool

PASSENGERS = [{"name": "Asha Rao", "age": 31, "gender": "FEMALE"}]


@pytest.fixture
def tools(api: TestClient):
    settings = FlightAPISettings(retry_backoff_s=0.0)
    http = FlakyHTTP(api)
    return build_flight_tools(FlightAPIClient(settings, http=http), settings), http  # type: ignore[arg-type]


def booking_args(key: str = "key-0000001", flight_id: str = "FL-1001", **overrides):
    args = {
        "flight_id": flight_id, "departure_date": day(20), "passengers": PASSENGERS,
        "contact_email": "asha@example.com", "contact_phone": "+919876543210", "idempotency_key": key,
    }
    return {**args, **overrides}


def test_the_agent_has_exactly_the_tools_the_api_supports(tools) -> None:
    assert set(tools[0]) == {"check_flight_availability", "book_flight", "get_booking_status", "cancel_flight_booking"}


def test_availability_returns_api_flights_only(tools, api: TestClient) -> None:
    result = run_tool(tools[0], "check_flight_availability", {
        "origin": "Bangalore", "destination": "Mumbai", "departure_date": day(10), "passengers": 1,
    })
    assert result.ok and result.data["count"] == 3
    assert [f["flight_id"] for f in result.data["flights"]] == ["FL-1001", "FL-1002", "FL-1003"]


def test_availability_for_an_unserved_route_is_an_empty_list_not_an_error(tools) -> None:
    result = run_tool(tools[0], "check_flight_availability", {
        "origin": "Delhi", "destination": "Chennai", "departure_date": day(10), "passengers": 1,
    })
    assert result.ok and result.data["count"] == 0


def test_invalid_arguments_never_reach_the_api(tools) -> None:
    registry, http = tools
    result = run_tool(registry, "check_flight_availability", {
        "origin": "Bangalore", "destination": "Mumbai", "departure_date": day(10), "passengers": 0,
    })
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS" and "passengers" in result.error.details
    result = run_tool(registry, "book_flight", booking_args(contact_email="not-an-email", passengers=[]))
    assert result.error.code == "INVALID_ARGUMENTS"
    assert http.requests == []


def test_passenger_age_over_120_is_rejected_before_any_call(tools) -> None:
    registry, http = tools
    bad = [{"name": "Asha Rao", "age": 150, "gender": "FEMALE"}]
    assert run_tool(registry, "book_flight", booking_args(passengers=bad)).error.code == "INVALID_ARGUMENTS"
    assert http.requests == []


def test_booking_is_created_once_and_replayed_with_the_same_key(tools, api: TestClient) -> None:
    registry, _ = tools
    first = run_tool(registry, "book_flight", booking_args())
    again = run_tool(registry, "book_flight", booking_args())
    assert first.ok and first.http_status == 201
    assert again.ok and again.http_status == 200
    assert again.data["booking_id"] == first.data["booking_id"]


def test_same_key_with_different_content_is_a_conflict(tools) -> None:
    registry, _ = tools
    run_tool(registry, "book_flight", booking_args())
    result = run_tool(registry, "book_flight", booking_args(flight_id="FL-1002"))
    assert not result.ok and result.error.code == "IDEMPOTENCY_KEY_CONFLICT"


def test_unknown_flight_is_reported_by_the_api(tools) -> None:
    result = run_tool(tools[0], "book_flight", booking_args(flight_id="FL-0000"))
    assert not result.ok and result.error.code == "FLIGHT_NOT_FOUND"


def test_booking_status_and_cancel(tools, api: TestClient) -> None:
    registry, _ = tools
    booking = book_via_api(api)
    status = run_tool(registry, "get_booking_status", {"booking_id": booking["booking_id"]})
    assert status.ok and status.data["status"] == "CONFIRMED"
    cancelled = run_tool(registry, "cancel_flight_booking", {"booking_id": booking["booking_id"]})
    assert cancelled.ok and cancelled.data["status"] == "CANCELLED"
    again = run_tool(registry, "cancel_flight_booking", {"booking_id": booking["booking_id"]})
    assert not again.ok and again.error.code == "BOOKING_ALREADY_CANCELLED"


def test_unknown_booking(tools) -> None:
    result = run_tool(tools[0], "get_booking_status", {"booking_id": "FB-NOPE00"})
    assert not result.ok and result.error.code == "BOOKING_NOT_FOUND"


def test_unsafe_booking_ids_are_rejected_locally(tools) -> None:
    registry, http = tools
    result = run_tool(registry, "get_booking_status", {"booking_id": "../../etc/passwd"})
    assert result.error.code == "INVALID_ARGUMENTS" and http.requests == []


def test_unknown_tool_name() -> None:
    result = run_tool({}, "refund_flight", {})
    assert not result.ok and result.error.code == "UNKNOWN_TOOL"


def test_the_last_flight_seats_can_be_sold_only_once(api: TestClient, tiny_flight: str, tools) -> None:
    registry, _ = tools
    three = [{"name": f"P{i} Person", "age": 30, "gender": "OTHER"} for i in range(3)]
    ok = run_tool(registry, "book_flight", booking_args("key-seats-01", tiny_flight, passengers=three))
    assert ok.ok
    full = run_tool(registry, "book_flight", booking_args("key-seats-02", tiny_flight))
    assert not full.ok and full.error.code == "SEATS_UNAVAILABLE"
