"""The five tools: argument validation, compact results, retry policy. Never raise, never leak HTTP objects."""

from __future__ import annotations

import pytest

from tests.hotel_agent.conftest import FlakyHTTP, book_via_api, day
from travel_agent.config import load_settings
from travel_agent.hotel_agent.api_client import HotelAPIClient
from travel_agent.hotel_agent.schemas import ToolResult
from travel_agent.hotel_agent.tools import build_hotel_tools, run_tool

CI, CO = day(20), day(22)
GOOD_BOOKING = {
    "hotel_id": "HT-1002", "check_in": CI, "check_out": CO, "rooms": 1, "guests": 2,
    "guest_name": "Asha Rao", "guest_email": "asha@example.com", "guest_phone": "+919876543210",
    "idempotency_key": "tool-test-key-001",
}


@pytest.fixture
def tools(api, agent_settings):
    def build(script=None, default="ok"):
        http = FlakyHTTP(api, script, default)
        built = build_hotel_tools(HotelAPIClient(agent_settings.hotel_api, http=http), agent_settings.hotel_api)
        built["_http"] = http  # type: ignore[assignment]
        return built

    return build


def test_exactly_five_tools_with_expected_names(tools):
    names = {n for n in tools() if not n.startswith("_")}
    assert names == {
        "check_hotel_availability", "get_hotel_details", "book_hotel", "get_booking_status", "cancel_hotel_booking",
    }


def test_availability_returns_compact_validated_data(tools):
    result = run_tool(tools(), "check_hotel_availability", {"city": "Pune", "check_in": CI, "check_out": CO, "guests": 2})
    assert result.ok and result.data["hotels"] and result.error is None
    assert {"hotel_id", "name", "price_per_night"} <= set(result.data["hotels"][0])


def test_details_by_id(tools):
    result = run_tool(tools(), "get_hotel_details", {"hotel_id": "HT-1001"})
    assert result.ok and result.data["name"] == "Lakeview Grand Bengaluru" and result.data["amenities"]


def test_unknown_hotel_returns_api_error_code(tools):
    result = run_tool(tools(), "get_hotel_details", {"hotel_id": "HT-9999"})
    assert not result.ok and result.error.code == "HOTEL_NOT_FOUND" and not result.error.retryable


def test_book_status_cancel_roundtrip(tools):
    t = tools()
    booked = run_tool(t, "book_hotel", GOOD_BOOKING)
    assert booked.ok and booked.http_status == 201
    booking_id = booked.data["booking_id"]
    assert run_tool(t, "get_booking_status", {"booking_id": booking_id}).data["status"] == "CONFIRMED"
    cancelled = run_tool(t, "cancel_hotel_booking", {"booking_id": booking_id})
    assert cancelled.ok and cancelled.data["status"] == "CANCELLED"
    again = run_tool(t, "cancel_hotel_booking", {"booking_id": booking_id})
    assert not again.ok and again.error.code == "BOOKING_ALREADY_CANCELLED"


def test_same_idempotency_key_never_double_books(tools, api):
    t = tools()
    first = run_tool(t, "book_hotel", GOOD_BOOKING)
    second = run_tool(t, "book_hotel", GOOD_BOOKING)
    assert first.data["booking_id"] == second.data["booking_id"] and second.http_status == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("guest_email", "nope"), ("guest_phone", "12345"), ("rooms", 0), ("rooms", 9), ("guests", 0),
        ("guest_name", "A"), ("idempotency_key", "short"), ("hotel_id", "HT 1002; DROP"),
    ],
)
def test_invalid_booking_arguments_never_reach_the_api(tools, field, value):
    t = tools()
    result = run_tool(t, "book_hotel", {**GOOD_BOOKING, field: value})
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS" and field in result.error.details
    assert t["_http"].requests == []


def test_missing_arguments_are_reported(tools):
    result = run_tool(tools(), "check_hotel_availability", {"city": "Pune"})
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS"
    assert {"check_in", "check_out"} <= set(result.error.details)


def test_unknown_tool_is_an_error_result(tools):
    result = run_tool(tools(), "delete_everything", {})
    assert not result.ok and result.error.code == "UNKNOWN_TOOL"


def test_read_tools_retry_transient_errors(tools):
    t = tools(script=["connect"])
    result = run_tool(t, "get_hotel_details", {"hotel_id": "HT-1001"})
    assert result.ok and result.attempts == 2


def test_read_tools_give_up_after_max_attempts(tools):
    t = tools(default="connect")
    result = run_tool(t, "get_hotel_details", {"hotel_id": "HT-1001"})
    assert not result.ok and result.error.code == "CONNECTION_ERROR" and result.error.retryable
    assert len(t["_http"].requests) == 2 and result.attempts == 2


def test_cancel_is_never_retried(tools, api):
    booking = book_via_api(api)
    t = tools(script=["connect"])
    result = run_tool(t, "cancel_hotel_booking", {"booking_id": booking["booking_id"]})
    assert not result.ok and len(t["_http"].requests) == 1


def test_book_timeout_after_send_is_flagged_unknown_then_replays(tools):
    t = tools(script=["send_timeout"])
    result = run_tool(t, "book_hotel", GOOD_BOOKING)
    # first attempt applied on the server and timed out; retry with the same key returns the original booking
    assert result.ok and result.http_status == 200 and result.attempts == 2


def test_unreachable_booking_marks_outcome_unknown_only_when_it_may_have_been_applied(tools):
    refused = run_tool(tools(default="connect"), "book_hotel", GOOD_BOOKING)
    assert not refused.error.outcome_unknown  # request never left the machine
    slow = run_tool(tools(default="timeout"), "book_hotel", GOOD_BOOKING)
    assert slow.error.outcome_unknown and slow.error.code == "TIMEOUT"


def test_results_are_plain_serialisable_models(tools):
    result = run_tool(tools(), "get_hotel_details", {"hotel_id": "HT-1001"})
    assert isinstance(result, ToolResult) and ToolResult.model_validate_json(result.model_dump_json()) == result


def test_default_settings_point_at_the_hotel_api_port():
    assert load_settings(env={}).hotel_api.base_url.endswith(":8001")
