"""Failure handling: API down, slow, broken; unknown booking outcomes are verified, never guessed."""

from __future__ import annotations

from tests.hotel_agent.conftest import book_via_api, day, say
from travel_agent.hotel_agent.schemas import TaskStatus

CI, CO = day(20), day(22)
SEARCH = f"hotels in Pune {CI} to {CO}"
BOOK = f"Book HT-1003 from {CI} to {CO} for 2 guests and 1 room. Name: Meera Nair, meera@example.com, 9123456780"


def assert_friendly(result):
    assert "Traceback" not in result.message and "Exception" not in result.message
    assert "Error(" not in result.message and result.message.strip()


def test_api_down_on_search_is_friendly(make_agent):
    agent = make_agent(default="connect")
    result = say(agent, SEARCH)
    assert result.status is TaskStatus.FAILED and "trouble reaching the hotel system" in result.message
    assert "try again" in result.message and "Nothing was changed" in result.message
    assert_friendly(result)
    assert len(agent.http.requests) == 2  # one retry with backoff, then gave up


def test_search_recovers_after_transient_failures(make_agent):
    agent = make_agent(script=["timeout"])
    result = say(agent, SEARCH)
    assert result.status is TaskStatus.COMPLETED and len(agent.http.requests) == 2


def test_garbage_response_is_not_trusted(make_agent):
    agent = make_agent(default="garbage")
    result = say(agent, SEARCH)
    assert result.status is TaskStatus.FAILED and "couldn't understand" in result.message
    assert_friendly(result)


def test_server_500_on_status_lookup(make_agent, api):
    booking = book_via_api(api)
    agent = make_agent(default="500")
    result = say(agent, f"status of {booking['booking_id']}")
    assert result.status is TaskStatus.FAILED and "couldn't look up the booking" in result.message
    assert_friendly(result)


def test_booking_when_api_unreachable_does_not_claim_success(make_agent, api):
    agent = make_agent(script=["ok", "ok", "connect", "connect", "connect"])  # precheck search+details ok, then book fails
    say(agent, BOOK)
    result = say(agent, "yes")
    assert result.status is not TaskStatus.COMPLETED
    assert "confirmed" not in result.message.lower().replace("not confirmed", "")
    assert_friendly(result)


def test_book_timeout_after_send_is_verified_by_retry_with_same_key(make_agent, api):
    agent = make_agent()
    say(agent, BOOK)
    # next HTTP call is the booking POST: server applies it, the client sees a timeout
    agent.http.script = ["send_timeout"] * 3
    result = say(agent, "yes")
    assert result.status is TaskStatus.UNVERIFIED
    assert "retry" in result.message.lower() and "won't" in result.message.lower() or "not" in result.message.lower()

    retried = say(agent, "retry")
    assert retried.status is TaskStatus.COMPLETED and retried.data["booking"]["status"] == "CONFIRMED"
    # exactly one booking exists: replaying the idempotency key returned the original
    booked = [h for h in api.get("/api/v1/hotels/search", params={
        "city": "Bangalore", "check_in": CI, "check_out": CO}).json()["data"]["hotels"] if h["hotel_id"] == "HT-1003"][0]
    assert booked["rooms_available"] == booked["total_rooms"] - 1


def test_cancel_timeout_is_resolved_by_status_lookup(make_agent, api):
    booking = book_via_api(api)
    agent = make_agent()
    say(agent, f"cancel {booking['booking_id']}")
    agent.http.script = ["send_timeout"]  # cancel applied on server, response lost
    result = say(agent, "yes")
    assert result.status is TaskStatus.COMPLETED and "cancelled" in result.message.lower()
    assert api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]["status"] == "CANCELLED"


def test_cancel_timeout_with_unknown_state_is_reported_honestly(make_agent, api):
    booking = book_via_api(api)
    agent = make_agent()
    say(agent, f"cancel {booking['booking_id']}")
    agent.http.script = ["timeout", "connect", "connect", "connect"]  # cancel not applied; lookups fail
    result = say(agent, "yes")
    assert result.status is TaskStatus.UNVERIFIED
    assert "cancelled" not in result.message.lower().replace("not been cancelled", "").replace("isn't cancelled", "")
    assert_friendly(result)
    assert api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]["status"] == "CONFIRMED"


def test_unexpected_exception_in_a_node_is_contained(make_agent):
    agent = make_agent()

    def boom() -> None:
        raise RuntimeError("secret internal detail")

    object.__setattr__(agent._deps, "today", boom)
    result = say(agent, SEARCH)
    assert result.status is TaskStatus.FAILED
    assert "secret internal detail" not in result.message
    assert_friendly(result)
    assert result.error and result.error["code"] == "AGENT_ERROR"


def test_threads_are_isolated(make_agent):
    agent = make_agent()
    say(agent, "I need a hotel", thread="a")
    result = say(agent, "Pune", thread="b")
    assert result.status is TaskStatus.COMPLETED  # thread b has no pending question
    again = say(agent, "Mumbai", thread="a")
    assert again.missing_params == ["check_in", "check_out"]
