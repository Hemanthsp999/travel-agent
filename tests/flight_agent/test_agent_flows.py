"""End-to-end conversations against the real Flight API (in-process). Rules-only understanding."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from tests.flight_agent.conftest import (
    CONTACT_LINE, PASSENGER_LINE, TINY_FLIGHT, FlakyHTTP, api_booking, book_via_api, day, nice, say,
)
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.schemas import ExecutionStatus as S, FlightTask, Intent

D = nice(10)
TRIP = f"book a flight from Bangalore to Mumbai on {D} for 2 passengers"
SECOND_PASSENGER = "Ravi Rao, 35, male"


def api_flights(api: TestClient, origin: str, destination: str, offset: int = 10, passengers: int = 2) -> list[dict]:
    response = api.get("/api/v1/flights/search", params={
        "origin": origin, "destination": destination, "departure_date": day(offset).isoformat(), "passengers": passengers,
    })
    assert response.status_code == 200, response.text
    return response.json()["data"]["flights"]


def reach_confirmation(agent: FlightAgent, pick: str = "the second one", thread: str = "t1"):
    result = say(agent, TRIP, pick, PASSENGER_LINE, SECOND_PASSENGER, CONTACT_LINE, thread=thread)
    assert result.status is S.PENDING, result.message
    return result


def writes(agent: FlightAgent) -> list[tuple[str, str]]:
    return agent.http.writes()  # type: ignore[attr-defined]


# ------------------------------------------------------------------------------ search


def test_search_lists_exactly_the_api_flights(agent: FlightAgent, api: TestClient) -> None:
    result = say(agent, f"flights from Bangalore to Mumbai on {D} for 2 passengers")
    expected = api_flights(api, "Bangalore", "Mumbai")
    assert result.status is S.SUCCESS and result.intent is Intent.CHECK_FLIGHT_AVAILABILITY
    assert [f["flight_id"] for f in result.data["flights"]] == [f["flight_id"] for f in expected]
    for flight in expected:
        assert flight["flight_number"] in result.message
        assert f"{flight['price_per_passenger']:,}" in result.message
    assert not writes(agent)


def test_search_with_no_flights_says_so(agent: FlightAgent) -> None:
    result = say(agent, f"flights from Delhi to Chennai on {D} for 1 passenger")
    assert result.status is S.SUCCESS and result.data["count"] == 0
    assert "couldn't find any flights" in result.message


def test_search_uses_the_requested_route(agent: FlightAgent, api: TestClient) -> None:
    result = say(agent, f"show me flights from Delhi to Mumbai on {D} for 1 person")
    expected = {f["flight_id"] for f in api_flights(api, "Delhi", "Mumbai", passengers=1)}
    assert {f["flight_id"] for f in result.data["flights"]} == expected and len(expected) == 3


def test_search_asks_for_missing_date_then_answers(agent: FlightAgent) -> None:
    asked = say(agent, "flights from Bangalore to Mumbai for 2 passengers")
    assert asked.status is S.REQUIRES_USER_INPUT and asked.missing_parameters == ["departure_date"]
    assert "What date" in asked.message
    answered = say(agent, D)
    assert answered.status is S.SUCCESS and answered.data["query"]["departure_date"] == day(10).isoformat()


def test_search_asks_for_missing_passenger_count(agent: FlightAgent) -> None:
    asked = say(agent, f"any flights from Bangalore to Mumbai on {D}?")
    assert asked.missing_parameters == ["passenger_count"]
    answered = say(agent, "just me")
    assert answered.status is S.SUCCESS and answered.data["query"]["passengers"] == 1


def test_missing_values_are_asked_one_at_a_time(agent: FlightAgent) -> None:
    first = say(agent, "I need a flight")
    assert first.status is S.REQUIRES_USER_INPUT and "from" in first.message.lower()
    second = say(agent, "Bangalore")
    assert "fly to" in second.message and second.missing_parameters[0] == "destination"
    third = say(agent, "Mumbai")
    assert third.missing_parameters[0] == "departure_date"


def test_date_beyond_booking_window_is_rejected_by_api_and_asked_again(agent: FlightAgent) -> None:
    result = say(agent, f"flights from Bangalore to Mumbai on {nice(300)} for 2 passengers")
    assert result.status is S.REQUIRES_USER_INPUT and result.missing_parameters == ["departure_date"]
    assert "doesn't work" in result.message


def test_same_origin_and_destination_is_not_searched(agent: FlightAgent) -> None:
    result = say(agent, f"flights from Mumbai to Mumbai on {D} for 1")
    assert result.status is S.REQUIRES_USER_INPUT and "different" in result.message
    assert not agent.http.requests  # type: ignore[attr-defined]


def test_follow_up_search_reuses_remembered_route_and_party(agent: FlightAgent) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D} for 2 passengers")
    result = say(agent, f"what about {nice(12)}?")
    assert result.status is S.SUCCESS
    assert result.data["query"] == {
        "origin": "Bangalore", "destination": "Mumbai", "departure_date": day(12).isoformat(), "passengers": 2,
    }


def test_only_max_shown_flights_are_listed_and_show_all_lists_the_rest(make_agent: Callable[..., FlightAgent]) -> None:
    agent = make_agent(max_shown=2)
    result = say(agent, f"flights from Bangalore to Mumbai on {D} for 1 passenger")
    assert "1 more available" in result.message and result.data["count"] == 3
    everything = say(agent, "show all")
    assert "QP-1303" in everything.message


def test_details_and_comparison_come_from_the_search_results(agent: FlightAgent, api: TestClient) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D} for 2 passengers")
    flights = api_flights(api, "Bangalore", "Mumbai")
    details = say(agent, "tell me about the second one")
    assert flights[1]["aircraft"] in details.message and flights[1]["flight_id"] in details.message
    compared = say(agent, "compare them")
    assert compared.data["flights"] and "Lowest price: IndiGo 6E-501" in compared.message


def test_details_without_any_search_ask_for_a_search(agent: FlightAgent) -> None:
    result = say(agent, "tell me about the first one")
    assert result.status is S.REQUIRES_USER_INPUT and "search" in result.message
    assert not agent.http.requests  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- booking


def test_booking_happy_path_matches_the_api(agent: FlightAgent, api: TestClient) -> None:
    asked = say(agent, TRIP)
    assert asked.status is S.REQUIRES_USER_INPUT and asked.missing_parameters[0] == "flight"
    assert "Which one would you like to book" in asked.message
    pending = reach_confirmation(agent)
    assert pending.status is S.PENDING and "Shall I go ahead" in pending.message
    assert writes(agent) == []  # nothing is booked before the user says yes

    done = say(agent, "yes, go ahead")
    assert done.status is S.SUCCESS and done.intent is Intent.BOOK_FLIGHT
    booking = done.data["booking"]
    stored = api_booking(api, booking["booking_id"])
    assert stored["status"] == "CONFIRMED" and stored["flight"]["flight_id"] == "FL-1002"
    assert [p["name"] for p in stored["passengers"]] == ["Asha Rao", "Ravi Rao"]
    assert stored["contact"] == {"email": "asha@example.com", "phone": "+919876543210"}
    assert booking["booking_id"] in done.message and f"{stored['total_price']:,}" in done.message
    assert len(writes(agent)) == 1


def test_booking_with_everything_in_one_message(agent: FlightAgent) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D} for 1 passenger")
    pending = say(
        agent, "book the first one for Asha Rao, 31, female, email asha@example.com phone +919876543210",
    )
    assert pending.status is S.PENDING and "IndiGo 6E-501" in pending.message and "Asha Rao" in pending.message
    assert say(agent, "yes").status is S.SUCCESS


def test_agent_asks_for_each_missing_passenger_detail(agent: FlightAgent) -> None:
    result = say(agent, TRIP, "the second one")
    assert result.status is S.REQUIRES_USER_INPUT
    assert result.missing_parameters[:3] == ["passenger_1_name", "passenger_1_age", "passenger_1_gender"]
    result = say(agent, "Asha Rao")
    assert result.missing_parameters[:2] == ["passenger_1_age", "passenger_1_gender"]
    result = say(agent, "31, female")
    assert result.missing_parameters[0] == "passenger_2_name" and "passenger 2 of 2" in result.message


def test_contact_details_are_required(agent: FlightAgent) -> None:
    result = say(agent, TRIP, "the first one", PASSENGER_LINE, SECOND_PASSENGER)
    assert result.missing_parameters == ["contact_email", "contact_phone"]
    result = say(agent, "asha@example.com")
    assert result.missing_parameters == ["contact_phone"]


def test_only_an_explicit_yes_books(agent: FlightAgent) -> None:
    reach_confirmation(agent)
    still_pending = say(agent, "hmm, let me think")
    assert still_pending.status is S.PENDING and not writes(agent)
    assert say(agent, "yes please").status is S.SUCCESS and len(writes(agent)) == 1


def test_declining_the_booking_books_nothing(agent: FlightAgent, api: TestClient) -> None:
    reach_confirmation(agent)
    result = say(agent, "no")
    assert result.status is S.FAILED and result.error["code"] == "USER_DECLINED"
    assert not writes(agent)


def test_changing_the_date_at_confirmation_rechecks_and_asks_again(agent: FlightAgent) -> None:
    reach_confirmation(agent)
    result = say(agent, f"change the date to {nice(14)}")
    assert result.status is S.PENDING and day(14).strftime("%b %Y") in result.message and f"{day(14).day} " in result.message
    assert not writes(agent)
    done = say(agent, "yes")
    assert done.status is S.SUCCESS and done.data["booking"]["departure_date"] == day(14).isoformat()


def test_unknown_flight_id_is_not_booked(agent: FlightAgent) -> None:
    result = say(agent, f"book flight FL-9999 from Bangalore to Mumbai on {D} for 1 passenger")
    assert result.status is S.FAILED and result.error["code"] == "FLIGHT_UNAVAILABLE"
    assert "FL-9999" in result.message and not writes(agent)


def test_a_flight_that_was_never_shown_cannot_be_booked_by_number(agent: FlightAgent) -> None:
    result = say(agent, f"book 6E-9999 from Bangalore to Mumbai on {D} for 1 passenger")
    assert result.status is S.REQUIRES_USER_INPUT and "6E-9999" in result.message
    assert not writes(agent)


def test_not_enough_seats_at_search_time(agent: FlightAgent, api: TestClient, tiny_flight: str) -> None:
    book_via_api(api, TINY_FLIGHT, offset=10, passengers=2)
    result = say(agent, f"book a flight from Testville to Nowhereburg on {D} for 2 passengers")
    assert result.status is S.SUCCESS and result.data["count"] == 0 and not writes(agent)


def test_seats_taken_between_confirmation_and_booking(agent: FlightAgent, api: TestClient, tiny_flight: str) -> None:
    result = say(agent, f"book a flight from Testville to Nowhereburg on {D} for 2 passengers", "the first one",
                 PASSENGER_LINE, SECOND_PASSENGER, CONTACT_LINE)
    assert result.status is S.PENDING and "Test Air TA-1" in result.message
    book_via_api(api, TINY_FLIGHT, offset=10, passengers=2)  # someone else takes the seats first
    failed = say(agent, "yes")
    assert failed.status is S.FAILED and "Only 1 seat" in failed.message
    assert failed.error["code"] == "SEATS_UNAVAILABLE"


def test_invalid_passenger_data_from_api_is_asked_again(agent: FlightAgent) -> None:
    pending = say(agent, f"flights from Bangalore to Mumbai on {D} for 1 passenger",
                  "book the first one for Asha Rao, 150, female, email asha@example.com phone +919876543210")
    # age 150 is rejected before any booking is sent
    assert pending.status is S.REQUIRES_USER_INPUT and "age" in " ".join(pending.missing_parameters)
    assert not writes(agent)


def test_second_identical_booking_warns_about_duplicate(agent: FlightAgent) -> None:
    reach_confirmation(agent)
    first = say(agent, "yes")
    assert first.status is S.SUCCESS
    again = say(agent, f"book AI-502 from Bangalore to Mumbai on {D} for 2 passengers")
    assert again.status is S.PENDING and "already booked" in again.message and first.data["booking"]["booking_id"] in again.message


def test_a_stray_yes_after_booking_does_not_book_again(agent: FlightAgent) -> None:
    reach_confirmation(agent)
    say(agent, "yes")
    result = say(agent, "yes")
    assert result.status is S.UNKNOWN and len(writes(agent)) == 1


# ---------------------------------------------------------------------- cancel / status


def test_cancel_needs_confirmation_then_cancels(agent: FlightAgent, api: TestClient) -> None:
    booking = book_via_api(api)
    pending = say(agent, f"please cancel my booking {booking['booking_id']}")
    assert pending.status is S.PENDING and booking["booking_id"] in pending.message
    assert api_booking(api, booking["booking_id"])["status"] == "CONFIRMED" and not writes(agent)
    done = say(agent, "yes")
    assert done.status is S.SUCCESS and done.data["booking"]["status"] == "CANCELLED"
    assert api_booking(api, booking["booking_id"])["status"] == "CANCELLED"
    assert "can't say anything about a refund" in done.message


def test_declining_a_cancellation_keeps_the_booking(agent: FlightAgent, api: TestClient) -> None:
    booking = book_via_api(api)
    say(agent, f"cancel {booking['booking_id']}")
    result = say(agent, "no, keep it")
    assert result.status is S.FAILED and result.error["code"] == "USER_DECLINED"
    assert api_booking(api, booking["booking_id"])["status"] == "CONFIRMED" and not writes(agent)


def test_cancel_without_id_asks_for_it(agent: FlightAgent, api: TestClient) -> None:
    asked = say(agent, "I want to cancel my flight")
    assert asked.status is S.REQUIRES_USER_INPUT and asked.missing_parameters == ["booking_id"]
    booking = book_via_api(api)
    assert say(agent, booking["booking_id"]).status is S.PENDING


def test_cancel_unknown_booking_asks_again_and_cancels_nothing(agent: FlightAgent) -> None:
    result = say(agent, "cancel booking FB-0000000000")
    assert result.status is S.REQUIRES_USER_INPUT and "couldn't find a booking" in result.message
    assert not writes(agent)


def test_cancelling_twice_reports_already_cancelled(agent: FlightAgent, api: TestClient) -> None:
    booking = book_via_api(api)
    assert api.post(f"/api/v1/bookings/{booking['booking_id']}/cancel").status_code == 200
    result = say(agent, f"cancel {booking['booking_id']}")
    assert result.status is S.SUCCESS and "already cancelled" in result.message and not writes(agent)


def test_status_comes_from_the_api(agent: FlightAgent, api: TestClient) -> None:
    booking = book_via_api(api, "FL-1004", passengers=2)
    result = say(agent, f"what's the status of {booking['booking_id']}?")
    assert result.status is S.SUCCESS and result.intent is Intent.GET_BOOKING_STATUS
    stored = {k: v for k, v in api_booking(api, booking["booking_id"]).items() if v is not None}
    assert result.data["booking"] == stored
    assert "CONFIRMED" in result.message and "6E-2102" in result.message and "Traveller 1" in result.message


def test_status_of_unknown_booking_is_not_invented(agent: FlightAgent) -> None:
    result = say(agent, "status of FB-FFFFFFFFFF")
    assert result.status is S.REQUIRES_USER_INPUT and "couldn't find a booking" in result.message


def test_book_then_check_status_then_cancel_in_one_conversation(agent: FlightAgent, api: TestClient) -> None:
    reach_confirmation(agent)
    booked = say(agent, "yes").data["booking"]["booking_id"]
    status = say(agent, "what is the status of my booking?")
    assert booked in status.message and status.status is S.SUCCESS
    assert say(agent, "cancel it").status is S.PENDING
    assert say(agent, "yes").status is S.SUCCESS
    assert api_booking(api, booked)["status"] == "CANCELLED"


# --------------------------------------------------------------- unsupported / unknown


def test_unsupported_requests_are_refused_without_api_calls(agent: FlightAgent) -> None:
    for text in ("can I get a refund?", "I want to modify my booking", "do you offer web check-in?", "find me a hotel in Goa"):
        result = say(agent, text)
        assert result.status is S.FAILED and result.error["code"] == "UNSUPPORTED_OPERATION", text
    assert not agent.http.requests  # type: ignore[attr-defined]


def test_unrecognised_input_is_reported_as_unknown(agent: FlightAgent) -> None:
    result = say(agent, "what's the weather like?")
    assert result.status is S.UNKNOWN and result.error["code"] == "INTENT_NOT_RECOGNIZED"


def test_help_and_greeting(agent: FlightAgent) -> None:
    result = say(agent, "hello")
    assert result.status is S.SUCCESS and "flight assistant" in result.message


def test_giving_up_after_repeated_non_answers(agent: FlightAgent) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D}")
    results = [say(agent, "dunno") for _ in range(3)]
    gave_up = [r for r in results if r.error and r.error["code"] == "MISSING_PARAMETERS"]
    assert len(gave_up) == 1 and gave_up[0].status is S.FAILED


def test_abort_drops_the_current_request(agent: FlightAgent) -> None:
    say(agent, TRIP, "the first one")
    result = say(agent, "never mind")
    assert result.status is S.FAILED and result.error["code"] == "USER_ABORTED"
    assert say(agent, f"flights from Bangalore to Mumbai on {D} for 1").status is S.SUCCESS


def test_threads_are_independent(agent: FlightAgent) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D}", thread="a")
    other = say(agent, "2 passengers", thread="b")
    assert other.status is not S.SUCCESS  # thread b has no route, so the bare answer cannot start a search
    assert say(agent, "2 passengers", thread="a").status is S.SUCCESS


def test_structured_task_with_hints(agent: FlightAgent) -> None:
    task = FlightTask(
        message="find flights", intent=Intent.CHECK_FLIGHT_AVAILABILITY,
        params={"origin": "Bangalore", "destination": "Mumbai", "departure_date": day(10).isoformat(), "passenger_count": 2},
    )
    result = agent.run_task(task)
    assert result.status is S.SUCCESS and result.data["count"] == 3


def test_flaky_http_helper_counts_requests(agent: FlightAgent) -> None:
    say(agent, f"flights from Bangalore to Mumbai on {D} for 1")
    http: FlakyHTTP = agent.http  # type: ignore[attr-defined]
    assert [m for m, _ in http.requests] == ["GET"]
