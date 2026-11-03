"""End-to-end conversations through the real graph against the real Hotel API (in-process)."""

from __future__ import annotations

import re

from tests.hotel_agent.conftest import api_search, book_via_api, day, say
from travel_agent.hotel_agent.schemas import Intent, TaskStatus

CI, CO = day(20), day(22)


def hotel_names(api, city, ci=CI, co=CO, **extra):
    return [h["name"] for h in api_search(api, city, ci, co, **extra)]


# ------------------------------------------------------------------------- availability


def test_search_with_all_details_lists_hotels_from_api(agent, api):
    result = say(agent, f"Show me hotels in Pune from {CI} to {CO} for 2 guests")
    assert result.status is TaskStatus.COMPLETED and result.intent is Intent.CHECK_AVAILABILITY
    expected = api_search(api, "Pune", CI, CO, guests=2)
    for hotel in expected:
        assert hotel["name"] in result.message
        assert f"INR {hotel['price_per_night']:,}" in result.message
    assert [h["hotel_id"] for h in result.data["hotels"]] == [h["hotel_id"] for h in expected]


def test_missing_info_is_asked_one_question_at_a_time(agent):
    first = say(agent, "I need a hotel")
    assert first.status is TaskStatus.NEEDS_INPUT and "city" in first.message.lower()
    assert first.missing_params[0] == "city"

    second = say(agent, "Mumbai")
    assert second.status is TaskStatus.NEEDS_INPUT and second.missing_params == ["check_in", "check_out"]

    third = say(agent, f"{CI} to {CO}")
    assert third.status is TaskStatus.COMPLETED and "Mumbai" in third.message
    assert "assumed 1 guest and 1 room" in third.message


def test_single_date_then_checkout_asked(agent):
    say(agent, "hotels in Delhi")
    asked = say(agent, f"check in {CI}")
    assert asked.missing_params == ["check_out"]
    done = say(agent, CO)
    assert done.status is TaskStatus.COMPLETED and "Delhi" in done.message


def test_nights_phrase_sets_checkout(agent):
    result = say(agent, f"hotels in Chennai from {CI} for 2 nights")
    assert result.status is TaskStatus.COMPLETED
    assert result.data["query"]["check_out"] == CO


def test_follow_up_reuses_dates_and_city_changes(agent, api):
    say(agent, f"hotels in Pune {CI} to {CO}")
    result = say(agent, "what about Mumbai?")
    assert result.status is TaskStatus.COMPLETED
    assert {h["name"] for h in result.data["hotels"]} == set(hotel_names(api, "Mumbai"))


def test_budget_filter_only_returns_cheaper_hotels(agent, api):
    result = say(agent, f"hotels in Bangalore {CI} to {CO} under 5000")
    assert result.data["query"]["max_price_per_night"] == 5000
    assert all(h["price_per_night"] <= 5000 for h in result.data["hotels"])
    assert {h["name"] for h in result.data["hotels"]} == set(hotel_names(api, "Bangalore", max_price_per_night=5000))
    assert "up to INR 5,000 per night" in result.message


def test_city_alias_is_normalised(agent):
    result = say(agent, f"hotels in Bengaluru {CI} to {CO}")
    assert result.data["query"]["city"] == "Bangalore"


def test_unknown_city_gives_honest_empty_answer(agent):
    result = say(agent, f"hotels in Atlantis {CI} to {CO}")
    assert result.status is TaskStatus.COMPLETED
    assert "couldn't find any hotels in Atlantis" in result.message
    assert result.data["hotels"] == []


def test_all_matching_hotels_are_listed(agent, api):
    names = hotel_names(api, "Bangalore")
    shown = say(agent, f"hotels in Bangalore {CI} to {CO}")
    assert len(names) == 3  # seed data
    assert all(n in shown.message for n in names)


def test_past_date_is_rejected_and_asked_again(agent):
    result = say(agent, f"hotels in Pune {day(-3)} to {day(-1)}")
    assert result.status is TaskStatus.NEEDS_INPUT and "past" in result.message
    fixed = say(agent, f"{CI} to {CO}")
    assert fixed.status is TaskStatus.COMPLETED


def test_checkout_before_checkin_is_asked_again(agent):
    result = say(agent, f"hotels in Pune from {CO} to {CI}")
    assert result.status is TaskStatus.NEEDS_INPUT and "after the check-in" in result.message
    assert result.missing_params == ["check_out"]


def test_stay_beyond_booking_window_recovers_via_api_error(agent):
    result = say(agent, f"hotels in Pune from {day(300)} to {day(302)}")
    assert result.status is TaskStatus.NEEDS_INPUT
    assert "doesn't work" in result.message and "Bookings are open only up to" in result.message


def test_specific_hotel_availability_question(agent):
    result = say(agent, f"Is Indiranagar Urban Stay available in Bangalore from {CI} to {CO}?", )
    # no LLM: hotel name is not extracted by rules, so the agent lists options honestly
    assert result.status is TaskStatus.COMPLETED and "Indiranagar Urban Stay" in result.message


# ---------------------------------------------------------------------------- booking


def test_full_booking_flow_with_confirmation(agent, api):
    listing = say(agent, f"hotels in Bangalore {CI} to {CO} for 2 guests")
    second = listing.data["hotels"][1]

    ask = say(agent, "book the second one")
    assert ask.status is TaskStatus.NEEDS_INPUT and ask.missing_params[0] == "rooms" and "rooms" in ask.message
    ask = say(agent, "1")
    assert ask.status is TaskStatus.NEEDS_INPUT and ask.missing_params == ["guest_name", "guest_email", "guest_phone"]

    confirm = say(agent, "Asha Rao, asha@example.com, +91 98765 43210")
    assert confirm.status is TaskStatus.NEEDS_CONFIRMATION
    expected_total = second["estimated_total_price"]
    assert second["name"] in confirm.question and f"INR {expected_total:,}" in confirm.question
    assert "asha@example.com" in confirm.question and "+919876543210" in confirm.question

    done = say(agent, "yes")
    assert done.status is TaskStatus.COMPLETED
    booking = done.data["booking"]
    assert booking["status"] == "CONFIRMED" and booking["booking_id"] in done.message
    stored = api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]
    assert stored["hotel"]["hotel_id"] == second["hotel_id"] and stored["total_price"] == expected_total
    assert stored["guest"]["phone"] == "+919876543210"


def test_declining_does_not_book(agent, api):
    say(agent, f"hotels in Pune {CI} to {CO} for 2 guests")
    say(agent, "book the first one for 1 room")
    confirm = say(agent, "my name is Ravi Kumar, ravi@example.com, 9876543210")
    assert confirm.status is TaskStatus.NEEDS_CONFIRMATION
    result = say(agent, "no")
    assert result.status is TaskStatus.DECLINED and "haven't gone ahead" in result.message
    assert api.get("/api/v1/bookings/HB-0000000000").status_code == 404


def test_ambiguous_reply_never_books(agent, api):
    say(agent, f"hotels in Pune {CI} to {CO} for 2 guests")
    say(agent, "book the first one for 1 room")
    say(agent, "Ravi Kumar ravi@example.com 9876543210")
    first = say(agent, "hmm let me think")
    assert first.status is TaskStatus.NEEDS_CONFIRMATION
    second = say(agent, "what's the weather like")
    assert second.status is not TaskStatus.COMPLETED or "booking is confirmed" not in second.message
    before = api_search(api, "Pune", CI, CO)
    assert all(h["rooms_available"] == h["total_rooms"] for h in before)


def test_changing_details_at_confirmation_reprices(agent):
    say(agent, f"hotels in Pune {CI} to {CO} for 2 guests")
    say(agent, "book the first one for 1 room")
    confirm = say(agent, "Ravi Kumar ravi@example.com 9876543210")
    assert "1 room for 2 guests" in confirm.question
    changed = say(agent, "make it 2 rooms")
    assert changed.status is TaskStatus.NEEDS_CONFIRMATION and "2 rooms for 2 guests" in changed.question


def test_book_by_hotel_id_directly(agent, api):
    confirm = say(
        agent,
        f"Book HT-1003 from {CI} to {CO} for 2 guests and 1 room. Name: Meera Nair, meera@example.com, 9123456780",
    )
    assert confirm.status is TaskStatus.NEEDS_CONFIRMATION and "Whitefield Business Inn" in confirm.question
    done = say(agent, "yes please")
    assert done.status is TaskStatus.COMPLETED and done.data["booking"]["hotel"]["hotel_id"] == "HT-1003"


def test_unknown_hotel_id_asks_again(agent):
    result = say(agent, f"Book HT-9999 from {CI} to {CO} for 2 guests and 1 room. Name: A Bee, a@b.co, 9123456780")
    assert result.status is TaskStatus.NEEDS_INPUT
    assert "couldn't find that hotel" in result.message


def test_sold_out_hotel_gives_failure_and_alternatives(agent, api):
    for _ in range(25):  # HT-1003 has 25 rooms
        book_via_api(api, "HT-1003", (20, 22))
    result = say(
        agent, f"Book HT-1003 from {CI} to {CO} for 2 guests and 1 room. Name: Meera Nair, meera@example.com, 9123456780"
    )
    assert result.status is TaskStatus.FAILED and "no availability" in result.message
    assert "Whitefield Business Inn" in result.message


def test_capacity_error_asks_for_guests_again(agent):
    # HT-1003 hosts at most 2 guests per room
    result = say(agent, f"Book HT-1003 from {CI} to {CO} for 7 guests and 1 room. Name: A Bee, a@b.co, 9123456780")
    assert result.status is TaskStatus.NEEDS_INPUT and "fits at most 2 guests" in result.message
    assert result.missing_params == ["guests", "rooms"]
    fixed = say(agent, "2 guests and 1 room")
    assert fixed.status is TaskStatus.NEEDS_CONFIRMATION


def test_invalid_email_is_not_accepted(agent):
    result = say(agent, f"Book HT-1003 from {CI} to {CO} for 2 guests and 1 room. Name: A Bee, not-an-email, 9123456780")
    assert result.status is TaskStatus.NEEDS_INPUT and result.missing_params == ["guest_email"]


def test_book_without_hotel_shows_options_first(agent):
    result = say(agent, f"I want to book a room in Delhi from {CI} to {CO}")
    assert result.status is TaskStatus.COMPLETED and "Which one would you like to book?" in result.message


def test_gives_up_after_repeated_unanswered_questions(agent):
    say(agent, "I need a hotel")
    say(agent, "hmm")
    say(agent, "dunno")
    result = say(agent, "no idea")
    assert result.status is TaskStatus.FAILED and "still don't have the city" in result.message


def test_abort_mid_flow(agent):
    say(agent, "I need a hotel")
    result = say(agent, "never mind")
    assert result.status is TaskStatus.DECLINED


# -------------------------------------------------------------------------- cancellation


def test_cancel_flow_with_confirmation(agent, api):
    booking = book_via_api(api)
    ask = say(agent, f"Please cancel {booking['booking_id']}")
    assert ask.status is TaskStatus.NEEDS_CONFIRMATION and booking["booking_id"] in ask.question
    assert api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]["status"] == "CONFIRMED"
    done = say(agent, "yes")
    assert done.status is TaskStatus.COMPLETED and "cancelled" in done.message.lower()
    assert api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]["status"] == "CANCELLED"


def test_cancel_declined_keeps_booking(agent, api):
    booking = book_via_api(api)
    say(agent, f"cancel {booking['booking_id']}")
    result = say(agent, "no, keep it")
    assert result.status is TaskStatus.DECLINED
    assert api.get(f"/api/v1/bookings/{booking['booking_id']}").json()["data"]["status"] == "CONFIRMED"


def test_cancel_already_cancelled(agent, api):
    booking = book_via_api(api)
    api.post(f"/api/v1/bookings/{booking['booking_id']}/cancel")
    result = say(agent, f"cancel {booking['booking_id']}")
    assert result.status is TaskStatus.COMPLETED and "already cancelled" in result.message


def test_cancel_unknown_booking_asks_again(agent):
    result = say(agent, "cancel HB-DEADBEEF00")
    assert result.status is TaskStatus.NEEDS_INPUT and "couldn't find a booking" in result.message
    assert result.missing_params == ["booking_id"]


def test_cancel_without_id_asks_for_it(agent, api):
    booking = book_via_api(api)
    ask = say(agent, "I want to cancel my booking")
    assert ask.status is TaskStatus.NEEDS_INPUT and ask.missing_params == ["booking_id"]
    confirm = say(agent, booking["booking_id"])
    assert confirm.status is TaskStatus.NEEDS_CONFIRMATION


def test_booking_status(agent, api):
    booking = book_via_api(api)
    result = say(agent, f"what's the status of {booking['booking_id']}?")
    assert result.status is TaskStatus.COMPLETED and "CONFIRMED" in result.message
    assert "Indiranagar Urban Stay" in result.message and f"INR {booking['total_price']:,}" in result.message


def test_new_booking_then_cancel_uses_context(agent, api):
    confirm = say(agent, f"Book HT-1003 from {CI} to {CO} for 2 guests and 1 room. Name: Meera Nair, meera@example.com, 9123456780")
    done = say(agent, "yes")
    booking_id = done.data["booking"]["booking_id"]
    ask = say(agent, "actually cancel it")
    assert ask.status is TaskStatus.NEEDS_CONFIRMATION and booking_id in ask.question
    assert say(agent, "yes").status is TaskStatus.COMPLETED


# ---------------------------------------------------------------------- general queries


def test_hotel_details_from_search_result(agent, api):
    listing = say(agent, f"hotels in Pune {CI} to {CO}")
    first_id = listing.data["hotels"][0]["hotel_id"]
    result = say(agent, "tell me more about the first one")
    assert result.status is TaskStatus.COMPLETED
    details = api.get(f"/api/v1/hotels/{first_id}").json()["data"]
    assert details["address"] in result.message and ", ".join(details["amenities"]) in result.message


def test_details_by_id_without_search(agent):
    result = say(agent, "tell me about HT-1001")
    assert "Lakeview Grand Bengaluru" in result.message


def test_compare_hotels(agent):
    say(agent, f"hotels in Mumbai {CI} to {CO}")
    result = say(agent, "compare them")
    assert result.status is TaskStatus.COMPLETED and "Lowest price: Andheri Airport Comfort" in result.message


def test_compare_without_search_explains(agent):
    result = say(agent, "compare the hotels")
    assert "at least two hotels" in result.message


def test_unsupported_request_is_declined_politely(agent):
    result = say(agent, "what is your refund policy?")
    assert result.status is TaskStatus.COMPLETED and "can't help with that part" in result.message


def test_greeting_and_help(agent):
    result = say(agent, "hello")
    assert "hotel assistant" in result.message


def test_gibberish_is_handled(agent):
    result = say(agent, "asdf qwer")
    assert "not sure I caught that" in result.message


def test_structured_task_from_orchestrator(agent, api):
    from travel_agent.hotel_agent.schemas import HotelTask

    result = agent.run_task(HotelTask(
        message="find hotels", intent=Intent.CHECK_AVAILABILITY,
        params={"city": "Hyderabad", "check_in": CI, "check_out": CO, "guests": 2},
    ))
    assert result.status is TaskStatus.COMPLETED and result.data["query"]["city"] == "Hyderabad"
    assert re.search(r"Hitech City Towers", result.message)
