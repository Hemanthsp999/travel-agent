"""Rule-based extraction: dates, numbers, ids, contact details and yes/no handling."""

from __future__ import annotations

from datetime import date

import pytest

from travel_agent.hotel_agent.parsing import (
    classify_confirmation, extract_params, is_abort, is_retry, parse_dates,
)

TODAY = date(2026, 6, 15)  # a Monday


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-07-01", [date(2026, 7, 1)]),
        ("1 Jul 2026", [date(2026, 7, 1)]),
        ("July 1st", [date(2026, 7, 1)]),
        ("from 5 to 8 August", [date(2026, 8, 5), date(2026, 8, 8)]),
        ("1 Jan", [date(2027, 1, 1)]),  # already past this year -> next occurrence
        ("tomorrow", [date(2026, 6, 16)]),
        ("day after tomorrow", [date(2026, 6, 17)]),
        ("today", [date(2026, 6, 15)]),
        ("next friday", [date(2026, 6, 19)]),
        ("monday", [date(2026, 6, 22)]),  # never "today" for a weekday name
        ("30 Feb", []),  # impossible date is ignored, not guessed
    ],
)
def test_parse_dates(text, expected):
    assert parse_dates(text, TODAY)[0] == expected


def test_two_dates_keep_textual_order():
    dates, _ = parse_dates("checkout 12 Aug 2026, checkin 10 Aug 2026", TODAY)
    assert dates == [date(2026, 8, 12), date(2026, 8, 10)]


@pytest.mark.parametrize(
    "text, key, value",
    [
        ("for 3 guests", "guests", 3),
        ("for two people", "guests", 2),
        ("just me", "guests", 1),
        ("2 rooms please", "rooms", 2),
        ("a room", "rooms", 1),
        ("under 5000", "max_price_per_night", 5000),
        ("budget of Rs. 7,500", "max_price_per_night", 7500),
        ("below 4k", "max_price_per_night", 4000),
        ("rated 4.5", "min_rating", 4.5),
        ("Cancel hb-1a2b3c4d5e", "booking_id", "HB-1A2B3C4D5E"),
        ("book ht-1003", "hotel_id", "HT-1003"),
        ("mail me at Asha.Rao@Example.com", "guest_email", "asha.rao@example.com"),
        ("my number is +91 98765-43210", "guest_phone", "+919876543210"),
        ("call 9876543210", "guest_phone", "9876543210"),
        ("my name is asha rao and my email is a@b.co", "guest_name", "Asha Rao"),
    ],
)
def test_extract_single_values(text, key, value):
    assert extract_params(text, TODAY).params[key] == value


def test_nights_are_extracted_separately():
    assert extract_params("for 3 nights", TODAY).nights == 3
    assert extract_params("for three nights", TODAY).nights == 3


def test_iso_date_is_not_mistaken_for_a_phone_number():
    assert "guest_phone" not in extract_params("stay 2026-07-01 to 2026-07-03", TODAY).params


def test_numbers_inside_dates_are_not_guest_counts():
    found = extract_params("5 to 8 August", TODAY)
    assert "guests" not in found.params and "rooms" not in found.params


@pytest.mark.parametrize(
    "awaiting, reply, key, value",
    [
        ("guests", "3", "guests", 3),
        ("rooms", "two", "rooms", 2),
        ("city", "navi mumbai", "city", "Navi Mumbai"),
        ("guest_name", "Ravi Kumar", "guest_name", "Ravi Kumar"),
        ("guest_name", "I'm Ravi Kumar", "guest_name", "Ravi Kumar"),
    ],
)
def test_bare_replies_use_the_question_asked(awaiting, reply, key, value):
    assert extract_params(reply, TODAY, awaiting=awaiting).params[key] == value


def test_bare_number_without_a_question_is_ignored():
    assert extract_params("3", TODAY).params == {}


def test_yes_is_not_taken_as_a_name():
    assert "guest_name" not in extract_params("yes", TODAY, awaiting="guest_name").params


@pytest.mark.parametrize("text", ["yes", "Yes!", "yes please", "sure", "go ahead", "Confirm", "book it", "okay"])
def test_confirmations(text):
    assert classify_confirmation(text, "book", False) == "confirmed"


@pytest.mark.parametrize("text", ["no", "nope", "don't", "not now", "wait", "no, keep it", "cancel", "stop"])
def test_declines(text):
    assert classify_confirmation(text, "book", False) == "declined"


@pytest.mark.parametrize("text", ["hmm", "what's the weather", "maybe", "can you repeat?", "yes but make it 2 rooms"])
def test_unclear_replies_are_never_consent(text):
    assert classify_confirmation(text, "book", False) != "confirmed"


def test_new_information_means_changed():
    assert classify_confirmation("yes", "book", has_new_info=True) == "changed"


def test_cancel_confirmation_accepts_cancel_it():
    assert classify_confirmation("yes cancel it", "cancel", False) == "confirmed"


def test_abort_and_retry_detection():
    assert is_abort("never mind") and is_abort("forget it") and not is_abort("sure")
    assert is_retry("please retry") and is_retry("try again") and not is_retry("yes")
