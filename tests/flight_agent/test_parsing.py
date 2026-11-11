"""Rule-based extraction: exact values only, no guessing."""

from __future__ import annotations

from datetime import date

import pytest

from travel_agent.flight_agent.parsing import (
    canonical_key, extract, is_known_city, merge_passengers, normalize_place, parse_route,
)

TODAY = date(2026, 11, 2)


@pytest.mark.parametrize("text, expected", [
    ("flights from Bangalore to Mumbai", ("Bangalore", "Mumbai")),
    ("flights from BLR to BOM", ("Bangalore", "Mumbai")),
    ("Delhi -> Chennai", ("Delhi", "Chennai")),
    ("I want to go to Mumbai from Delhi", ("Delhi", "Mumbai")),
    ("between Hyderabad and Bangalore", ("Hyderabad", "Bangalore")),
    ("flights from Pune to Goa", ("Pune", "Goa")),
    ("book a flight", (None, None)),
])
def test_route_parsing(text: str, expected: tuple[str | None, str | None]) -> None:
    assert parse_route(text) == expected


def test_a_bare_reply_fills_only_the_slot_being_asked() -> None:
    assert parse_route("Mumbai", awaiting="destination") == (None, "Mumbai")
    assert parse_route("Mumbai", awaiting="origin") == ("Mumbai", None)
    assert parse_route("Mumbai") == (None, None)


def test_place_normalisation() -> None:
    assert normalize_place("BLR") == "Bangalore"
    assert normalize_place("bengaluru") == "Bangalore"
    assert canonical_key("BOM") == canonical_key("mumbai")
    assert is_known_city("Delhi") and not is_known_city("Atlantis")


def test_dates_are_parsed_by_code() -> None:
    assert extract("on 12 Nov", TODAY).departure_date == date(2026, 11, 12)
    assert extract("tomorrow", TODAY).departure_date == date(2026, 11, 3)
    assert extract("tommorow", TODAY).departure_date == date(2026, 11, 3)
    assert extract("2026-12-05", TODAY).departure_date == date(2026, 12, 5)
    assert extract("sometime soon", TODAY).departure_date is None


def test_route_parsing_strips_date_typos_from_city_names() -> None:
    text = "Hi, I'm looking for flights from Bengaluru to mumbai tommorow, get me tickets"
    assert parse_route(text) == ("Bangalore", "Mumbai")


def test_two_dates_are_flagged_instead_of_picking_silently() -> None:
    found = extract("from 12 Nov to 15 Nov", TODAY)
    assert found.extra_dates == 1


@pytest.mark.parametrize("text, count", [
    ("for 3 passengers", 3),
    ("2 adults and 1 child", 3),
    ("for two people", 2),
    ("just me", 1),
    ("a flight to Mumbai", None),
])
def test_passenger_counts(text: str, count: int | None) -> None:
    assert extract(text, TODAY).passenger_count == count


def test_a_bare_number_is_a_count_only_when_one_was_asked_for() -> None:
    assert extract("3", TODAY, awaiting="passenger_count").passenger_count == 3
    assert extract("3", TODAY).passenger_count is None


def test_dates_are_not_mistaken_for_counts() -> None:
    found = extract("flight on 12 Nov", TODAY)
    assert found.passenger_count is None and found.departure_date == date(2026, 11, 12)


def test_identifiers_are_extracted_verbatim() -> None:
    found = extract("cancel fb-ab12cd for flight FL-1004 (6E-501)", TODAY)
    assert found.booking_id == "FB-AB12CD" and found.flight_id == "FL-1004" and found.flight_number == "6E-501"


def test_contact_details() -> None:
    found = extract("mail Asha@Example.com or call +91 98765 43210", TODAY, awaiting="contact_details")
    assert found.email == "asha@example.com" and found.phone is not None
    assert "".join(ch for ch in found.phone if ch.isdigit()) == "919876543210"


def test_passengers_while_awaiting_details() -> None:
    found = extract("Asha Rao, 31, female and Ravi Rao, 35, male", TODAY, awaiting="passenger_details")
    assert merge_passengers([], found.passengers, 2) == [
        {"name": "Asha Rao", "age": 31, "gender": "FEMALE"},
        {"name": "Ravi Rao", "age": 35, "gender": "MALE"},
    ]


def test_partial_passenger_details_are_kept_as_pieces() -> None:
    assert extract("Asha Rao", TODAY, awaiting="passenger_details").passengers == [{"name": "Asha Rao"}]
    pieces = extract("31, female", TODAY, awaiting="passenger_details").passengers
    assert pieces == [{"age": 31}, {"gender": "FEMALE"}]


def test_passengers_inside_a_sentence_need_a_complete_record() -> None:
    complete = extract("book the first one for Asha Rao, 31, female", TODAY, allow_passengers=True)
    assert complete.passengers == [{"name": "Asha Rao", "age": 31, "gender": "FEMALE"}]
    for text in ("book it for 2 people on 12 Nov", "book the first one for Asha", "book flight to Mumbai 2 other"):
        assert extract(text, TODAY, allow_passengers=True).passengers == []


def test_implausible_age_is_not_accepted() -> None:
    assert extract("Asha Rao 150 female", TODAY, awaiting="passenger_details").passengers[0].get("age") is None


def test_merge_never_invents_values() -> None:
    people = merge_passengers([], [{"name": "Asha Rao"}], 2)
    people = merge_passengers(people, [{"age": 31, "gender": "FEMALE"}], 2)
    people = merge_passengers(people, [{"name": "Ravi Rao"}], 2)
    people = merge_passengers(people, [{"name": "Extra Person"}], 2)  # beyond the count: ignored
    assert people == [{"name": "Asha Rao", "age": 31, "gender": "FEMALE"}, {"name": "Ravi Rao"}]
