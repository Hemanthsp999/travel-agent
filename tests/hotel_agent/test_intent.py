"""Intent classification: LLM output is validated and grounded; rules are the safety net."""

from __future__ import annotations

import pytest

from tests.hotel_agent.conftest import FakeLLM, day, say
from travel_agent.hotel_agent.intent import (
    ClassifyContext, classify, find_hotels_by_name, grounded, normalize_city, resolve_hotel, rule_city,
    rule_classify,
)
from travel_agent.hotel_agent.schemas import GeneralTopic, Intent, TaskStatus

HOTELS = [
    {"hotel_id": "HT-1001", "name": "Lakeview Grand Bengaluru", "price_per_night": 9500, "rating": 4.6},
    {"hotel_id": "HT-1002", "name": "Indiranagar Urban Stay", "price_per_night": 5200, "rating": 4.2},
    {"hotel_id": "HT-1003", "name": "Whitefield Business Inn", "price_per_night": 3400, "rating": 3.9},
]


def payload(intent="CHECK_AVAILABILITY", confidence=0.9, topic="NONE", city=None, hotel=None, guest=None):
    return {"intent": intent, "confidence": confidence, "general_topic": topic,
            "city": city, "hotel_name": hotel, "guest_name": guest}


@pytest.mark.parametrize(
    "text, intent",
    [
        ("Show me hotels in Pune", Intent.CHECK_AVAILABILITY),
        ("any rooms available tomorrow?", Intent.CHECK_AVAILABILITY),
        ("I want to book the second one", Intent.BOOK_HOTEL),
        ("please reserve HT-1002", Intent.BOOK_HOTEL),
        ("cancel my booking HB-0123456789", Intent.CANCEL_BOOKING),
        ("what is the status of HB-0123456789", Intent.GET_BOOKING_STATUS),
        ("HB-0123456789", Intent.GET_BOOKING_STATUS),
        ("compare them", Intent.GENERAL_HOTEL_QUERY),
        ("tell me more about HT-1001", Intent.GENERAL_HOTEL_QUERY),
        ("hello", Intent.GENERAL_HOTEL_QUERY),
        ("what's your refund policy", Intent.GENERAL_HOTEL_QUERY),
        ("blah blah", Intent.UNKNOWN),
    ],
)
def test_rule_classifier(text, intent):
    assert rule_classify(text).intent is intent


def test_rule_topics():
    assert rule_classify("compare them").topic is GeneralTopic.COMPARE_HOTELS
    assert rule_classify("what's your refund policy").topic is GeneralTopic.UNSUPPORTED
    assert rule_classify("show me more options").topic is GeneralTopic.SHOW_MORE


@pytest.mark.parametrize(
    "text, city",
    [
        ("hotels in Bengaluru", "Bangalore"),
        ("stay in new delhi", "Delhi"),
        ("somewhere near Madras", "Chennai"),
        ("hotels in Goa from 5 Aug", "Goa"),
        ("hotels in Navi Mumbai for 2", "Mumbai"),
        ("check in at asha@example.com", None),
        ("in the morning", None),
        ("in advance", None),
    ],
)
def test_rule_city(text, city):
    assert rule_city(text) == city


def test_llm_result_is_used_when_confident():
    llm = FakeLLM(payload("CHECK_AVAILABILITY", 0.9, city="Pune"))
    result = classify(llm, "got any rooms in Pune?", ClassifyContext())
    assert result.intent is Intent.CHECK_AVAILABILITY and result.source == "llm" and result.city == "Pune"
    assert llm.calls


def test_hallucinated_city_is_dropped():
    llm = FakeLLM(payload(city="Paris"))
    result = classify(llm, "any rooms tomorrow?", ClassifyContext())
    assert result.city is None


def test_hallucinated_hotel_and_guest_names_are_dropped():
    llm = FakeLLM(payload("BOOK_HOTEL", hotel="Taj Mahal Palace", guest="John Smith"))
    result = classify(llm, "book the cheapest one", ClassifyContext())
    assert result.hotel_name is None and result.guest_name is None


def test_low_confidence_falls_back_to_rules():
    llm = FakeLLM(payload("CANCEL_BOOKING", 0.2))
    result = classify(llm, "show me hotels in Pune", ClassifyContext())
    assert result.source == "rules" and result.intent is Intent.CHECK_AVAILABILITY


def test_implausible_cancel_is_rejected():
    llm = FakeLLM(payload("CANCEL_BOOKING", 0.95))
    result = classify(llm, "show me hotels in Pune", ClassifyContext())
    assert result.intent is Intent.CHECK_AVAILABILITY


def test_implausible_booking_is_rejected():
    llm = FakeLLM(payload("BOOK_HOTEL", 0.95))
    result = classify(llm, "what hotels are in Pune", ClassifyContext())
    assert result.intent is not Intent.BOOK_HOTEL


def test_llm_crash_falls_back_to_rules():
    result = classify(FakeLLM(None), "show me hotels in Pune", ClassifyContext())
    assert result.intent is Intent.CHECK_AVAILABILITY and result.source == "rules"


def test_llm_unknown_does_not_override_clear_rules():
    result = classify(FakeLLM(payload("UNKNOWN", 0.9)), "cancel HB-0123456789", ClassifyContext())
    assert result.intent is Intent.CANCEL_BOOKING


def test_grounded_and_normalisation():
    assert grounded("Pune", "hotels in pune please") and not grounded("Paris", "hotels in pune")
    assert not grounded(None, "x") and not grounded("a", "a b")
    assert normalize_city("BENGALURU") == "Bangalore" and normalize_city("goa") == "Goa"


# ---------------------------------------------------------------------- hotel references


@pytest.mark.parametrize(
    "text, hotel_id",
    [
        ("the first one", "HT-1001"), ("2nd", "HT-1002"), ("option 3", "HT-1003"), ("the last one", "HT-1003"),
        ("the cheapest", "HT-1003"), ("best rated", "HT-1001"), ("book HT-1002", "HT-1002"),
        ("Indiranagar", "HT-1002"), ("the Whitefield one", "HT-1003"), ("Lakeview Grand Bengaluru", "HT-1001"),
    ],
)
def test_resolve_hotel(text, hotel_id):
    hit, ambiguous = resolve_hotel(text, HOTELS)
    assert hit and hit["hotel_id"] == hotel_id and not ambiguous


def test_out_of_range_or_unknown_reference_resolves_to_nothing():
    assert resolve_hotel("the fifth one", HOTELS) == (None, [])
    assert resolve_hotel("HT-9999", HOTELS) == (None, [])
    assert resolve_hotel("Taj Mahal", HOTELS) == (None, [])
    assert resolve_hotel("the first one", []) == (None, [])


def test_bare_digit_needs_permission():
    assert resolve_hotel("2", HOTELS) == (None, [])
    assert resolve_hotel("2", HOTELS, allow_bare_number=True)[0]["hotel_id"] == "HT-1002"


def test_shared_word_is_ambiguous():
    both = [HOTELS[0], {**HOTELS[0], "hotel_id": "HT-1099", "name": "Lakeview Palace"}]
    hit, ambiguous = resolve_hotel("the Lakeview", both)
    assert hit is None and len(ambiguous) == 2


def test_single_shown_hotel_can_be_called_it():
    assert resolve_hotel("book it", HOTELS[:1])[0]["hotel_id"] == "HT-1001"
    assert find_hotels_by_name("", HOTELS) == []


# ------------------------------------------------------ through the whole agent with an LLM


def test_agent_ignores_llm_inventions_end_to_end(make_agent):
    agent = make_agent(llm=FakeLLM(payload("CHECK_AVAILABILITY", 0.9, city="Paris")))
    result = say(agent, f"any rooms {day(20)} to {day(22)}?")
    assert result.status is TaskStatus.NEEDS_INPUT and result.missing_params == ["city"]


def test_agent_survives_llm_failure_end_to_end(make_agent):
    agent = make_agent(llm=FakeLLM(None))
    result = say(agent, f"hotels in Pune {day(20)} to {day(22)}")
    assert result.status is TaskStatus.COMPLETED


def test_keyword_followups_beat_a_vague_llm_label():
    llm = FakeLLM(payload("CHECK_AVAILABILITY", 0.9, topic="BUDGET_SEARCH"))
    result = classify(llm, "tell me about the cheapest one", ClassifyContext())
    assert result.intent is Intent.GENERAL_HOTEL_QUERY and result.topic is GeneralTopic.HOTEL_DETAILS


def test_named_city_keeps_the_llm_search_label():
    llm = FakeLLM(payload("CHECK_AVAILABILITY", 0.9, city="Pune"))
    result = classify(llm, "hotels in Pune with gym amenities", ClassifyContext())
    assert result.intent is Intent.CHECK_AVAILABILITY and result.topic is GeneralTopic.NONE


def test_topic_is_cleared_for_non_general_intents():
    llm = FakeLLM(payload("GET_BOOKING_STATUS", 0.9, topic="UNSUPPORTED"))
    assert classify(llm, "status of HB-0123456789", ClassifyContext()).topic is GeneralTopic.NONE


def test_book_verb_beats_a_search_label_when_no_city_named():
    llm = FakeLLM(payload("CHECK_AVAILABILITY", 0.9))
    result = classify(llm, "book the second one", ClassifyContext())
    assert result.intent is Intent.BOOK_HOTEL


def test_status_label_needs_booking_wording():
    llm = FakeLLM(payload("GET_BOOKING_STATUS", 0.9))
    result = classify(llm, "hotels in Mumbai 2026-12-02 to 2026-12-04", ClassifyContext(current_intent="GET_BOOKING_STATUS"))
    assert result.intent is Intent.CHECK_AVAILABILITY
