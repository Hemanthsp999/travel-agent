"""Intent classification: rules decide what is unambiguous, the LLM output is validated and grounded."""

from __future__ import annotations

import pytest

from tests.flight_agent.conftest import FakeLLM
from travel_agent.flight_agent.intent import ClassifyContext, classify, grounded, rule_classify
from travel_agent.flight_agent.schemas import GeneralTopic as T, Intent as I

CTX = ClassifyContext()


@pytest.mark.parametrize("text, intent", [
    ("flights from Bangalore to Mumbai tomorrow", I.CHECK_FLIGHT_AVAILABILITY),
    ("is there any flight available?", I.CHECK_FLIGHT_AVAILABILITY),
    ("book flight FL-1001", I.BOOK_FLIGHT),
    ("I'll take the second one", I.BOOK_FLIGHT),
    ("cancel my booking FB-ABC123", I.CANCEL_FLIGHT),
    ("cancellation please", I.CANCEL_FLIGHT),
    ("what's the status of FB-ABC123", I.GET_BOOKING_STATUS),
    ("FB-ABC123", I.GET_BOOKING_STATUS),
    ("hello", I.GENERAL_FLIGHT_QUERY),
    ("asdf qwer", I.UNKNOWN),
])
def test_rules(text: str, intent: I) -> None:
    assert rule_classify(text).intent is intent


@pytest.mark.parametrize("text, topic", [
    ("tell me about the first one", T.FLIGHT_DETAILS),
    ("compare them", T.COMPARE_FLIGHTS),
    ("show all", T.SHOW_MORE),
    ("what's your baggage allowance?", T.UNSUPPORTED),
    ("I want a refund", T.UNSUPPORTED),
    ("find me a hotel in Goa", T.UNSUPPORTED),
    ("can you change my seat?", T.UNSUPPORTED),
    ("hi", T.HELP),
])
def test_general_topics(text: str, topic: T) -> None:
    found = rule_classify(text)
    assert found.intent is I.GENERAL_FLIGHT_QUERY and found.topic is topic


def test_cancelling_by_booking_id_is_not_treated_as_unsupported() -> None:
    assert rule_classify("cancel FB-ABC123, I want a refund").intent is I.CANCEL_FLIGHT


def test_without_a_model_rules_are_used() -> None:
    assert classify(None, "book the first one", CTX).source == "rules"


def test_llm_label_is_used_when_supported_by_the_wording() -> None:
    llm = FakeLLM({"intent": "CANCEL_FLIGHT", "confidence": 0.9, "general_topic": "NONE", "origin": None, "destination": None})
    found = classify(llm, "please cancel it", CTX)
    assert found.intent is I.CANCEL_FLIGHT and found.source == "llm"


def test_implausible_llm_intent_is_ignored() -> None:
    llm = FakeLLM({"intent": "CANCEL_FLIGHT", "confidence": 0.99, "general_topic": "NONE", "origin": None, "destination": None})
    assert classify(llm, "flights from Bangalore to Mumbai", CTX).intent is I.CHECK_FLIGHT_AVAILABILITY


def test_low_confidence_llm_output_falls_back_to_rules() -> None:
    llm = FakeLLM({"intent": "BOOK_FLIGHT", "confidence": 0.1, "general_topic": "NONE", "origin": None, "destination": None})
    assert classify(llm, "any flights to Mumbai?", CTX).intent is I.CHECK_FLIGHT_AVAILABILITY


def test_model_failure_falls_back_to_rules() -> None:
    assert classify(FakeLLM(None), "cancel FB-ABC123", CTX).intent is I.CANCEL_FLIGHT


def test_explicit_book_wording_beats_a_vague_search_label() -> None:
    llm = FakeLLM({"intent": "CHECK_FLIGHT_AVAILABILITY", "confidence": 0.9, "general_topic": "NONE", "origin": None, "destination": None})
    assert classify(llm, "book the cheapest one", CTX).intent is I.BOOK_FLIGHT


def test_unsupported_wording_beats_a_search_label() -> None:
    llm = FakeLLM({"intent": "CHECK_FLIGHT_AVAILABILITY", "confidence": 0.9, "general_topic": "NONE", "origin": None, "destination": None})
    found = classify(llm, "what is the baggage allowance", CTX)
    assert found.intent is I.GENERAL_FLIGHT_QUERY and found.topic is T.UNSUPPORTED


def test_cities_from_the_llm_must_appear_in_the_message() -> None:
    llm = FakeLLM({"intent": "CHECK_FLIGHT_AVAILABILITY", "confidence": 0.9, "general_topic": "NONE", "origin": "Paris", "destination": "Mumbai"})
    found = classify(llm, "any flights to Mumbai", CTX)
    assert found.origin is None and found.destination == "Mumbai"


def test_topic_is_only_kept_for_general_queries() -> None:
    llm = FakeLLM({"intent": "BOOK_FLIGHT", "confidence": 0.9, "general_topic": "HELP", "origin": None, "destination": None})
    assert classify(llm, "book it", CTX).topic is T.NONE


def test_grounding_helper() -> None:
    assert grounded("Mumbai", "flights to mumbai!")
    assert not grounded("Paris", "flights to Mumbai")
    assert not grounded(None, "anything") and not grounded("A", "A flight")
