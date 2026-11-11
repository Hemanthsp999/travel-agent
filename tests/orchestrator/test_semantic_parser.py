from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.orchestrator.graph import Orchestrator
from travel_agent.orchestrator.semantic_parser import extract_semantic_request


def test_extract_semantic_request_parses_fixed_schema():
    text = (
        "I want to travel from Bengaluru to Goa for three days. "
        "Find me a flight and a hotel. I prefer a hotel near the beach."
    )

    parsed = extract_semantic_request(text)

    assert parsed["origin"] == "Bangalore"
    assert parsed["destination"] == "Goa"
    assert parsed["hotel_city"] == "Goa"
    assert parsed["nights"] == 3
    assert parsed["preferences"] == ["beach"]
    assert parsed["passengers"] == 1


def test_extract_semantic_request_handles_common_date_typos():
    parsed = extract_semantic_request("I am travelling tommorow from Bengaluru to Mumbai")

    assert parsed["origin"] == "Bangalore"
    assert parsed["destination"] == "Mumbai"
    assert parsed["travel_date"] is not None


def test_extract_semantic_request_extracts_hotel_city_from_hotel_only_queries():
    parsed = extract_semantic_request("I need a hotel in Goa for 3 nights")

    assert parsed["hotel_city"] == "Goa"
    assert parsed["nights"] == 3


def test_orchestrator_keeps_semantic_slots_in_execution_plan():
    orch = Orchestrator(checkpointer=InMemorySaver())
    result = orch.chat(
        "I want to travel from Bengaluru to Goa for three days. Find me a flight and a hotel.",
        thread_id="semantic-slot-test",
    )

    slots = result["execution_plan"].get("semantic_slots")
    assert slots is not None
    assert slots["origin"] == "Bangalore"
    assert slots["destination"] == "Goa"
    assert slots["hotel_city"] == "Goa"
    assert slots["nights"] == 3
