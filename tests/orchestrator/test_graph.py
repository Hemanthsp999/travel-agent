from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.orchestrator.graph import Orchestrator


class FakeAgent:
    def __init__(self, status="SUCCESS", message="verified result"):
        self.calls = []
        self.status = status
        self.message = message

    def run_task(self, task):
        self.calls.append(task)
        return SimpleNamespace(status=self.status, intent=None, data={}, error=None,
                               requires_user_input=self.status == "REQUIRES_USER_INPUT",
                               missing_parameters=[], message=self.message, question=None)


def setup(hotel=None, flight=None):
    h, f = hotel or FakeAgent(), flight or FakeAgent()
    return Orchestrator(h, f, checkpointer=InMemorySaver()), h, f


def test_hotel_and_flight_requests_route_only_to_their_agent():
    orch, hotel, flight = setup()
    assert orch.chat("Find hotels in Bangalore", "h")['execution_status'] == "SUCCESS"
    assert len(hotel.calls) == 1 and not flight.calls
    orch.chat("Find flights from Bangalore to Mumbai", "f")
    assert len(flight.calls) == 1


def test_both_domains_fan_out_and_aggregate():
    orch, hotel, flight = setup()
    result = orch.chat("Find a flight and hotel for Mumbai", "both")
    assert result["execution_status"] == "SUCCESS"
    assert len(hotel.calls) == len(flight.calls) == 1


def test_ambiguous_cancellation_does_not_invoke_an_agent():
    orch, hotel, flight = setup()
    result = orch.chat("Cancel my booking", "ambiguous")
    assert result["execution_status"] == "REQUIRES_USER_INPUT"
    assert not hotel.calls and not flight.calls


def test_partial_failure_is_reported():
    orch, hotel, flight = setup(FakeAgent("FAILED", "Hotel lookup failed"), FakeAgent())
    result = orch.chat("Find a flight and hotel for Mumbai", "partial")
    assert result["execution_status"] == "PARTIAL_SUCCESS"
    assert "Hotel lookup failed" in result["final_response"]


def test_pending_followup_keeps_agent_thread():
    h = FakeAgent("REQUIRES_USER_INPUT", "What date?")
    orch, _, _ = setup(hotel=h)
    orch.chat("I need a hotel in Mumbai", "resume")
    orch.chat("November 10", "resume")
    assert [t.thread_id for t in h.calls] == ["resume:hotel", "resume:hotel"]


def test_completed_flight_query_does_not_block_a_new_hotel_query_in_same_thread():
    orch, hotel, flight = setup()
    orch.chat("Find flights from Bangalore to Mumbai", "cross-domain")
    result = orch.chat("Find hotels in Goa for 3 nights", "cross-domain")

    assert result["execution_plan"]["domains"] == ["hotel"]
    assert result["execution_status"] == "SUCCESS"
    assert len(hotel.calls) == 1
    assert len(flight.calls) == 1


def test_followup_book_one_reuses_previous_domain():
    orch, hotel, flight = setup()
    orch.chat("Find flights from Bangalore to Mumbai", "book-one")
    result = orch.chat("book one", "book-one")

    assert result["execution_plan"]["domains"] == ["flight"]
    assert result["execution_status"] == "SUCCESS"
    assert len(flight.calls) == 2
    assert len(hotel.calls) == 0


def test_orchestrator_passes_structured_context_to_subagents():
    orch, hotel, flight = setup()
    orch.chat("Find flights from Bangalore to Mumbai on 10 Nov for 2 passengers", "context-check")

    assert flight.calls[0].intent.value == "CHECK_FLIGHT_AVAILABILITY"
    assert flight.calls[0].params["origin"] == "Bangalore"
    assert flight.calls[0].params["destination"] == "Mumbai"
    assert flight.calls[0].params["departure_date"]
    assert flight.calls[0].params["passenger_count"] == 2
    assert hotel.calls == []


def test_retrieval_policy_queries_are_answered_humanely_without_unknown_fallback():
    orch, hotel, flight = setup()
    result = orch.chat("what's the refund policy?", "policy-query")

    assert result["execution_status"] == "SUCCESS"
    assert "refund" in result["final_response"].lower()
    assert "not sure i caught that" not in result["final_response"].lower()
    assert len(flight.calls) == 0
    assert len(hotel.calls) == 0


def test_cancellation_policy_question_is_answered_from_rag_not_cancel_flow():
    orch, hotel, flight = setup()
    result = orch.chat("what is the cancellation policy?", "policy-question")

    assert result["execution_status"] == "SUCCESS"
    assert "cancellation" in result["final_response"].lower()
    assert "cancel it" not in result["final_response"].lower()
    assert len(flight.calls) == 0
    assert len(hotel.calls) == 0


def test_hotel_policy_question_prefers_hotel_policy_over_flight_policy():
    orch, hotel, flight = setup()
    result = orch.chat("what is the hotel cancellation policy?", "hotel-policy")

    assert result["execution_status"] == "SUCCESS"
    assert "48 hours" in result["final_response"]
    assert "standard rooms" in result["final_response"].lower()
    assert "24 hours" not in result["final_response"]
    assert len(flight.calls) == 0
    assert len(hotel.calls) == 0


def test_flight_policy_question_prefers_flight_policy_over_hotel_policy():
    orch, hotel, flight = setup()
    result = orch.chat("what is the flight cancellation policy?", "flight-policy")

    assert result["execution_status"] == "SUCCESS"
    assert "24 hours" in result["final_response"]
    assert "refundable fares" in result["final_response"].lower()
    assert "48 hours" not in result["final_response"]
    assert len(flight.calls) == 0
    assert len(hotel.calls) == 0


def test_name_question_gets_a_conversational_response_without_agent_clarification():
    orch, hotel, flight = setup()

    result = orch.chat("May I know your name?", "assistant-name")

    assert result["execution_status"] == "SUCCESS"
    assert "travel assistant" in result["final_response"].lower()
    assert "hotel reservation or a flight booking" not in result["final_response"].lower()
    assert not hotel.calls
    assert not flight.calls


def test_unknown_conversational_request_asks_an_open_friendly_question():
    orch, hotel, flight = setup()

    result = orch.chat("I have a question", "open-clarification")

    assert result["execution_status"] == "SUCCESS"
    assert "tell me a little more" in result["final_response"].lower()
    assert "hotel reservation or a flight booking" not in result["final_response"].lower()
    assert not hotel.calls
    assert not flight.calls


def test_subagent_results_are_presented_by_the_orchestrator_in_a_human_tone():
    orch, hotel, flight = setup()

    result = orch.chat("Find hotels in Bangalore", "humanized-result")

    assert result["final_response"].startswith("Of course —")
    assert "verified result" in result["final_response"]
    assert len(hotel.calls) == 1
    assert not flight.calls
