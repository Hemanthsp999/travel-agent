from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.orchestrator.graph import Orchestrator
from travel_agent.retrieval import KnowledgeRetriever


class FakeAgent:
    def __init__(self):
        self.calls = []

    def run_task(self, task):
        self.calls.append(task)
        return type(
            "Result",
            (),
            {
                "status": "SUCCESS",
                "intent": None,
                "data": {},
                "error": None,
                "requires_user_input": False,
                "missing_parameters": [],
                "message": "Handled by subagent.",
                "question": None,
            },
        )()


def test_knowledge_retriever_indexes_mock_entity_specific_policies():
    retriever = KnowledgeRetriever()

    hotel_policy = retriever.retrieve("What is the cancellation policy for HT-1001?")
    other_hotel_policy = retriever.retrieve("What is the cancellation policy for HT-1002?")
    flight_policy = retriever.retrieve("What is the flight cancellation policy for FL-1001?")

    assert "HT-1001" in hotel_policy["context"]
    assert "48 hours" in hotel_policy["context"]
    assert "HT-1002" in other_hotel_policy["context"]
    assert "24 hours" in other_hotel_policy["context"]
    assert "FL-1001" in flight_policy["context"]
    assert "24 hours" in flight_policy["context"]


def test_policy_question_uses_rag_without_invoking_subagents():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    result = agent.chat("What is the hotel cancellation policy?", thread_id="rag-policy")

    assert result["execution_status"] == "SUCCESS"
    assert "48 hours" in result["final_response"]
    assert not hotel.calls
    assert not flight.calls
    assert result["retrieved_knowledge"]["source_refs"]


def test_rag_policy_answer_formats_each_condition_as_a_separate_bullet():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    result = agent.chat("What is the hotel cancellation policy?", thread_id="rag-policy-format")
    response = result["final_response"]
    points = [line for line in response.splitlines() if line.startswith("- ")]

    assert response.startswith("Here’s what I found:")
    assert "**Hotel cancellation policy**" in response
    assert response.count("Here’s what I found:") == 1
    assert len(points) >= 3
    assert any("48 hours" in point for point in points)
    assert any("Flexible rates" in point for point in points)
    assert not hotel.calls
    assert not flight.calls


def test_entity_specific_rule_answer_separates_distinct_policy_facts():
    knowledge = KnowledgeRetriever().retrieve("What are the rules for hotel HT-1001?")
    answer = knowledge["answer"]

    assert "**Lakeview Grand Bengaluru (HT-1001) rules**" in answer
    assert "- Standard check-in is 2:00 PM." in answer
    assert "- Check-out is 11:00 AM." in answer
    assert "- Maximum guests per room: 3." in answer


def test_retrieved_entity_context_is_forwarded_to_the_relevant_subagent():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    agent.chat("Book hotel HT-1001", thread_id="rag-context")

    assert len(hotel.calls) == 1
    assert "HT-1001" in hotel.calls[0].context["retrieved_context"]
    assert hotel.calls[0].context["retrieval_metadata"]["domain"] == "hotel"
    assert not flight.calls


def test_live_flight_price_enquiry_still_calls_the_flight_subagent():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    result = agent.chat("What is the price for flight FL-1001?", thread_id="rag-live-price")

    assert result["execution_status"] == "SUCCESS"
    assert len(flight.calls) == 1
    assert flight.calls[0].context is None
    assert not hotel.calls


def test_booking_request_with_a_policy_keyword_still_calls_the_subagent():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    agent.chat("Book a flight with checked baggage from Bangalore to Mumbai", thread_id="rag-booking")

    assert len(flight.calls) == 1
    assert not hotel.calls


def test_cancellation_how_to_faq_uses_rag_but_cancellation_action_calls_agent():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    answer = agent.chat("How do I cancel my flight booking?", thread_id="rag-cancel-faq")
    assert answer["execution_status"] == "SUCCESS"
    assert "does not cancel" in answer["final_response"].lower()
    assert not hotel.calls
    assert not flight.calls

    agent.chat("Cancel my flight booking", thread_id="rag-cancel-action")
    assert len(flight.calls) == 1


def test_refund_eligibility_followup_retains_previously_retrieved_hotel_policy():
    hotel, flight = FakeAgent(), FakeAgent()
    agent = Orchestrator(hotel, flight, checkpointer=InMemorySaver(), retriever=KnowledgeRetriever())

    agent.chat("What is the cancellation policy for HT-1001?", thread_id="rag-refund-followup")
    result = agent.chat("Am I eligible for a refund?", thread_id="rag-refund-followup")

    assert result["execution_status"] == "SUCCESS"
    assert "HT-1001" in result["retrieved_knowledge"]["context"]
    assert "48 hours" in result["final_response"]
    assert "can’t confirm refund eligibility" in result["final_response"]
    assert "check-in date and room rate type" in result["final_response"]
    assert not hotel.calls
    assert not flight.calls


def test_refund_eligibility_without_context_asks_for_booking_details():
    knowledge = KnowledgeRetriever().retrieve("Am I eligible for a refund?")

    assert knowledge["answer"] is not None
    assert "can’t determine refund eligibility" in knowledge["answer"]
    assert "hotel or a flight" in knowledge["answer"]
    assert not any(term in knowledge["answer"].lower() for term in ("you are eligible", "you qualify"))
