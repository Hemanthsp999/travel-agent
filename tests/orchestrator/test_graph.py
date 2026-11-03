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
