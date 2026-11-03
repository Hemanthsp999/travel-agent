from travel_agent.orchestrator.cli import run_chat


class FakeOrchestrator:
    def __init__(self):
        self.turns = []

    def chat(self, message, thread_id):
        self.turns.append((message, thread_id))
        return {"final_response": "What date would you like?"}


def test_cli_keeps_thread_and_relays_interactive_question():
    agent = FakeOrchestrator()
    inputs = iter(["I need a hotel", "November 10", "quit"])
    output = []
    run_chat(agent, "trip-1", read=lambda _: next(inputs), write=output.append)
    assert agent.turns == [("I need a hotel", "trip-1"), ("November 10", "trip-1")]
    assert any("What date would you like?" in line for line in output)
