from __future__ import annotations

from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from travel_agent.orchestrator.graph import Orchestrator


class FakeAgent:
    def __init__(self, domain: str, *, status: str = "SUCCESS", message: str | None = None):
        self.domain = domain
        self.status = status
        self.message = message or f"{domain.title()} request handled successfully."
        self.calls: list[object] = []

    def run_task(self, task):
        self.calls.append(task)
        return SimpleNamespace(
            status=self.status,
            intent=None,
            data={"domain": self.domain},
            error=None,
            requires_user_input=self.status in {"REQUIRES_USER_INPUT", "PENDING"},
            missing_parameters=["date"] if self.status in {"REQUIRES_USER_INPUT", "PENDING"} else [],
            message=self.message,
            question="What date do you want?" if self.status in {"REQUIRES_USER_INPUT", "PENDING"} else None,
        )


def _check(query: str, *, expected_status: str, expected_domains: set[str] | None = None, expected_calls: dict[str, int] | None = None):
    hotel_agent = FakeAgent("hotel")
    flight_agent = FakeAgent("flight")
    orchestrator = Orchestrator(hotel_agent, flight_agent, checkpointer=InMemorySaver())

    result = orchestrator.chat(query, thread_id=f"score-{abs(hash(query)) % 100000}")

    assert result["execution_status"] == expected_status, (
        f"Query: {query!r}\nExpected status: {expected_status}\nActual status: {result['execution_status']}\nResponse: {result.get('final_response')}"
    )

    if expected_domains is not None:
        passed_domains = set()
        if hotel_agent.calls:
            passed_domains.add("hotel")
        if flight_agent.calls:
            passed_domains.add("flight")
        assert passed_domains == expected_domains, (
            f"Query: {query!r}\nExpected domains: {expected_domains}\nActual domains: {passed_domains}"
        )

    if expected_calls is not None:
        actual_calls = {"hotel": len(hotel_agent.calls), "flight": len(flight_agent.calls)}
        assert actual_calls == expected_calls, (
            f"Query: {query!r}\nExpected calls: {expected_calls}\nActual calls: {actual_calls}"
        )

    return result


def test_agent_query_battery():
    cases = [
        {
            "query": "Find flights from Bangalore to Mumbai",
            "expected_status": "SUCCESS",
            "expected_domains": {"flight"},
            "expected_calls": {"hotel": 0, "flight": 1},
        },
        {
            "query": "Find hotels in Bangalore for 3 nights",
            "expected_status": "SUCCESS",
            "expected_domains": {"hotel"},
            "expected_calls": {"hotel": 1, "flight": 0},
        },
        {
            "query": "Book a flight and hotel in Goa",
            "expected_status": "SUCCESS",
            "expected_domains": {"flight", "hotel"},
            "expected_calls": {"hotel": 1, "flight": 1},
        },
        {
            "query": "Cancel my booking",
            "expected_status": "REQUIRES_USER_INPUT",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "What can you do?",
            "expected_status": "SUCCESS",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "Book a train to Mumbai",
            "expected_status": "UNSUPPORTED",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "I need a hotel in Pune",
            "expected_status": "SUCCESS",
            "expected_domains": {"hotel"},
            "expected_calls": {"hotel": 1, "flight": 0},
        },
        {
            "query": "I need a flight to Delhi",
            "expected_status": "SUCCESS",
            "expected_domains": {"flight"},
            "expected_calls": {"hotel": 0, "flight": 1},
        },
        {
            "query": "Find a taxi to the airport",
            "expected_status": "UNSUPPORTED",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "Hello there",
            "expected_status": "SUCCESS",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "Plan a trip to Goa for 4 nights",
            "expected_status": "SUCCESS",
            "expected_domains": set(),
            "expected_calls": {"hotel": 0, "flight": 0},
        },
        {
            "query": "Find flights and hotels for Delhi",
            "expected_status": "SUCCESS",
            "expected_domains": {"flight", "hotel"},
            "expected_calls": {"hotel": 1, "flight": 1},
        },
    ]

    passed = 0
    total = len(cases)

    for item in cases:
        query = item["query"]
        try:
            _check(
                query,
                expected_status=item["expected_status"],
                expected_domains=item["expected_domains"],
                expected_calls=item["expected_calls"],
            )
            passed += 1
            print(f"PASS: {query} -> {item['expected_status']}")
        except AssertionError as exc:
            print(f"FAIL: {query}\n{exc}\n")

    score = (passed / total) * 100
    print(f"\nAgent performance score: {score:.2f}% ({passed}/{total})")

    assert passed == total, f"Query battery failed: {passed}/{total} passed."
