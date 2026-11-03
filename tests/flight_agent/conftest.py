"""Fixtures for flight agent tests: the real Flight API (in-process, temp SQLite), a scripted LLM, and
failure-injecting HTTP wrappers. No test talks to a model or to the network."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_apis.common.db import connect, write_transaction
from mock_apis.flight_api.app import create_app
from mock_apis.flight_api.config import FlightSettings, load_settings as load_api_settings
from mock_apis.flight_api.seed import ensure_inventory
from travel_agent.config import Settings, load_settings
from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.api_client import FlightAPIClient
from travel_agent.flight_agent.schemas import FlightTaskResult

TINY_FLIGHT = "FL-9001"


def day(offset: int) -> date:
    return date.today() + timedelta(days=offset)


def nice(offset: int) -> str:
    """Date as a user would type it, e.g. '14 Nov'."""
    d = day(offset)
    return f"{d.day} {d.strftime('%b')}"


@pytest.fixture
def api_settings(tmp_path: Path) -> FlightSettings:
    return load_api_settings(tmp_path / "flight.db", horizon_days=200)


@pytest.fixture
def api(api_settings: FlightSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=api_settings)) as client:
        yield client


@pytest.fixture
def tiny_flight(api: TestClient, api_settings: FlightSettings) -> str:
    """A 3-seat flight Testville -> Nowhereburg so tests can sell it out quickly."""
    conn = connect(api_settings.db_path)
    try:
        with write_transaction(conn):
            conn.execute(
                """INSERT INTO flights VALUES (?, 'Test Air', 'TA', 'TA-1', 'Testville', 'TST',
                   'Nowhereburg', 'NWB', '23:30', 60, 2000, 'INR', 3, 'Cessna')""",
                (TINY_FLIGHT,),
            )
            ensure_inventory(conn, api_settings.inventory_horizon_days)
    finally:
        conn.close()
    return TINY_FLIGHT


@pytest.fixture
def agent_settings(tmp_path: Path) -> Settings:
    base = load_settings(env={})
    return replace(
        base,
        flight_api=replace(base.flight_api, retry_backoff_s=0.0),
        agent=replace(base.agent, use_llm=False),
        graph=replace(base.graph, artifacts_dir=tmp_path / "artifacts", remote_png=False),
    )


class FakeLLM:
    """Returns a scripted classification; records prompts. `payload=None` raises (model failure)."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def generate_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
        self.calls.append(user)
        if self.payload is None:
            raise RuntimeError("model crashed")
        return self.payload


class FlakyHTTP:
    """Wraps the API test client; `script` lists per-call behaviours consumed in order.

    "ok"           -> pass through
    "connect"      -> raise ConnectError before sending
    "timeout"      -> raise ReadTimeout before sending
    "send_timeout" -> perform the request on the server, then raise ReadTimeout (outcome unknown)
    "500"          -> HTTP 500 without sending
    "garbage"      -> HTTP 200 with a non-JSON body
    """

    def __init__(self, inner: TestClient, script: list[str] | None = None, default: str = "ok") -> None:
        self.inner, self.script, self.default = inner, list(script or []), default
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append((method, url))
        action = self.script.pop(0) if self.script else self.default
        if action == "connect":
            raise httpx.ConnectError("refused")
        if action == "timeout":
            raise httpx.ReadTimeout("slow")
        if action == "500":
            return httpx.Response(500, json={"success": False})
        if action == "garbage":
            return httpx.Response(200, text="<html>oops</html>")
        response = self.inner.request(method, url, **kwargs)
        if action == "send_timeout":
            raise httpx.ReadTimeout("slow after send")
        return response

    def writes(self) -> list[tuple[str, str]]:
        return [r for r in self.requests if r[0] in ("POST", "PATCH", "DELETE", "PUT")]


@pytest.fixture
def make_agent(api: TestClient, agent_settings: Settings) -> Callable[..., FlightAgent]:
    counter = itertools.count(1)

    def factory(llm: FakeLLM | None = None, script: list[str] | None = None, default: str = "ok",
                http: FlakyHTTP | None = None, max_shown: int | None = None) -> FlightAgent:
        http = http or FlakyHTTP(api, script, default)
        settings = replace(agent_settings, agent=replace(agent_settings.agent, use_llm=llm is not None))
        if max_shown is not None:
            settings = replace(settings, agent=replace(settings.agent, max_flights_shown=max_shown))
        agent = FlightAgent(
            settings, llm=llm, client=FlightAPIClient(settings.flight_api, http=http),  # type: ignore[arg-type]
            new_key=lambda: f"test-key-{next(counter):06d}", load_model=False, save_artifacts=False,
        )
        agent.http = http  # type: ignore[attr-defined]
        return agent

    return factory


@pytest.fixture
def agent(make_agent: Callable[..., FlightAgent]) -> FlightAgent:
    return make_agent()


def say(agent: FlightAgent, *messages: str, thread: str = "t1") -> FlightTaskResult:
    """Send messages in order and return the last result."""
    result: FlightTaskResult | None = None
    for message in messages:
        result = agent.chat(message, thread)
    assert result is not None
    return result


PASSENGER_LINE = "Asha Rao, 31, female"
CONTACT_LINE = "asha@example.com +919876543210"


def book_via_api(api: TestClient, flight_id: str = "FL-1001", offset: int = 30, passengers: int = 1) -> dict[str, Any]:
    """Create a booking directly through the API and return its data."""
    response = api.post(
        "/api/v1/bookings",
        json={
            "flight_id": flight_id, "departure_date": day(offset).isoformat(),
            "passengers": [{"name": f"Traveller {i}", "age": 30 + i, "gender": "OTHER"} for i in range(passengers)],
            "contact": {"email": "t@example.com", "phone": "+919876543210"},
        },
        headers={"Idempotency-Key": f"direct-{flight_id}-{offset}-{passengers}-{id(api)}"},
    )
    assert response.status_code in (200, 201), response.text
    return response.json()["data"]


def api_booking(api: TestClient, booking_id: str) -> dict[str, Any]:
    response = api.get(f"/api/v1/bookings/{booking_id}")
    assert response.status_code == 200, response.text
    return response.json()["data"]
