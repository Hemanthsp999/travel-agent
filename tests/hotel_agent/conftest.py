"""Fixtures for hotel agent tests: the real Hotel API (in-process, temp SQLite), a scripted LLM, and
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

from mock_apis.hotel_api.app import create_app
from mock_apis.hotel_api.config import HotelSettings, load_settings as load_api_settings
from travel_agent.config import Settings, load_settings
from travel_agent.hotel_agent.agent import HotelAgent
from travel_agent.hotel_agent.api_client import HotelAPIClient
from travel_agent.hotel_agent.schemas import HotelTaskResult


def day(offset: int) -> str:
    """ISO date `offset` days from today (the API validates against the real clock)."""
    return (date.today() + timedelta(days=offset)).isoformat()


@pytest.fixture
def api_settings(tmp_path: Path) -> HotelSettings:
    return load_api_settings(tmp_path / "hotel.db", horizon_days=200)


@pytest.fixture
def api(api_settings: HotelSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=api_settings)) as client:
        yield client


@pytest.fixture
def agent_settings(tmp_path: Path) -> Settings:
    base = load_settings(env={})
    return replace(
        base,
        hotel_api=replace(base.hotel_api, retry_backoff_s=0.0),
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
    """Wraps the API client; `script` is a list of per-call behaviours consumed in order.

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


@pytest.fixture
def make_agent(api: TestClient, agent_settings: Settings) -> Callable[..., HotelAgent]:
    counter = itertools.count(1)

    def factory(llm: FakeLLM | None = None, script: list[str] | None = None, default: str = "ok",
                http: FlakyHTTP | None = None) -> HotelAgent:
        http = http or FlakyHTTP(api, script, default)
        settings = replace(agent_settings, agent=replace(agent_settings.agent, use_llm=llm is not None))
        agent = HotelAgent(
            settings, llm=llm, client=HotelAPIClient(settings.hotel_api, http=http),  # type: ignore[arg-type]
            new_key=lambda: f"test-key-{next(counter):06d}", load_model=False, save_artifacts=False,
        )
        agent.http = http  # type: ignore[attr-defined]
        return agent

    return factory


@pytest.fixture
def agent(make_agent: Callable[..., HotelAgent]) -> HotelAgent:
    return make_agent()


def say(agent: HotelAgent, *messages: str, thread: str = "t1") -> HotelTaskResult:
    """Send messages in order and return the last result."""
    result: HotelTaskResult | None = None
    for message in messages:
        result = agent.chat(message, thread)
    assert result is not None
    return result


def book_via_api(api: TestClient, hotel_id: str = "HT-1002", nights: tuple[int, int] = (10, 12)) -> dict[str, Any]:
    """Create a booking directly through the API and return its data."""
    response = api.post(
        "/api/v1/bookings",
        json={
            "hotel_id": hotel_id, "check_in": day(nights[0]), "check_out": day(nights[1]), "rooms": 1, "guests": 2,
            "guest": {"name": "Asha Rao", "email": "asha@example.com", "phone": "+919876543210"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


def api_search(api: TestClient, city: str, check_in: str, check_out: str, **extra: Any) -> list[dict[str, Any]]:
    response = api.get("/api/v1/hotels/search", params={"city": city, "check_in": check_in, "check_out": check_out, **extra})
    assert response.status_code == 200, response.text
    return response.json()["data"]["hotels"]
