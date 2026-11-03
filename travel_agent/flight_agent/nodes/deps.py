"""Dependencies and helpers shared by all nodes (injected so tests can fake the clock, keys, model and API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable
from uuid import uuid4

from langchain_core.tools import BaseTool

from travel_agent.config import Settings
from travel_agent.flight_agent.parsing import canonical_key
from travel_agent.flight_agent.state import FlightAgentState
from travel_agent.llm.loader import LLM


def _new_key() -> str:
    return f"flight-agent-{uuid4().hex}"


@dataclass
class AgentDeps:
    settings: Settings
    tools: dict[str, BaseTool]
    llm: LLM | None = None
    today: Callable[[], date] = date.today
    new_key: Callable[[], str] = field(default=_new_key)


def shown_flights(state: FlightAgentState) -> list[dict[str, Any]]:
    """Flights from the latest search that the user has actually been shown, in display order."""
    search = state.get("last_search") or {}
    by_id = {f["flight_id"]: f for f in search.get("flights", [])}
    return [by_id[i] for i in search.get("shown_ids", []) if i in by_id]


def current_query(state: FlightAgentState) -> dict[str, Any] | None:
    """The search the current trip details describe, or None while something is missing."""
    keys = ("origin", "destination", "departure_date", "passenger_count")
    if not all(state.get(k) for k in keys):
        return None
    return {
        "origin": state["origin"], "destination": state["destination"],
        "departure_date": state["departure_date"], "passengers": state["passenger_count"],
    }


def same_query(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    return (
        canonical_key(a["origin"]) == canonical_key(b["origin"])
        and canonical_key(a["destination"]) == canonical_key(b["destination"])
        and a["departure_date"] == b["departure_date"]
        and a["passengers"] == b["passengers"]
    )


def make_outcome(
    kind: str, data: dict[str, Any] | None = None, error: dict[str, Any] | None = None, **fields: Any
) -> dict[str, Any]:
    """Uniform outcome dict: `kind` selects the reply template, `data` is returned to callers."""
    return {"kind": kind, "data": data or {}, "error": error, **fields}


def compact_flight(flight: dict[str, Any]) -> dict[str, Any]:
    """Flight fields passed on to callers (a subset of the API record, never invented)."""
    keys = (
        "flight_id", "airline", "flight_number", "origin", "destination", "departure_date", "departure_time",
        "arrival_time", "duration_minutes", "price_per_passenger", "total_price", "currency", "seats_available",
    )
    return {k: flight[k] for k in keys if k in flight}
