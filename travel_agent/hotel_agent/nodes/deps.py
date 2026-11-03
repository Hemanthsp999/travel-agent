"""Dependencies shared by all nodes (injected so tests can fake the clock, keys, model and API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable
from uuid import uuid4

from langchain_core.tools import BaseTool

from travel_agent.config import Settings
from travel_agent.llm.loader import LLM


def _new_key() -> str:
    return f"hotel-agent-{uuid4().hex}"


@dataclass
class AgentDeps:
    settings: Settings
    tools: dict[str, BaseTool]
    llm: LLM | None = None
    today: Callable[[], date] = date.today
    new_key: Callable[[], str] = field(default=_new_key)


def shown_hotels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Hotels from the latest search that the user has actually been shown, in display order."""
    search = state.get("last_search") or {}
    by_id = {h["hotel_id"]: h for h in search.get("hotels", [])}
    return [by_id[i] for i in search.get("shown_ids", []) if i in by_id]


def make_outcome(kind: str, data: dict[str, Any] | None = None, error: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    """Uniform outcome dict: `kind` selects the reply template, `data` is returned to callers."""
    return {"kind": kind, "data": data or {}, "error": error, **fields}
