"""Checkpointed state for the travel supervisor; subagent state stays in its own graph."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class OrchestratorState(TypedDict, total=False):
    user_request: str
    conversation_history: list[dict[str, str]]
    identified_intent: str
    identified_domains: list[str]
    requested_operations: list[str]
    execution_plan: dict[str, Any]
    selected_subagents: list[str]
    hotel_task_status: str
    flight_task_status: str
    hotel_result: dict[str, Any] | None
    flight_result: dict[str, Any] | None
    subagent_errors: Annotated[dict[str, Any], _merge]
    missing_information: list[str]
    execution_status: str
    final_response: str
    pending_domains: list[str]
