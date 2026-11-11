"""Graph state. Small, serialisable, and limited to what later steps need.

Facts shown to the user are always rebuilt from API data stored here (`last_search`, `booking`,
`pending_action`), never from model text.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class HotelAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    task: dict[str, Any]  # optional orchestrator hints: {"intent": ..., "params": {...}}

    # understanding
    intent: str
    last_intent: str  # intent of the previous finished task
    general_topic: str
    params: dict[str, Any]  # city, check_in, check_out, guests, rooms, hotel_id, guest_name, ...
    missing_params: list[str]
    awaiting: str | None  # what the last question asked for
    ask_counts: dict[str, int]
    validation_note: str | None

    # remembered API facts
    last_search: dict[str, Any]  # {"query": {...}, "hotels": [...], "shown_ids": [...]}
    selected_hotel: dict[str, Any] | None
    known_bookings: list[str]

    # action control
    pending_action: dict[str, Any] | None  # booking/cancel waiting for confirmation
    retryable_action: dict[str, Any] | None  # booking whose outcome is unknown or never sent
    confirmation: str | None  # confirmed | declined | changed
    tool_call: dict[str, Any] | None
    api_result: dict[str, Any] | None
    invite_booking: bool  # a search was triggered by a booking request: ask which hotel to book
    stage: str
    force_stage: str | None
    next: str

    # Knowledge-injected context from retrieval layer
    retrieved_context: str | None  # Retrieved knowledge relevant to this task
    retrieval_metadata: dict[str, Any] | None  # {"confidence", "sources", "domain", ...}

    # outcome
    outcome: dict[str, Any]
    result: dict[str, Any]
