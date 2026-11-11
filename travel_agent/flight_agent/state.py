"""Graph state. Small, serialisable, and limited to what later steps need.

Facts shown to the user are always rebuilt from API data stored here (`last_search`, `booking_details`,
`pending_action`), never from model text. Field names follow the task specification:

    conversation_history -> `messages`        user_request -> `user_request`
    identified_intent    -> `identified_intent`   origin / destination / departure_date / passenger_count
    passenger_details    -> `passenger_details`   selected_flight -> `selected_flight`
    missing_parameters   -> `missing_parameters`  user_confirmation_status -> `user_confirmation_status`
    api_execution_status -> `api_execution_status` booking_details / cancellation_details
    error_information    -> `error_information`   final_response -> `final_response`
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FlightAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation history (checkpointed per thread)
    task: dict[str, Any]  # optional orchestrator hints: {"intent": ..., "params": {...}}
    user_request: str  # newest user message

    # understanding
    identified_intent: str
    last_intent: str  # intent of the previous finished task
    general_topic: str
    sort_hint: str  # cheapest | fastest | earliest | latest | ""
    missing_parameters: list[str]
    awaiting: str | None  # what the last question asked for
    ask_counts: dict[str, int]
    validation_note: str | None

    # trip and traveller details (carried over within a conversation)
    origin: str | None
    destination: str | None
    departure_date: str | None  # ISO date
    passenger_count: int | None
    passenger_details: list[dict[str, Any]]  # [{"name", "age", "gender"}] (entries may be partial)
    contact: dict[str, str]  # {"email", "phone"}

    # task scoped
    booking_id: str | None
    selected_flight: dict[str, Any] | None  # {"flight_id", ...} (full API data once verified)
    verified_key: str | None  # query + flight the API has confirmed as available

    # remembered API facts
    last_search: dict[str, Any]  # {"query": {...}, "flights": [...], "shown_ids": [...]}
    known_bookings: list[str]
    booking_signatures: dict[str, str]  # request signature -> booking_id (duplicate guard)

    # action control
    pending_action: dict[str, Any] | None  # booking/cancel waiting for confirmation
    retryable_action: dict[str, Any] | None  # booking whose outcome is unknown or never sent
    user_confirmation_status: str | None  # confirmed | declined | changed
    invite_booking: bool  # a search was triggered by a booking request: ask which flight to book
    next: str

    # Knowledge-injected context from retrieval layer
    retrieved_context: str | None  # Retrieved knowledge relevant to this task
    retrieval_metadata: dict[str, Any] | None  # {"confidence", "sources", "domain", ...}

    # outcome
    api_execution_status: str  # ExecutionStatus value of the last API-backed step
    booking_details: dict[str, Any] | None
    cancellation_details: dict[str, Any] | None
    error_information: dict[str, Any] | None
    final_response: str
    outcome: dict[str, Any]
    result: dict[str, Any]
