"""Ask-user node: ask one natural question and pause the graph until the user answers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from travel_agent.flight_agent.nodes.deps import AgentDeps, compact_flight, make_outcome, shown_flights
from travel_agent.flight_agent.params import ask_group, awaiting_for
from travel_agent.flight_agent.responses import ask_text, render
from travel_agent.flight_agent.schemas import Intent
from travel_agent.flight_agent.state import FlightAgentState


def make_ask_user(deps: AgentDeps):
    max_shown = deps.settings.agent.max_flights_shown

    def ask_user(state: FlightAgentState) -> dict[str, Any]:
        missing = state.get("missing_parameters") or []
        group = ask_group(missing)
        intent = state.get("identified_intent", "")
        note = state.get("validation_note")
        data: dict[str, Any] = {}

        if group[0] == "flight" and intent == Intent.BOOK_FLIGHT.value and (state.get("last_search") or {}).get("flights"):
            search = state["last_search"]
            outcome = make_outcome(
                "search_results", query=search["query"], flights=search["flights"],
                shown_ids=search["shown_ids"], sort_hint=state.get("sort_hint", ""), invite_booking=True, note=note,
            )
            question = render(outcome, max_shown)
            data = {"query": search["query"], "flights": [compact_flight(f) for f in shown_flights(state)]}
        else:
            ctx = {
                "origin": state.get("origin"), "destination": state.get("destination"),
                "passenger_count": state.get("passenger_count"), "passenger_details": state.get("passenger_details"),
                "intent": intent, "known_bookings": state.get("known_bookings"),
            }
            question = ask_text(group, ctx, note, (state.get("ask_counts") or {}).get(group[0], 0))

        answer = interrupt({
            "kind": "input", "question": question, "missing_params": missing, "intent": intent or None, "data": data,
        })
        counts = dict(state.get("ask_counts") or {})
        counts[group[0]] = counts.get(group[0], 0) + 1
        return {
            "messages": [AIMessage(content=question), HumanMessage(content=str(answer))],
            "awaiting": awaiting_for(group),
            "ask_counts": counts,
            "validation_note": None,
            "invite_booking": False,
        }

    return ask_user
