"""Select-tool node: choose the next API call from state. Pure rules, no LLM."""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.nodes.deps import AgentDeps
from travel_agent.hotel_agent.schemas import GeneralTopic, Intent
from travel_agent.hotel_agent.state import HotelAgentState


def _call(name: str, args: dict[str, Any], stage: str) -> dict[str, Any]:
    return {"tool_call": {"name": name, "args": args}, "stage": stage, "next": "execute_tool", "force_stage": None}


def make_select_tool(deps: AgentDeps):
    def select_tool(state: HotelAgentState) -> dict[str, Any]:
        intent = Intent(state["intent"])
        params = state.get("params") or {}
        pending = state.get("pending_action")
        confirmed = state.get("confirmation") == "confirmed" and pending is not None

        if state.get("force_stage") == "cancel_verify" and pending:
            return _call("get_booking_status", {"booking_id": pending["booking_id"]}, "cancel_verify")

        if intent is Intent.CHECK_AVAILABILITY:
            args = {
                "city": params["city"], "check_in": params["check_in"], "check_out": params["check_out"],
                "guests": params.get("guests") or 1, "rooms": params.get("rooms") or 1,
            }
            for key in ("max_price_per_night", "min_rating"):
                if params.get(key):
                    args[key] = params[key]
            return _call("check_hotel_availability", args, "search")

        if intent is Intent.BOOK_HOTEL:
            if confirmed and pending["type"] == "book":
                return _call("book_hotel", {**pending["payload"], "idempotency_key": pending["idempotency_key"]}, "book")
            selected = state.get("selected_hotel")
            hotel_id = params.get("hotel_id")
            if hotel_id and not (selected and selected["hotel_id"] == hotel_id):
                return _call("get_hotel_details", {"hotel_id": hotel_id}, "hotel_lookup")
            city = (selected or {}).get("city") or params["city"]
            args = {
                "city": city, "check_in": params["check_in"], "check_out": params["check_out"],
                "guests": params["guests"], "rooms": params["rooms"],
            }
            return _call("check_hotel_availability", args, "precheck")

        if intent is Intent.CANCEL_BOOKING:
            if confirmed and pending["type"] == "cancel":
                return _call("cancel_hotel_booking", {"booking_id": pending["booking_id"]}, "cancel")
            return _call("get_booking_status", {"booking_id": params["booking_id"]}, "booking_lookup")

        if intent is Intent.GET_BOOKING_STATUS:
            return _call("get_booking_status", {"booking_id": params["booking_id"]}, "status")

        if intent is Intent.GENERAL_HOTEL_QUERY and state.get("general_topic") == GeneralTopic.HOTEL_DETAILS.value:
            return _call("get_hotel_details", {"hotel_id": params["hotel_id"]}, "details")

        raise ValueError(f"No tool for intent {intent.value}")

    return select_tool
