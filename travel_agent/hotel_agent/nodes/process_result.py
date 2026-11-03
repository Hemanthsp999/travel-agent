"""Process-result node: verify what the API returned, update memory, and decide the next step.

Success is only ever reported when the API response itself proves it (status field, matching IDs).
"""

from __future__ import annotations

from typing import Any

from travel_agent.hotel_agent.errors import UNREACHABLE, plan_for_error
from travel_agent.hotel_agent.intent import find_hotels_by_name
from travel_agent.hotel_agent.nodes.deps import AgentDeps, make_outcome
from travel_agent.hotel_agent.responses import booking_prompt, cancel_prompt
from travel_agent.hotel_agent.schemas import Intent, ToolResult
from travel_agent.hotel_agent.state import HotelAgentState

RETRY_HINT = " Say \"retry\" and I'll try again."


def _known(state: HotelAgentState, booking_id: str) -> list[str]:
    known = list(state.get("known_bookings") or [])
    return known if booking_id in known else known + [booking_id]


def _query(params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    query = {
        "city": data.get("city") or params["city"],
        "check_in": data.get("check_in") or params["check_in"],
        "check_out": data.get("check_out") or params["check_out"],
        "guests": data.get("guests") or params.get("guests") or 1,
        "rooms": data.get("rooms") or params.get("rooms") or 1,
    }
    for key in ("max_price_per_night", "min_rating"):
        if params.get(key):
            query[key] = params[key]
    return query


def _assumptions(params: dict[str, Any]) -> list[str]:
    assumed = [label for key, label in (("guests", "1 guest"), ("rooms", "1 room")) if not params.get(key)]
    return [f"I assumed {' and '.join(assumed)}."] if assumed else []


def make_process_result(deps: AgentDeps):
    max_shown = deps.settings.agent.max_hotels_shown

    # ------------------------------------------------------------------ stage handlers

    def search(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        params, hotels = state["params"], data["hotels"]
        query = _query(params, data)
        shown_ids = [h["hotel_id"] for h in hotels[:max_shown]]
        update: dict[str, Any] = {
            "last_search": {"query": query, "hotels": hotels, "shown_ids": shown_ids},
            "selected_hotel": None, "next": "respond",
        }
        if not hotels:
            return {**update, "outcome": make_outcome("no_results", {"query": query, "hotels": []}, query=query)}
        extra: dict[str, Any] = {}
        focus_query = params.get("hotel_id") or params.get("hotel_name")
        if focus_query:
            match = (
                [h for h in hotels if h["hotel_id"] == params.get("hotel_id")] if params.get("hotel_id")
                else find_hotels_by_name(params["hotel_name"], hotels)
            )
            if len(match) == 1:
                extra["focus_available"] = match[0]["name"]
                others = [h for h in hotels if h is not match[0]]
                hotels = [match[0]] + others
                update["last_search"] = {"query": query, "hotels": hotels, "shown_ids": [h["hotel_id"] for h in hotels[:max_shown]]}
            else:
                extra["focus_missing"] = params.get("hotel_name") or params["hotel_id"]
        outcome = make_outcome(
            "search_results", {"query": query, "hotels": hotels}, query=query, hotels=hotels,
            shown_ids=update["last_search"]["shown_ids"], assumptions=_assumptions(params),
            invite_booking=bool(state.get("invite_booking")), **extra,
        )
        return {**update, "outcome": outcome}

    def hotel_lookup(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        params = dict(state["params"])
        if data.get("city") and not params.get("city"):
            params["city"] = data["city"]
        return {"selected_hotel": data, "params": params, "next": "select_tool"}

    def precheck(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        params, hotels = state["params"], data["hotels"]
        selected = state.get("selected_hotel")
        if params.get("hotel_id"):
            matches = [h for h in hotels if h["hotel_id"] == params["hotel_id"]]
        else:
            matches = find_hotels_by_name(params["hotel_name"], hotels)
        if len(matches) > 1:
            query = _query(params, data)
            names = ", ".join(h["name"] for h in matches)
            params = {k: v for k, v in params.items() if k != "hotel_name"}
            return {
                "params": params, "validation_note": f"More than one hotel matches: {names}.", "next": "validate",
                "last_search": {"query": query, "hotels": matches, "shown_ids": [h["hotel_id"] for h in matches]},
            }
        if not matches:
            label = (selected or {}).get("name") or params.get("hotel_name") or params["hotel_id"]
            per_room = (selected or {}).get("max_guests_per_room")
            if per_room and params["guests"] > per_room * params["rooms"]:
                note = (
                    f"{label} fits at most {per_room * params['rooms']} guests in "
                    f"{params['rooms']} room(s) ({per_room} per room)."
                )
                trimmed = {k: v for k, v in params.items() if k not in ("guests", "rooms")}
                return {"params": trimmed, "validation_note": note, "next": "validate"}
            query = _query(params, data)
            ids = [h["hotel_id"] for h in hotels[:max_shown]]
            outcome = make_outcome(
                "booking_unavailable", {"hotels": hotels[:max_shown], "query": query}, hotel_name=label,
                alternatives=hotels[:max_shown],
            )
            return {
                "outcome": outcome, "next": "respond", "selected_hotel": None,
                "last_search": {"query": query, "hotels": hotels, "shown_ids": ids},
            }
        hotel = matches[0]
        payload = {
            "hotel_id": hotel["hotel_id"], "check_in": params["check_in"], "check_out": params["check_out"],
            "rooms": params["rooms"], "guests": params["guests"], "guest_name": params["guest_name"],
            "guest_email": params["guest_email"], "guest_phone": params["guest_phone"],
        }
        quote = {
            "hotel_name": hotel["name"], "locality": hotel.get("locality"), "nights": hotel.get("nights"),
            "price_per_night": hotel.get("price_per_night"), "currency": hotel.get("currency"),
            "estimated_total_price": hotel.get("estimated_total_price"),
        }
        pending = {
            "type": "book", "payload": payload, "quote": quote, "idempotency_key": deps.new_key(),
            "prompt": booking_prompt(payload, quote), "intent": "BOOK_HOTEL",
        }
        return {"pending_action": pending, "selected_hotel": hotel, "next": "confirm"}

    def book(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        pending = state["pending_action"]
        hotel_id = (data.get("hotel") or {}).get("hotel_id")
        proven = data.get("status") == "CONFIRMED" and (hotel_id is None or hotel_id == pending["payload"]["hotel_id"])
        if not proven:
            message = (
                "The hotel system's answer was unclear, so I can't confirm this booking"
                + (f" (it mentioned booking {data.get('booking_id')})" if data.get("booking_id") else "")
                + ". Please ask me to check the booking ID before relying on it."
            )
            return {
                "outcome": make_outcome("booking_unverified", {"booking": data}, message=message), "next": "respond",
                "known_bookings": _known(state, data["booking_id"]) if data.get("booking_id") else state.get("known_bookings", []),
                "pending_action": None,
            }
        return {
            "booking": data, "known_bookings": _known(state, data["booking_id"]), "pending_action": None,
            "retryable_action": None, "selected_hotel": None, "next": "respond",
            "outcome": make_outcome("booking_success", {"booking": data}, booking=data),
        }

    def booking_lookup(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        known = _known(state, data["booking_id"])
        if data["status"] == "CANCELLED":
            outcome = make_outcome("already_cancelled", {"booking": data}, booking_id=data["booking_id"])
            return {"outcome": outcome, "next": "respond", "known_bookings": known}
        pending = {
            "type": "cancel", "booking_id": data["booking_id"], "booking": data, "prompt": cancel_prompt(data),
            "intent": "CANCEL_BOOKING",
        }
        return {"pending_action": pending, "known_bookings": known, "next": "confirm"}

    def cancel(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        pending = state["pending_action"]
        if data.get("status") == "CANCELLED" and data.get("booking_id") == pending["booking_id"]:
            return {
                "pending_action": None, "next": "respond",
                "outcome": make_outcome("cancel_success", {"booking": data}, booking=data),
            }
        message = "The hotel system's answer was unclear, so I can't confirm the cancellation. Ask me to check the booking status."
        return {"outcome": make_outcome("cancel_unverified", {"booking": data}, message=message), "next": "respond", "pending_action": None}

    def cancel_verify(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("status") == "CANCELLED":
            return {
                "pending_action": None, "next": "respond", "force_stage": None,
                "outcome": make_outcome("cancel_success", {"booking": data}, booking=data, verified_by_lookup=True),
            }
        message = f"Booking {data['booking_id']} is still {data['status']}, so the cancellation did not go through. Nothing else was changed."
        return {
            "outcome": make_outcome("cancel_failed", {"booking": data}, message=message), "next": "respond",
            "pending_action": None, "force_stage": None,
        }

    def status(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "booking": data, "known_bookings": _known(state, data["booking_id"]), "next": "respond",
            "outcome": make_outcome("booking_status", {"booking": data}, booking=data),
        }

    def details(state: HotelAgentState, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "selected_hotel": data, "next": "respond",
            "outcome": make_outcome("hotel_details", {"hotel": data}, hotel=data),
        }

    handlers = {
        "search": search, "hotel_lookup": hotel_lookup, "precheck": precheck, "book": book,
        "booking_lookup": booking_lookup, "cancel": cancel, "cancel_verify": cancel_verify,
        "status": status, "details": details,
    }

    # ---------------------------------------------------------------------- failures

    def on_error(state: HotelAgentState, stage: str, result: ToolResult) -> dict[str, Any]:
        error = result.error.model_dump() if result.error else {"code": "UNKNOWN", "message": ""}
        pending = state.get("pending_action")
        code = error["code"]

        if stage == "book" and error.get("outcome_unknown"):
            message = (
                "I sent the booking request but didn't get a clear answer, so I can't tell if it went through. "
                "I have NOT treated it as booked. Say \"retry\" and I'll resend the exact same request "
                "(it can't create a duplicate), or give me a booking ID to check."
            )
            return {
                "outcome": make_outcome("booking_unverified", error=error, message=message),
                "retryable_action": pending, "pending_action": None, "next": "respond",
            }
        if stage == "cancel" and error.get("outcome_unknown"):
            return {"force_stage": "cancel_verify", "next": "select_tool"}
        if stage == "cancel_verify":
            message = (
                "I couldn't confirm whether the cancellation worked because the hotel system isn't answering. "
                "Please ask me to check the booking status in a moment."
            )
            return {"outcome": make_outcome("cancel_unverified", error=error, message=message),
                    "pending_action": None, "force_stage": None, "next": "respond"}

        plan = plan_for_error(stage, error)
        if plan.kind == "already_cancelled":
            booking_id = (pending or {}).get("booking_id") or state["params"].get("booking_id", "")
            return {"outcome": make_outcome("already_cancelled", error=error, booking_id=booking_id),
                    "pending_action": None, "next": "respond"}
        if plan.recover:
            params = {k: v for k, v in state["params"].items() if k not in plan.clear}
            update: dict[str, Any] = {
                "params": params, "validation_note": plan.note, "pending_action": None,
                "next": "general_query" if state.get("intent") == Intent.GENERAL_HOTEL_QUERY.value else "validate",
            }
            if "hotel_id" in plan.clear:
                update["selected_hotel"] = None
            return update

        message = plan.message
        update = {"pending_action": None, "next": "respond"}
        if stage == "book" and code in UNREACHABLE:
            update["retryable_action"] = pending
            message += RETRY_HINT
        update["outcome"] = make_outcome(plan.kind, error=error, message=message)
        return update

    def process_result(state: HotelAgentState) -> dict[str, Any]:
        result = ToolResult.model_validate(state["api_result"])
        stage = state["stage"]
        if not result.ok:
            return on_error(state, stage, result)
        return handlers[stage](state, result.data or {})

    return process_result
