"""User-facing text. Every fact (airline, time, price, ID, status) is rendered from API data in code.

The functions take plain dicts produced by the nodes, so wording can change without touching logic.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_agent.flight_agent.schemas import ExecutionStatus

FIELD_LABELS = {
    "origin": "departure city", "destination": "destination city", "departure_date": "travel date",
    "passenger_count": "number of passengers", "flight": "flight", "contact_email": "email address",
    "contact_phone": "phone number", "booking_id": "booking ID",
}

STATUS_BY_KIND = {
    "search_results": ExecutionStatus.SUCCESS, "no_results": ExecutionStatus.SUCCESS,
    "flight_details": ExecutionStatus.SUCCESS, "comparison": ExecutionStatus.SUCCESS,
    "show_all": ExecutionStatus.SUCCESS, "booking_success": ExecutionStatus.SUCCESS,
    "cancel_success": ExecutionStatus.SUCCESS, "already_cancelled": ExecutionStatus.SUCCESS,
    "booking_status": ExecutionStatus.SUCCESS, "help": ExecutionStatus.SUCCESS,
    "no_context": ExecutionStatus.REQUIRES_USER_INPUT, "unrecognized": ExecutionStatus.UNKNOWN,
    "unsupported": ExecutionStatus.FAILED, "booking_unavailable": ExecutionStatus.FAILED,
    "booking_failed": ExecutionStatus.FAILED, "cancel_failed": ExecutionStatus.FAILED,
    "error": ExecutionStatus.FAILED, "gave_up": ExecutionStatus.FAILED, "aborted": ExecutionStatus.FAILED,
    "declined": ExecutionStatus.FAILED, "booking_unverified": ExecutionStatus.UNKNOWN,
    "cancel_unverified": ExecutionStatus.UNKNOWN,
}
# Outcomes that are failures of the *task*, with the error code reported to the orchestrator.
ERROR_CODE_BY_KIND = {
    "unsupported": "UNSUPPORTED_OPERATION", "gave_up": "MISSING_PARAMETERS", "aborted": "USER_ABORTED",
    "declined": "USER_DECLINED", "booking_unavailable": "FLIGHT_UNAVAILABLE", "booking_failed": "BOOKING_FAILED",
    "cancel_failed": "CANCELLATION_FAILED", "error": "AGENT_ERROR", "booking_unverified": "BOOKING_OUTCOME_UNKNOWN",
    "cancel_unverified": "CANCELLATION_OUTCOME_UNKNOWN", "unrecognized": "INTENT_NOT_RECOGNIZED",
}


# --------------------------------------------------------------------------- formatting


def fmt_date(value: str | date) -> str:
    d = date.fromisoformat(value[:10]) if isinstance(value, str) else value
    return f"{d.day} {d.strftime('%b %Y')}"


def hhmm(iso_datetime: str) -> str:
    return iso_datetime[11:16]


def money(amount: int | float | None, currency: str | None = None) -> str:
    return f"{currency or 'INR'} {int(amount):,}" if amount is not None else "price not provided"


def duration(minutes: int | None) -> str:
    if minutes is None:
        return ""
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _route(query: dict[str, Any]) -> str:
    return f"{query['origin']} to {query['destination']}"


def flight_line(flight: dict[str, Any], index: int | None = None, passengers: int | None = None) -> str:
    head = f"{index}. " if index else "- "
    bits = [f"departs {hhmm(flight['departure_time'])}, arrives {hhmm(flight['arrival_time'])}"]
    if flight.get("duration_minutes") is not None:
        bits[0] += f" ({duration(flight['duration_minutes'])})"
    bits.append(f"{money(flight['price_per_passenger'], flight.get('currency'))} per passenger")
    if passengers and passengers > 1 and flight.get("total_price") is not None:
        bits.append(f"{money(flight['total_price'], flight.get('currency'))} for {passengers}")
    if flight.get("seats_available") is not None:
        bits.append(f"{_plural(flight['seats_available'], 'seat')} left")
    return f"{head}{flight['airline']} {flight['flight_number']}: {', '.join(bits)}"


def passenger_lines(passengers: list[dict[str, Any]]) -> list[str]:
    return [
        f"  {i}. {p['name']}" + (f", {p['age']}" if p.get("age") is not None else "")
        + (f", {str(p['gender']).lower()}" if p.get("gender") else "")
        for i, p in enumerate(passengers, 1)
    ]


# ----------------------------------------------------------------------------- questions


def ask_text(group: list[str], ctx: dict[str, Any], note: str | None, attempt: int) -> str:
    """One natural question for the first group of missing values. `ctx` holds known state."""
    lead = f"{note} " if note else ""
    first = group[0]
    origin, destination = ctx.get("origin"), ctx.get("destination")
    if first == "origin":
        to = f"Flying to {destination}, lovely. " if destination else "Happy to help with your flight! "
        return lead + to + "Which city will you be flying from?"
    if first == "destination":
        frm = f"Flying from {origin}, got it. " if origin else "Happy to help with your flight! "
        return lead + frm + "Where would you like to fly to?"
    if first == "departure_date":
        route = f" from {origin} to {destination}" if origin and destination else ""
        return lead + f"What date would you like to travel{route}? (for example 10 Nov)"
    if first == "passenger_count":
        return lead + "How many passengers will be travelling?"
    if first == "flight":
        return lead + "Which flight would you like?"
    if first.startswith("passenger_"):
        return lead + _passenger_question(group, ctx)
    if first in ("contact_email", "contact_phone"):
        labels = [FIELD_LABELS[k] for k in group]
        joined = " and ".join(labels)
        return lead + f"Almost there! What {'is' if len(labels) == 1 else 'are'} the contact {joined} for the booking?"
    if first == "booking_id":
        action = {"CANCEL_FLIGHT": "cancel", "GET_BOOKING_STATUS": "look up"}.get(ctx.get("intent", ""), "use")
        known = ctx.get("known_bookings") or []
        hint = f" (From this chat: {', '.join(known)}.)" if len(known) > 1 else ""
        sure = "" if note else "Sure. "
        return lead + f"{sure}What's the booking ID you'd like me to {action}? It looks like FB-XXXXXXXXXX.{hint}"
    return lead + "Could you tell me a bit more?"


def _passenger_question(group: list[str], ctx: dict[str, Any]) -> str:
    index = int(group[0].split("_")[1])
    total = ctx.get("passenger_count") or index
    known = (ctx.get("passenger_details") or [])[index - 1:index]
    name = known[0].get("name") if known else None
    fields = [k.split("_", 2)[2] for k in group]
    labels = {"name": "full name", "age": "age", "gender": "gender (male, female or other)"}
    wanted = [labels[f] for f in fields]
    joined = wanted[0] if len(wanted) == 1 else ", ".join(wanted[:-1]) + f" and {wanted[-1]}"
    who = "the passenger" if total == 1 else f"passenger {index} of {total}"
    if name:
        return f"Thanks! For {name}, what is their {joined}?"
    return f"Great! For {who}, could you share the {joined}? For example: Asha Rao, 31, female."


def booking_prompt(payload: dict[str, Any], quote: dict[str, Any], note: str | None = None) -> str:
    flight = quote["flight"]
    count = len(payload["passengers"])
    cur = flight.get("currency")
    lines = [
        "Here is what I'm about to book:",
        f"- Flight: {flight['airline']} {flight['flight_number']} ({flight['flight_id']}), "
        f"{flight['origin']['city']} ({flight['origin']['code']}) to {flight['destination']['city']} ({flight['destination']['code']})",
        f"- Date: {fmt_date(flight['departure_date'])}, departs {hhmm(flight['departure_time'])}, arrives {hhmm(flight['arrival_time'])}",
        f"- Passengers ({count}):",
        *passenger_lines(payload["passengers"]),
        f"- Contact: {payload['contact_email']}, {payload['contact_phone']}",
        f"- Price: {money(flight['price_per_passenger'], cur)} per passenger, "
        f"{money(quote.get('total_price'), cur)} in total",
    ]
    if note:
        lines.append(f"Note: {note}")
    lines.append("Shall I go ahead and book it? (yes / no)")
    return "\n".join(lines)


def cancel_prompt(booking: dict[str, Any]) -> str:
    flight = booking.get("flight") or {}
    label = f"{flight.get('airline', 'flight')} {flight.get('flight_number', '')}".strip()
    when = f" on {fmt_date(booking['departure_date'])}" if booking.get("departure_date") else ""
    total = f" (total {money(booking['total_price'], booking.get('currency'))})" if booking.get("total_price") is not None else ""
    return (
        f"I found booking {booking['booking_id']}: {label}{when}{total}. "
        "Cancelling can't be undone. Do you want me to cancel it? (yes / no)"
    )


# ------------------------------------------------------------------------------- answers


def booking_block(booking: dict[str, Any]) -> str:
    flight = booking.get("flight") or {}
    lines = [f"Booking {booking['booking_id']} is {booking['status']}."]
    if flight.get("airline"):
        route = ""
        if flight.get("origin") and flight.get("destination"):
            route = f", {flight['origin']['city']} ({flight['origin']['code']}) to {flight['destination']['city']} ({flight['destination']['code']})"
        lines.append(f"- Flight: {flight['airline']} {flight.get('flight_number', '')} ({flight['flight_id']}){route}")
    if booking.get("departure_date"):
        times = ""
        if flight.get("departure_time") and flight.get("arrival_time"):
            times = f", departs {hhmm(flight['departure_time'])}, arrives {hhmm(flight['arrival_time'])}"
        lines.append(f"- Date: {fmt_date(booking['departure_date'])}{times}")
    if booking.get("passengers"):
        lines.append(f"- Passengers ({len(booking['passengers'])}):")
        lines.extend(passenger_lines(booking["passengers"]))
    if booking.get("total_price") is not None:
        per = f" ({money(booking['price_per_passenger'], booking.get('currency'))} per passenger)" if booking.get("price_per_passenger") is not None else ""
        lines.append(f"- Total: {money(booking['total_price'], booking.get('currency'))}{per}")
    contact = booking.get("contact") or {}
    if contact.get("email") or contact.get("phone"):
        lines.append("- Contact: " + ", ".join(v for v in (contact.get("email"), contact.get("phone")) if v))
    if booking.get("cancelled_at"):
        lines.append(f"- Cancelled at: {booking['cancelled_at']}")
    return "\n".join(lines)


def details_text(flight: dict[str, Any], passengers: int | None) -> str:
    lines = [
        f"{flight['airline']} {flight['flight_number']} ({flight['flight_id']}), "
        f"{flight['origin']['city']} ({flight['origin']['code']}) to {flight['destination']['city']} ({flight['destination']['code']}):",
        f"- Date: {fmt_date(flight['departure_date'])}",
        f"- Departs {hhmm(flight['departure_time'])}, arrives {hhmm(flight['arrival_time'])}"
        + (f" ({duration(flight['duration_minutes'])})" if flight.get("duration_minutes") is not None else ""),
    ]
    if flight.get("aircraft"):
        lines.append(f"- Aircraft: {flight['aircraft']}")
    lines.append(f"- Price: {money(flight['price_per_passenger'], flight.get('currency'))} per passenger")
    if passengers and passengers > 1 and flight.get("total_price") is not None:
        lines.append(f"- For {passengers} passengers: {money(flight['total_price'], flight.get('currency'))}")
    if flight.get("seats_available") is not None:
        lines.append(f"- Seats left (at my last search): {flight['seats_available']}")
    return "\n".join(lines)


def comparison_text(flights: list[dict[str, Any]], passengers: int | None) -> str:
    lines = ["Here's how they compare:"] + [flight_line(f, passengers=passengers) for f in flights]
    cheapest = min(flights, key=lambda f: f["price_per_passenger"])
    lines.append(f"Lowest price: {cheapest['airline']} {cheapest['flight_number']}.")
    timed = [f for f in flights if f.get("duration_minutes") is not None]
    if len(timed) > 1:
        quick = min(timed, key=lambda f: f["duration_minutes"])
        lines.append(f"Shortest: {quick['airline']} {quick['flight_number']} ({duration(quick['duration_minutes'])}).")
    lines.append(f"Earliest departure: {min(flights, key=lambda f: f['departure_time'])['flight_number']}.")
    return "\n".join(lines)


def _sort_line(flights: list[dict[str, Any]], hint: str) -> str:
    if hint == "cheapest":
        best = min(flights, key=lambda f: f["price_per_passenger"])
        return f"Cheapest: {best['airline']} {best['flight_number']} at {money(best['price_per_passenger'], best.get('currency'))} per passenger."
    if hint == "fastest":
        timed = [f for f in flights if f.get("duration_minutes") is not None]
        if timed:
            best = min(timed, key=lambda f: f["duration_minutes"])
            return f"Fastest: {best['airline']} {best['flight_number']} ({duration(best['duration_minutes'])})."
    if hint == "earliest":
        best = min(flights, key=lambda f: f["departure_time"])
        return f"Earliest: {best['airline']} {best['flight_number']}, departing {hhmm(best['departure_time'])}."
    if hint == "latest":
        best = max(flights, key=lambda f: f["departure_time"])
        return f"Latest: {best['airline']} {best['flight_number']}, departing {hhmm(best['departure_time'])}."
    return ""


def search_text(outcome: dict[str, Any], max_shown: int, invite: bool | None = None) -> str:
    query, flights = outcome["query"], outcome["flights"]
    shown_ids = set(outcome.get("shown_ids", []))
    shown = [f for f in flights if f["flight_id"] in shown_ids] or flights[:max_shown]
    count = query["passengers"]
    head = (
        f"Good news! I found {_plural(len(flights), 'flight')} from {_route(query)} on {fmt_date(query['departure_date'])} "
        f"for {_plural(count, 'passenger')}"
    )
    lines = [head + ":"] + [flight_line(f, i, count) for i, f in enumerate(shown, 1)]
    if len(flights) > len(shown):
        lines.append(f"({len(flights) - len(shown)} more available. Say \"show all\" to see them.)")
    if outcome.get("sort_hint") and (extra := _sort_line(flights, outcome["sort_hint"])):
        lines.append(extra)
    if outcome.get("note"):
        lines.append(f"Note: {outcome['note']}")
    invite = outcome.get("invite_booking") if invite is None else invite
    lines.append(
        "Which one would you like to book? You can say the number, the flight number or something like \"the cheapest\"."
        if invite else "Want details on any of them, or shall I book one?"
    )
    return "\n".join(lines)


def no_results_text(outcome: dict[str, Any]) -> str:
    query = outcome["query"]
    return (
        f"I couldn't find any flights from {_route(query)} on {fmt_date(query['departure_date'])} "
        f"with seats for {_plural(query['passengers'], 'passenger')}. "
        "Try another date, a smaller group or a different route."
    )


HELP_TEXT = (
    "I'm your flight assistant. I can check flight availability between two cities, book a flight, "
    "look up a booking's status and details, and cancel a booking. What would you like to do?"
)
UNSUPPORTED_TEXT = (
    "I can't help with that part. The flight system lets me search flights, book, look up a booking and cancel one "
    "- but not things like changes, refunds, baggage, seat choice or check-in. Which of the supported things would you like?"
)
UNRECOGNIZED_TEXT = (
    "I'm not sure I caught that. I can check flights, book one, look up a booking, or cancel a booking. What would you like?"
)


def render(outcome: dict[str, Any], max_shown: int = 4) -> str:
    """Final reply for an outcome dict."""
    kind = outcome["kind"]
    if kind == "search_results":
        return search_text(outcome, max_shown)
    if kind == "show_all":
        return search_text({**outcome, "shown_ids": [f["flight_id"] for f in outcome["flights"]]}, max_shown, invite=False)
    if kind == "no_results":
        return no_results_text(outcome)
    if kind == "flight_details":
        return details_text(outcome["flight"], outcome.get("passengers")) + "\nWould you like to book it?"
    if kind == "comparison":
        return comparison_text(outcome["flights"], outcome.get("passengers"))
    if kind == "booking_success":
        return (
            "All done! Your flight is booked.\n" + booking_block(outcome["booking"])
            + "\nWould you like help with anything else?"
        )
    if kind == "booking_status":
        return booking_block(outcome["booking"])
    if kind == "cancel_success":
        note = " (I double-checked this with the flight system.)" if outcome.get("verified_by_lookup") else ""
        return (
            "Done, that booking is cancelled" + note + ".\n" + booking_block(outcome["booking"])
            + "\nThe flight system doesn't report refund details, so I can't say anything about a refund."
        )
    if kind == "already_cancelled":
        return f"Booking {outcome['booking_id']} was already cancelled, so there is nothing more to do."
    if kind == "booking_unavailable":
        alt = outcome.get("alternatives") or []
        tail = ("\nThese have seats instead:\n" + "\n".join(flight_line(f, i, outcome.get("passengers")) for i, f in enumerate(alt, 1))) if alt else ""
        query = outcome["query"]
        return (
            f"Sorry, {outcome['flight_label']} isn't available from {_route(query)} on {fmt_date(query['departure_date'])} "
            f"for {_plural(query['passengers'], 'passenger')} (it doesn't run then, or has too few seats), "
            f"so I haven't booked anything.{tail}"
        )
    if kind in {"booking_failed", "cancel_failed", "error", "booking_unverified", "cancel_unverified", "no_context"}:
        return outcome["message"]
    if kind == "declined":
        what = "booking" if outcome.get("action") == "book" else "cancellation"
        return f"No problem, I haven't gone ahead with the {what}. Anything else I can do?"
    if kind == "aborted":
        return "Okay, I've dropped that request. What would you like to do next?"
    if kind == "gave_up":
        labels = ", ".join(FIELD_LABELS.get(k, k.replace("_", " ")) for k in outcome.get("missing", []))
        return f"I still don't have the {labels}, so I can't continue. Tell me when you have it and we'll pick it up again."
    if kind == "help":
        return HELP_TEXT
    if kind == "unrecognized":
        return UNRECOGNIZED_TEXT
    return UNSUPPORTED_TEXT
