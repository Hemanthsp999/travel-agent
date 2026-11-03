"""User-facing text. Every fact (name, price, ID, date, status) is rendered from API data in code.

The functions take plain dicts produced by the nodes, so wording can change without touching logic.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_agent.hotel_agent.schemas import TaskStatus

FIELD_LABELS = {
    "city": "city", "check_in": "check-in date", "check_out": "check-out date", "hotel": "hotel",
    "guests": "number of guests", "rooms": "number of rooms", "guest_name": "guest name",
    "guest_email": "email address", "guest_phone": "phone number", "booking_id": "booking ID",
}

STATUS_BY_KIND = {
    "search_results": TaskStatus.COMPLETED, "no_results": TaskStatus.COMPLETED,
    "hotel_details": TaskStatus.COMPLETED, "comparison": TaskStatus.COMPLETED, "show_all": TaskStatus.COMPLETED,
    "booking_success": TaskStatus.COMPLETED, "cancel_success": TaskStatus.COMPLETED,
    "already_cancelled": TaskStatus.COMPLETED, "booking_status": TaskStatus.COMPLETED,
    "help": TaskStatus.COMPLETED, "unsupported": TaskStatus.COMPLETED, "no_context": TaskStatus.COMPLETED,
    "booking_unavailable": TaskStatus.FAILED, "booking_failed": TaskStatus.FAILED, "cancel_failed": TaskStatus.FAILED,
    "error": TaskStatus.FAILED, "gave_up": TaskStatus.FAILED, "aborted": TaskStatus.DECLINED,
    "declined": TaskStatus.DECLINED, "booking_unverified": TaskStatus.UNVERIFIED,
    "cancel_unverified": TaskStatus.UNVERIFIED,
}


def fmt_date(value: str | date) -> str:
    d = date.fromisoformat(value) if isinstance(value, str) else value
    return f"{d.day} {d.strftime('%b %Y')}"


def money(amount: int | float | None, currency: str | None = None) -> str:
    return f"{currency or 'INR'} {int(amount):,}" if amount is not None else "price not provided"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _stay(query: dict[str, Any]) -> str:
    parts = [f"{fmt_date(query['check_in'])} to {fmt_date(query['check_out'])}"]
    if query.get("guests"):
        parts.append(_plural(query["guests"], "guest"))
    if query.get("rooms"):
        parts.append(_plural(query["rooms"], "room"))
    return ", ".join(parts)


def _filters(query: dict[str, Any]) -> str:
    bits = []
    if query.get("max_price_per_night"):
        bits.append(f"up to {money(query['max_price_per_night'])} per night")
    if query.get("min_rating"):
        bits.append(f"rated {query['min_rating']}+")
    return f" ({', '.join(bits)})" if bits else ""


def hotel_line(hotel: dict[str, Any], index: int | None = None) -> str:
    head = f"{index}. " if index else "- "
    place = f" ({hotel['locality']})" if hotel.get("locality") else ""
    bits = []
    if hotel.get("rating") is not None:
        bits.append(f"rating {hotel['rating']}")
    if hotel.get("price_per_night") is not None:
        bits.append(f"{money(hotel['price_per_night'], hotel.get('currency'))}/night")
    if hotel.get("estimated_total_price") is not None:
        bits.append(f"{money(hotel['estimated_total_price'], hotel.get('currency'))} total")
    if hotel.get("rooms_available") is not None:
        bits.append(f"{_plural(hotel['rooms_available'], 'room')} free")
    return f"{head}{hotel['name']}{place}" + (f": {', '.join(bits)}" if bits else "")


# ----------------------------------------------------------------------------- questions


def ask_text(group: list[str], params: dict[str, Any], note: str | None, attempt: int, intent: str) -> str:
    """One natural question for the first group of missing values."""
    lead = f"{note} " if note else ""
    first = group[0]
    if first == "city":
        return lead + "Happy to help! Which city are you looking to stay in?"
    if first in ("check_in", "check_out"):
        city = f" in {params['city']}" if params.get("city") else ""
        if group == ["check_out"]:
            return lead + f"Got it, check-in on {fmt_date(params['check_in'])}. What date will you check out?"
        if group == ["check_in"]:
            return lead + f"And what's your check-in date{city}?"
        return lead + f"Lovely! What are your check-in and check-out dates{city}?"
    if first == "hotel":
        return lead + "Which hotel would you like? You can give me its name or ID."
    if first in ("guests", "rooms"):
        if len(group) == 2:
            return lead + "How many guests will be staying, and how many rooms would you like?"
        return lead + ("How many guests will be staying?" if first == "guests" else "How many rooms would you like?")
    if first in ("guest_name", "guest_email", "guest_phone"):
        labels = [FIELD_LABELS[k] for k in group]
        joined = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
        tail = " for the booking" if len(group) > 1 else ""
        return lead + f"Almost there! Could you share the {joined}{tail}?"
    if first == "booking_id":
        action = "cancel" if intent == "CANCEL_BOOKING" else "look up"
        return lead + f"Sure. What's the booking ID you'd like me to {action}? It looks like HB-XXXXXXXXXX."
    return lead + "Could you tell me a bit more?"


def booking_prompt(payload: dict[str, Any], quote: dict[str, Any]) -> str:
    cur = quote.get("currency")
    lines = [
        "Here is what I'm about to book:",
        f"- Hotel: {quote['hotel_name']}" + (f" ({quote['locality']})" if quote.get("locality") else ""),
        f"- Stay: {fmt_date(payload['check_in'])} to {fmt_date(payload['check_out'])}"
        + (f" ({_plural(quote['nights'], 'night')})" if quote.get("nights") else ""),
        f"- {_plural(payload['rooms'], 'room')} for {_plural(payload['guests'], 'guest')}",
    ]
    if quote.get("price_per_night") is not None:
        lines.append(f"- Price: {money(quote['price_per_night'], cur)} per night")
    if quote.get("estimated_total_price") is not None:
        lines.append(f"- Estimated total: {money(quote['estimated_total_price'], cur)}")
    lines.append(f"- Guest: {payload['guest_name']}, {payload['guest_email']}, {payload['guest_phone']}")
    lines.append("Shall I go ahead and book it? (yes / no)")
    return "\n".join(lines)


def cancel_prompt(booking: dict[str, Any]) -> str:
    hotel = (booking.get("hotel") or {}).get("name", "this hotel")
    stay = (
        f", {fmt_date(booking['check_in'])} to {fmt_date(booking['check_out'])}"
        if booking.get("check_in") and booking.get("check_out") else ""
    )
    total = f" (total {money(booking['total_price'], booking.get('currency'))})" if booking.get("total_price") is not None else ""
    return (
        f"I found booking {booking['booking_id']} at {hotel}{stay}{total}. "
        "Cancelling can't be undone. Do you want me to cancel it? (yes / no)"
    )


# ------------------------------------------------------------------------------- answers


def booking_block(booking: dict[str, Any]) -> str:
    hotel = booking.get("hotel") or {}
    lines = [f"Booking {booking['booking_id']} is {booking['status']}."]
    if hotel.get("name"):
        place = f" ({hotel['locality']})" if hotel.get("locality") else ""
        lines.append(f"- Hotel: {hotel['name']}{place}")
    if booking.get("check_in") and booking.get("check_out"):
        nights = f" ({_plural(booking['nights'], 'night')})" if booking.get("nights") else ""
        lines.append(f"- Stay: {fmt_date(booking['check_in'])} to {fmt_date(booking['check_out'])}{nights}")
    if booking.get("rooms") and booking.get("guests"):
        lines.append(f"- {_plural(booking['rooms'], 'room')} for {_plural(booking['guests'], 'guest')}")
    if booking.get("total_price") is not None:
        lines.append(f"- Total: {money(booking['total_price'], booking.get('currency'))}")
    if (booking.get("guest") or {}).get("name"):
        lines.append(f"- Guest: {booking['guest']['name']}")
    if booking.get("cancelled_at"):
        lines.append(f"- Cancelled at: {booking['cancelled_at']}")
    return "\n".join(lines)


def details_text(hotel: dict[str, Any]) -> str:
    lines = [f"{hotel['name']}" + (f" in {hotel['city']}" if hotel.get("city") else "") + ":"]
    if hotel.get("address"):
        lines.append(f"- Address: {hotel['address']}")
    if hotel.get("rating") is not None:
        stars = f", {hotel['star_category']}-star" if hotel.get("star_category") else ""
        lines.append(f"- Rating: {hotel['rating']}{stars}")
    if hotel.get("price_per_night") is not None:
        lines.append(f"- Price: {money(hotel['price_per_night'], hotel.get('currency'))} per night")
    if hotel.get("max_guests_per_room"):
        lines.append(f"- Up to {_plural(hotel['max_guests_per_room'], 'guest')} per room")
    if hotel.get("amenities"):
        lines.append(f"- Amenities: {', '.join(hotel['amenities'])}")
    return "\n".join(lines)


def comparison_text(hotels: list[dict[str, Any]]) -> str:
    lines = ["Here's how they compare:"]
    for hotel in hotels:
        extra = f", {', '.join(hotel['amenities'][:4])}" if hotel.get("amenities") else ""
        lines.append(hotel_line(hotel) + extra)
    priced = [h for h in hotels if h.get("price_per_night") is not None]
    rated = [h for h in hotels if h.get("rating") is not None]
    if len(priced) > 1:
        lines.append(f"Lowest price: {min(priced, key=lambda h: h['price_per_night'])['name']}.")
    if len(rated) > 1:
        lines.append(f"Highest rating: {max(rated, key=lambda h: h['rating'])['name']}.")
    return "\n".join(lines)


def search_text(outcome: dict[str, Any], max_shown: int) -> str:
    query, hotels = outcome["query"], outcome["hotels"]
    shown = [h for h in hotels if h["hotel_id"] in set(outcome.get("shown_ids", []))] or hotels[:max_shown]
    head = f"Good news! I found {_plural(len(hotels), 'hotel')} in {query['city']} for {_stay(query)}{_filters(query)}"
    if outcome.get("focus_available"):
        head = f"Yes! {outcome['focus_available']} has rooms free in {query['city']} for {_stay(query)}{_filters(query)}"
    elif outcome.get("focus_missing"):
        head = (
            f"I couldn't match {outcome['focus_missing']} to a hotel with rooms free, but I found "
            f"{_plural(len(hotels), 'hotel')} in {query['city']} for {_stay(query)}{_filters(query)}"
        )
    lines = [head + ":"] + [hotel_line(h, i) for i, h in enumerate(shown, 1)]
    if len(hotels) > len(shown):
        lines.append(f"({len(hotels) - len(shown)} more available. Say \"show all\" to see them.)")
    if outcome.get("assumptions"):
        lines.append("Note: " + " ".join(outcome["assumptions"]))
    lines.append(
        "Which one would you like to book?" if outcome.get("invite_booking")
        else "Want details on any of them, or shall I book one?"
    )
    return "\n".join(lines)


def no_results_text(outcome: dict[str, Any]) -> str:
    query = outcome["query"]
    tip = " Try other dates" + (", a higher budget" if query.get("max_price_per_night") else "")
    tip += (", or a lower rating filter" if query.get("min_rating") else "") + ", or another city."
    return f"I couldn't find any hotels in {query['city']} with rooms free for {_stay(query)}{_filters(query)}.{tip}"


HELP_TEXT = (
    "I'm your hotel assistant. I can check hotel availability, share hotel details, book a room, "
    "check a booking's status, and cancel a booking. What would you like to do?"
)
UNSUPPORTED_TEXT = (
    "I can't help with that part yet. I can check availability, show hotel details, book, "
    "look up a booking, and cancel one. Which of those would you like?"
)


def render(outcome: dict[str, Any], max_shown: int = 3) -> str:
    """Final reply for an outcome dict."""
    kind = outcome["kind"]
    if kind == "search_results":
        return search_text(outcome, max_shown)
    if kind == "show_all":
        return search_text({**outcome, "shown_ids": [h["hotel_id"] for h in outcome["hotels"]]}, max_shown)
    if kind == "no_results":
        return no_results_text(outcome)
    if kind == "hotel_details":
        return details_text(outcome["hotel"]) + "\nWould you like to check availability or book it?"
    if kind == "comparison":
        return comparison_text(outcome["hotels"])
    if kind == "booking_success":
        return (
            "All done! Your booking is confirmed.\n" + booking_block(outcome["booking"])
            + "\nWould you like help with anything else?"
        )
    if kind == "booking_status":
        return booking_block(outcome["booking"])
    if kind == "cancel_success":
        note = " (I double-checked this with the hotel system.)" if outcome.get("verified_by_lookup") else ""
        return "Done, that booking is cancelled" + note + ".\n" + booking_block(outcome["booking"])
    if kind == "already_cancelled":
        return f"Booking {outcome['booking_id']} was already cancelled, so there is nothing more to do."
    if kind == "booking_unavailable":
        alt = outcome.get("alternatives") or []
        tail = ("\nThese are free instead:\n" + "\n".join(hotel_line(h, i) for i, h in enumerate(alt, 1))) if alt else ""
        return f"Sorry, {outcome['hotel_name']} has no availability for those dates and party size, so I haven't booked anything.{tail}"
    if kind in {"booking_failed", "cancel_failed", "error"}:
        return outcome["message"]
    if kind in {"booking_unverified", "cancel_unverified"}:
        return outcome["message"]
    if kind == "declined":
        what = "booking" if outcome.get("action") == "book" else "cancellation"
        return f"No problem, I haven't gone ahead with the {what}. Anything else I can do?"
    if kind == "aborted":
        return "Okay, I've dropped that request. What would you like to do next?"
    if kind == "gave_up":
        labels = ", ".join(FIELD_LABELS.get(k, k) for k in outcome.get("missing", []))
        return f"I still don't have the {labels}, so I can't continue. Tell me when you have it and we'll pick it up again."
    if kind == "no_context":
        return outcome["message"]
    if kind == "help":
        return HELP_TEXT
    return UNSUPPORTED_TEXT
