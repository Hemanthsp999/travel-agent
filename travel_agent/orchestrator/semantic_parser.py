"""Convert free-form travel requests into a fixed, structured schema."""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from travel_agent.flight_agent.parsing import CITY_ALIASES, normalize_place, parse_route
from travel_agent.hotel_agent.parsing import NUMBER_WORDS, parse_dates


PREFERENCE_HINTS = (
    "beach",
    "airport",
    "pool",
    "wifi",
    "parking",
    "breakfast",
    "quiet",
    "budget",
    "luxury",
    "downtown",
    "city center",
    "family",
    "pet friendly",
)


def _to_int(token: str | None) -> int | None:
    if token is None:
        return None
    value = token.strip().lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _extract_passengers(text: str) -> int:
    match = re.search(r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:passengers?|people|persons?|adults?|travellers?|traveler|pax)\b", text, re.I)
    if match:
        value = _to_int(match.group("count"))
        if value is not None:
            return value
    if re.search(r"\b(?:solo|alone|just me|myself)\b", text, re.I):
        return 1
    return 1


def _extract_nights(text: str) -> int | None:
    patterns = [
        r"\bfor\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b",
        r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+night(?:s)?\b",
        r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+trip\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = _to_int(match.group("count"))
            if value is not None:
                return value
    return None


def _extract_preferences(text: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for hint in PREFERENCE_HINTS:
        if hint in lower and hint not in found:
            found.append(hint)
    return found


def _extract_date(text: str) -> str | None:
    dates, _ = parse_dates(text, date.today())
    if not dates:
        return None
    return dates[0].isoformat()


def _extract_hotel_city(text: str) -> str | None:
    route_origin, route_destination = parse_route(text)
    if route_origin and route_destination:
        return normalize_place(route_destination)

    matches: list[tuple[int, str]] = []
    for city_name, canonical in sorted(CITY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(city_name)}\b", text, re.I):
            matches.append((text.lower().find(city_name.lower()), canonical))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def extract_semantic_request(text: str) -> dict[str, Any]:
    """Return a stable schema for travel intent extraction."""
    if not text or not text.strip():
        return {
            "origin": None,
            "destination": None,
            "travel_date": None,
            "hotel_city": None,
            "nights": None,
            "preferences": [],
            "passengers": 1,
        }

    source_text = text.strip()
    origin, destination = parse_route(source_text)
    normalized_origin = normalize_place(origin) if origin else None
    normalized_destination = normalize_place(destination) if destination else None

    hotel_city = normalized_destination or normalized_origin or _extract_hotel_city(source_text)
    nights = _extract_nights(source_text)
    preferences = _extract_preferences(source_text)
    passengers = _extract_passengers(source_text)
    travel_date = _extract_date(source_text)

    return {
        "origin": normalized_origin,
        "destination": normalized_destination,
        "travel_date": travel_date,
        "hotel_city": hotel_city,
        "nights": nights,
        "preferences": preferences,
        "passengers": passengers,
    }
