"""Deterministic text parsing for flight requests: route, dates, counts, IDs, contacts and passengers.

Everything numeric or identifying is extracted here, never by the LLM, so values are exact.
Domain-agnostic helpers (dates, yes/no replies, e-mail) are reused from the hotel agent's public parsing API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from travel_agent.hotel_agent.parsing import (  # shared, domain-agnostic text utilities
    EMAIL_RE,
    NUMBER_WORDS,
    classify_confirmation,
    is_abort,
    is_retry,
    parse_dates,
)

__all__ = [
    "Extracted", "extract", "classify_confirmation", "is_abort", "is_retry", "merge_passengers",
    "normalize_place", "canonical_key", "is_known_city",
]

# Recognition list only: it lets free text like "Pune to Goa" be read as places. Whether a route is
# served is always decided by the Flight API, never by this table.
CITY_ALIASES = {
    "bengaluru": "Bangalore", "bangalore": "Bangalore", "bombay": "Mumbai", "mumbai": "Mumbai",
    "delhi": "Delhi", "new delhi": "Delhi", "hyderabad": "Hyderabad", "chennai": "Chennai", "madras": "Chennai",
    "kolkata": "Kolkata", "calcutta": "Kolkata", "pune": "Pune", "goa": "Goa", "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur", "kochi": "Kochi", "cochin": "Kochi",
}
IATA_CODES = {"BLR": "Bangalore", "BOM": "Mumbai", "DEL": "Delhi", "HYD": "Hyderabad", "MAA": "Chennai"}

_CITY_RE = re.compile(r"\b(" + "|".join(sorted(CITY_ALIASES, key=len, reverse=True)) + r")\b", re.I)
_CODE_RE = re.compile(r"\b(" + "|".join(IATA_CODES) + r")\b")  # codes only count when upper-case
_NUM = r"(?:\d+|" + "|".join(NUMBER_WORDS) + r")"

TRAVELLER_AWAITING = {"passenger_details", "contact_details"}  # replies carrying names, ages, e-mail, phone

BOOKING_ID_RE = re.compile(r"\bFB-[A-Z0-9]{4,}\b", re.I)
FLIGHT_ID_RE = re.compile(r"\bFL-\d{3,5}\b", re.I)
_FLIGHT_NO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]|[0-9][A-Za-z])-(\d{2,4})\b")
_FLIGHT_NO_SPACED_RE = re.compile(r"\b([A-Z][A-Z0-9]|[0-9][A-Z]) (\d{3,4})\b")  # upper-case only: "6E 501"
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s\-()]{8,18}\d(?![\w])")

# ------------------------------------------------------------------------------- places

_PLACE_STOP = {
    "on", "for", "at", "in", "next", "this", "coming", "tomorrow", "today", "tonight", "with", "under", "by",
    "around", "the", "and", "please", "pls", "to", "from", "leaving", "departing", "flight", "flights", "ticket",
    "tickets", "airport", "a", "an", "my", "me", "us", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "passengers", "passenger", "people", "person", "adults", "adult", "travellers", "pax",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "january",
    "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "evening",
    "night", "afternoon", "here", "there", "it", "that", "this", "is", "are", "can", "could", "would", "want",
    "need", "book", "booking", "check", "find", "show", "search", "cheapest", "earliest", "fastest", "latest",
    "yes", "no", "ok", "okay", "thanks", "thank", "you", "hi", "hello", "just", "only", "solo", "alone",
    "oneway", "one-way", "round", "trip", "return", "of", "cancel", "status", "id", "number", "flightbooking",
    "all", "more", "any", "some", "what", "which", "how", "who", "why", "when", "where", "help", "about", "details",
    "option", "options", "other", "else", "refund", "change", "modify", "list", "see", "give", "tell", "get", "fly",
    "go", "going", "travel", "traveling", "travelling", "available", "availability", "seat", "seats", "price", "prices",
}
_PLACE = r"[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*){0,2}"
_CAP_PLACE = r"[A-Z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*){0,2}"
_FROM_TO_RE = re.compile(rf"\bfrom\s+(?P<o>{_PLACE})\s+(?:to|→|->|towards)\s+(?P<d>{_PLACE})", re.I)
_BETWEEN_RE = re.compile(rf"\bbetween\s+(?P<o>{_PLACE})\s+and\s+(?P<d>{_PLACE})", re.I)
_TO_FROM_RE = re.compile(rf"\bto\s+(?P<d>{_PLACE})\s+from\s+(?P<o>{_PLACE})", re.I)
_ARROW_RE = re.compile(rf"(?P<o>{_PLACE})\s*(?:→|->|—>)\s*(?P<d>{_PLACE})")
_ORIGIN_MARK = re.compile(r"\b(?:from|leaving|departing|out of|ex)\s*$", re.I)
_DEST_MARK = re.compile(r"\b(?:to|into|towards|for|reach|arrive in|arriving in|fly to|going to)\s*$", re.I)


def canonical_key(place: str | None) -> str:
    """Comparable form of a place (aliases and IATA codes collapse to the same key)."""
    return re.sub(r"[^a-z]+", " ", normalize_place(place or "").lower()).strip()


def normalize_place(raw: str) -> str:
    """Canonical city name for known aliases/codes, otherwise the user's words in Title Case."""
    text = raw.strip()
    if text.upper() in IATA_CODES and text.isupper():
        return IATA_CODES[text.upper()]
    key = re.sub(r"[^a-z ]+", " ", text.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    return " ".join(w[:1].upper() + w[1:] for w in key.split())


def is_known_city(raw: str) -> bool:
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", raw.lower())).strip()
    return key in CITY_ALIASES or (raw.strip().isupper() and raw.strip() in IATA_CODES)


def _clean_place(raw: str | None) -> str | None:
    """Trim leading/trailing filler words ('Mumbai on' -> 'Mumbai'); None if nothing place-like remains."""
    if not raw:
        return None
    words = raw.replace("→", " ").split()
    while words and words[-1].lower().strip(".'-") in _PLACE_STOP:
        words.pop()
    while words and words[0].lower().strip(".'-") in _PLACE_STOP:
        words.pop(0)
    if not words:
        return None
    place = " ".join(words)
    return normalize_place(place) if len(place) >= 2 else None


def _known_cities(text: str) -> list[tuple[int, int, str]]:
    found = [(m.start(), m.end(), CITY_ALIASES[m.group(1).lower()]) for m in _CITY_RE.finditer(text)]
    found += [(m.start(), m.end(), IATA_CODES[m.group(1)]) for m in _CODE_RE.finditer(text)]
    return sorted(found)


def parse_route(
    text: str, awaiting: str | None = None, have_origin: bool = False, have_destination: bool = False
) -> tuple[str | None, str | None]:
    """Origin and destination named in `text` (either may be None)."""
    for pattern in (_FROM_TO_RE, _BETWEEN_RE, _ARROW_RE):
        if match := pattern.search(text):
            origin, dest = _clean_place(match["o"]), _clean_place(match["d"])
            if origin and dest:
                return origin, dest
    if match := _TO_FROM_RE.search(text):
        origin, dest = _clean_place(match["o"]), _clean_place(match["d"])
        if origin and dest:
            return origin, dest

    cities = _known_cities(text)
    origin = dest = None
    unmarked: list[str] = []
    unmarked_at: list[int] = []
    dest_at = -1
    for start, _end, city in cities:
        before = text[max(0, start - 18):start]
        if _ORIGIN_MARK.search(before) and origin is None:
            origin = city
        elif _DEST_MARK.search(before) and dest is None:
            dest, dest_at = city, start
        else:
            unmarked.append(city)
            unmarked_at.append(start)
    if origin is None and dest is None and len(unmarked) >= 2:
        return unmarked[0], unmarked[1]
    if origin is None and dest is not None and len(unmarked) == 1 and unmarked_at[0] < dest_at:
        return unmarked[0], dest  # "Bangalore to Mumbai"
    for city in unmarked:
        if origin is None and (awaiting == "origin" or (awaiting != "destination" and have_destination and not have_origin)):
            origin = city
        elif dest is None and (awaiting == "destination" or (awaiting != "origin" and have_origin and not have_destination)):
            dest = city
    if origin is None and dest is None:
        # Unknown cities need an explicit marker and a capital letter ("to Pune"), so "to book" is not a place.
        if match := re.search(rf"\bfrom\s+({_CAP_PLACE})", text):
            origin = _clean_place(match.group(1))
        if match := re.search(rf"\bto\s+({_CAP_PLACE})", text):
            dest = _clean_place(match.group(1))
    if awaiting in ("origin", "destination") and origin is None and dest is None:
        place = _bare_place(text)
        if place:
            return (place, None) if awaiting == "origin" else (None, place)
    return origin, dest


def _bare_place(text: str) -> str | None:
    stripped = re.sub(r"^(?:from|to|in|at)\s+", "", text.strip(" .!?,"), flags=re.I)
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", stripped) and len(stripped.split()) <= 3:
        return _clean_place(stripped) if stripped.lower() not in _PLACE_STOP else None
    return None


# ----------------------------------------------------------------------------- extraction


@dataclass
class Extracted:
    """Parameters found in one user message by rules alone."""

    origin: str | None = None
    destination: str | None = None
    departure_date: date | None = None
    extra_dates: int = 0  # more than one date mentioned (range or return trip)
    round_trip: bool = False
    passenger_count: int | None = None
    booking_id: str | None = None
    flight_id: str | None = None
    flight_number: str | None = None
    email: str | None = None
    phone: str | None = None
    passengers: list[dict[str, Any]] = field(default_factory=list)  # pieces, see `merge_passengers`
    sort_hint: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def has_values(self) -> bool:
        return any((
            self.origin, self.destination, self.departure_date, self.passenger_count, self.booking_id,
            self.flight_id, self.flight_number, self.email, self.phone, self.passengers,
        ))


def _to_int(token: str) -> int | None:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBER_WORDS.get(token)


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in used)


_COUNT_RE = re.compile(
    rf"\b({_NUM})\s*(?:adults?|children|child|kids?|passengers?|people|persons?|pax|travell?ers|tickets?|seats?|members|of us)\b",
    re.I,
)
_FOR_COUNT_RE = re.compile(rf"\bfor\s+({_NUM})\b(?!\s*(?:st|nd|rd|th|nights?|days?|hours?|[/-]))", re.I)
_SOLO_RE = re.compile(r"\b(?:just me|only me|myself|solo|alone|one person|single passenger)\b", re.I)
_ROUND_TRIP_RE = re.compile(r"\b(?:round[- ]?trip|return(?:ing)?(?: flight| ticket| journey)?|two[- ]way|and back)\b", re.I)
_SORT_RE = (
    ("cheapest", re.compile(r"\b(?:cheapest|lowest (?:price|fare|cost)|least expensive|most affordable)\b", re.I)),
    ("fastest", re.compile(r"\b(?:fastest|quickest|shortest|least time)\b", re.I)),
    ("earliest", re.compile(r"\b(?:earliest|first flight of the day|earlier)\b", re.I)),
    ("latest", re.compile(r"\b(?:latest|last flight)\b", re.I)),
)


def extract(
    text: str,
    today: date,
    awaiting: str | None = None,
    *,
    have_origin: bool = False,
    have_destination: bool = False,
    allow_passengers: bool = False,
) -> Extracted:
    """Pull every rule-detectable parameter out of `text`. `awaiting` lets bare replies be understood."""
    out = Extracted()
    lowered = text.lower()
    typing_traveller_data = awaiting in TRAVELLER_AWAITING

    if match := BOOKING_ID_RE.search(text):
        out.booking_id = match.group().upper()
        out.spans.append(match.span())
    if match := FLIGHT_ID_RE.search(text):
        out.flight_id = match.group().upper()
        out.spans.append(match.span())
    if match := EMAIL_RE.search(text):
        out.email = match.group().lower()
        out.spans.append(match.span())
    if phone := _find_phone(text, out.spans):
        out.phone = phone[0]
        out.spans.append(phone[1])
    for pattern in (_FLIGHT_NO_RE, _FLIGHT_NO_SPACED_RE):
        for match in pattern.finditer(text):
            if out.flight_number or _overlaps(match.span(), out.spans) or match.group(1).upper() == "FL":
                continue
            out.flight_number = f"{match.group(1).upper()}-{match.group(2)}"
            out.spans.append(match.span())

    if not typing_traveller_data:
        dates, date_spans = parse_dates(text, today)
        out.spans.extend(date_spans)
        if dates:
            out.departure_date, out.extra_dates = dates[0], len(dates) - 1
        out.round_trip = bool(_ROUND_TRIP_RE.search(text))
        out.origin, out.destination = parse_route(text, awaiting, have_origin, have_destination)
        out.passenger_count = _passenger_count(text, lowered, awaiting, out.spans)
        for hint, pattern in _SORT_RE:
            if pattern.search(text):
                out.sort_hint = hint
                break

    if typing_traveller_data or allow_passengers:
        out.passengers = parse_passenger_pieces(text, out.spans, strict=not typing_traveller_data)
    return out


def _passenger_count(text: str, lowered: str, awaiting: str | None, spans: list[tuple[int, int]]) -> int | None:
    counts = [_to_int(m.group(1)) for m in _COUNT_RE.finditer(text) if not _overlaps(m.span(), spans)]
    counts = [c for c in counts if c is not None]
    if counts:
        return sum(counts)
    if match := _FOR_COUNT_RE.search(text):
        if not _overlaps(match.span(1), spans):
            return _to_int(match.group(1))
    if _SOLO_RE.search(text):
        return 1
    if awaiting == "passenger_count":
        return _to_int(lowered.strip(" .!?"))
    return None


def _find_phone(text: str, used: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    for match in _PHONE_RE.finditer(text):
        if _overlaps(match.span(), used):
            continue
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 15 and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", match.group().strip()):
            return ("+" if match.group().strip().startswith("+") else "") + digits, match.span()
    return None


# --------------------------------------------------------------------------- passengers

_GENDER_WORDS = {
    "male": "MALE", "man": "MALE", "boy": "MALE", "gentleman": "MALE",
    "female": "FEMALE", "woman": "FEMALE", "girl": "FEMALE", "lady": "FEMALE",
    "other": "OTHER", "nonbinary": "OTHER", "non-binary": "OTHER",
}
_GENDER_LETTERS = {"m": "MALE", "f": "FEMALE"}
_NAME_NOISE = {
    "age", "aged", "years", "year", "old", "yrs", "yr", "yo", "name", "is", "my", "passenger", "passengers",
    "traveller", "traveler", "gender", "sex", "and", "the", "a", "an", "i", "am", "im", "its", "it's", "first",
    "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "of", "for", "with", "email",
    "phone", "mobile", "number", "contact", "no", "yes", "ok", "okay", "please", "pls", "thanks", "thank", "you",
    "nope", "nah", "yeah", "sure", "cancel", "book", "booking", "flight", "flights", "hello", "hi", "retry",
    "confirm", "go", "ahead", "that", "this", "it", "is", "are", "was", "will", "be", "me", "myself", "same",
}
_LEAD_RE = re.compile(
    r"\b(?:my name is|the name is|name is|name:|this is|it'?s|i am|i'?m|passenger\s*\d*\s*(?:name)?:?|"
    r"traveller\s*\d*:?|traveler\s*\d*:?)\s*",
    re.I,
)
_AGE_RE = re.compile(r"(?:\bage[d]?\s*:?\s*)?\b(\d{1,3})\b(?:\s*(?:years?(?: old)?|yrs?|y/?o))?", re.I)
_COMPLETE_RE = re.compile(
    r"[A-Z][A-Za-z.'-]+(?: [A-Z][A-Za-z.'-]+){0,3},?\s+\d{1,3}(?:\s*(?:years?(?: old)?|yrs?|y/?o))?,?\s+"
    r"(?i:male|female|other|man|woman)\b"
)


def _parse_piece(piece: str) -> dict[str, Any]:
    """One name / age / gender fragment (any subset)."""
    cleaned = _LEAD_RE.sub(" ", piece)
    record: dict[str, Any] = {}
    for match in _AGE_RE.finditer(cleaned):
        if int(match.group(1)) <= 120:
            record["age"] = int(match.group(1))
            cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]
            break
    tokens = re.findall(r"[A-Za-z][A-Za-z.'-]*", cleaned)
    name_tokens: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in _GENDER_WORDS:
            record["gender"] = _GENDER_WORDS[low]
        elif low in _GENDER_LETTERS and ("age" in record or len(tokens) == 1):
            record["gender"] = _GENDER_LETTERS[low]
        elif low in _NAME_NOISE:
            if name_tokens:
                break  # name ended ("Asha Rao and ...")
        else:
            name_tokens.append(token)
    name = " ".join(t[:1].upper() + t[1:] for t in name_tokens)
    if 2 <= len(name) <= 100 and len(name_tokens) <= 4 and not is_known_city(name):
        record["name"] = name
    return record


def parse_passenger_pieces(text: str, used: list[tuple[int, int]], strict: bool) -> list[dict[str, Any]]:
    """Fragments of passenger data in textual order. `strict` keeps only complete name+age+gender records."""
    masked = list(text)
    for start, end in used:
        masked[start:end] = " " * (end - start)
    body = "".join(masked)
    if strict:  # inside a longer sentence only a complete "Name, age, gender" record is trusted
        found = (_parse_piece(m.group()) for m in _COMPLETE_RE.finditer(body))
        return [r for r in found if {"name", "age", "gender"} <= r.keys()]
    pieces = [p for p in re.split(r"[;\n|,&]|\band\b", body, flags=re.I) if p.strip()]
    return [r for r in (_parse_piece(p) for p in pieces) if r]


def merge_passengers(existing: list[dict[str, Any]], pieces: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    """Fill passenger records in order from parsed fragments; never invents values."""
    people = [dict(p) for p in existing]
    limit = count or 9
    for piece in pieces:
        target: dict[str, Any] | None = None
        if "name" in piece:
            target = next((p for p in people if "name" not in p), None)
            if target is None and len(people) < limit:
                target = {}
                people.append(target)
        else:
            keys = [k for k in ("age", "gender") if k in piece]
            target = next((p for p in people if any(k not in p for k in keys)), None)
        if target is None:
            continue
        for key, value in piece.items():
            if key not in target or key == "name":
                target[key] = value
    return people
