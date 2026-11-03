"""Deterministic text parsing: dates, numbers, IDs, contact details and yes/no replies.

Everything numeric or identifying is extracted here, never by the LLM, so values are exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(?:\d+|" + "|".join(NUMBER_WORDS) + r")"
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ORD = r"(?:st|nd|rd|th)?"
_SEP = r"\s*(?:-|–|to|till|until|through)\s*"
_YEAR = r"(?:,?\s*(?P<y>\d{4}))?"

_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
    re.compile(rf"(?P<d1>\d{{1,2}}){_ORD}{_SEP}(?P<d2>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<m>{_MONTH})\b\.?{_YEAR}", re.I),
    re.compile(rf"(?P<m>{_MONTH})\b\.?\s+(?P<d1>\d{{1,2}}){_ORD}{_SEP}(?P<d2>\d{{1,2}}){_ORD}(?!\d){_YEAR}", re.I),
    re.compile(r"(?<![\d/-])(?P<d>\d{1,2})[/-](?P<m>\d{1,2})[/-](?P<y>\d{4})(?!\d)"),
    re.compile(r"(?<![\d/-])(?P<d>\d{1,2})/(?P<m>\d{1,2})(?![\d/])"),
    re.compile(rf"(?<![\d-])(?P<d>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<m>{_MONTH})\b\.?{_YEAR}", re.I),
    re.compile(rf"\b(?P<m>{_MONTH})\b\.?\s+(?P<d>\d{{1,2}}){_ORD}(?!\d){_YEAR}", re.I),
    re.compile(r"\b(?P<rel>day after tomorrow|tomorrow|tonight|today)\b", re.I),
    re.compile(r"\b(?:(?P<next>next|this|coming|on)\s+)?(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
]

BOOKING_ID_RE = re.compile(r"\bHB-[A-Z0-9]{4,}\b", re.I)
HOTEL_ID_RE = re.compile(r"\bHT-[A-Z0-9]{2,}\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s\-()]{8,18}\d(?![\w])")


@dataclass
class Extracted:
    """Parameters found in one user message by rules alone."""

    params: dict[str, Any] = field(default_factory=dict)
    dates: list[date] = field(default_factory=list)
    nights: int | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)


def _to_int(token: str) -> int | None:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBER_WORDS.get(token)


def _month(token: str) -> int:
    return MONTHS[token[:3].lower()]


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in used)


def _resolve(day: int, month: int, year: int | None, today: date) -> date | None:
    """Build a date; with no explicit year pick the next occurrence on or after today."""
    try:
        if year is not None:
            return date(year, month, day)
        candidate = date(today.year, month, day)
        return candidate if candidate >= today else date(today.year + 1, month, day)
    except ValueError:
        return None


def parse_dates(text: str, today: date) -> tuple[list[date], list[tuple[int, int]]]:
    """Return dates in textual order plus the character spans they came from."""
    found: list[tuple[int, date]] = []
    spans: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, spans):
                continue
            dates = _match_dates(match, today)
            if dates:
                spans.append(span)
                found.extend((span[0], d) for d in dates)
    found.sort(key=lambda item: item[0])
    return [d for _, d in found], spans


def _match_dates(match: re.Match[str], today: date) -> list[date]:
    g = match.groupdict()
    year = int(g["y"]) if g.get("y") else None
    if g.get("rel"):
        offset = {"today": 0, "tonight": 0, "tomorrow": 1, "day after tomorrow": 2}[g["rel"].lower()]
        return [today + timedelta(days=offset)]
    if g.get("wd"):
        delta = (WEEKDAYS[g["wd"].lower()] - today.weekday()) % 7
        return [today + timedelta(days=delta or 7)]
    month = int(g["m"]) if g["m"].isdigit() else _month(g["m"])
    if not 1 <= month <= 12:
        return []
    if g.get("d1"):
        first = _resolve(int(g["d1"]), month, year, today)
        second = _resolve(int(g["d2"]), month, year, today)
        return [first, second] if first and second else []
    single = _resolve(int(g["d"]), month, year, today)
    return [single] if single else []


def extract_params(text: str, today: date, awaiting: str | None = None) -> Extracted:
    """Pull every rule-detectable parameter out of `text`. `awaiting` lets bare replies be understood."""
    out = Extracted()
    params = out.params
    lowered = text.lower()

    out.dates, out.spans = parse_dates(text, today)

    if match := BOOKING_ID_RE.search(text):
        params["booking_id"] = match.group().upper()
    if match := HOTEL_ID_RE.search(text):
        params["hotel_id"] = match.group().upper()
    if match := EMAIL_RE.search(text):
        params["guest_email"] = match.group().lower()
        out.spans.append(match.span())

    if phone := _find_phone(text, out.spans):
        params["guest_phone"] = phone

    if match := re.search(rf"\b({_NUM})\s*nights?\b", lowered):
        out.nights = _to_int(match.group(1))
    if match := re.search(rf"\b({_NUM})\s*(?:guests?|people|persons?|adults?|pax|travell?ers|members)\b", lowered):
        params["guests"] = _to_int(match.group(1))
    elif re.search(r"\b(?:just me|only me|myself|solo|alone)\b", lowered):
        params["guests"] = 1
    if match := re.search(rf"\b({_NUM})\s*(?:rooms?)\b", lowered):
        params["rooms"] = _to_int(match.group(1))
    if match := re.search(r"\b(?:single|a) room\b", lowered):
        params.setdefault("rooms", 1)

    if match := re.search(
        r"(?:under|below|within|less than|up ?to|upto|max(?:imum)?|budget(?: of| is)?|not more than|cheaper than)"
        r"\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*)\s*(k)?\b",
        lowered,
    ):
        value = int(match.group(1).replace(",", "")) * (1000 if match.group(2) else 1)
        if value > 0:
            params["max_price_per_night"] = value
    if match := re.search(r"(?:rating|rated)\D{0,12}?([1-5](?:\.\d)?)\b|\b([1-5](?:\.\d)?)\s*\+?\s*rating", lowered):
        params["min_rating"] = float(match.group(1) or match.group(2))

    if name := _find_name(text, awaiting):
        params["guest_name"] = name

    _bare_reply(text, lowered, awaiting, params, out)
    return out


def _bare_reply(text: str, lowered: str, awaiting: str | None, params: dict[str, Any], out: Extracted) -> None:
    """Interpret one-word answers (e.g. '2', 'Goa') using the question that was asked."""
    stripped = lowered.strip(" .!?")
    if awaiting in {"guests", "rooms"} and awaiting not in params:
        number = _to_int(stripped)
        if number is not None:
            params[awaiting] = number
    elif awaiting == "city" and "city" not in params:
        if re.fullmatch(r"[a-z][a-z .'-]{1,40}", stripped) and len(stripped.split()) <= 3:
            params["city"] = " ".join(w.capitalize() for w in stripped.split())
    elif awaiting == "guest_name" and "guest_name" not in params:
        cleaned = EMAIL_RE.sub(" ", text)
        cleaned = _PHONE_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", re.sub(r"[,;]+", " ", cleaned)).strip(" .!?")
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", cleaned) and len(cleaned.split()) <= 4:
            if cleaned.lower() not in _YES | _NO:
                params["guest_name"] = _clean_name(cleaned)


_NAME_STOP = {"and", "my", "email", "phone", "mobile", "number", "with", "at", "contact", "is", "no"}
_NAME_RE = re.compile(
    r"(?:my name is|name is|name:|guest name(?: is|:)?|under the name|booking name(?: is|:)?)\s+"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3})",
    re.I,
)
_IM_RE = re.compile(r"\b(?:i am|i'm|this is)\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2})", re.I)


def _clean_name(raw: str) -> str | None:
    words: list[str] = []
    for word in raw.replace(",", " ").split():
        if word.lower() in _NAME_STOP:
            break
        words.append(word)
    name = " ".join(w[:1].upper() + w[1:] for w in words)
    return name if len(name) >= 2 else None


def _find_name(text: str, awaiting: str | None) -> str | None:
    if match := _NAME_RE.search(text):
        return _clean_name(match.group(1))
    if awaiting == "guest_name" and (match := _IM_RE.search(text)):
        return _clean_name(match.group(1))
    return None


def _find_phone(text: str, used: list[tuple[int, int]]) -> str | None:
    for match in _PHONE_RE.finditer(text):
        if _overlaps(match.span(), used):
            continue
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 15 and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", match.group().strip()):
            return ("+" if match.group().strip().startswith("+") else "") + digits
    return None


# ---------------------------------------------------------------- yes / no / abort / retry

_YES = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct", "proceed",
    "go ahead", "please do", "do it", "book it", "go for it", "sounds good", "that's right", "thats right",
    "absolutely", "definitely", "yes please", "yes confirm", "confirm it", "cancel it", "yes cancel it",
    "yes book it", "yes go ahead", "yes proceed", "please proceed", "please go ahead", "cancel it please",
}
_NO = {"no", "n", "nope", "nah"}
_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed"}
_COURTESY = {
    "please", "pls", "go", "ahead", "do", "it", "that", "this", "book", "cancel", "proceed", "thanks", "thank",
    "you", "confirm", "sure", "definitely", "absolutely", "now", "the", "booking", "and", "is", "fine",
}
_NEGATION_RE = re.compile(
    r"\b(?:no|nope|nah|don'?t|do not|stop|never ?mind|not now|wait|hold on|abort|not yet|keep it|leave it)\b", re.I
)
_ABORT_RE = re.compile(r"\b(?:never ?mind|forget it|start over|scrap that|drop it|leave it|stop)\b", re.I)
_RETRY_RE = re.compile(r"\b(?:retry|try again|check again|resubmit|re-?send|once more)\b", re.I)


def classify_confirmation(text: str, action: str, has_new_info: bool) -> str:
    """Return 'confirmed', 'declined' or 'changed'. Anything unclear is never treated as consent."""
    cleaned = re.sub(r"[^\w' ]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if has_new_info:
        return "changed"
    if _NEGATION_RE.search(cleaned):
        return "declined"
    if action == "book" and re.search(r"\bcancel\b", cleaned):
        return "declined"
    if cleaned in _YES and len(cleaned.split()) <= 5:
        return "confirmed"
    words = cleaned.split()
    if words and len(words) <= 6 and words[0] in _YES_WORDS and all(w in _COURTESY for w in words[1:]):
        return "confirmed"
    return "changed"


def is_abort(text: str) -> bool:
    return bool(_ABORT_RE.search(text))


def is_retry(text: str) -> bool:
    return bool(_RETRY_RE.search(text))

