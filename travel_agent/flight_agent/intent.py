"""Intent classification (LLM with rule fallback), grounding of LLM text, and flight references."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from travel_agent.flight_agent.parsing import BOOKING_ID_RE, FLIGHT_ID_RE, parse_route
from travel_agent.flight_agent.schemas import GeneralTopic, Intent, IntentClassification
from travel_agent.llm.loader import LLM
from travel_agent.llm.structured import generate_structured
from travel_agent.prompts import load_prompt

logger = logging.getLogger(__name__)

_CANCEL_RE = re.compile(r"\bcancel(?:l?ation|l?ed|l?ing)?\b|\bcall off\b", re.I)
_BOOK_RE = re.compile(
    r"\b(?:book(?:ing)?|reserve|reservation|buy|purchase|i'?ll take|go with|take (?:it|this|that|the)|"
    r"proceed|choose|select|pick|want (?:this|that|the|it)|this one|that one|the (?:first|second|third|last) one|"
    r"get me (?:a|the) (?:ticket|seat|flight))\b",
    re.I,
)
_BOOK_VERB_RE = re.compile(r"\b(?:book(?:ing)?|reserve|buy|purchase)\b", re.I)
_STATUS_RE = re.compile(
    r"\b(?:status|booking details|(?:check|look up|lookup|retrieve|show) (?:on )?(?:my|the) (?:booking|reservation|ticket)|"
    r"my (?:booking|reservation|ticket)|is (?:my|the) (?:booking|reservation|ticket)|who(?:'s| is) (?:on|travelling)|"
    r"passengers? (?:on|in) (?:my|the) booking)\b",
    re.I,
)
_UNSUPPORTED_RE = re.compile(
    r"\b(?:refunds?|polic(?:y|ies)|baggage|luggage|meals?|upgrade|modify|modification|amend|reschedule|"
    r"web check-?in|check-?in online|boarding pass|lounge|visa|hotels?|payments?|pay|invoice|delay(?:ed)?|"
    r"live status|track(?:ing)?|loyalty|miles|points|coupon|discount|promo|wheelchair|insurance|multi-?city|"
    r"seat (?:selection|map|number)|choose (?:a|my) seat)\b|"
    r"\bchange (?:my|the|our) (?:booking|reservation|flight|date|seat|name)\b|\bcheck-?in\b",
    re.I,
)
_DETAILS_RE = re.compile(
    r"\b(?:details?|tell me (?:more )?about|more (?:info|information)|aircraft|duration|how long|which airline|"
    r"what time (?:does|is|will)|when does|how many seats|seats left|arrival time|departure time)\b",
    re.I,
)
_COMPARE_RE = re.compile(r"\b(?:compare|versus|vs\.?|difference between|which (?:one )?is better)\b", re.I)
_SHOW_MORE_RE = re.compile(
    r"\b(?:more options|show (?:me )?(?:all|more)|other options|see all|all (?:the )?(?:flights|options)|anything else)\b", re.I
)
_HELP_RE = re.compile(r"^\s*(?:hi|hello|hey|help|good (?:morning|afternoon|evening))\b|\bwhat can you (?:do|help)\b", re.I)
_SEARCH_RE = re.compile(
    r"\b(?:find|show|search|look(?:ing)? for|available|availability|flights?|options|fly|flying|travel|"
    r"cheapest|earliest|fastest|any (?:seats|tickets)|is there)\b",
    re.I,
)

ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}


@dataclass
class Classification:
    intent: Intent = Intent.UNKNOWN
    topic: GeneralTopic = GeneralTopic.NONE
    confidence: float = 0.0
    origin: str | None = None
    destination: str | None = None
    source: str = "rules"


@dataclass
class ClassifyContext:
    awaiting: str | None = None
    current_intent: str | None = None
    known_flights: list[str] = field(default_factory=list)
    recent: str = ""


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def grounded(value: str | None, text: str) -> bool:
    """True if `value` literally appears in the user's text (guards against LLM invention)."""
    return bool(value) and normalize(value) in normalize(text) and len(normalize(value)) >= 2  # type: ignore[arg-type]


# -------------------------------------------------------------------------- classification


def rule_classify(text: str) -> Classification:
    """Keyword-based fallback. Conservative: returns UNKNOWN unless the wording is clear."""
    has_booking_id = bool(BOOKING_ID_RE.search(text))
    if _UNSUPPORTED_RE.search(text) and not (has_booking_id and _CANCEL_RE.search(text)):
        return Classification(Intent.GENERAL_FLIGHT_QUERY, GeneralTopic.UNSUPPORTED, 0.7)
    if _CANCEL_RE.search(text):
        return Classification(Intent.CANCEL_FLIGHT, confidence=0.85)
    if _STATUS_RE.search(text) or (has_booking_id and not _BOOK_RE.search(text)):
        return Classification(Intent.GET_BOOKING_STATUS, confidence=0.8)
    if _COMPARE_RE.search(text):
        return Classification(Intent.GENERAL_FLIGHT_QUERY, GeneralTopic.COMPARE_FLIGHTS, 0.8)
    if _SHOW_MORE_RE.search(text):
        return Classification(Intent.GENERAL_FLIGHT_QUERY, GeneralTopic.SHOW_MORE, 0.8)
    origin, destination = parse_route(text)
    if _DETAILS_RE.search(text) and not (origin and destination) and not _BOOK_VERB_RE.search(text):
        return Classification(Intent.GENERAL_FLIGHT_QUERY, GeneralTopic.FLIGHT_DETAILS, 0.7)
    if _BOOK_RE.search(text):
        return Classification(Intent.BOOK_FLIGHT, confidence=0.75)
    if _HELP_RE.search(text):
        return Classification(Intent.GENERAL_FLIGHT_QUERY, GeneralTopic.HELP, 0.8)
    if _SEARCH_RE.search(text) or (origin and destination):
        return Classification(Intent.CHECK_FLIGHT_AVAILABILITY, confidence=0.65)
    return Classification()


def mentions_booking(text: str) -> bool:
    """True if the wording asks to book/reserve something."""
    return bool(_BOOK_RE.search(text))


def _plausible(intent: Intent, text: str) -> bool:
    """Reject LLM intents that have no support in the wording (cheap guard against misroutes)."""
    if intent is Intent.CANCEL_FLIGHT:
        return bool(_CANCEL_RE.search(text))
    if intent is Intent.BOOK_FLIGHT:
        return bool(_BOOK_RE.search(text))
    if intent is Intent.GET_BOOKING_STATUS:
        return bool(BOOKING_ID_RE.search(text) or re.search(r"\b(?:status|booking|reservation|ticket)\b", text, re.I))
    return True


def _llm_classify(llm: LLM | None, text: str, ctx: ClassifyContext, threshold: float) -> Classification | None:
    user = (
        f"message: {text!r}\nawaiting: {ctx.awaiting or 'none'}\ncurrent_intent: {ctx.current_intent or 'none'}\n"
        f"known_flights: {', '.join(ctx.known_flights) or 'none'}\nrecent:\n{ctx.recent or 'none'}"
    )
    parsed = generate_structured(llm, load_prompt("flight_agent"), user, IntentClassification)
    if parsed is None or parsed.confidence < threshold:
        return None
    intent = Intent(parsed.intent)
    if not _plausible(intent, text):
        logger.info("LLM intent %s rejected: no supporting wording", intent.value)
        return None
    return Classification(
        intent=intent,
        topic=GeneralTopic(parsed.general_topic),
        confidence=parsed.confidence,
        origin=parsed.origin if grounded(parsed.origin, text) else None,
        destination=parsed.destination if grounded(parsed.destination, text) else None,
        source="llm",
    )


def _names_route(text: str) -> bool:
    origin, destination = parse_route(text)
    return bool(origin and destination)


def _rules_override(rules: Classification, llm_result: Classification, text: str) -> bool:
    """Explicit keywords beat a vaguer LLM label.

    The 3B model tends to label 'book the cheapest one' or 'tell me about the first one' as a new
    search. An explicit keyword wins unless the user clearly named a whole route to search.
    """
    if rules.intent is Intent.BOOK_FLIGHT and llm_result.intent in (Intent.CHECK_FLIGHT_AVAILABILITY, Intent.GENERAL_FLIGHT_QUERY):
        return not _names_route(text)
    if rules.intent is not Intent.GENERAL_FLIGHT_QUERY or rules.topic in (GeneralTopic.NONE, GeneralTopic.HELP):
        return False
    if llm_result.intent is Intent.GENERAL_FLIGHT_QUERY:
        return llm_result.topic is not rules.topic
    if llm_result.intent is Intent.CHECK_FLIGHT_AVAILABILITY:
        return not _names_route(text)
    return False


def classify(llm: LLM | None, text: str, ctx: ClassifyContext, threshold: float = 0.5) -> Classification:
    """LLM first (validated and grounded), rules as fallback. Rule keywords may fill gaps."""
    rules = rule_classify(text)
    result = _llm_classify(llm, text, ctx, threshold) if llm is not None else None
    if result is None or (result.intent is Intent.UNKNOWN and rules.intent is not Intent.UNKNOWN):
        result = rules
    elif _rules_override(rules, result, text):
        result.intent, result.topic, result.confidence, result.source = (
            rules.intent, rules.topic, rules.confidence, "rules+llm",
        )
    if result.intent is not Intent.GENERAL_FLIGHT_QUERY:
        result.topic = GeneralTopic.NONE  # the small model sets topics inconsistently; only GENERAL uses one
    return result


# ---------------------------------------------------------------------- flight references


def _clock(departure: str) -> tuple[int, int]:
    hour, minute = departure[11:16].split(":")
    return int(hour), int(minute)


_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b", re.I)
_PARTS_OF_DAY = {
    "morning": lambda h: 4 <= h < 12,
    "afternoon": lambda h: 12 <= h < 17,
    "evening": lambda h: 17 <= h < 21,
    "night": lambda h: h >= 21 or h < 4,
}


def resolve_flight(
    text: str, shown: list[dict[str, Any]], allow_bare_number: bool = False
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve 'the second one', 'cheapest', 'FL-1001', '6E-501', '6:15 am' or an airline name.

    Only flights the user was shown can be chosen. Returns (flight, ambiguous_candidates); both
    empty means no reference was recognised. A bare digit counts only when `allow_bare_number`.
    """
    if not shown:
        return None, []
    lowered = text.lower()
    if match := FLIGHT_ID_RE.search(text):
        wanted = match.group().upper()
        return next((f for f in shown if f["flight_id"].upper() == wanted), None), []
    for flight in shown:
        if re.search(rf"\b{re.escape(flight['flight_number'].lower())}\b", lowered):
            return flight, []
    if match := re.search(r"(?:option|number|no\.?|#)\s*(\d)\b", lowered) or (
        allow_bare_number and re.fullmatch(r"\s*(\d)\s*[.!]?\s*", lowered)
    ):
        index = int(match.group(1))
        return (shown[index - 1], []) if 1 <= index <= len(shown) else (None, [])
    if re.search(r"\b(?:cheapest|lowest (?:price|fare)|least expensive)\b", lowered):
        return min(shown, key=lambda f: f["price_per_passenger"]), []
    if re.search(r"\b(?:fastest|quickest|shortest)\b", lowered):
        timed = [f for f in shown if f.get("duration_minutes") is not None]
        if timed:
            return min(timed, key=lambda f: f["duration_minutes"]), []
    if re.search(r"\b(?:earliest|first flight of the day)\b", lowered):
        return min(shown, key=lambda f: f["departure_time"]), []
    if re.search(r"\blatest\b", lowered):
        return max(shown, key=lambda f: f["departure_time"]), []
    for word, index in ORDINALS.items():
        if re.search(rf"\b{word}\b", lowered):
            return (shown[index - 1], []) if index <= len(shown) else (None, [])
    if re.search(r"\blast one\b|\bthe last\b", lowered):
        return shown[-1], []
    if match := _TIME_RE.search(lowered):
        hits = [f for f in shown if _matches_time(match, f)]
        if len(hits) == 1:
            return hits[0], []
    for part, in_part in _PARTS_OF_DAY.items():
        if re.search(rf"\b{part}\b", lowered):
            hits = [f for f in shown if in_part(_clock(f["departure_time"])[0])]
            if len(hits) == 1:
                return hits[0], []
            if len(hits) > 1:
                return None, hits
    airline_hits = [
        f for f in shown
        if normalize(f["airline"]) in normalize(text)
        or (f.get("airline_code") and re.search(rf"\b{re.escape(f['airline_code'].lower())}\b", lowered))
    ]
    if len(airline_hits) == 1:
        return airline_hits[0], []
    if len(airline_hits) > 1:
        return None, airline_hits
    if len(shown) == 1 and re.search(r"\b(?:it|this|that|this one|that one)\b", lowered):
        return shown[0], []
    return None, []


def _matches_time(match: re.Match[str], flight: dict[str, Any]) -> bool:
    hour, minute = _clock(flight["departure_time"])
    if match.group(3):  # 12-hour clock with am/pm
        wanted_h = int(match.group(1)) % 12 + (12 if match.group(3).lower() == "pm" else 0)
        return (wanted_h, int(match.group(2) or 0)) == (hour, minute)
    return (int(match.group(4)), int(match.group(5))) == (hour, minute)
