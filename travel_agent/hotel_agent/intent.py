"""Intent classification (LLM with rule fallback), grounding of LLM text, and hotel references."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from travel_agent.hotel_agent.schemas import GeneralTopic, Intent, IntentClassification
from travel_agent.llm.loader import LLM
from travel_agent.llm.structured import generate_structured
from travel_agent.prompts import load_prompt

logger = logging.getLogger(__name__)

CITY_ALIASES = {
    "bengaluru": "Bangalore", "bangalore": "Bangalore", "bombay": "Mumbai", "mumbai": "Mumbai",
    "delhi": "Delhi", "new delhi": "Delhi", "hyderabad": "Hyderabad", "chennai": "Chennai",
    "madras": "Chennai", "pune": "Pune",
}
_NOT_A_CITY = {
    "the", "a", "an", "my", "your", "this", "that", "next", "hotel", "hotels", "room", "rooms", "night",
    "nights", "days", "advance", "total", "general", "case", "mind", "stay", "person", "people", "it",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "tonight", "weekend", "budget", "price", "rating", "me", "us", "one",
}

_CITY_STOP = {"from", "for", "on", "to", "between", "with", "under", "below", "within", "and", "check", "checking"}

_CANCEL_RE = re.compile(r"\bcancel(?:l?ation|l?ed|l?ing)?\b|\bcall off\b", re.I)
_BOOK_RE = re.compile(
    r"\b(?:book(?:ing)?|reserve|reservation|i'?ll take|go with|take (?:it|this|that|the)|stay at|proceed|"
    r"choose|select|pick|want (?:this|that|the|it)|this one|that one|the (?:first|second|third|last) one)\b", re.I)
_STATUS_RE = re.compile(
    r"\b(?:status|booking details|check (?:on )?(?:my|the) (?:booking|reservation)|my (?:booking|reservation)|"
    r"is (?:my|the) (?:booking|reservation))\b", re.I)
_UNSUPPORTED_RE = re.compile(
    r"\b(?:refund|polic(?:y|ies)|reviews?|photos?|pictures?|payments?|pay|modify|modification|amend|reschedule|"
    r"extend|upgrade|flights?|loyalty|points|coupon|discount|check-?in time|check-?out time|pets?|invoice)\b|"
    r"\bchange (?:my|the|our) (?:booking|reservation|dates?)\b", re.I)
_DETAILS_RE = re.compile(
    r"\b(?:details?|tell me (?:more )?about|amenit(?:y|ies)|facilit(?:y|ies)|address|does (?:it|the \w+|this)\b.*\bhave|"
    r"what(?:'s| is| are) (?:the )?(?:price|rating|location)|more (?:info|information) (?:on|about))\b", re.I)
_COMPARE_RE = re.compile(r"\b(?:compare|versus|vs\.?|difference between|which (?:one )?is better)\b", re.I)
_SHOW_MORE_RE = re.compile(
    r"\b(?:more options|show (?:me )?(?:all|more)|other options|see all|all (?:the )?(?:hotels|options)|anything else)\b", re.I)
_HELP_RE = re.compile(r"^\s*(?:hi|hello|hey|help|good (?:morning|afternoon|evening))\b|\bwhat can you (?:do|help)\b", re.I)
_SEARCH_RE = re.compile(
    r"\b(?:find|show|search|look(?:ing)? for|available|availability|rooms?|hotels?|stay|options|vacanc(?:y|ies)|accommodation)\b", re.I)

ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}
_GENERIC_NAME_TOKENS = {
    "hotel", "hotels", "inn", "stay", "suites", "suite", "residency", "palace", "towers", "tower", "grand",
    "express", "comfort", "central", "business", "urban", "boutique", "heritage", "city", "bay", "the", "and",
}


@dataclass
class Classification:
    intent: Intent = Intent.UNKNOWN
    topic: GeneralTopic = GeneralTopic.NONE
    confidence: float = 0.0
    city: str | None = None
    hotel_name: str | None = None
    guest_name: str | None = None
    source: str = "rules"


@dataclass
class ClassifyContext:
    awaiting: str | None = None
    current_intent: str | None = None
    known_hotels: list[str] = field(default_factory=list)
    recent: str = ""


# ------------------------------------------------------------------------------ helpers


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def grounded(value: str | None, text: str) -> bool:
    """True if `value` literally appears in the user's text (guards against LLM invention)."""
    return bool(value) and normalize(value) in normalize(text) and len(normalize(value)) >= 2  # type: ignore[arg-type]


def normalize_city(raw: str) -> str:
    key = normalize(raw)
    return CITY_ALIASES.get(key, " ".join(w.capitalize() for w in key.split()))


def rule_city(text: str, known_only: bool = False) -> str | None:
    """Find a city: known names first, then (unless `known_only`) an 'in/at <word(s)>' phrase."""
    norm = normalize(text)
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{alias}\b", norm):
            return CITY_ALIASES[alias]
    if known_only:
        return None
    match = re.search(
        r"\b(?:in|at|near|around)\s+([A-Za-z][A-Za-z]*(?: [A-Za-z]+)?)(?=\s+(?:from|for|on|between|with|under|below|within|"
        r"next|this|tomorrow|today|and|check|checking)\b|[,.?!]|\s*\d|$)",
        text,
        re.I,
    )
    if match:
        words = match.group(1).split()
        while words and words[-1].lower() in _NOT_A_CITY | _CITY_STOP:
            words.pop()
        if words and words[0].lower() not in _NOT_A_CITY:
            return normalize_city(" ".join(words))
    return None


# -------------------------------------------------------------------------- classification


def rule_classify(text: str) -> Classification:
    """Keyword-based fallback. Conservative: returns UNKNOWN unless the wording is clear."""
    has_id = bool(re.search(r"\bHB-[A-Z0-9]{4,}\b", text, re.I))
    if _UNSUPPORTED_RE.search(text) and not (has_id and _CANCEL_RE.search(text)):
        return Classification(Intent.GENERAL_HOTEL_QUERY, GeneralTopic.UNSUPPORTED, 0.7)
    if _CANCEL_RE.search(text):
        return Classification(Intent.CANCEL_BOOKING, confidence=0.85)
    if _STATUS_RE.search(text) or (has_id and not _BOOK_RE.search(text)):
        return Classification(Intent.GET_BOOKING_STATUS, confidence=0.8)
    if _COMPARE_RE.search(text):
        return Classification(Intent.GENERAL_HOTEL_QUERY, GeneralTopic.COMPARE_HOTELS, 0.8)
    if _SHOW_MORE_RE.search(text):
        return Classification(Intent.GENERAL_HOTEL_QUERY, GeneralTopic.SHOW_MORE, 0.8)
    if _DETAILS_RE.search(text):
        return Classification(Intent.GENERAL_HOTEL_QUERY, GeneralTopic.HOTEL_DETAILS, 0.7)
    if _BOOK_RE.search(text):
        return Classification(Intent.BOOK_HOTEL, confidence=0.75)
    if _HELP_RE.search(text):
        return Classification(Intent.GENERAL_HOTEL_QUERY, GeneralTopic.HELP, 0.8)
    if _SEARCH_RE.search(text):
        return Classification(Intent.CHECK_AVAILABILITY, confidence=0.65)
    return Classification()


def _plausible(intent: Intent, text: str) -> bool:
    """Reject LLM intents that have no support in the wording (cheap guard against misroutes)."""
    if intent is Intent.CANCEL_BOOKING:
        return bool(_CANCEL_RE.search(text))
    if intent is Intent.BOOK_HOTEL:
        return bool(_BOOK_RE.search(text))
    if intent is Intent.GET_BOOKING_STATUS:
        return bool(re.search(r"\bHB-[A-Z0-9]{4,}\b|\b(?:status|booking|reservation)\b", text, re.I))
    return True


def _llm_classify(llm: LLM | None, text: str, ctx: ClassifyContext, threshold: float) -> Classification | None:
    user = (
        f"message: {text!r}\nawaiting: {ctx.awaiting or 'none'}\ncurrent_intent: {ctx.current_intent or 'none'}\n"
        f"known_hotels: {', '.join(ctx.known_hotels) or 'none'}\nrecent:\n{ctx.recent or 'none'}"
    )
    parsed = generate_structured(llm, load_prompt("hotel_agent"), user, IntentClassification)
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
        city=parsed.city if grounded(parsed.city, text) else None,
        hotel_name=parsed.hotel_name if grounded(parsed.hotel_name, text) else None,
        guest_name=parsed.guest_name if grounded(parsed.guest_name, text) else None,
        source="llm",
    )


def _rules_override(rules: Classification, llm_result: Classification, text: str) -> bool:
    """Explicit keywords for follow-up questions beat a vaguer LLM label.

    The 3B model tends to label 'tell me about the cheapest one' as a new search. A keyword hit for
    details / compare / show-more / unsupported wins unless the user clearly named a city to search.
    """
    if rules.intent is Intent.BOOK_HOTEL and llm_result.intent in (Intent.CHECK_AVAILABILITY, Intent.GENERAL_HOTEL_QUERY):
        return rule_city(text, known_only=True) is None  # 'book the second one' is not a new search
    if rules.intent is not Intent.GENERAL_HOTEL_QUERY or rules.topic in (GeneralTopic.NONE, GeneralTopic.HELP):
        return False
    if llm_result.intent is Intent.GENERAL_HOTEL_QUERY:
        return llm_result.topic is not rules.topic
    if llm_result.intent is Intent.CHECK_AVAILABILITY:
        return rules.topic is not GeneralTopic.UNSUPPORTED and rule_city(text, known_only=True) is None
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
    if result.intent is not Intent.GENERAL_HOTEL_QUERY:
        result.topic = GeneralTopic.NONE  # the small model sets topics inconsistently; only GENERAL uses one
    if result.city is None and result.intent in {Intent.CHECK_AVAILABILITY, Intent.BOOK_HOTEL}:
        result.city = rule_city(text, known_only=result.intent is Intent.BOOK_HOTEL)
    if result.city:
        result.city = normalize_city(result.city)
    return result


# ----------------------------------------------------------------------- hotel references


def _distinctive_tokens(name: str) -> set[str]:
    skip = _GENERIC_NAME_TOKENS | {w for alias in CITY_ALIASES for w in alias.split()}
    return {t for t in normalize(name).split() if t not in skip and len(t) > 2}


def find_hotels_by_name(query: str, hotels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hotels whose name matches `query` fully or by a distinctive word. Several hits mean ambiguity."""
    norm = normalize(query)
    if not norm:
        return []
    exact = [
        h for h in hotels
        if normalize(h["name"]) in norm or (len(norm) >= 5 and norm in normalize(h["name"]))
    ]
    if exact:
        return exact
    words = set(norm.split())
    return [h for h in hotels if _distinctive_tokens(h["name"]) & words]


def resolve_hotel(
    text: str, shown: list[dict[str, Any]], allow_bare_number: bool = False
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve 'the second one', 'cheapest', 'HT-1002' or a hotel name against hotels the user was shown.

    Returns (hotel, ambiguous_candidates). Both empty means no reference was recognised.
    A bare digit ('2') counts as an option number only when `allow_bare_number` is set.
    """
    if not shown:
        return None, []
    if match := re.search(r"\bHT-[A-Z0-9]{2,}\b", text, re.I):
        wanted = match.group().upper()
        hit = next((h for h in shown if h["hotel_id"].upper() == wanted), None)
        return (hit, []) if hit else (None, [])
    lowered = text.lower()
    for word, index in ORDINALS.items():
        if re.search(rf"\b{word}\b", lowered):
            return (shown[index - 1], []) if index <= len(shown) else (None, [])
    if match := re.search(r"(?:option|number|no\.?|#)\s*(\d)\b", lowered) or (
        allow_bare_number and re.fullmatch(r"\s*(\d)\s*[.!]?\s*", lowered)
    ):
        index = int(match.group(1))
        return (shown[index - 1], []) if 1 <= index <= len(shown) else (None, [])
    if re.search(r"\blast one\b|\bthe last\b", lowered):
        return shown[-1], []
    priced = [h for h in shown if h.get("price_per_night") is not None]
    if re.search(r"\b(?:cheapest|lowest price|least expensive)\b", lowered) and priced:
        return min(priced, key=lambda h: h["price_per_night"]), []
    rated = [h for h in shown if h.get("rating") is not None]
    if re.search(r"\b(?:best rated|highest rated|top rated|best one)\b", lowered) and rated:
        return max(rated, key=lambda h: h["rating"]), []
    hits = find_hotels_by_name(text, shown)
    if len(hits) == 1:
        return hits[0], []
    if len(hits) > 1:
        return None, hits
    if len(shown) == 1 and re.search(r"\b(?:it|this|that|this one|that one)\b", lowered):
        return shown[0], []
    return None, []
