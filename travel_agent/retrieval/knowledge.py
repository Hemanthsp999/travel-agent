"""FAISS-backed retrieval for FAQs and policies attached to mock travel entities."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from mock_apis.flight_api.config import load_settings as load_flight_settings
from mock_apis.hotel_api.config import load_settings as load_hotel_settings

logger = logging.getLogger(__name__)

_DIMENSIONS = 4096
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "tell", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}
_POLICY_TERMS = re.compile(
    r"\b(?:faq|polic(?:y|ies)|rules?|refunds?|refundable|cancell?ation fee|"
    r"cancell?ation policy|baggage|luggage|bags?|carry[\s-]?on|check[\s-]?in|"
    r"check[\s-]?out|amenities|pets?|smoking|meal|seat selection)\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^\s*(?:what|how|when|where|why|which|who|can i|could i|do i|does|is|are|"
    r"tell me about|explain)\b",
    re.IGNORECASE,
)
_OPERATIONAL_LOOKUP = re.compile(
    r"\b(?:price|cost|how much|availability|available|search|find|show|status|"
    r"schedule|booking id|booking reference|reservation status|seats?)\b",
    re.IGNORECASE,
)
_DIRECT_ACTION = re.compile(
    r"\b(?:book|reserve|find|search|check availability|look up|cancel)\s+"
    r"(?:my|this|the|a|an)\s+(?:hotel|flight|booking|reservation)\b",
    re.IGNORECASE,
)
_PROCEDURAL_CANCELLATION = re.compile(
    r"\bhow\s+(?:do i|can i|to)\s+cancel\b",
    re.IGNORECASE,
)
_DOMAIN_TERMS = {
    "hotel": re.compile(r"\b(?:hotel|hotels|room|rooms|stay|property|accommodation)\b", re.I),
    "flight": re.compile(r"\b(?:flight|flights|airline|airlines|airfare|baggage|luggage|fare)\b", re.I),
}
_POLICY_KIND = re.compile(
    r"\b(?:cancel|cancell?ation|refund|refundable|fee)\b", re.IGNORECASE
)
_ELIGIBILITY_QUESTION = re.compile(
    r"\b(?:eligible|eligibility|qualify|qualified|entitled|can i get|will i get|"
    r"do i get|can i receive|will i receive)\b",
    re.IGNORECASE,
)
_RULE_KIND = re.compile(
    r"\b(?:rule|rules|baggage|luggage|bag|carry[\s-]?on|check[\s-]?in|"
    r"check[\s-]?out|amenit|pet|smoking|meal|seat)\b",
    re.IGNORECASE,
)
_ALIASES = {
    "cancellations": "cancel",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "cancelling": "cancel",
    "policies": "policy",
    "rules": "rule",
    "refunds": "refund",
    "refundable": "refund",
    "refunded": "refund",
    "fees": "fee",
    "amenities": "amenity",
    "hotels": "hotel",
    "rooms": "room",
    "flights": "flight",
    "airlines": "airline",
    "bags": "bag",
    "baggage": "bag",
    "luggage": "bag",
    "checkout": "check out",
    "checkin": "check in",
}
_SYNONYMS = {
    "cancel": ("cancellation", "terminate", "call off"),
    "refund": ("reimbursement", "money back", "return"),
    "bag": ("baggage", "luggage", "carry on", "checked bag"),
    "hotel": ("accommodation", "property", "stay"),
    "flight": ("airline", "airfare", "plane", "ticket"),
    "rule": ("policy", "restriction", "allowance", "terms"),
    "check in": ("arrival", "registration"),
    "check out": ("departure", "checkout"),
}

_HOTEL_CANCELLATION: dict[str, tuple[int, str]] = {
    "HT-1001": (48, "one night's room rate"),
    "HT-1002": (24, "50% of the first night's room rate"),
    "HT-1003": (48, "one night's room rate"),
    "HT-1004": (72, "one night's room rate"),
    "HT-1005": (24, "50% of the first night's room rate"),
    "HT-1006": (48, "one night's room rate"),
    "HT-1007": (48, "one night's room rate"),
    "HT-1008": (24, "one night's room rate"),
    "HT-1009": (72, "one night's room rate"),
    "HT-1010": (24, "one night's room rate"),
    "HT-1011": (48, "one night's room rate"),
    "HT-1012": (24, "one night's room rate"),
    "HT-1013": (48, "one night's room rate"),
    "HT-1014": (24, "one night's room rate"),
}
_FLIGHT_CANCELLATION: dict[str, int] = {
    "FL-1001": 24,
    "FL-1002": 24,
    "FL-1003": 12,
    "FL-1004": 24,
    "FL-1005": 24,
    "FL-1006": 12,
    "FL-1007": 24,
    "FL-1008": 24,
    "FL-1009": 12,
    "FL-1010": 24,
    "FL-1011": 12,
    "FL-1012": 24,
    "FL-1013": 24,
    "FL-1014": 24,
    "FL-1015": 24,
    "FL-1016": 24,
    "FL-1017": 12,
    "FL-1018": 24,
}
_AIRLINE_BAGGAGE = {
    "IndiGo": "7 kg cabin baggage; checked baggage allowance is fare-dependent.",
    "Air India": "7 kg cabin baggage; checked baggage allowance is fare-dependent.",
    "Akasa Air": "7 kg cabin baggage; checked baggage allowance is fare-dependent.",
    "SpiceJet": "7 kg cabin baggage; checked baggage allowance is fare-dependent.",
}


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    text: str
    domain: str
    document_type: str
    source: str
    entity_id: str | None = None
    entity_name: str | None = None
    entity_aliases: tuple[str, ...] = ()


def is_knowledge_query(query: str) -> bool:
    """Distinguish policy/FAQ questions from requests to perform a booking operation."""
    if _PROCEDURAL_CANCELLATION.search(query):
        return True
    if _DIRECT_ACTION.search(query):
        return False
    if _POLICY_TERMS.search(query):
        return True
    if not _QUESTION_START.search(query) or _OPERATIONAL_LOOKUP.search(query):
        return False
    return not any(pattern.search(query) for pattern in _DOMAIN_TERMS.values())


def _features(text: str) -> dict[str, float]:
    tokens = [
        _ALIASES.get(token, token)
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS
    ]
    if re.search(r"\bwhat can you do\b", text, re.IGNORECASE):
        tokens.extend(("service", "help", "capability"))
    features: dict[str, float] = {}
    for token in tokens:
        features[token] = features.get(token, 0.0) + 1.0
        for synonym in _SYNONYMS.get(token, ()):
            features[synonym] = features.get(synonym, 0.0) + 0.35
    for left, right in zip(tokens, tokens[1:]):
        feature = f"{left} {right}"
        features[feature] = features.get(feature, 0.0) + 0.5
    return features


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(_DIMENSIONS, dtype=np.float32)
    for feature, weight in _features(text).items():
        slot = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little") % _DIMENSIONS
        vector[slot] += weight
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def _format_knowledge_sections(matches: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for match in matches:
        domain = match["domain"]
        document_type = match["document_type"]
        entity_name = match.get("entity_name")
        entity_id = match.get("entity_id")
        if entity_name:
            subject = f"{entity_name} ({entity_id})"
        elif domain == "hotel":
            subject = "Hotel"
        elif domain == "flight":
            subject = "Flight"
        else:
            subject = "Travel"

        if document_type == "cancellation":
            heading = f"{subject} cancellation policy"
        elif document_type == "rules":
            heading = f"{subject} rules"
        else:
            heading = "Travel FAQ"

        text = match["text"].strip()
        if document_type in {"cancellation", "rules"} and ": " in text:
            text = text.split(": ", 1)[1]
        points = [
            point.strip().rstrip(".")
            for point in re.split(r"(?<=[.!?])\s+|;\s+", text)
            if point.strip()
        ]
        points = [point[0].upper() + point[1:] for point in points]
        if points:
            sections.append(f"**{heading}**\n" + "\n".join(f"- {point}." for point in points))

    return "\n\n".join(sections)


def _format_knowledge_answer(matches: list[dict[str, Any]]) -> str:
    return f"Here’s what I found:\n\n{_format_knowledge_sections(matches)}"


def _eligibility_guidance(
    query: str, matches: list[dict[str, Any]], prior_knowledge: dict[str, Any] | None
) -> str | None:
    if not _ELIGIBILITY_QUESTION.search(query):
        return None

    domains = {match["domain"] for match in matches if match["domain"] in {"hotel", "flight"}}
    if prior_knowledge:
        domains.update(
            match.get("domain")
            for match in prior_knowledge.get("matches", [])
            if match.get("domain") in {"hotel", "flight"}
        )
    entities = [match for match in matches if match.get("entity_id")]

    if len(entities) == 1:
        entity = entities[0]
        if entity["domain"] == "hotel":
            question = "What are your check-in date and room rate type (standard or flexible)?"
        else:
            question = "What is your departure date, and was the fare refundable?"
        subject = f" for {entity.get('entity_name') or entity['entity_id']}"
        return (
            f"I can help check that, but I can’t confirm refund eligibility without the booking’s "
            f"rate and cancellation timing. Here’s the policy{subject}:\n\n"
            f"{_format_knowledge_sections(matches)}\n\n{question}"
        )

    if domains == {"hotel"}:
        question = "Which hotel is the booking for, and was the rate standard or flexible?"
    elif domains == {"flight"}:
        question = "Which flight is the booking for, and was the fare refundable?"
    else:
        question = "Is this for a hotel or a flight, and what is the booking reference?"
    return (
        "I can help check, but I can’t determine refund eligibility from the general policy alone. "
        "Eligibility depends on the specific booking’s rate or fare and when it is cancelled. "
        f"{question}"
    )


def _read_rows(path: Path, table: str) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.warning("Could not read mock %s database at %s: %s", table, path, exc)
        return None


def _hotel_rows() -> list[dict[str, Any]]:
    path = load_hotel_settings().db_path
    rows = _read_rows(path, "hotels")
    if rows:
        for row in rows:
            try:
                row["amenities"] = json.loads(row["amenities"])
            except (TypeError, json.JSONDecodeError):
                row["amenities"] = []
        return rows

    from mock_apis.hotel_api.seed import HOTELS

    logger.info("Using hotel mock seed data to build the knowledge index.")
    return [
        {
            "hotel_id": hotel_id,
            "name": name,
            "city": city,
            "max_guests_per_room": capacity,
            "amenities": amenities,
        }
        for hotel_id, name, city, _locality, _address, _rating, _stars, _price, _rooms, capacity, amenities in HOTELS
    ]


def _flight_rows() -> list[dict[str, Any]]:
    path = load_flight_settings().db_path
    rows = _read_rows(path, "flights")
    if rows:
        return rows

    from mock_apis.flight_api.seed import FLIGHTS

    logger.info("Using flight mock seed data to build the knowledge index.")
    return [
        {
            "flight_id": flight_id,
            "airline": airline,
            "flight_number": flight_number,
            "origin_city": origin,
            "destination_city": destination,
            "departure_time": departure_time,
            "aircraft": aircraft,
        }
        for flight_id, airline, _code, flight_number, origin, destination, departure_time, _duration, _fare, _seats, aircraft in FLIGHTS
    ]


def _build_documents() -> list[KnowledgeDocument]:
    documents = [
        KnowledgeDocument(
            "faq:services",
            "Travel assistant FAQ and services. I can search the mock inventory and help book or look up "
            "hotel and flight reservations. Availability, prices, and booking status are checked "
            "through the hotel or flight agent, not guessed from policy documents.",
            "general",
            "faq",
            "faq:services",
        ),
        KnowledgeDocument(
            "faq:cancellation-process",
            "Travel assistant FAQ about cancellation: ask the assistant to cancel a hotel or flight "
            "booking and provide its booking reference. The hotel or flight agent checks the live "
            "booking and asks for confirmation before submitting a cancellation. A policy answer "
            "does not cancel a reservation or guarantee a refund.",
            "general",
            "faq",
            "faq:cancellation-process",
        ),
        KnowledgeDocument(
            "faq:availability",
            "Travel assistant FAQ about availability and prices: hotel room availability, flight "
            "seats, and current prices are checked through a live mock API search. A policy or "
            "previous search result does not guarantee current availability.",
            "general",
            "faq",
            "faq:availability",
        ),
        KnowledgeDocument(
            "policy:hotel:overview",
            "General mock hotel cancellation policy: standard rooms can be cancelled up to 48 hours "
            "before check-in without penalty. Flexible rates are refundable until check-in. "
            "Individual hotel and rate terms may differ; check the selected booking's terms.",
            "hotel",
            "cancellation",
            "mock-policy:hotel-overview",
        ),
        KnowledgeDocument(
            "policy:flight:overview",
            "General mock flight cancellation policy: refundable fares can be cancelled up to "
            "24 hours before departure. Non-refundable fares are not refunded unless the airline "
            "cancels the flight. The exact fare rules can differ by flight and fare type.",
            "flight",
            "cancellation",
            "mock-policy:flight-overview",
        ),
        KnowledgeDocument(
            "rules:hotel:overview",
            "General mock hotel rules: standard check-in starts at 2:00 PM and check-out is by "
            "11:00 AM. Guest limits, amenities, and property rules vary by hotel; confirm them "
            "with the selected property. Pet and special-request rules must be confirmed with "
            "the hotel.",
            "hotel",
            "rules",
            "mock-policy:hotel-overview",
        ),
        KnowledgeDocument(
            "rules:flight:overview",
            "General mock flight rules: baggage allowance, check-in deadlines, and seat conditions "
            "depend on the airline and fare. Confirm the rules for the selected flight and ticket.",
            "flight",
            "rules",
            "mock-policy:flight-overview",
        ),
    ]

    for row in _hotel_rows():
        hotel_id = str(row["hotel_id"])
        name = str(row["name"])
        city = str(row["city"])
        amenities = row.get("amenities") or []
        if isinstance(amenities, str):
            amenities = [amenities]
        amenity_text = ", ".join(str(item) for item in amenities) or "not listed"
        capacity = row.get("max_guests_per_room", "not specified")
        aliases = (hotel_id, name)
        documents.append(
            KnowledgeDocument(
                f"rules:hotel:{hotel_id}",
                f"Mock rules for {name} ({hotel_id}) in {city}: standard check-in is 2:00 PM;"
                f" check-out is 11:00 AM; maximum guests per room: {capacity}. Listed "
                f"amenities: {amenity_text}. Other property-specific requests must be confirmed "
                "with the hotel. These are illustrative mock rules.",
                "hotel",
                "rules",
                f"mock-hotel:{hotel_id}",
                hotel_id,
                name,
                aliases,
            )
        )
        hours, late_charge = _HOTEL_CANCELLATION.get(hotel_id, (48, "one night's room rate"))
        documents.append(
            KnowledgeDocument(
                f"cancellation:hotel:{hotel_id}",
                f"Mock cancellation policy for {name} ({hotel_id}): cancel at least {hours} hours "
                f"before check-in for no cancellation charge; later cancellations may incur "
                f"{late_charge}. The selected rate's booking terms take precedence. This is an "
                "illustrative mock policy.",
                "hotel",
                "cancellation",
                f"mock-hotel:{hotel_id}",
                hotel_id,
                name,
                aliases,
            )
        )

    for row in _flight_rows():
        flight_id = str(row["flight_id"])
        airline = str(row["airline"])
        number = str(row["flight_number"])
        origin = str(row["origin_city"])
        destination = str(row["destination_city"])
        aliases = (flight_id, number, airline, f"{airline} {number}")
        baggage = _AIRLINE_BAGGAGE.get(
            airline, "Baggage allowance is fare-dependent; confirm it on the ticket."
        )
        documents.append(
            KnowledgeDocument(
                f"rules:flight:{flight_id}",
                f"Mock rules for flight {number} ({flight_id}), {airline}, from {origin} to "
                f"{destination} at {row.get('departure_time', 'the listed departure time')}: "
                f"{baggage} Aircraft: {row.get('aircraft', 'not listed')}. Check-in and seat "
                "conditions depend on the fare and must be confirmed on the ticket. These are "
                "illustrative mock rules.",
                "flight",
                "rules",
                f"mock-flight:{flight_id}",
                flight_id,
                number,
                aliases,
            )
        )
        hours = _FLIGHT_CANCELLATION.get(flight_id, 24)
        documents.append(
            KnowledgeDocument(
                f"cancellation:flight:{flight_id}",
                f"Mock cancellation policy for flight {number} ({flight_id}), {airline}, "
                f"{origin} to {destination}: refundable fares can be cancelled at least {hours} "
                "hours before departure without a cancellation charge. Non-refundable fares are "
                "not refunded unless the airline cancels the flight. The fare's ticket terms "
                "take precedence. This is an illustrative mock policy.",
                "flight",
                "cancellation",
                f"mock-flight:{flight_id}",
                flight_id,
                number,
                aliases,
            )
        )
    return documents


class KnowledgeRetriever:
    """A small in-memory FAISS index shared by the supervisor and both subagents."""

    def __init__(self, documents: list[KnowledgeDocument] | None = None) -> None:
        self.documents = documents if documents is not None else _build_documents()
        self._index = faiss.IndexFlatIP(_DIMENSIONS)
        if self.documents:
            vectors = np.stack([_embed(doc.text) for doc in self.documents])
            self._index.add(vectors)

    def _matching_entities(self, query: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
        found: set[str] = set()
        entities = {
            (doc.entity_id, doc.entity_aliases)
            for doc in self.documents
            if doc.entity_id is not None
        }
        for entity_id, aliases in entities:
            for alias in aliases:
                normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
                if normalized_alias and re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
                    normalized,
                ):
                    found.add(entity_id)
                    break
        return found

    def _search(
        self,
        query: str,
        *,
        domains: set[str] | None = None,
        entity_ids: set[str] | None = None,
        generic_only: bool = False,
        document_type: str | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        if not self.documents:
            return []
        scores, indexes = self._index.search(_embed(query).reshape(1, -1), len(self.documents))
        matches = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0 or float(score) < 0.08:
                continue
            doc = self.documents[int(index)]
            if domains and doc.domain not in domains and doc.domain != "general":
                continue
            if entity_ids and doc.entity_id not in entity_ids:
                continue
            if generic_only and doc.entity_id is not None:
                continue
            if document_type and doc.document_type != document_type:
                continue
            matches.append(
                {
                    "document_id": doc.document_id,
                    "text": doc.text,
                    "domain": doc.domain,
                    "document_type": doc.document_type,
                    "source": doc.source,
                    "entity_id": doc.entity_id,
                    "entity_name": doc.entity_name,
                    "score": float(score),
                }
            )
        matches.sort(key=lambda result: result["score"], reverse=True)

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in matches:
            grouping = match["entity_id"] or match["domain"]
            if grouping in seen:
                continue
            selected.append(match)
            seen.add(grouping)
            if len(selected) == limit:
                break
        return selected

    def retrieve(
        self, query: str, prior_knowledge: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return ranked context and a grounded answer candidate for the supervisor."""
        entity_ids = self._matching_entities(query)
        domains = {domain for domain, pattern in _DOMAIN_TERMS.items() if pattern.search(query)}
        knowledge_query = is_knowledge_query(query)
        search_query = query
        if not entity_ids and prior_knowledge:
            previous_matches = prior_knowledge.get("matches", [])
            if not domains or any(
                match.get("domain") in domains for match in previous_matches
            ):
                entity_ids = {
                    match["entity_id"]
                    for match in previous_matches
                    if match.get("entity_id")
                    and (not domains or match.get("domain") in domains)
                }
                if entity_ids:
                    search_query = " ".join(
                        part
                        for part in (query, prior_knowledge.get("query", ""))
                        if part
                    )
            if not domains:
                domains.update(
                    match["domain"]
                    for match in previous_matches
                    if match.get("domain") in {"hotel", "flight"}
                )
        if _OPERATIONAL_LOOKUP.search(query) and not knowledge_query:
            matches = []
        elif entity_ids:
            document_type = (
                "cancellation" if _POLICY_KIND.search(query)
                else "rules" if _RULE_KIND.search(query)
                else None
            )
            matches = self._search(
                search_query,
                entity_ids=entity_ids,
                domains=domains or None,
                document_type=document_type,
                limit=4,
            )
        elif knowledge_query:
            if not domains and _POLICY_KIND.search(query):
                domains = {"hotel", "flight"}
            matches = self._search(
                search_query,
                domains=domains or None,
                generic_only=True,
                limit=4,
            )
        else:
            matches = self._search(
                search_query,
                domains=domains or None,
                generic_only=True,
                limit=3,
            )

        context = "\n\n".join(match["text"] for match in matches)
        sources = list(dict.fromkeys(match["source"] for match in matches))
        matched_domains = list(dict.fromkeys(match["domain"] for match in matches))
        answer = None
        if knowledge_query and matches:
            answer = _eligibility_guidance(query, matches, prior_knowledge) or _format_knowledge_answer(matches)
        return {
            "domain": matched_domains[0] if len(matched_domains) == 1 else "multiple" if matched_domains else None,
            "context": context,
            "source_refs": sources,
            "confidence": max((match["score"] for match in matches), default=0.0),
            "matches": matches,
            "answer": answer,
            "is_knowledge_query": knowledge_query,
            "query": query,
            "needs_booking_details": bool(
                knowledge_query and _ELIGIBILITY_QUESTION.search(query)
            ),
        }


@lru_cache(maxsize=1)
def get_knowledge_retriever() -> KnowledgeRetriever:
    """Build the shared index once per process."""
    return KnowledgeRetriever()
