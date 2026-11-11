"""Conservative, dependency-free top-level intent/domain classification."""
from __future__ import annotations
import re

HOTEL = re.compile(r"\bhotel|hotels|accommodation|room|stay\b", re.I)
FLIGHT = re.compile(r"\bflight|flights|airfare|airline|plane|fly|ticket\b", re.I)
BOOK = re.compile(r"\bbook|booking|reserve|reservation\b", re.I)
CANCEL = re.compile(r"\bcancel|cancellation\b", re.I)
AVAIL = re.compile(r"\bfind|search|available|availability|check|show\b", re.I)
HELP = re.compile(r"\bhelp|what can you|what services|plan a trip\b", re.I)
SOCIAL = re.compile(
    r"\b(?:hello|hi|hey|good morning|good afternoon|good evening|"
    r"how are you|how's it going|thanks|thank you|"
    r"who are you|what is your name|what's your name|"
    r"may i know your name|can i know your name|could i know your name)\b",
    re.I,
)
UNSUPPORTED = re.compile(r"\btrain|rail|bus|car rental|taxi\b", re.I)

def classify(text: str, pending: list[str] | None = None) -> dict:
    h, f = bool(HOTEL.search(text)), bool(FLIGHT.search(text))
    social = bool(SOCIAL.search(text)) and not any(
        pattern.search(text) for pattern in (BOOK, CANCEL, AVAIL, UNSUPPORTED)
    )
    if social:
        domains = []
    elif not h and not f and pending:
        domains = list(pending)
    elif h and f:
        domains = ["hotel", "flight"]
    elif h:
        domains = ["hotel"]
    elif f:
        domains = ["flight"]
    elif re.search(r"\b(cancel|booking|reservation)\b", text, re.I):
        domains = []
    elif HELP.search(text) or social:
        domains = []
    else:
        domains = []
    if UNSUPPORTED.search(text): op = "UNSUPPORTED"
    elif CANCEL.search(text): op = "CANCEL"
    elif BOOK.search(text): op = "BOOK"
    elif AVAIL.search(text) or h or f: op = "CHECK_AVAILABILITY"
    elif HELP.search(text) or social: op = "GENERAL_QUERY"
    else: op = "UNKNOWN"
    domain = "BOTH" if len(domains) == 2 else (domains[0].upper() if domains else ("GENERAL" if op == "GENERAL_QUERY" else "UNKNOWN"))
    if social and re.search(r"\b(?:who are you|what is your name|what's your name|may i know your name|can i know your name|could i know your name)\b", text, re.I):
        general_topic = "IDENTITY"
    elif social and re.search(r"\b(?:thanks|thank you)\b", text, re.I):
        general_topic = "THANKS"
    elif social and re.search(r"\b(?:how are you|how's it going)\b", text, re.I):
        general_topic = "WELLBEING"
    elif social:
        general_topic = "GREETING"
    elif HELP.search(text):
        general_topic = "HELP"
    else:
        general_topic = "NONE"
    return {"domain": domain, "domains": domains, "operation": op, "general_topic": general_topic}
