"""Conservative, dependency-free top-level intent/domain classification."""
from __future__ import annotations
import re

HOTEL = re.compile(r"\bhotel|hotels|accommodation|room|stay\b", re.I)
FLIGHT = re.compile(r"\bflight|flights|airfare|airline|plane|fly|ticket\b", re.I)
BOOK = re.compile(r"\bbook|booking|reserve|reservation\b", re.I)
CANCEL = re.compile(r"\bcancel|cancellation\b", re.I)
AVAIL = re.compile(r"\bfind|search|available|availability|check|show\b", re.I)
HELP = re.compile(r"\bhelp|what can you|what services|plan a trip\b", re.I)
UNSUPPORTED = re.compile(r"\btrain|rail|bus|car rental|taxi\b", re.I)

def classify(text: str, pending: list[str] | None = None) -> dict:
    h, f = bool(HOTEL.search(text)), bool(FLIGHT.search(text))
    if not h and not f and pending:
        domains = list(pending)
    elif h and f:
        domains = ["hotel", "flight"]
    elif h:
        domains = ["hotel"]
    elif f:
        domains = ["flight"]
    elif re.search(r"\b(cancel|booking|reservation)\b", text, re.I):
        domains = []
    elif HELP.search(text):
        domains = []
    else:
        domains = []
    if UNSUPPORTED.search(text): op = "UNSUPPORTED"
    elif CANCEL.search(text): op = "CANCEL"
    elif BOOK.search(text): op = "BOOK"
    elif AVAIL.search(text) or h or f: op = "CHECK_AVAILABILITY"
    elif HELP.search(text): op = "GENERAL_QUERY"
    else: op = "UNKNOWN"
    domain = "BOTH" if len(domains) == 2 else (domains[0].upper() if domains else ("GENERAL" if op == "GENERAL_QUERY" else "UNKNOWN"))
    return {"domain": domain, "domains": domains, "operation": op}
