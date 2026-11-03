"""Normalize the existing public agent results without merging their internal state."""
from __future__ import annotations
from typing import Any

def normalize(result: Any) -> dict:
    status = getattr(result, "status", "UNKNOWN")
    status = getattr(status, "value", status)
    status_map = {"completed": "SUCCESS", "needs_input": "REQUIRES_USER_INPUT", "needs_confirmation": "PENDING", "SUCCESS": "SUCCESS", "FAILED": "FAILED", "PENDING": "PENDING", "REQUIRES_USER_INPUT": "REQUIRES_USER_INPUT", "UNKNOWN": "UNKNOWN", "unverified": "UNKNOWN", "declined": "FAILED"}
    msg = getattr(result, "message", "")
    return {"intent": getattr(getattr(result, "intent", None), "value", None), "status": status_map.get(str(status), "UNKNOWN"),
            "data": getattr(result, "data", {}) or {}, "error": getattr(result, "error", None),
            "requires_user_input": bool(getattr(result, "requires_user_input", False)) or status in ("needs_input", "needs_confirmation", "PENDING", "REQUIRES_USER_INPUT"),
            "missing_parameters": getattr(result, "missing_parameters", getattr(result, "missing_params", [])) or [],
            "message": msg, "question": getattr(result, "question", None)}
