"""Schema-constrained model calls that never raise: failures return None so callers can fall back."""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from travel_agent.llm.loader import LLM

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def generate_structured(llm: LLM | None, system: str, user: str, model: type[T]) -> T | None:
    """Ask the model for JSON matching `model`; return None on any failure or invalid output."""
    if llm is None:
        return None
    try:
        raw = llm.generate_json(system, user, model.model_json_schema())
        return model.model_validate(raw)
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        logger.warning("LLM output rejected: %s", exc)
    except Exception as exc:  # model runtime errors must not crash the conversation
        logger.warning("LLM call failed: %s", exc)
    return None
