"""Short-term memory helpers: bound the message history, never summarise with an LLM.

Facts that matter (parameters, last search, known bookings) live in structured state, so old
messages can be dropped safely and nothing needs a lossy, model-written summary.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage


def prune_messages(messages: list[AnyMessage], keep: int) -> list[RemoveMessage]:
    """Remove all but the newest `keep` messages (no-op when the history is short)."""
    if len(messages) <= keep:
        return []
    return [RemoveMessage(id=m.id) for m in messages[:-keep] if m.id]


def last_user_text(messages: list[AnyMessage]) -> str:
    """Text of the newest human message, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def recent_dialogue(messages: list[AnyMessage], limit: int = 4) -> str:
    """Compact `role: text` transcript of the last few messages for LLM context."""
    lines = []
    for message in messages[-limit:]:
        role = "user" if isinstance(message, HumanMessage) else "agent"
        lines.append(f"{role}: {str(message.content)[:300]}")
    return "\n".join(lines)
