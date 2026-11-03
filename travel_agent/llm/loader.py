"""Singleton llama.cpp loader shared by every agent (one model instance per process)."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Protocol

from travel_agent.config import ModelSettings, Settings, get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The local model cannot be loaded (missing file or runtime)."""


class LLM(Protocol):
    """What the agents need from a model: schema-constrained JSON generation."""

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None
    ) -> dict[str, Any]: ...


class LlamaCppLLM:
    """llama.cpp chat model using grammar-constrained JSON output."""

    def __init__(self, settings: ModelSettings) -> None:
        if not settings.model_path.exists():
            raise LLMUnavailable(f"Model file not found: {settings.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMUnavailable("llama-cpp-python is not installed") from exc
        self._settings = settings
        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe
        self._llama = Llama(
            model_path=str(settings.model_path),
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_batch=settings.n_batch,
            use_mmap=settings.use_mmap,
            verbose=settings.verbose,
        )

    def generate_json(
        self, system: str, user: str, schema: dict[str, Any], max_tokens: int | None = None
    ) -> dict[str, Any]:
        with self._lock:
            out = self._llama.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object", "schema": schema},
                temperature=self._settings.routing_temperature,
                max_tokens=max_tokens or self._settings.routing_max_tokens,
            )
        return json.loads(out["choices"][0]["message"]["content"])


_instance: LlamaCppLLM | None = None
_instance_lock = threading.Lock()


def get_llm(settings: Settings | None = None) -> LlamaCppLLM:
    """Load the model once and reuse it for every agent."""
    global _instance
    with _instance_lock:
        if _instance is None:
            settings = settings or get_settings()
            logger.info("Loading %s", settings.model.model_path)
            _instance = LlamaCppLLM(settings.model)
        return _instance
