"""`FlightAgent`: the public entry point (used by the CLI now, by an orchestrator later)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Iterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from travel_agent.config import Settings, get_settings
from travel_agent.flight_agent.api_client import FlightAPIClient
from travel_agent.flight_agent.graph import build_flight_graph
from travel_agent.flight_agent.nodes.deps import AgentDeps
from travel_agent.flight_agent.schemas import ExecutionStatus, FlightTask, FlightTaskResult, Intent
from travel_agent.flight_agent.tools import build_flight_tools
from travel_agent.llm.loader import LLM, LLMUnavailable, get_llm
from travel_agent.memory.checkpointer import create_checkpointer

logger = logging.getLogger(__name__)

FALLBACK_ERROR = "Sorry, something went wrong on my side. I can't tell what was completed, so please check before retrying."


class FlightAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        client: FlightAPIClient | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        today: Callable[[], date] = date.today,
        new_key: Callable[[], str] | None = None,
        save_artifacts: bool = True,
        load_model: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        if llm is None and load_model and self.settings.agent.use_llm:
            llm = self._try_load_model()
        client = client or FlightAPIClient(self.settings.flight_api)
        extra = {"new_key": new_key} if new_key else {}
        self._deps = AgentDeps(
            settings=self.settings, tools=build_flight_tools(client, self.settings.flight_api), llm=llm, today=today, **extra
        )
        self.graph = build_flight_graph(
            self._deps, checkpointer or create_checkpointer(None), save_artifacts=save_artifacts
        )

    def _try_load_model(self) -> LLM | None:
        try:
            return get_llm(self.settings)
        except LLMUnavailable as exc:
            logger.warning("Running with rule-based understanding only: %s", exc)
            return None

    # ------------------------------------------------------------------------ public API

    def run_task(self, task: FlightTask) -> FlightTaskResult:
        """Run one structured task. If the thread is waiting for an answer, `message` is that answer."""
        return self._run(task)

    def chat(self, message: str, thread_id: str = "default") -> FlightTaskResult:
        return self._run(FlightTask(message=message, thread_id=thread_id))

    def stream_chat(self, message: str, thread_id: str = "default") -> Iterator[str]:
        """Yield node names as they run, then finish; fetch the answer with `last_result(thread_id)`."""
        yield from self._events(FlightTask(message=message, thread_id=thread_id))

    def last_result(self, thread_id: str = "default") -> FlightTaskResult:
        return self._result_from_state(thread_id, [])

    # ---------------------------------------------------------------------------- internals

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.settings.agent.recursion_limit}

    def _graph_input(self, task: FlightTask) -> Any:
        config = self._config(task.thread_id)
        if self.graph.get_state(config).next:  # paused on a question: this message is the answer
            return Command(resume=task.message)
        hints = {"intent": task.intent.value if task.intent else None, "params": task.params}
        return {"messages": [HumanMessage(content=task.message)], "task": hints}

    def _events(self, task: FlightTask) -> Iterator[str]:
        for update in self.graph.stream(self._graph_input(task), self._config(task.thread_id), stream_mode="updates"):
            yield from (name for name in update if not name.startswith("__"))

    def _run(self, task: FlightTask) -> FlightTaskResult:
        nodes: list[str] = []
        try:
            nodes = list(self._events(task))
            return self._result_from_state(task.thread_id, nodes)
        except Exception:
            logger.exception("Flight agent failed")
            return FlightTaskResult(
                thread_id=task.thread_id, status=ExecutionStatus.FAILED, message=FALLBACK_ERROR,
                error={"code": "AGENT_ERROR", "message": "Unexpected internal error."}, nodes_run=nodes,
            )

    def _result_from_state(self, thread_id: str, nodes: list[str]) -> FlightTaskResult:
        snapshot = self.graph.get_state(self._config(thread_id))
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if interrupts:
            payload = interrupts[0].value
            waiting_confirmation = payload["kind"] == "confirmation"
            return FlightTaskResult(
                thread_id=thread_id,
                status=ExecutionStatus.PENDING if waiting_confirmation else ExecutionStatus.REQUIRES_USER_INPUT,
                intent=Intent(payload["intent"]) if payload.get("intent") else None,
                message=payload["question"], question=payload["question"], requires_user_input=True,
                missing_parameters=payload.get("missing_params", []), data=payload.get("data", {}),
                outcome="awaiting_confirmation" if waiting_confirmation else "awaiting_input", nodes_run=nodes,
            )
        result = dict(snapshot.values.get("result") or {})
        if not result:
            raise RuntimeError("graph finished without a result")
        return FlightTaskResult(thread_id=thread_id, nodes_run=nodes, **result)
