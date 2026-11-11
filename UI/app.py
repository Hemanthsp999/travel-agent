from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any, Callable

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from travel_agent.config import get_settings
from travel_agent.flight_agent import FlightAgent
from travel_agent.hotel_agent import HotelAgent
from travel_agent.memory.checkpointer import create_checkpointer
from travel_agent.orchestrator.graph import Orchestrator


st.set_page_config(page_title="Travel Agent Console", page_icon="✈️", layout="wide")

QUICK_PROMPTS = [
    ("✈️ Flights BLR → BOM", "Find flights from Bangalore to Mumbai"),
    ("🏨 Hotel in Mumbai", "Book a hotel in Mumbai for 2 nights"),
    ("🧳 Flight + hotel", "Flights to Delhi tomorrow and a hotel for 3 nights"),
]

CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.hero {padding: .9rem 1.1rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, rgba(56,132,255,.16), rgba(155,89,255,.14));
       border: 1px solid rgba(128,128,128,.25);}
.hero h2 {margin:0; font-size:1.45rem;}
.hero span {opacity:.75; font-size:.9rem;}
.badge {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600;
        border:1px solid rgba(128,128,128,.4); margin-right:.35rem;}
.badge.run {background:rgba(255,165,0,.18); border-color:orange; animation:pulse 1.2s infinite;}
.badge.ok {background:rgba(46,160,67,.18); border-color:rgba(46,160,67,.7);}
.badge.err {background:rgba(220,53,69,.18); border-color:#dc3545;}
.badge.idle {opacity:.7;}
.timeline {border-left:2px solid rgba(128,128,128,.35); margin:.4rem 0 .4rem .5rem; padding-left:.9rem;}
.tl-item {position:relative; margin-bottom:.55rem; font-size:.88rem;}
.tl-item:before {content:''; position:absolute; left:-1.32rem; top:.35rem; width:.65rem; height:.65rem;
                 border-radius:50%; background:rgba(46,160,67,.9);}
.tl-item.live:before {background:orange; animation:pulse 1.2s infinite;}
.tl-item small {opacity:.65; display:block;}
.logline {font-family:monospace; font-size:.76rem; opacity:.8; white-space:pre-wrap; word-break:break-word;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(255,165,0,.55);} 70%{box-shadow:0 0 0 8px rgba(255,165,0,0);} 100%{box-shadow:0 0 0 0 rgba(255,165,0,0);}}
@media (max-width: 768px) {
  .block-container {padding: .75rem .6rem;}
  .hero h2 {font-size:1.15rem;}
}
</style>
"""


# --------------------------------------------------------------------------- #
# Real-time node events
# --------------------------------------------------------------------------- #
class StreamingGraph:
    """Wraps a compiled LangGraph so that .invoke() reports every node as it runs.

    Orchestrator.chat() keeps working exactly as before; we only swap invoke()
    for stream(), forward the node updates to a listener and return the final state.
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self.listener: Callable[[str, tuple, Any], None] | None = None

    @staticmethod
    def _merge_live_state(update: Any) -> dict[str, Any]:
        if not isinstance(update, dict):
            return {}
        return {
            key: value for key, value in update.items()
            if key in {"selected_subagents", "execution_status", "pending_domains", "missing_information",
                       "next_action", "identified_domains", "hotel_task_status", "flight_task_status",
                       "hotel_result", "flight_result", "execution_plan", "subagent_errors"}
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self.listener is None:
            return self._graph.invoke(input, config, **kwargs)

        final_state: Any = None
        started = False
        try:
            for ns, mode, data in self._graph.stream(
                input, config, stream_mode=["updates", "values"], subgraphs=True, **kwargs
            ):
                started = True
                if mode == "values" and not ns:
                    final_state = data
                elif mode == "updates" and isinstance(data, dict):
                    for node, update in data.items():
                        self.listener(str(node), tuple(ns), update)
        except TypeError:
            if started:
                raise
            return self._graph.invoke(input, config, **kwargs)
        return final_state


@st.cache_resource(show_spinner="Loading agents…")
def build_orchestrator() -> Orchestrator:
    settings = get_settings()
    settings = replace(settings, agent=replace(settings.agent, use_llm=False))
    hotel = HotelAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db), load_model=False)
    flight = FlightAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db), load_model=False)
    agent = Orchestrator(hotel, flight, checkpointer=create_checkpointer(settings.checkpoint_db))
    try:
        agent.graph = StreamingGraph(agent.graph)
    except Exception:  # noqa: BLE001 - fall back to log-only status
        pass
    return agent


class LogCollector(logging.Handler):
    def __init__(self, on_line: Callable[[str], None]) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.on_line = on_line

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        self.lines.append(line)
        self.on_line(line)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def node_icon(name: str) -> str:
    n = name.lower()
    for keys, icon in (
        (("classif", "intent", "understand"), "🧭"),
        (("rout", "plan", "dispatch"), "🔀"),
        (("hotel",), "🏨"),
        (("flight",), "✈️"),
        (("clarif", "missing", "ask"), "❓"),
        (("respond", "final", "answer", "compose"), "💬"),
    ):
        if any(k in n for k in keys):
            return icon
    return "⚙️"


def pretty(name: str) -> str:
    return name.replace("_", " ").strip().title()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_answer(result: Any, snapshot: dict[str, Any]) -> str:
    """Find the agent's reply no matter how chat() shaped its return value."""
    if isinstance(result, str) and result.strip():
        return result
    for source in (as_dict(result), snapshot):
        text = source.get("final_response")
        if isinstance(text, str) and text.strip():
            return text
    for source in (as_dict(result), snapshot):
        messages = source.get("messages") or []
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
            if content and role in ("ai", "assistant"):
                return content if isinstance(content, str) else str(content)
    return "The agent finished but did not return a response. Check the Logs tab in the inspector."


def render_timeline(events: list[dict[str, Any]], live: bool) -> str:
    if not events:
        return "<div class='timeline'><div class='tl-item live'>Waiting for the first step…</div></div>" if live else ""
    items = []
    for i, ev in enumerate(events):
        is_live = live and i == len(events) - 1
        sub = f" <small>subgraph: {escape(' › '.join(ev['ns']))}</small>" if ev["ns"] else ""
        detail = f"<small>{escape(ev['detail'])}</small>" if ev["detail"] else ""
        items.append(
            f"<div class='tl-item {'live' if is_live else ''}'>{ev['icon']} <b>{escape(pretty(ev['node']))}</b>"
            f" <small>+{ev['t']:.2f}s</small>{sub}{detail}</div>"
        )
    return f"<div class='timeline'>{''.join(items)}</div>"


def summarize_agent_state(result: dict[str, Any]) -> list[dict[str, str]]:
    payload = as_dict(result)
    selected = payload.get("selected_subagents") or payload.get("identified_domains") or payload.get("pending_domains") or []
    rows: list[dict[str, str]] = []

    for domain in selected:
        domain_name = str(domain).lower()
        domain_result = as_dict(payload.get(f"{domain_name}_result"))
        status = str(payload.get(f"{domain_name}_task_status") or domain_result.get("status") or "UNKNOWN").upper()
        message = str(domain_result.get("message") or domain_result.get("question") or "No update yet").strip()
        rows.append({"agent": pretty(domain_name), "status": status, "detail": message})

    if not rows:
        status = str(payload.get("execution_status") or "IDLE").upper()
        action = str(payload.get("next_action") or "waiting").upper()
        rows.append({"agent": "Orchestrator", "status": status, "detail": f"Next action: {action}"})

    return rows


def render_monitor(ph: Any, run: dict[str, Any] | None) -> None:
    """Draw the right-hand Agent monitor from a run record."""
    with ph.container():
        if not run:
            st.markdown("<span class='badge idle'>● Idle</span> Waiting for a query", unsafe_allow_html=True)
            st.caption("Live agent steps appear here while a query runs.")
            return

        state = run["state"]
        badge = {"running": ("run", "● Running"), "complete": ("ok", "● Completed"), "error": ("err", "● Failed")}[state]
        st.markdown(
            f"<span class='badge {badge[0]}'>{badge[1]}</span> <span style='opacity:.75'>{run['elapsed']:.1f}s</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Query: {run['query']}")
        st.markdown(render_timeline(run["events"], live=state == "running"), unsafe_allow_html=True)

        merged_state = {**as_dict(run.get("result")), **as_dict(run.get("agent_state"))}
        agent_rows = summarize_agent_state(merged_state)
        current_event = run["events"][-1] if run.get("events") else {}
        current_node = str(current_event.get("node") or "idle")
        current_status = str((merged_state.get("execution_status") or "RUNNING")).upper()
        next_action = str((merged_state.get("next_action") or "waiting")).upper()
        primary_agent = agent_rows[0]["agent"] if agent_rows else "Orchestrator"
        primary_status = agent_rows[0]["status"] if agent_rows else current_status
        primary_detail = agent_rows[0]["detail"] if agent_rows else "Waiting for the next step"

        st.markdown(
            f"<div style='border:1px solid rgba(128,128,128,.3); border-radius:12px; padding:.75rem .8rem; margin-top:.8rem; background:rgba(56,132,255,.04);'>"
            f"<div style='display:flex; justify-content:space-between; gap:.5rem; align-items:center;'><b>Current phase</b><span style='font-weight:600; color:orange;'>{escape(pretty(current_node))}</span></div>"
            f"<div style='margin-top:.45rem; display:flex; flex-wrap:wrap; gap:.45rem; font-size:.82rem;'><span class='badge run'>Status: {escape(current_status)}</span> <span class='badge idle'>Next: {escape(next_action)}</span></div>"
            f"<div style='margin-top:.55rem;'><b>{escape(primary_agent)}</b> · <span style='font-weight:600;'>{escape(primary_status)}</span></div>"
            f"<div style='opacity:.8; font-size:.82rem; margin-top:.2rem;'>{escape(primary_detail)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption("Agent state")
        for row in agent_rows:
            color = "orange" if row["status"] in {"RUNNING", "PENDING", "REQUIRES_USER_INPUT"} else "green" if row["status"] in {"SUCCESS", "COMPLETED"} else "red" if row["status"] in {"FAILED", "ERROR"} else "gray"
            st.markdown(
                f"<div style='border:1px solid rgba(128,128,128,.3); border-radius:10px; padding:.55rem .7rem; margin-bottom:.35rem;'><div style='display:flex; justify-content:space-between; gap:.5rem; align-items:center;'><b>{escape(row['agent'])}</b><span style='color:{color}; font-weight:600;'>{escape(row['status'])}</span></div><div style='opacity:.8; font-size:.82rem; margin-top:.2rem;'>{escape(row['detail'])}</div></div>",
                unsafe_allow_html=True,
            )

        if run.get("last_log") and state == "running":
            st.markdown(f"<div class='logline'>{escape(run['last_log'])}</div>", unsafe_allow_html=True)

        if state != "running":
            result = as_dict(run.get("result"))
            c1, c2 = st.columns(2)
            c1.metric("Status", str(result.get("execution_status") or state))
            c2.metric("Steps", len(run["events"]))
            missing = result.get("missing_information")
            if missing:
                st.warning(f"Needs more info: {missing}")
            if run.get("error"):
                st.error(run["error"])


def suggestions_for(result: dict[str, Any]) -> list[str]:
    plan = as_dict(result.get("execution_plan"))
    domains = plan.get("domains") or []
    if result.get("missing_information"):
        return []
    out: list[str] = []
    if "flight" in domains and "hotel" not in domains:
        out.append("Also find a hotel there")
    if "hotel" in domains and "flight" not in domains:
        out.append("Also find flights there")
    if domains:
        out += ["Show cheaper options", "Show the best-rated option"]
    return out[:3]


def queue_query(text: str) -> None:
    st.session_state.queued = text


def render_details(entry: dict[str, Any]) -> None:
    result = as_dict(entry.get("result"))
    t_plan, t_state, t_logs = st.tabs(["Plan", "Checkpoint", "Logs"])
    with t_plan:
        st.json({
            "execution_status": result.get("execution_status"),
            "execution_plan": result.get("execution_plan"),
            "pending_domains": result.get("pending_domains"),
            "missing_information": result.get("missing_information"),
        }, expanded=True)
    with t_state:
        st.json(entry.get("snapshot") or {}, expanded=False)
    with t_logs:
        logs = entry.get("logs") or []
        st.code("\n".join(logs) if logs else "No logs captured.", language="text")


def render_debug_panel(thread_id: str, history: list[dict[str, Any]], run: dict[str, Any] | None) -> None:
    if not history:
        st.info("No agent flow has run yet for this thread.")
        return

    choices = [f"{idx + 1}. {entry.get('query', 'Untitled query')}" for idx, entry in enumerate(history)]
    selected = st.selectbox("Select debug run", range(len(history)), format_func=lambda idx: choices[idx], index=len(history) - 1)

    entry = history[selected]
    display_run = run if run and run.get("query") == entry.get("query") else {
        "query": entry.get("query"),
        "state": "complete",
        "events": entry.get("events", []),
        "result": entry.get("result", {}),
        "logs": entry.get("logs", []),
        "last_log": entry.get("logs", [])[-1] if entry.get("logs") else "",
        "elapsed": entry.get("elapsed", 0.0),
    }

    flow_tab, state_tab, logs_tab = st.tabs(["Flow", "State", "Logs"])
    with flow_tab:
        st.caption(f"Thread: {thread_id} · Query {selected + 1}/{len(history)}")
        st.markdown(render_timeline(display_run.get("events", []), live=display_run.get("state") == "running"), unsafe_allow_html=True)
        if display_run.get("last_log"):
            st.markdown(f"<div class='logline'>{escape(display_run['last_log'])}</div>", unsafe_allow_html=True)

    with state_tab:
        result = as_dict(display_run.get("result"))
        st.json({
            "query": display_run.get("query"),
            "state": display_run.get("state"),
            "execution_status": result.get("execution_status"),
            "execution_plan": result.get("execution_plan"),
            "pending_domains": result.get("pending_domains"),
            "missing_information": result.get("missing_information"),
            "result": result,
            "snapshot": entry.get("snapshot") if entry else {},
        }, expanded=False)

    with logs_tab:
        logs = entry.get("logs") if entry else display_run.get("logs", [])
        st.code("\n".join(logs) if logs else "No logs captured.", language="text")


def render_assistant(entry: dict[str, Any], idx: int, thread_id: str, is_last: bool) -> None:
    if entry.get("error"):
        st.error(entry["content"])
    else:
        st.markdown(entry["content"])
        st.caption(f"⏱ {entry.get('elapsed', 0):.1f}s · {len(entry.get('events', []))} steps")
    with st.expander("Inspect agent run"):
        render_details(entry)
    if is_last:
        sugg = suggestions_for(as_dict(entry.get("result")))
        if sugg:
            cols = st.columns(len(sugg))
            for i, text in enumerate(sugg):
                cols[i].button(text, key=f"sug-{thread_id}-{idx}-{i}", on_click=queue_query, args=(text,),
                               use_container_width=True)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_query(query: str, thread_id: str, history: list[dict[str, Any]], monitor_ph: Any) -> None:
    agent = build_orchestrator()
    started = time.time()
    run: dict[str, Any] = {"query": query, "events": [], "logs": [], "state": "running", "elapsed": 0.0, "last_log": ""}
    st.session_state.last_run[thread_id] = run

    graph = agent.graph if isinstance(agent.graph, StreamingGraph) else None

    with st.chat_message("assistant", avatar="🤖"):
        status = st.status("Agent is working…", expanded=True)
        with status:
            line_ph = st.empty()

        def refresh() -> None:
            run["elapsed"] = time.time() - started
            render_monitor(monitor_ph, run)

        def on_node(node: str, ns: tuple, update: Any) -> None:
            keys = ", ".join(update.keys()) if isinstance(update, dict) else ""
            run["events"].append({
                "node": node, "ns": ns, "icon": node_icon(node),
                "t": time.time() - started, "detail": f"updated: {keys}" if keys else "",
            })
            run["agent_state"] = {**as_dict(run.get("agent_state")), **StreamingGraph._merge_live_state(update)}
            status.update(label=f"{node_icon(node)} {pretty(node)}…")
            line_ph.markdown(f"{node_icon(node)} **{pretty(node)}** finished at +{time.time() - started:.1f}s")
            refresh()

        def on_log(line: str) -> None:
            run["logs"].append(line)
            run["last_log"] = line
            refresh()

        collector = LogCollector(on_log)
        collector.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(collector)
        if graph:
            graph.listener = on_node
        refresh()

        entry: dict[str, Any]
        try:
            result = agent.chat(query, thread_id)
            try:
                snap = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
                snapshot = dict(snap.values) if snap and snap.values else {}
            except Exception:  # noqa: BLE001
                snapshot = {}
            answer = extract_answer(result, snapshot)
            run.update(state="complete", result=as_dict(result) or snapshot, elapsed=time.time() - started)
            status.update(label=f"Completed in {run['elapsed']:.1f}s", state="complete", expanded=False)
            entry = {"role": "assistant", "content": answer, "query": query, "result": run["result"], "snapshot": snapshot,
                     "logs": collector.lines, "events": run["events"], "elapsed": run["elapsed"], "run": dict(run)}
        except Exception as exc:  # noqa: BLE001
            run.update(state="error", error=str(exc), elapsed=time.time() - started)
            status.update(label="Agent failed", state="error", expanded=True)
            entry = {"role": "assistant", "content": f"Something went wrong: `{exc}`", "query": query, "error": True,
                     "result": {}, "snapshot": {}, "logs": collector.lines, "events": run["events"],
                     "elapsed": run["elapsed"], "run": dict(run)}
        finally:
            root_logger.removeHandler(collector)
            root_logger.setLevel(previous_level)
            if graph:
                graph.listener = None
            render_monitor(monitor_ph, run)

        history.append(entry)
        render_assistant(entry, len(history) - 1, thread_id, is_last=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    ss.setdefault("thread_id", "ui-demo-thread")
    ss.setdefault("histories", {})
    ss.setdefault("last_run", {})

    # ---- sidebar
    with st.sidebar:
        st.header("Session")
        ss.thread_id = st.text_input("Thread ID", value=ss.thread_id, help="Same ID continues the conversation.")
        c1, c2 = st.columns(2)
        if c1.button("New thread", use_container_width=True):
            ss.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
            st.rerun()
        if c2.button("Clear chat", use_container_width=True):
            ss.histories[ss.thread_id] = []
            ss.last_run.pop(ss.thread_id, None)
            st.rerun()
        st.divider()
        st.subheader("Quick prompts")
        for i, (label, text) in enumerate(QUICK_PROMPTS):
            st.button(label, key=f"qp-{i}", on_click=queue_query, args=(text,), use_container_width=True)
        st.divider()
        others = [t for t in ss.histories if t != ss.thread_id and ss.histories[t]]
        if others:
            st.caption("Other threads this session: " + ", ".join(f"`{t}`" for t in others))

    thread_id: str = ss.thread_id
    history: list[dict[str, Any]] = ss.histories.setdefault(thread_id, [])

    # chat input must live at the page root so it stays pinned to the bottom
    query = st.chat_input("Ask about flights or hotels…") or ss.pop("queued", None)

    tabs = st.tabs(["Chat", "Debug"])

    with tabs[0]:
        st.markdown(
            f"<div class='hero'><h2>✈️ Travel Agent Console</h2>"
            f"<span>Thread <code>{escape(thread_id)}</code> · live agent monitor on the right</span></div>",
            unsafe_allow_html=True,
        )

        chat_col, monitor_col = st.columns([1.7, 1], gap="large")

        with monitor_col:
            st.subheader("Agent monitor")
            monitor_ph = st.empty()
            render_monitor(monitor_ph, ss.last_run.get(thread_id))

        with chat_col:
            if not history and not query:
                st.info("Pick a quick prompt from the sidebar or type a request below.")
            for idx, entry in enumerate(history):
                avatar = "🧑" if entry["role"] == "user" else "🤖"
                with st.chat_message(entry["role"], avatar=avatar):
                    if entry["role"] == "user":
                        st.markdown(entry["content"])
                    else:
                        render_assistant(entry, idx, thread_id, is_last=idx == len(history) - 1)

            if query:
                history.append({"role": "user", "content": query, "query": query})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(query)
                run_query(query, thread_id, history, monitor_ph)

    with tabs[1]:
        render_debug_panel(thread_id, history, ss.last_run.get(thread_id))


if __name__ == "__main__":
    main()