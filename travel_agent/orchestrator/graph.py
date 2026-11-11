"""LangGraph supervisor with a dynamic ReAct loop: observe -> decide -> act -> re-observe."""
from __future__ import annotations
from datetime import date, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from travel_agent.flight_agent import FlightAgent, FlightTask
from travel_agent.flight_agent.schemas import Intent as FlightIntent
from travel_agent.hotel_agent import HotelAgent, HotelTask
from travel_agent.hotel_agent.schemas import Intent as HotelIntent
from travel_agent.memory.checkpointer import create_checkpointer
from travel_agent.orchestrator.adapter import normalize
from travel_agent.orchestrator.intent import classify
from travel_agent.orchestrator.semantic_parser import extract_semantic_request
from travel_agent.orchestrator.state import OrchestratorState
from travel_agent.prompts import load_prompt
from travel_agent.retrieval import KnowledgeRetriever, get_knowledge_retriever

MAX_REACT_ITERATIONS = 6


def _history(state):
    return (state.get("conversation_history") or []) + [{"role": "user", "content": state["user_request"]}]


def _retrieve(state, retriever: KnowledgeRetriever):
    knowledge = retriever.retrieve(
        state["user_request"],
        prior_knowledge=state.get("retrieved_knowledge"),
    )
    return {
        "retrieved_knowledge": knowledge,
        "knowledge_query": knowledge["is_knowledge_query"],
    }


def _observe(state):
    prior_domains = state.get("pending_domains") or state.get("identified_domains") or state.get("selected_subagents") or []
    c = classify(state["user_request"], prior_domains)
    semantic_slots = extract_semantic_request(state["user_request"])
    loop_count = int(state.get("loop_count", 0)) + 1
    return {
        "conversation_history": _history(state),
        "identified_intent": c["operation"],
        "identified_domains": c["domains"],
        "requested_operations": [c["operation"]],
        "selected_subagents": c["domains"],
        "execution_plan": {**c, "semantic_slots": semantic_slots},
        "semantic_slots": semantic_slots,
        "supervisor_prompt": load_prompt("orchestrator"),
        "hotel_task_status": "NOT_SELECTED",
        "flight_task_status": "NOT_SELECTED",
        "hotel_result": None,
        "flight_result": None,
        "subagent_errors": {},
        "missing_information": [],
        "loop_count": loop_count,
        "next_action": "execute" if c["domains"] else "clarify",
    }


def _decide(state):
    plan = state.get("execution_plan", {})
    if state.get("knowledge_query"):
        return {
            "next_action": "respond",
            "execution_status": "SUCCESS" if (state.get("retrieved_knowledge") or {}).get("answer") else "REQUIRES_USER_INPUT",
            "pending_domains": [],
            "identified_domains": [],
            "selected_subagents": [],
            "execution_plan": {**plan, "domains": [], "operation": "GENERAL_QUERY", "route": "rag"},
        }
    if plan.get("operation") == "GENERAL_QUERY":
        return {
            "next_action": "respond",
            "execution_status": "SUCCESS",
            "selected_subagents": [],
            "identified_domains": [],
            "execution_plan": {**plan, "domains": []},
        }
    domains = plan.get("domains") or state.get("pending_domains") or []
    if not domains:
        op = plan.get("operation")
        if op == "UNSUPPORTED":
            return {
                "next_action": "done",
                "execution_status": "UNSUPPORTED",
                "final_response": "This system currently supports hotel and flight operations only.",
                "pending_domains": [],
            }
        if op == "GENERAL_QUERY":
            return {
                "next_action": "done",
                "execution_status": "SUCCESS",
                "final_response": "I can help you find, book, or cancel hotels and flights. What would you like to do?",
                "pending_domains": [],
            }
        return {
            "next_action": "done",
            "execution_status": "REQUIRES_USER_INPUT",
            "final_response": "I can help with that. Is this about a hotel reservation or a flight booking?",
            "pending_domains": [],
        }
    return {"next_action": "act", "selected_subagents": domains, "execution_status": "RUNNING", "pending_domains": domains}


def _task_params_for_domain(domain: str, slots: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if domain == "flight":
        if slots.get("origin"):
            params["origin"] = slots["origin"]
        if slots.get("destination"):
            params["destination"] = slots["destination"]
        if slots.get("travel_date"):
            params["departure_date"] = slots["travel_date"]
        if slots.get("passengers"):
            params["passenger_count"] = int(slots["passengers"])
        return params

    if slots.get("hotel_city"):
        params["city"] = slots["hotel_city"]
    if slots.get("travel_date"):
        params["check_in"] = slots["travel_date"]
        nights = int(slots.get("nights") or 1)
        try:
            checkout = date.fromisoformat(slots["travel_date"]) + timedelta(days=nights)
            params["check_out"] = checkout.isoformat()
        except ValueError:
            pass
    if slots.get("passengers"):
        params["guests"] = int(slots["passengers"])
    if slots.get("preferences"):
        params["preferences"] = slots["preferences"]
    return params


def _intent_for_domain(domain: str, plan: dict[str, Any]) -> Any:
    op = (plan.get("operation") or "").upper()
    if domain == "flight":
        mapping = {
            "CHECK_AVAILABILITY": FlightIntent.CHECK_FLIGHT_AVAILABILITY,
            "BOOK": FlightIntent.BOOK_FLIGHT,
            "CANCEL": FlightIntent.CANCEL_FLIGHT,
            "GENERAL_QUERY": FlightIntent.GENERAL_FLIGHT_QUERY,
            "UNKNOWN": FlightIntent.UNKNOWN,
        }
        return mapping.get(op, FlightIntent.CHECK_FLIGHT_AVAILABILITY)

    mapping = {
        "CHECK_AVAILABILITY": HotelIntent.CHECK_AVAILABILITY,
        "BOOK": HotelIntent.BOOK_HOTEL,
        "CANCEL": HotelIntent.CANCEL_BOOKING,
        "GENERAL_QUERY": HotelIntent.GENERAL_HOTEL_QUERY,
        "UNKNOWN": HotelIntent.UNKNOWN,
    }
    return mapping.get(op, HotelIntent.CHECK_AVAILABILITY)


def _hotel(state, runtime):
    agent: HotelAgent = runtime.context["hotel"]
    tid = runtime.context["thread_id"] + ":hotel"
    plan = state.get("execution_plan", {})
    slots = state.get("semantic_slots") or {}
    task = HotelTask(
        message=state["user_request"],
        thread_id=tid,
        intent=_intent_for_domain("hotel", plan),
        params=_task_params_for_domain("hotel", slots),
        context=_subagent_context(state),
    )
    try:
        result = normalize(agent.run_task(task))
        return {"hotel_result": result, "hotel_task_status": result["status"]}
    except Exception as exc:
        result = {"intent": None, "status": "FAILED", "data": {}, "error": {"code": "SUBAGENT_FAILURE", "message": str(exc)}, "requires_user_input": False, "missing_parameters": [], "message": "Hotel agent execution failed."}
        return {"hotel_result": result, "hotel_task_status": "FAILED", "subagent_errors": {"hotel": result["error"]}}


def _flight(state, runtime):
    agent: FlightAgent = runtime.context["flight"]
    tid = runtime.context["thread_id"] + ":flight"
    plan = state.get("execution_plan", {})
    slots = state.get("semantic_slots") or {}
    task = FlightTask(
        message=state["user_request"],
        thread_id=tid,
        intent=_intent_for_domain("flight", plan),
        params=_task_params_for_domain("flight", slots),
        context=_subagent_context(state),
    )
    try:
        result = normalize(agent.run_task(task))
        return {"flight_result": result, "flight_task_status": result["status"]}
    except Exception as exc:
        result = {"intent": None, "status": "FAILED", "data": {}, "error": {"code": "SUBAGENT_FAILURE", "message": str(exc)}, "requires_user_input": False, "missing_parameters": [], "message": "Flight agent execution failed."}
        return {"flight_result": result, "flight_task_status": "FAILED", "subagent_errors": {"flight": result["error"]}}


def _reobserve(state):
    loop_count = int(state.get("loop_count", 0))
    selected = state.get("selected_subagents") or state.get("identified_domains") or state.get("pending_domains") or []
    statuses = []
    pending = []
    missing = []
    details = []

    for d in selected:
        result = state.get(f"{d}_result") or {"status": "UNKNOWN", "message": "No verified result was returned."}
        statuses.append(result.get("status", "UNKNOWN"))
        if result.get("requires_user_input"):
            pending.append(d)
            missing.extend(result.get("missing_parameters", []))
        details.append(result.get("question") or result.get("message") or result.get("status") or "No result")

    if "FAILED" in statuses and "SUCCESS" in statuses:
        status = "PARTIAL_SUCCESS"
    elif any(s in ("PENDING", "REQUIRES_USER_INPUT") for s in statuses):
        status = "PENDING"
    elif all(s == "SUCCESS" for s in statuses):
        status = "SUCCESS"
    elif all(s == "FAILED" for s in statuses):
        status = "FAILED"
    else:
        status = "UNKNOWN"

    if pending:
        next_action = "done"
        final_response = "I need a couple of details to continue with your request. " + " ".join(details)
        execution_status = "REQUIRES_USER_INPUT"
        missing_information = missing
        pending_domains = pending
    elif status == "PARTIAL_SUCCESS":
        final_response = "I handled part of your request. " + " ".join(details)
        execution_status = "PARTIAL_SUCCESS"
        pending_domains = []
        missing_information = []
        next_action = "done"
    elif status == "SUCCESS":
        final_response = " ".join(details) if details else "I have the information needed."
        execution_status = "SUCCESS"
        pending_domains = []
        missing_information = []
        next_action = "done"
    elif status == "FAILED":
        final_response = " ".join(details) if details else "The request could not be completed."
        execution_status = "FAILED"
        pending_domains = []
        missing_information = []
        next_action = "done"
    elif loop_count < MAX_REACT_ITERATIONS:
        final_response = "I am reviewing the latest result to determine the next step."
        execution_status = "RUNNING"
        pending_domains = []
        missing_information = []
        next_action = "observe"
    else:
        final_response = "I reached the maximum iteration limit while handling this request."
        execution_status = "FAILED"
        pending_domains = []
        missing_information = []
        next_action = "done"

    return {
        "execution_status": execution_status,
        "final_response": final_response,
        "pending_domains": pending_domains,
        "missing_information": missing_information,
        "next_action": next_action,
    }


def _respond(state):
    plan = state.get("execution_plan", {})
    domains = plan.get("domains", [])
    if state.get("knowledge_query"):
        knowledge = state.get("retrieved_knowledge") or {}
        answer = knowledge.get("answer")
        if not answer:
            answer = (
                "I’m sorry, but I couldn’t find a matching policy or FAQ in my travel guide. "
                "Please tell me the hotel name or flight number so I can check the relevant rules."
            )
        return {
            "final_response": answer,
            "execution_status": "SUCCESS" if knowledge.get("answer") else "REQUIRES_USER_INPUT",
            "pending_domains": [],
            "missing_information": [] if knowledge.get("answer") else ["hotel name or flight number"],
        }
    if not domains:
        if plan.get("operation") == "UNSUPPORTED":
            return {"final_response": "I’m sorry, I can currently help with hotel and flight travel. If you’re planning a trip, I’d be happy to help with either of those.", "execution_status": "UNSUPPORTED", "pending_domains": []}
        if plan.get("operation") == "GENERAL_QUERY":
            topic = plan.get("general_topic")
            replies = {
                "IDENTITY": "Of course! I’m your travel assistant. You can call me your travel companion. How can I help with your trip?",
                "GREETING": "Hello! It’s lovely to hear from you. How can I help with your travel plans today?",
                "THANKS": "You’re very welcome! I’m happy to help with your travel plans whenever you need.",
                "WELLBEING": "I’m doing well, thank you for asking! I’m ready to help with your travel plans. How are you?",
                "HELP": "Absolutely! I can help you search for and book hotels or flights, look up bookings, explain travel policies, and guide you through cancellations. What can I help you with?",
            }
            msg = replies.get(topic, "I’d be happy to help. What would you like to know about your trip?")
            status = "SUCCESS"
        elif plan.get("operation") in {"BOOK", "CANCEL"}:
            action = "booking" if plan.get("operation") == "BOOK" else "cancellation"
            msg = f"I can help with that {action}. Is it for a hotel or a flight?"
            status = "REQUIRES_USER_INPUT"
        else:
            msg = "I’d be happy to help. Could you tell me a little more about what you need?"
            status = "SUCCESS"
        return {"final_response": msg, "execution_status": status, "pending_domains": []}

    statuses = []
    pending = []
    missing = []
    details = []
    for d in domains:
        r = state.get(f"{d}_result") or {"status": "UNKNOWN", "message": "No verified result was returned."}
        statuses.append(r["status"])
        if r.get("requires_user_input"):
            pending.append(d)
            missing += r.get("missing_parameters", [])
        message = r.get("question") or r.get("message") or r["status"]
        details.append(message)

    if "FAILED" in statuses and "SUCCESS" in statuses:
        status = "PARTIAL_SUCCESS"
    elif any(s in ("PENDING", "REQUIRES_USER_INPUT") for s in statuses):
        status = "PENDING"
    elif all(s == "SUCCESS" for s in statuses):
        status = "SUCCESS"
    elif all(s == "FAILED" for s in statuses):
        status = "FAILED"
    else:
        status = "UNKNOWN"

    if pending:
        pending_statuses = [state.get(f"{domain}_result", {}).get("status") for domain in pending]
        if "PENDING" in pending_statuses:
            msg = "Before I proceed, please confirm: " + " ".join(details)
        else:
            msg = "Of course — I just need one more detail to help with that. " + " ".join(details)
    elif status == "PARTIAL_SUCCESS":
        msg = "I was able to complete part of your request. " + " ".join(details)
    elif status == "FAILED":
        msg = "I’m sorry, I couldn’t complete that just now. " + " ".join(details)
    else:
        msg = "Of course — " + " ".join(details) if details else "I have the information needed."

    return {"final_response": msg, "execution_status": status, "pending_domains": pending, "missing_information": missing}


def _route_after_decide(state):
    domains = state.get("selected_subagents") or state.get("identified_domains") or []
    if not domains:
        return "respond"
    return "hotel" if len(domains) == 1 and domains[0] == "hotel" else "flight" if len(domains) == 1 and domains[0] == "flight" else "hotel"


def _route_after_reobserve(state):
    if state.get("next_action") == "observe" and int(state.get("loop_count", 0)) < MAX_REACT_ITERATIONS:
        return "observe"
    return "respond"


def _subagent_context(state: OrchestratorState) -> dict[str, Any] | None:
    knowledge = state.get("retrieved_knowledge") or {}
    if not knowledge.get("context"):
        return None
    return {
        "retrieved_context": knowledge["context"],
        "retrieval_metadata": {
            "domain": knowledge.get("domain"),
            "source_refs": knowledge.get("source_refs", []),
            "confidence": knowledge.get("confidence", 0.0),
        },
    }


def build_orchestrator_graph(
    hotel: HotelAgent,
    flight: FlightAgent,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    retriever: KnowledgeRetriever | None = None,
):
    knowledge_retriever = retriever or get_knowledge_retriever()
    b = StateGraph(OrchestratorState, context_schema=dict)
    b.add_node("observe", _observe)
    b.add_node("retrieve", lambda state: _retrieve(state, knowledge_retriever))
    b.add_node("decide", _decide)
    b.add_node("hotel", _hotel)
    b.add_node("flight", _flight)
    b.add_node("reobserve", _reobserve)
    b.add_node("respond", _respond)

    b.add_edge(START, "observe")
    b.add_edge("observe", "retrieve")
    b.add_edge("retrieve", "decide")

    from langgraph.types import Send

    def fanout(s):
        domains = s.get("selected_subagents") or s.get("identified_domains") or state_get(s, "pending_domains") or []
        if not domains:
            return "respond"
        if len(domains) == 1:
            return domains[0]
        return [Send(d, s) for d in domains]

    b.add_conditional_edges("decide", fanout, {"hotel": "hotel", "flight": "flight", "respond": "respond"})
    b.add_edge("hotel", "reobserve")
    b.add_edge("flight", "reobserve")
    b.add_conditional_edges("reobserve", _route_after_reobserve, {"observe": "observe", "respond": "respond"})
    b.add_edge("respond", END)
    return b.compile(checkpointer=checkpointer or create_checkpointer(None)).with_config({"configurable": {"orchestrator": True}})


def state_get(state, key):
    return state.get(key, [])


class Orchestrator:
    def __init__(
        self,
        hotel: HotelAgent | None = None,
        flight: FlightAgent | None = None,
        *,
        checkpointer=None,
        retriever: KnowledgeRetriever | None = None,
    ):
        self.hotel = hotel or HotelAgent()
        self.flight = flight or FlightAgent()
        self.graph = build_orchestrator_graph(
            self.hotel, self.flight, checkpointer=checkpointer, retriever=retriever
        )

    def chat(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        effective_thread_id = thread_id if thread_id not in (None, "", "default") else f"default-{__import__('uuid').uuid4().hex[:8]}"
        result = self.graph.invoke({
            "user_request": message,
            "iteration_limit": MAX_REACT_ITERATIONS,
            "supervisor_prompt": load_prompt("orchestrator"),
        }, {"configurable": {"thread_id": effective_thread_id}}, context={"hotel": self.hotel, "flight": self.flight, "thread_id": effective_thread_id})
        return {
            k: result.get(k)
            for k in (
                "execution_status", "final_response", "execution_plan", "hotel_result",
                "flight_result", "pending_domains", "missing_information", "retrieved_knowledge",
            )
        }
