"""LangGraph supervisor with conditional routing and parallel fan-out/fan-in."""
from __future__ import annotations
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from travel_agent.flight_agent import FlightAgent, FlightTask
from travel_agent.hotel_agent import HotelAgent, HotelTask
from travel_agent.orchestrator.adapter import normalize
from travel_agent.orchestrator.intent import classify
from travel_agent.orchestrator.state import OrchestratorState
from travel_agent.memory.checkpointer import create_checkpointer

def _history(state):
    return (state.get("conversation_history") or []) + [{"role": "user", "content": state["user_request"]}]

def _analyze(state):
    c = classify(state["user_request"], state.get("pending_domains"))
    return {"conversation_history": _history(state), "identified_intent": c["operation"],
            "identified_domains": c["domains"], "requested_operations": [c["operation"]],
            "selected_subagents": c["domains"], "execution_plan": c,
            "hotel_task_status": "NOT_SELECTED", "flight_task_status": "NOT_SELECTED",
            "hotel_result": None, "flight_result": None, "subagent_errors": {}, "missing_information": []}

def _hotel(state, runtime):
    agent: HotelAgent = runtime.context["hotel"]
    tid = runtime.context["thread_id"] + ":hotel"
    try:
        result = normalize(agent.run_task(HotelTask(message=state["user_request"], thread_id=tid)))
        return {"hotel_result": result, "hotel_task_status": result["status"]}
    except Exception as exc:
        result = {"intent": None, "status": "FAILED", "data": {}, "error": {"code":"SUBAGENT_FAILURE", "message":str(exc)}, "requires_user_input":False,"missing_parameters":[],"message":"Hotel agent execution failed."}
        return {"hotel_result":result,"hotel_task_status":"FAILED","subagent_errors":{"hotel":result["error"]}}

def _flight(state, runtime):
    agent: FlightAgent = runtime.context["flight"]
    tid = runtime.context["thread_id"] + ":flight"
    try:
        result = normalize(agent.run_task(FlightTask(message=state["user_request"], thread_id=tid)))
        return {"flight_result": result, "flight_task_status": result["status"]}
    except Exception as exc:
        result = {"intent": None, "status": "FAILED", "data": {}, "error": {"code":"SUBAGENT_FAILURE", "message":str(exc)}, "requires_user_input":False,"missing_parameters":[],"message":"Flight agent execution failed."}
        return {"flight_result":result,"flight_task_status":"FAILED","subagent_errors":{"flight":result["error"]}}

def _respond(state):
    plan = state.get("execution_plan", {})
    domains = plan.get("domains", [])
    if not domains:
        if plan.get("operation") == "UNSUPPORTED":
            return {"final_response":"This system currently supports hotel and flight operations only.","execution_status":"UNSUPPORTED","pending_domains":[]}
        if plan.get("operation") == "GENERAL_QUERY":
            msg = "I can help you find, book, or cancel hotels and flights. What would you like to do?"
            status = "SUCCESS"
        else:
            msg = "I can help with that. Is this about a hotel reservation or a flight booking?"
            status = "REQUIRES_USER_INPUT"
        return {"final_response":msg,"execution_status":status,"pending_domains":[]}
    parts=[]; statuses=[]; pending=[]; missing=[]
    for d in domains:
        r=state.get(f"{d}_result") or {"status":"UNKNOWN","message":"No verified result was returned."}
        statuses.append(r["status"])
        if r.get("requires_user_input"):
            pending.append(d); missing += r.get("missing_parameters", [])
        message = r.get("question") or r.get("message") or r["status"]
        parts.append(f"To continue with your {d} request, {message}" if r.get("requires_user_input")
                     else f"{d.title()}: {message}")
    if "FAILED" in statuses and "SUCCESS" in statuses: status="PARTIAL_SUCCESS"
    elif any(s in ("PENDING","REQUIRES_USER_INPUT") for s in statuses): status="PENDING"
    elif all(s=="SUCCESS" for s in statuses): status="SUCCESS"
    elif all(s=="FAILED" for s in statuses): status="FAILED"
    else: status="UNKNOWN"
    return {"final_response":"\n".join(parts),"execution_status":status,"pending_domains":pending,"missing_information":missing}

def build_orchestrator_graph(hotel: HotelAgent, flight: FlightAgent, *, checkpointer: BaseCheckpointSaver | None = None):
    b=StateGraph(OrchestratorState, context_schema=dict)
    b.add_node("analyze",_analyze); b.add_node("hotel",_hotel); b.add_node("flight",_flight); b.add_node("respond",_respond)
    b.add_edge(START,"analyze")
    # Conditional Send edges fan out to both agents in parallel when required.
    from langgraph.types import Send
    def fanout(s):
        nodes=[Send(d, s) for d in s["identified_domains"]]
        return nodes or "respond"
    b.add_conditional_edges("analyze", fanout, {"hotel":"hotel","flight":"flight","respond":"respond"})
    b.add_edge("hotel","respond"); b.add_edge("flight","respond"); b.add_edge("respond",END)
    return b.compile(checkpointer=checkpointer or create_checkpointer(None)).with_config({"configurable": {"orchestrator": True}})

class Orchestrator:
    def __init__(self, hotel: HotelAgent | None=None, flight: FlightAgent | None=None, *, checkpointer=None):
        self.hotel=hotel or HotelAgent(); self.flight=flight or FlightAgent()
        self.graph=build_orchestrator_graph(self.hotel,self.flight,checkpointer=checkpointer)
    def chat(self,message:str,thread_id:str="default") -> dict[str,Any]:
        result=self.graph.invoke({"user_request":message},{"configurable":{"thread_id":thread_id}},context={"hotel":self.hotel,"flight":self.flight,"thread_id":thread_id})
        return {k:result.get(k) for k in ("execution_status","final_response","execution_plan","hotel_result","flight_result","pending_domains","missing_information")}
