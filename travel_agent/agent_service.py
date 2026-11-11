from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from travel_agent.config import get_settings
from travel_agent.flight_agent import FlightAgent
from travel_agent.hotel_agent import HotelAgent
from travel_agent.memory.checkpointer import create_checkpointer
from travel_agent.orchestrator.graph import Orchestrator


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field("ui-thread", min_length=1)


settings = get_settings()
settings = replace(settings, agent=replace(settings.agent, use_llm=False))

hotel = HotelAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db), load_model=False)
flight = FlightAgent(settings, checkpointer=create_checkpointer(settings.checkpoint_db), load_model=False)
agent = Orchestrator(hotel, flight, checkpointer=create_checkpointer(settings.checkpoint_db))

app = FastAPI(title="Travel Agent Backend Service", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "travel-agent"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    result = agent.chat(request.query, request.thread_id)
    return {
        "thread_id": request.thread_id,
        "response": result.get("final_response") or "No final response returned.",
        "status": result.get("execution_status"),
        "execution_plan": result.get("execution_plan"),
        "pending_domains": result.get("pending_domains"),
        "missing_information": result.get("missing_information"),
        "raw": result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("travel_agent.agent_service:app", host="0.0.0.0", port=8009, reload=False)
