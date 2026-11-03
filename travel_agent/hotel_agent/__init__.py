"""Hotel agent package: a LangGraph subagent that talks to the Hotel Mock API over HTTP.

Public surface for a future orchestrator: `HotelAgent`, `HotelTask`, `HotelTaskResult`.
"""

from travel_agent.hotel_agent.agent import HotelAgent
from travel_agent.hotel_agent.schemas import HotelTask, HotelTaskResult, Intent, TaskStatus

__all__ = ["HotelAgent", "HotelTask", "HotelTaskResult", "Intent", "TaskStatus"]
