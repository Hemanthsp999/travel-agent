"""Flight agent package: a LangGraph subagent that talks to the Flight Mock API over HTTP.

Public surface for a future orchestrator: `FlightAgent`, `FlightTask`, `FlightTaskResult`.
"""

from travel_agent.flight_agent.agent import FlightAgent
from travel_agent.flight_agent.schemas import ExecutionStatus, FlightTask, FlightTaskResult, Intent

__all__ = ["ExecutionStatus", "FlightAgent", "FlightTask", "FlightTaskResult", "Intent"]
