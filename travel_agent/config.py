"""Typed settings loaded from `configs/*.yaml` with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class ModelSettings:
    model_path: Path = Path("~/models/phi-2.Q4_K_M.gguf")
    n_ctx: int = 4096
    n_threads: int = 0
    n_batch: int = 256
    use_mmap: bool = True
    verbose: bool = False
    routing_temperature: float = 0.0
    reply_temperature: float = 0.7
    routing_max_tokens: int = 160


@dataclass(frozen=True)
class HotelAPISettings:
    base_url: str = "http://localhost:8001"
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 8.0
    max_attempts: int = 2
    retry_backoff_s: float = 0.5


@dataclass(frozen=True)
class FlightAPISettings:
    base_url: str = "http://localhost:8002"
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 8.0
    max_attempts: int = 2
    retry_backoff_s: float = 0.5


@dataclass(frozen=True)
class AgentSettings:
    max_hotels_shown: int = 3
    max_flights_shown: int = 4
    max_ask_attempts: int = 3
    recursion_limit: int = 40
    history_keep_messages: int = 12
    use_llm: bool = True
    llm_confidence_threshold: float = 0.5


@dataclass(frozen=True)
class GraphSettings:
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    remote_png: bool = True


@dataclass(frozen=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    hotel_api: HotelAPISettings = field(default_factory=HotelAPISettings)
    flight_api: FlightAPISettings = field(default_factory=FlightAPISettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    checkpoint_db: Path = PROJECT_ROOT / "data" / "checkpoints.db"


def _read_yaml(name: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / name
    return yaml.safe_load(path.read_text()) or {} if path.exists() else {}


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_dir: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Merge defaults, YAML files and environment overrides (env wins)."""
    config_dir = config_dir or CONFIG_DIR
    env = dict(os.environ) if env is None else env
    models = _read_yaml("models.yaml", config_dir)
    settings = _read_yaml("settings.yaml", config_dir)
    api, agent, memory, graph = (settings.get(k, {}) for k in ("hotel_api", "agent", "memory", "graph"))
    flight = settings.get("flight_api", {})
    temperature, max_tokens = models.get("temperature", {}), models.get("max_tokens", {})

    model = ModelSettings(
        model_path=Path(env.get("MODEL_PATH") or models.get("model_path") or ModelSettings.model_path).expanduser(),
        n_ctx=int(models.get("n_ctx", 4096)),
        n_threads=int(models.get("n_threads", 0)) or (os.cpu_count() or 4),
        n_batch=int(models.get("n_batch", 256)),
        use_mmap=bool(models.get("use_mmap", True)),
        verbose=bool(models.get("verbose", False)),
        routing_temperature=float(temperature.get("routing", 0.0)),
        reply_temperature=float(temperature.get("reply", 0.7)),
        routing_max_tokens=int(max_tokens.get("routing", 160)),
    )
    defaults = HotelAPISettings()
    hotel_api = HotelAPISettings(
        base_url=env.get("HOTEL_API_BASE_URL") or api.get("base_url", defaults.base_url),
        connect_timeout_s=float(api.get("connect_timeout_s", defaults.connect_timeout_s)),
        read_timeout_s=float(api.get("read_timeout_s", defaults.read_timeout_s)),
        max_attempts=int(api.get("max_attempts", defaults.max_attempts)),
        retry_backoff_s=float(api.get("retry_backoff_s", defaults.retry_backoff_s)),
    )
    flight_defaults = FlightAPISettings()
    flight_api = FlightAPISettings(
        base_url=env.get("FLIGHT_API_BASE_URL") or flight.get("base_url", flight_defaults.base_url),
        connect_timeout_s=float(flight.get("connect_timeout_s", flight_defaults.connect_timeout_s)),
        read_timeout_s=float(flight.get("read_timeout_s", flight_defaults.read_timeout_s)),
        max_attempts=int(flight.get("max_attempts", flight_defaults.max_attempts)),
        retry_backoff_s=float(flight.get("retry_backoff_s", flight_defaults.retry_backoff_s)),
    )
    agent_defaults = AgentSettings()
    use_llm = env.get("HOTEL_AGENT_USE_LLM")
    agent_settings = AgentSettings(
        max_hotels_shown=int(agent.get("max_hotels_shown", agent_defaults.max_hotels_shown)),
        max_flights_shown=int(agent.get("max_flights_shown", agent_defaults.max_flights_shown)),
        max_ask_attempts=int(agent.get("max_ask_attempts", agent_defaults.max_ask_attempts)),
        recursion_limit=int(agent.get("recursion_limit", agent_defaults.recursion_limit)),
        history_keep_messages=int(agent.get("history_keep_messages", agent_defaults.history_keep_messages)),
        use_llm=_bool(use_llm) if use_llm is not None else bool(agent.get("use_llm", True)),
        llm_confidence_threshold=float(agent.get("llm_confidence_threshold", agent_defaults.llm_confidence_threshold)),
    )
    checkpoint = env.get("CHECKPOINT_DB") or memory.get("checkpoint_db", "data/checkpoints.db")
    return Settings(
        model=model,
        hotel_api=hotel_api,
        flight_api=flight_api,
        agent=agent_settings,
        graph=GraphSettings(
            artifacts_dir=_project_path(graph.get("artifacts_dir", "artifacts")),
            remote_png=bool(graph.get("remote_png", True)),
        ),
        checkpoint_db=_project_path(checkpoint),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
