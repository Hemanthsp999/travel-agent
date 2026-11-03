"""Hotel API application factory. Run: `uvicorn mock_apis.hotel_api.app:app --port 8001`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from mock_apis.common.errors import register_error_handlers
from mock_apis.hotel_api.config import HotelSettings, load_settings
from mock_apis.hotel_api.routes import health_router, router
from mock_apis.hotel_api.seed import initialize_database


def create_app(db_path: str | Path | None = None, settings: HotelSettings | None = None) -> FastAPI:
    settings = settings or load_settings(db_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        initialize_database(settings)
        yield

    app = FastAPI(
        title="Hotel Mock API",
        version="1.0.0",
        description="Mock hotel booking backend: availability, booking and cancellation (SQLite).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(router)
    return app


app = create_app()
