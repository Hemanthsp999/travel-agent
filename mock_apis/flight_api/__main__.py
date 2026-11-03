"""`python -m mock_apis.flight_api` starts the Flight API on port 8002."""

import os

import uvicorn

from mock_apis.flight_api.config import DEFAULT_PORT

if __name__ == "__main__":
    uvicorn.run(
        "mock_apis.flight_api.app:app",
        host=os.getenv("FLIGHT_HOST", "127.0.0.1"),
        port=int(os.getenv("FLIGHT_PORT", DEFAULT_PORT)),
    )
