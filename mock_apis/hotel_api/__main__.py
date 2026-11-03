"""`python -m mock_apis.hotel_api` starts the Hotel API on port 8001."""

import os

import uvicorn

from mock_apis.hotel_api.config import DEFAULT_PORT

if __name__ == "__main__":
    uvicorn.run(
        "mock_apis.hotel_api.app:app",
        host=os.getenv("HOTEL_HOST", "127.0.0.1"),
        port=int(os.getenv("HOTEL_PORT", DEFAULT_PORT)),
    )
