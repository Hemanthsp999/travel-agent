"""Flight cancellation, seat restoration, concurrency and persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.common.errors import APIError
from mock_apis.flight_api import service
from mock_apis.flight_api.app import create_app
from mock_apis.flight_api.config import FlightSettings
from mock_apis.flight_api.schemas import BookingCreate
from tests.flight_api.helpers import booking_payload, day, seats_on

BOOK = "/api/v1/bookings"


def book(client: TestClient, *args, **kwargs) -> dict:
    response = client.post(BOOK, json=booking_payload(*args, **kwargs))
    assert response.status_code == 201, response.text
    return response.json()["data"]


def test_cancel_updates_status_and_restores_seats(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    booking = book(client, tiny_flight, passengers=3)
    assert seats_on(settings, tiny_flight) == 0

    response = client.post(f"{BOOK}/{booking['booking_id']}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "CANCELLED"
    assert body["data"]["cancelled_at"]
    assert seats_on(settings, tiny_flight) == 3

    fetched = client.get(f"{BOOK}/{booking['booking_id']}").json()["data"]
    assert fetched["status"] == "CANCELLED"
    assert len(fetched["passengers"]) == 3  # history is kept


def test_repeated_cancellation_does_not_restore_seats_twice(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    first = book(client, tiny_flight, passengers=1)
    second = book(client, tiny_flight, passengers=1)  # keeps one seat held
    assert seats_on(settings, tiny_flight) == 1

    assert client.post(f"{BOOK}/{first['booking_id']}/cancel").status_code == 200
    assert seats_on(settings, tiny_flight) == 2

    for _ in range(3):
        again = client.post(f"{BOOK}/{first['booking_id']}/cancel")
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "BOOKING_ALREADY_CANCELLED"
    assert seats_on(settings, tiny_flight) == 2
    assert client.get(f"{BOOK}/{second['booking_id']}").json()["data"]["status"] == "CONFIRMED"


def test_cancelling_only_affects_the_booked_date(
    client: TestClient, settings: FlightSettings
) -> None:
    booking = book(client, "FL-1001", offset=30, passengers=2)
    other = book(client, "FL-1001", offset=31, passengers=1)
    client.post(f"{BOOK}/{booking['booking_id']}/cancel")
    assert seats_on(settings, "FL-1001", 30) == 180
    assert seats_on(settings, "FL-1001", 31) == 179
    assert client.get(f"{BOOK}/{other['booking_id']}").json()["data"]["status"] == "CONFIRMED"


def test_cancelled_seats_can_be_rebooked(client: TestClient, tiny_flight: str) -> None:
    booking = book(client, tiny_flight, passengers=3)
    assert client.post(BOOK, json=booking_payload(tiny_flight)).status_code == 409
    client.post(f"{BOOK}/{booking['booking_id']}/cancel")
    assert client.post(BOOK, json=booking_payload(tiny_flight, passengers=3)).status_code == 201


def test_cancel_unknown_booking_returns_404(client: TestClient) -> None:
    response = client.post(f"{BOOK}/FB-DOESNOTEXIST/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOOKING_NOT_FOUND"


def test_concurrent_bookings_never_oversell(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    request = BookingCreate(**booking_payload(tiny_flight, passengers=1))

    def attempt(_: int) -> str:
        conn = connect(settings.db_path)
        try:
            service.create_booking(conn, settings, request)
            return "ok"
        except APIError as exc:
            return exc.code
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))

    assert results.count("ok") == 3
    assert results.count("SEATS_UNAVAILABLE") == 5
    assert seats_on(settings, tiny_flight) == 0


def test_concurrent_cancellations_restore_once(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    booking = book(client, tiny_flight, passengers=2)
    book(client, tiny_flight, passengers=1)

    def attempt(_: int) -> str:
        conn = connect(settings.db_path)
        try:
            service.cancel_booking(conn, booking["booking_id"])
            return "ok"
        except APIError as exc:
            return exc.code
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(attempt, range(6)))

    assert results.count("ok") == 1
    assert seats_on(settings, tiny_flight) == 2


def test_data_persists_after_restart(client: TestClient, settings: FlightSettings) -> None:
    booking = book(client, "FL-1001", passengers=2)
    cancelled = book(client, "FL-1002", passengers=1)
    client.post(f"{BOOK}/{cancelled['booking_id']}/cancel")
    seats_after = seats_on(settings, "FL-1001")

    # A brand-new application instance on the same database file simulates a restart.
    with TestClient(create_app(settings=settings)) as restarted:
        fetched = restarted.get(f"{BOOK}/{booking['booking_id']}").json()["data"]
        assert fetched["status"] == "CONFIRMED"
        assert fetched["total_price"] == booking["total_price"]
        assert [p["name"] for p in fetched["passengers"]] == [p["name"] for p in booking["passengers"]]
        assert restarted.get(f"{BOOK}/{cancelled['booking_id']}").json()["data"]["status"] == "CANCELLED"
        # Re-seeding on startup must not reset inventory.
        assert seats_on(settings, "FL-1001") == seats_after
        flights = restarted.get(
            "/api/v1/flights/search",
            params={"origin": "Bangalore", "destination": "Mumbai", "departure_date": day(30).isoformat()},
        ).json()["data"]["flights"]
        assert next(f for f in flights if f["flight_id"] == "FL-1001")["seats_available"] == seats_after

        # ...and the restarted service still protects against double cancellation.
        assert restarted.post(f"{BOOK}/{cancelled['booking_id']}/cancel").status_code == 409
        assert seats_on(settings, "FL-1002") == 162


def test_idempotency_survives_restart(client: TestClient, settings: FlightSettings) -> None:
    headers = {"Idempotency-Key": "restart-key-01"}
    first = client.post(BOOK, json=booking_payload(), headers=headers).json()["data"]
    with TestClient(create_app(settings=settings)) as restarted:
        again = restarted.post(BOOK, json=booking_payload(), headers=headers)
        assert again.status_code == 200
        assert again.json()["data"]["booking_id"] == first["booking_id"]
