"""Hotel cancellation, inventory restoration, concurrency and persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.common.errors import APIError
from mock_apis.hotel_api import service
from mock_apis.hotel_api.app import create_app
from mock_apis.hotel_api.config import HotelSettings
from mock_apis.hotel_api.schemas import BookingCreate
from tests.hotel_api.helpers import booking_payload, day, rooms_by_date

BOOK = "/api/v1/bookings"


def book(client: TestClient, **kwargs) -> dict:
    response = client.post(BOOK, json=booking_payload(**kwargs))
    assert response.status_code == 201, response.text
    return response.json()["data"]


def test_cancel_updates_status_and_restores_inventory(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    booking = book(client, hotel_id=tiny_hotel, nights=3, rooms=2, guests=2)
    assert rooms_by_date(settings, tiny_hotel, 30, 4) == [0, 0, 0, 2]

    response = client.post(f"{BOOK}/{booking['booking_id']}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "CANCELLED"
    assert body["data"]["cancelled_at"]
    assert rooms_by_date(settings, tiny_hotel, 30, 4) == [2, 2, 2, 2]

    fetched = client.get(f"{BOOK}/{booking['booking_id']}").json()["data"]
    assert fetched["status"] == "CANCELLED"


def test_repeated_cancellation_does_not_restore_inventory_twice(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    first = book(client, hotel_id=tiny_hotel, rooms=1, guests=1)
    second = book(client, hotel_id=tiny_hotel, rooms=1, guests=1)  # keeps one room held
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [0, 0]

    assert client.post(f"{BOOK}/{first['booking_id']}/cancel").status_code == 200
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [1, 1]

    for _ in range(3):
        again = client.post(f"{BOOK}/{first['booking_id']}/cancel")
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "BOOKING_ALREADY_CANCELLED"
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [1, 1]
    assert client.get(f"{BOOK}/{second['booking_id']}").json()["data"]["status"] == "CONFIRMED"


def test_cancelled_rooms_can_be_rebooked(client: TestClient, tiny_hotel: str) -> None:
    booking = book(client, hotel_id=tiny_hotel, rooms=2, guests=2)
    assert client.post(BOOK, json=booking_payload(tiny_hotel)).status_code == 409
    client.post(f"{BOOK}/{booking['booking_id']}/cancel")
    assert client.post(BOOK, json=booking_payload(tiny_hotel, rooms=2, guests=2)).status_code == 201


def test_cancel_unknown_booking_returns_404(client: TestClient) -> None:
    response = client.post(f"{BOOK}/HB-DOESNOTEXIST/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOOKING_NOT_FOUND"


def test_concurrent_bookings_never_oversell(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    request = BookingCreate(**booking_payload(tiny_hotel, rooms=1, guests=1))

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

    assert results.count("ok") == 2
    assert results.count("ROOMS_UNAVAILABLE") == 6
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [0, 0]


def test_concurrent_cancellations_restore_once(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    booking = book(client, hotel_id=tiny_hotel, rooms=1, guests=1)
    book(client, hotel_id=tiny_hotel, rooms=1, guests=1)

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
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [1, 1]


def test_data_persists_after_restart(client: TestClient, settings: HotelSettings) -> None:
    booking = book(client, hotel_id="HT-1001", rooms=2, guests=2)
    cancelled = book(client, hotel_id="HT-1001", start=40, rooms=1, guests=1)
    client.post(f"{BOOK}/{cancelled['booking_id']}/cancel")
    after_booking = rooms_by_date(settings, "HT-1001", 30, 2)

    # A brand-new application instance on the same database file simulates a restart.
    with TestClient(create_app(settings=settings)) as restarted:
        fetched = restarted.get(f"{BOOK}/{booking['booking_id']}").json()["data"]
        assert fetched["status"] == "CONFIRMED"
        assert fetched["total_price"] == booking["total_price"]
        assert restarted.get(f"{BOOK}/{cancelled['booking_id']}").json()["data"]["status"] == "CANCELLED"
        # Re-seeding on startup must not reset inventory.
        assert rooms_by_date(settings, "HT-1001", 30, 2) == after_booking
        search = restarted.get(
            "/api/v1/hotels/search",
            params={"city": "Bangalore", "check_in": day(30).isoformat(), "check_out": day(32).isoformat()},
        ).json()["data"]["hotels"]
        grand = next(h for h in search if h["hotel_id"] == "HT-1001")
        assert grand["rooms_available"] == after_booking[0]

        # ...and the restarted service still protects against double cancellation.
        assert restarted.post(f"{BOOK}/{cancelled['booking_id']}/cancel").status_code == 409
        assert rooms_by_date(settings, "HT-1001", 30, 2) == after_booking


def test_idempotency_survives_restart(client: TestClient, settings: HotelSettings) -> None:
    headers = {"Idempotency-Key": "restart-key-01"}
    first = client.post(BOOK, json=booking_payload(), headers=headers).json()["data"]
    with TestClient(create_app(settings=settings)) as restarted:
        again = restarted.post(BOOK, json=booking_payload(), headers=headers)
        assert again.status_code == 200
        assert again.json()["data"]["booking_id"] == first["booking_id"]
