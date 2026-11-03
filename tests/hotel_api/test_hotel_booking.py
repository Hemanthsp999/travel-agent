"""Hotel booking: success, validation, sold-out, overlapping stays, idempotency."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.hotel_api.config import HotelSettings
from tests.hotel_api.helpers import booking_payload, day, rooms_by_date

BOOK = "/api/v1/bookings"


def count_bookings(settings: HotelSettings) -> int:
    conn = connect(settings.db_path)
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM bookings").fetchone()["n"]
    finally:
        conn.close()


def test_booking_success_returns_details_and_updates_inventory(
    client: TestClient, settings: HotelSettings
) -> None:
    before = rooms_by_date(settings, "HT-1001", 30, 3)
    response = client.post(BOOK, json=booking_payload("HT-1001", start=30, nights=2, rooms=2, guests=4))
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True and "booked" in body["message"].lower()
    data = body["data"]
    assert data["booking_id"].startswith("HB-")
    assert data["status"] == "CONFIRMED"
    assert data["hotel"]["name"] == "Lakeview Grand Bengaluru"
    assert data["nights"] == 2
    assert data["total_price"] == 9500 * 2 * 2  # computed by the server
    assert rooms_by_date(settings, "HT-1001", 30, 3) == [before[0] - 2, before[1] - 2, before[2]]

    fetched = client.get(f"{BOOK}/{data['booking_id']}").json()["data"]
    assert fetched["booking_id"] == data["booking_id"]


def test_client_supplied_price_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(total_price=1))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_booking_validation_errors(client: TestClient) -> None:
    bad_email = booking_payload()
    bad_email["guest"]["email"] = "nope"
    cases = [
        booking_payload(check_out=day(30).isoformat()),  # zero nights
        booking_payload(check_in=day(35).isoformat(), check_out=day(30).isoformat()),
        booking_payload(rooms=0),
        booking_payload(guests=0),
        bad_email,
    ]
    for payload in cases:
        response = client.post(BOOK, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.post(BOOK, json={}).status_code == 422


def test_booking_in_the_past_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(start=-5))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DATE"


def test_booking_beyond_booking_window_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(start=500))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DATE"


def test_booking_unknown_hotel(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload("HT-0000"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HOTEL_NOT_FOUND"


def test_booking_over_guest_capacity(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload("HT-1003", guests=3))  # 2 guests per room
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CAPACITY_EXCEEDED"


def test_booking_unavailable_rooms_never_confirmed(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    first = client.post(BOOK, json=booking_payload(tiny_hotel, rooms=2, guests=2))
    assert first.status_code == 201
    second = client.post(BOOK, json=booking_payload(tiny_hotel, rooms=1, guests=1))
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "ROOMS_UNAVAILABLE"
    assert error["details"]["rooms_available"] == 0
    assert count_bookings(settings) == 1
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [0, 0]


def test_booking_more_rooms_than_exist_fails_without_changes(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    response = client.post(BOOK, json=booking_payload(tiny_hotel, rooms=3, guests=3))
    assert response.status_code == 409
    assert rooms_by_date(settings, tiny_hotel, 30, 2) == [2, 2]


def test_overlapping_reservations_are_tracked_per_date(
    client: TestClient, settings: HotelSettings, tiny_hotel: str
) -> None:
    # Hotel has 2 rooms. A: nights 30-32, B: nights 32-34 -> night 32 is full.
    assert client.post(BOOK, json=booking_payload(tiny_hotel, start=30, nights=3)).status_code == 201
    assert client.post(BOOK, json=booking_payload(tiny_hotel, start=32, nights=3)).status_code == 201
    assert rooms_by_date(settings, tiny_hotel, 30, 6) == [1, 1, 0, 1, 1, 2]

    # A request that touches the full night must fail and leave inventory untouched.
    blocked = client.post(BOOK, json=booking_payload(tiny_hotel, start=31, nights=2, rooms=1, guests=1))
    assert blocked.status_code == 409
    assert blocked.json()["error"]["code"] == "ROOMS_UNAVAILABLE"
    assert rooms_by_date(settings, tiny_hotel, 30, 6) == [1, 1, 0, 1, 1, 2]

    # Non-overlapping stays (the night before, and a free later night) still succeed.
    adjacent = client.post(BOOK, json=booking_payload(tiny_hotel, start=29, nights=1, rooms=2, guests=2))
    assert adjacent.status_code == 201
    free = client.post(BOOK, json=booking_payload(tiny_hotel, start=33, nights=1, rooms=1, guests=1))
    assert free.status_code == 201
    assert rooms_by_date(settings, tiny_hotel, 30, 6) == [1, 1, 0, 0, 1, 2]


def test_idempotent_retry_returns_same_booking_and_books_once(
    client: TestClient, settings: HotelSettings
) -> None:
    headers = {"Idempotency-Key": "retry-key-0001"}
    payload = booking_payload("HT-1001", rooms=1)
    before = rooms_by_date(settings, "HT-1001", 30, 2)

    first = client.post(BOOK, json=payload, headers=headers)
    second = client.post(BOOK, json=payload, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["data"]["booking_id"] == first.json()["data"]["booking_id"]
    assert count_bookings(settings) == 1
    assert rooms_by_date(settings, "HT-1001", 30, 2) == [before[0] - 1, before[1] - 1]


def test_idempotency_key_reused_with_different_request_conflicts(
    client: TestClient, settings: HotelSettings
) -> None:
    headers = {"Idempotency-Key": "retry-key-0002"}
    assert client.post(BOOK, json=booking_payload(rooms=1), headers=headers).status_code == 201
    response = client.post(BOOK, json=booking_payload(rooms=2), headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"
    assert count_bookings(settings) == 1


def test_booking_without_key_is_not_deduplicated(client: TestClient, settings: HotelSettings) -> None:
    client.post(BOOK, json=booking_payload())
    client.post(BOOK, json=booking_payload())
    assert count_bookings(settings) == 2


def test_idempotency_key_too_short_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(), headers={"Idempotency-Key": "x"})
    assert response.status_code == 422


def test_unknown_booking_id_returns_404(client: TestClient) -> None:
    response = client.get(f"{BOOK}/HB-DOESNOTEXIST")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOOKING_NOT_FOUND"
