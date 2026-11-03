"""Flight booking: success, validation, insufficient seats, idempotency."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest
from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.flight_api import service
from mock_apis.flight_api.config import FlightSettings
from tests.flight_api.helpers import booking_payload, passenger, seats_on

BOOK = "/api/v1/bookings"


def count_rows(settings: FlightSettings, table: str) -> int:
    conn = connect(settings.db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    finally:
        conn.close()


def test_booking_success_stores_passengers_and_updates_seats(
    client: TestClient, settings: FlightSettings
) -> None:
    before = seats_on(settings, "FL-1001")
    response = client.post(BOOK, json=booking_payload("FL-1001", passengers=3))
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True and "booked" in body["message"].lower()
    data = body["data"]
    assert data["booking_id"].startswith("FB-")
    assert data["status"] == "CONFIRMED"
    assert data["flight"]["flight_number"] == "6E-501"
    assert data["num_passengers"] == 3 == len(data["passengers"])
    assert data["total_price"] == 4800 * 3  # computed by the server
    assert seats_on(settings, "FL-1001") == before - 3
    assert count_rows(settings, "passengers") == 3

    fetched = client.get(f"{BOOK}/{data['booking_id']}").json()["data"]
    assert fetched["passengers"] == data["passengers"]


def test_seats_are_tracked_per_date(client: TestClient, settings: FlightSettings) -> None:
    client.post(BOOK, json=booking_payload("FL-1001", offset=30, passengers=2))
    assert seats_on(settings, "FL-1001", 30) == 178
    assert seats_on(settings, "FL-1001", 31) == 180


def test_client_supplied_price_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(total_price=1))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_booking_validation_errors(client: TestClient) -> None:
    def with_passengers(passengers: list[dict]) -> dict:
        payload = booking_payload()
        payload["passengers"] = passengers
        return payload

    bad_email = booking_payload()
    bad_email["contact"]["email"] = "nope"
    cases = [
        booking_payload(passengers=0),  # no passengers
        booking_payload(passengers=10),  # above the per-booking limit
        with_passengers([passenger(age=-1)]),
        with_passengers([passenger(gender="ROBOT")]),
        with_passengers([passenger(name="A")]),
        booking_payload(departure_date="soon"),
        bad_email,
    ]
    for payload in cases:
        response = client.post(BOOK, json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.post(BOOK, json={}).status_code == 422


def test_booking_unknown_flight(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload("FL-0000"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "FLIGHT_NOT_FOUND"


def test_booking_past_and_far_future_dates_rejected(client: TestClient) -> None:
    for offset in (-2, 500):
        response = client.post(BOOK, json=booking_payload(offset=offset))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_DATE"


def test_booking_departed_flight_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_now", lambda: datetime.combine(date.today(), time(23, 0)))
    response = client.post(BOOK, json=booking_payload(offset=0))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "FLIGHT_DEPARTED"


def test_insufficient_seats_never_confirmed(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    ok = client.post(BOOK, json=booking_payload(tiny_flight, passengers=2))
    assert ok.status_code == 201
    too_many = client.post(BOOK, json=booking_payload(tiny_flight, passengers=2))
    assert too_many.status_code == 409
    error = too_many.json()["error"]
    assert error["code"] == "SEATS_UNAVAILABLE"
    assert error["details"] == {"seats_requested": 2, "seats_available": 1}
    assert count_rows(settings, "bookings") == 1
    assert count_rows(settings, "passengers") == 2
    assert seats_on(settings, tiny_flight) == 1


def test_last_seat_can_be_booked_then_flight_is_sold_out(
    client: TestClient, settings: FlightSettings, tiny_flight: str
) -> None:
    assert client.post(BOOK, json=booking_payload(tiny_flight, passengers=3)).status_code == 201
    assert seats_on(settings, tiny_flight) == 0
    assert client.post(BOOK, json=booking_payload(tiny_flight, passengers=1)).status_code == 409


def test_idempotent_retry_returns_same_booking_and_books_once(
    client: TestClient, settings: FlightSettings
) -> None:
    headers = {"Idempotency-Key": "retry-key-0001"}
    payload = booking_payload("FL-1001", passengers=2)
    first = client.post(BOOK, json=payload, headers=headers)
    second = client.post(BOOK, json=payload, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["data"]["booking_id"] == first.json()["data"]["booking_id"]
    assert count_rows(settings, "bookings") == 1
    assert count_rows(settings, "passengers") == 2
    assert seats_on(settings, "FL-1001") == 178


def test_idempotency_key_reused_with_different_request_conflicts(
    client: TestClient, settings: FlightSettings
) -> None:
    headers = {"Idempotency-Key": "retry-key-0002"}
    assert client.post(BOOK, json=booking_payload(passengers=1), headers=headers).status_code == 201
    response = client.post(BOOK, json=booking_payload(passengers=2), headers=headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"
    assert count_rows(settings, "bookings") == 1


def test_booking_without_key_is_not_deduplicated(client: TestClient, settings: FlightSettings) -> None:
    client.post(BOOK, json=booking_payload())
    client.post(BOOK, json=booking_payload())
    assert count_rows(settings, "bookings") == 2


def test_idempotency_key_too_short_is_rejected(client: TestClient) -> None:
    response = client.post(BOOK, json=booking_payload(), headers={"Idempotency-Key": "x"})
    assert response.status_code == 422


def test_unknown_booking_id_returns_404(client: TestClient) -> None:
    response = client.get(f"{BOOK}/FB-DOESNOTEXIST")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "BOOKING_NOT_FOUND"
