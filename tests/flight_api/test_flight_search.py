"""Flight search, seed data and health."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest
from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.flight_api import service
from mock_apis.flight_api.config import FlightSettings
from tests.flight_api.helpers import day

SEARCH = "/api/v1/flights/search"


def search(client: TestClient, **overrides):
    params = {"origin": "Bangalore", "destination": "Mumbai", "departure_date": day(30).isoformat()}
    params.update(overrides)
    return client.get(SEARCH, params=params)


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["success"] is True
    assert body["data"] == {"service": "flight-api", "status": "ok"}


def test_seed_data_covers_required_routes(client: TestClient, settings: FlightSettings) -> None:
    conn = connect(settings.db_path)
    try:
        routes = {(r["origin_city"], r["destination_city"]) for r in conn.execute("SELECT * FROM flights")}
        count = conn.execute("SELECT COUNT(*) AS n FROM flights").fetchone()["n"]
        airlines = conn.execute("SELECT COUNT(DISTINCT airline) AS n FROM flights").fetchone()["n"]
    finally:
        conn.close()
    assert count >= 15
    assert airlines >= 3
    required = {
        ("Bangalore", "Mumbai"),
        ("Bangalore", "Delhi"),
        ("Delhi", "Mumbai"),
        ("Mumbai", "Chennai"),
        ("Hyderabad", "Bangalore"),
        ("Chennai", "Delhi"),
    }
    assert required <= routes


def test_search_returns_matching_flights_with_details(client: TestClient) -> None:
    response = search(client, passengers=2)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["count"] == len(data["flights"]) == 3
    for flight in data["flights"]:
        assert flight["origin"] == {"city": "Bangalore", "code": "BLR"}
        assert flight["destination"]["code"] == "BOM"
        assert flight["airline"] and flight["flight_number"]
        assert flight["total_price"] == flight["price_per_passenger"] * 2
        assert flight["seats_available"] >= 2
    times = [f["departure_time"] for f in data["flights"]]
    assert times == sorted(times)


def test_search_computes_arrival_time(client: TestClient) -> None:
    flight = next(f for f in search(client).json()["data"]["flights"] if f["flight_id"] == "FL-1001")
    departure = datetime.fromisoformat(flight["departure_time"])
    arrival = datetime.fromisoformat(flight["arrival_time"])
    assert (departure.hour, departure.minute) == (6, 15)
    assert (arrival - departure).total_seconds() == 105 * 60


def test_search_accepts_iata_codes_and_any_case(client: TestClient) -> None:
    assert search(client, origin="blr", destination="BOM").json()["data"]["count"] == 3
    assert search(client, origin="BANGALORE", destination="mumbai").json()["data"]["count"] == 3


def test_search_one_way_only_matches_requested_direction(client: TestClient) -> None:
    assert search(client, origin="Mumbai", destination="Bangalore").json()["data"]["count"] == 1


def test_search_with_no_matching_route_returns_empty_list(client: TestClient) -> None:
    response = search(client, origin="Pune", destination="Goa")
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 0


def test_search_excludes_flights_with_too_few_seats(client: TestClient, tiny_flight: str) -> None:
    route = {"origin": "Testville", "destination": "Nowhereburg"}
    assert search(client, passengers=3, **route).json()["data"]["count"] == 1
    assert search(client, passengers=4, **route).json()["data"]["count"] == 0


def test_search_hides_flights_that_already_departed_today(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_now", lambda: datetime.combine(date.today(), time(12, 0)))
    flights = search(client, departure_date=date.today().isoformat()).json()["data"]["flights"]
    # Of the Bangalore -> Mumbai departures (06:15, 11:30, 19:45) only 19:45 is still ahead at noon.
    assert [f["flight_number"] for f in flights] == ["QP-1303"]


def test_search_validation_errors(client: TestClient) -> None:
    same = search(client, origin="Delhi", destination="delhi")
    assert same.status_code == 422 and same.json()["error"]["code"] == "VALIDATION_ERROR"
    assert search(client, passengers=0).status_code == 422
    assert search(client, passengers=10).status_code == 422
    assert search(client, departure_date="tomorrow").status_code == 422
    assert client.get(SEARCH, params={"origin": "Delhi"}).status_code == 422
    assert search(client, cabin="business").status_code == 422


def test_search_rejects_past_and_far_future_dates(client: TestClient) -> None:
    past = search(client, departure_date=day(-1).isoformat())
    assert past.status_code == 422 and past.json()["error"]["code"] == "INVALID_DATE"
    far = search(client, departure_date=day(400).isoformat())
    assert far.status_code == 422 and far.json()["error"]["code"] == "INVALID_DATE"


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
