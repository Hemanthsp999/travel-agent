"""Hotel availability search, seed data and health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mock_apis.common.db import connect
from mock_apis.hotel_api.config import HotelSettings
from tests.hotel_api.helpers import day

SEARCH = "/api/v1/hotels/search"


def search(client: TestClient, **overrides):
    params = {"city": "Bangalore", "check_in": day(30).isoformat(), "check_out": day(32).isoformat()}
    params.update(overrides)
    return client.get(SEARCH, params=params)


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["success"] is True
    assert body["data"] == {"service": "hotel-api", "status": "ok"}


def test_seed_data_covers_required_cities(client: TestClient, settings: HotelSettings) -> None:
    conn = connect(settings.db_path)
    try:
        cities = {r["city"] for r in conn.execute("SELECT city FROM hotels")}
        count = conn.execute("SELECT COUNT(*) AS n FROM hotels").fetchone()["n"]
    finally:
        conn.close()
    assert count >= 10
    assert cities == {"Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune"}


def test_search_returns_available_hotels_with_prices(client: TestClient) -> None:
    response = search(client, rooms=2, guests=4)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]
    assert data["count"] == len(data["hotels"]) >= 3
    assert data["nights"] == 2
    for hotel in data["hotels"]:
        assert hotel["city"] == "Bangalore"
        assert hotel["rooms_available"] >= 2
        assert hotel["estimated_total_price"] == hotel["price_per_night"] * 2 * 2
        assert {"name", "rating", "locality", "amenities"} <= hotel.keys()
    prices = [h["price_per_night"] for h in data["hotels"]]
    assert prices == sorted(prices)


def test_search_city_is_case_insensitive(client: TestClient) -> None:
    assert search(client, city="mUmBaI").json()["data"]["count"] >= 3


def test_search_unknown_city_returns_empty_list(client: TestClient) -> None:
    response = search(client, city="Atlantis")
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 0


def test_search_filters_by_rating_price_and_guest_capacity(client: TestClient) -> None:
    top = search(client, min_rating=4.4).json()["data"]["hotels"]
    assert top and all(h["rating"] >= 4.4 for h in top)
    cheap = search(client, max_price_per_night=4000).json()["data"]["hotels"]
    assert cheap and all(h["price_per_night"] <= 4000 for h in cheap)
    # one room fits at most 3 guests, so 4 guests in 1 room matches nothing
    assert search(client, guests=4, rooms=1).json()["data"]["count"] == 0


def test_search_excludes_hotels_without_enough_rooms(client: TestClient, tiny_hotel: str) -> None:
    response = search(client, city="Testville", rooms=3, guests=3)
    assert response.json()["data"]["count"] == 0
    assert search(client, city="Testville", rooms=2, guests=3).json()["data"]["count"] == 1


def test_search_rejects_checkout_before_checkin(client: TestClient) -> None:
    response = search(client, check_in=day(32).isoformat(), check_out=day(30).isoformat())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_rejects_past_dates(client: TestClient) -> None:
    response = search(client, check_in=day(-3).isoformat(), check_out=day(-1).isoformat())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DATE"


def test_search_rejects_missing_or_malformed_params(client: TestClient) -> None:
    assert client.get(SEARCH, params={"city": "Pune"}).status_code == 422
    assert search(client, check_in="not-a-date").status_code == 422
    assert search(client, rooms=0).status_code == 422
    assert search(client, unknown_param="x").status_code == 422


def test_get_hotel_details_and_unknown_hotel(client: TestClient) -> None:
    ok = client.get("/api/v1/hotels/HT-1001")
    assert ok.status_code == 200
    assert ok.json()["data"]["name"] == "Lakeview Grand Bengaluru"
    missing = client.get("/api/v1/hotels/HT-0000")
    assert missing.status_code == 404
    assert missing.json() == {
        "success": False,
        "error": {"code": "HOTEL_NOT_FOUND", "message": "Hotel 'HT-0000' does not exist.", "details": None},
    }


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
