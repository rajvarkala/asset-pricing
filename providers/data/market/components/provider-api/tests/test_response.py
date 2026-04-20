from fastapi.testclient import TestClient

from provider_api.main import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_invalid_interval() -> None:
    client = TestClient(app)
    resp = client.get(
        "/instruments/historical/123/minute",
        params={"from": "2025-01-01", "to": "2025-01-31"},
    )
    assert resp.status_code == 400
    assert resp.json()["status"] == "error"
