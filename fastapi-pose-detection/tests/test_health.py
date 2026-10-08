from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_points_to_docs():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "pose-detection-api"
    assert body["docs"] == "/docs"
