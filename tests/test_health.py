def test_health_check_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body


def test_docs_available(client):
    response = client.get("/docs")
    assert response.status_code == 200
