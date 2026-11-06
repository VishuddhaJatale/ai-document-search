from app import main


def test_api_key_is_required_when_enabled(client, monkeypatch):
    monkeypatch.setattr(main, "REQUIRE_API_KEY", True)
    monkeypatch.setattr(main, "API_KEY", "test-secret")
    assert client.get("/documents").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_rate_limit_returns_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "REQUIRE_API_KEY", False)
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_SECONDS", 60)
    main._request_times.clear()
    assert client.get("/documents").status_code == 200
    response = client.get("/documents")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    main._request_times.clear()
