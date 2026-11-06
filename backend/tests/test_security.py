from app import main


def test_rate_limit_returns_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(main, "RATE_LIMIT_WINDOW_SECONDS", 60)
    main._request_times.clear()
    assert client.get("/documents").status_code == 200
    response = client.get("/documents")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    main._request_times.clear()
