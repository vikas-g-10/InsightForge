from settings import Settings

ALLOWED = "https://app.example.com"
BLOCKED = "https://evil.example.com"


def _client(make_client):
    return make_client(Settings.from_env({"ALLOWED_ORIGINS": ALLOWED}))


def test_allowed_origin_gets_cors_header(make_client):
    r = _client(make_client).get("/health", headers={"Origin": ALLOWED})
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_disallowed_origin_gets_no_cors_header(make_client):
    r = _client(make_client).get("/health", headers={"Origin": BLOCKED})
    assert "access-control-allow-origin" not in r.headers


def test_preflight_allowed_origin(make_client):
    r = _client(make_client).options(
        "/health",
        headers={"Origin": ALLOWED, "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_preflight_allows_standard_request_headers(make_client):
    r = _client(make_client).options(
        "/api/analyze",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type, accept",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ALLOWED


def test_preflight_disallowed_origin_is_rejected(make_client):
    r = _client(make_client).options(
        "/health",
        headers={"Origin": BLOCKED, "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_default_settings_allow_vite_dev_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_wildcard_and_credentials_never_sent(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] != "*"
    assert "access-control-allow-credentials" not in r.headers
