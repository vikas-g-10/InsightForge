from settings import Settings


def test_health_returns_expected_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "lite", "db_connected": False}


def test_default_mode_is_lite_when_env_unset():
    assert Settings.from_env({}).mode == "lite"


def test_configured_full_mode_is_reflected(make_client):
    client = make_client(Settings.from_env({"APP_MODE": "full"}))
    assert client.get("/health").json()["mode"] == "full"


def test_mode_is_case_and_whitespace_insensitive():
    assert Settings.from_env({"APP_MODE": " FULL "}).mode == "full"


def test_health_does_not_leak_environment(make_client, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "super-secret-value")
    body = make_client().get("/health").text
    assert "super-secret-value" not in body
    assert set(make_client().get("/health").json()) == {
        "status",
        "mode",
        "db_connected",
    }
