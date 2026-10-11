import re

import pytest


def _assert_envelope(response, status, code):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def _assert_no_leaks(text):
    assert "Traceback" not in text
    assert "File \"" not in text
    assert not re.search(r"\.py", text)
    assert "secret-value" not in text


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/nope")
    _assert_envelope(r, 404, "not_found")
    _assert_no_leaks(r.text)


def test_wrong_method_uses_error_envelope_and_keeps_allow_header(client):
    r = client.post("/health")
    _assert_envelope(r, 405, "method_not_allowed")
    assert "GET" in r.headers["allow"]


def test_validation_error_uses_envelope_without_echoing_input(make_client):
    def needs_int(n: int):
        return {"n": n}

    client = make_client(extra_routes=[("/_needs_int", needs_int)])
    r = client.get("/_needs_int", params={"n": "secret-value"})
    _assert_envelope(r, 422, "validation_error")
    _assert_no_leaks(r.text)


def test_unexpected_error_is_hidden_from_client_but_logged(
    make_client, monkeypatch, caplog
):
    monkeypatch.setenv("DB_PASSWORD", "secret-value")

    def boom():
        raise RuntimeError("db password is secret-value at /srv/app/db.py")

    client = make_client(extra_routes=[("/_boom", boom)])
    with caplog.at_level("ERROR", logger="insightforge"):
        r = client.get("/_boom")

    _assert_envelope(r, 500, "internal_error")
    _assert_no_leaks(r.text)
    assert "db password" not in r.text
    # Diagnostics stay server-side.
    assert any("/_boom" in rec.getMessage() for rec in caplog.records)
    assert any(rec.exc_info for rec in caplog.records)


@pytest.mark.parametrize("path", ["/health", "/nope"])
def test_responses_are_json(client, path):
    assert client.get(path).headers["content-type"].startswith("application/json")
