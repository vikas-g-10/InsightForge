"""Deterministic tests for Supabase Authentication, JWT verification, and user isolation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from auth import AuthenticatedUser, get_current_user, verify_token
from main import create_app
from settings import Settings

TEST_JWT_SECRET = "super-secret-test-jwt-key-32-chars-long!"
TEST_SUPABASE_URL = "https://testproj.supabase.co"


def create_test_token(
    sub: str = "user-1234-abcd",
    email: str = "user@example.com",
    role: str = "authenticated",
    aud: str = "authenticated",
    iss: str | None = f"{TEST_SUPABASE_URL}/auth/v1",
    exp_delta: int = 3600,
    secret: str = TEST_JWT_SECRET,
    alg: str = "HS256",
) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "email": email,
        "role": role,
        "aud": aud,
        "iat": now,
        "exp": now + exp_delta,
    }
    if iss is not None:
        payload["iss"] = iss
    return jwt.encode(payload, secret, algorithm=alg)


@pytest.fixture
def auth_settings():
    return Settings(
        supabase_url=TEST_SUPABASE_URL,
        supabase_jwt_secret=TEST_JWT_SECRET,
        supabase_jwt_audience="authenticated",
        auth_enabled=True,
    )


@pytest.fixture
def unauthenticated_client(auth_settings):
    """Client backed by real auth verification (no dependency overrides)."""
    app = create_app(auth_settings)
    # Clear any dependency overrides to ensure real verification runs
    app.dependency_overrides.clear()
    return TestClient(app)


# ==============================================================================
# 1. Token Verification Unit Tests
# ==============================================================================

def test_verify_valid_token_hs256(auth_settings):
    token = create_test_token(
        sub="usr-abc-123",
        email="alex@company.com",
        iss=f"{TEST_SUPABASE_URL}/auth/v1",
    )
    user = verify_token(token, auth_settings)
    assert user.user_id == "usr-abc-123"
    assert user.email == "alex@company.com"
    assert user.role == "authenticated"


def test_verify_expired_token(auth_settings):
    token = create_test_token(exp_delta=-300)
    with pytest.raises(Exception) as exc_info:
        verify_token(token, auth_settings)
    assert "expired" in str(exc_info.value).lower()


def test_verify_invalid_audience(auth_settings):
    token = create_test_token(aud="wrong-audience")
    with pytest.raises(Exception) as exc_info:
        verify_token(token, auth_settings)
    assert "audience" in str(exc_info.value).lower()


def test_verify_invalid_issuer(auth_settings):
    token = create_test_token(iss="https://malicious.issuer.com/auth/v1")
    with pytest.raises(Exception) as exc_info:
        verify_token(token, auth_settings)
    assert "issuer" in str(exc_info.value).lower()


def test_verify_invalid_signature(auth_settings):
    token = create_test_token(secret="wrong-secret-key-that-does-not-match!")
    with pytest.raises(Exception) as exc_info:
        verify_token(token, auth_settings)
    assert "signature" in str(exc_info.value).lower()


def test_verify_alg_none_rejected(auth_settings):
    payload = {"sub": "usr-1", "aud": "authenticated", "exp": int(time.time()) + 3600}
    # Craft token with alg "none"
    token = jwt.encode(payload, key="", algorithm="none")
    with pytest.raises(Exception) as exc_info:
        verify_token(token, auth_settings)
    assert "none" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()


# ==============================================================================
# 2. Endpoint Authentication Enforcement Tests
# ==============================================================================

def test_health_endpoint_public_without_token(unauthenticated_client):
    """Health check must remain accessible to deployment monitors without a token."""
    resp = unauthenticated_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_missing_bearer_token_rejected_401(unauthenticated_client):
    """Calling protected endpoint without Authorization header returns 401 unauthorized."""
    resp = unauthenticated_client.post(
        "/api/analyze",
        json={"reviews": [{"id": "1", "text": "Great device", "rating": 5}]},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "unauthorized"


def test_malformed_token_rejected_401(unauthenticated_client):
    resp = unauthenticated_client.post(
        "/api/analyze",
        headers={"Authorization": "Bearer not-a-valid-jwt-token"},
        json={"reviews": [{"id": "1", "text": "Great device", "rating": 5}]},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_expired_token_rejected_401(unauthenticated_client):
    expired = create_test_token(exp_delta=-600)
    resp = unauthenticated_client.post(
        "/api/analyze",
        headers={"Authorization": f"Bearer {expired}"},
        json={"reviews": [{"id": "1", "text": "Great device", "rating": 5}]},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_valid_token_accesses_protected_route(unauthenticated_client):
    valid_token = create_test_token(iss=f"{TEST_SUPABASE_URL}/auth/v1")
    resp = unauthenticated_client.post(
        "/api/analyze",
        headers={"Authorization": f"Bearer {valid_token}"},
        json={"reviews": [{"id": "1", "text": "Great battery life", "rating": 5}]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["reviews"]) == 1


def test_cors_preflight_options_bypasses_auth(unauthenticated_client):
    """CORS preflight (OPTIONS) requests must work without authentication."""
    resp = unauthenticated_client.options(
        "/api/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ==============================================================================
# 3. Comprehensive Inventory Check: ALL 12 /api/ Routes Protected
# ==============================================================================

@pytest.mark.parametrize(
    "method,path,kwargs",
    [
        ("POST", "/api/ingest/validate", {"data": {"text_column": "text"}}),
        ("POST", "/api/analyze", {"json": {"reviews": []}}),
        ("POST", "/api/analyze/file", {"data": {"mode": "lite"}}),
        ("POST", "/api/issues", {"json": {"reviews": []}}),
        ("POST", "/api/simulate", {"json": {"reviews": [], "fixes": {}}}),
        ("POST", "/api/emerging-issues", {"json": {"reviews": []}}),
        ("GET", "/api/vision/health", {}),
        ("POST", "/api/vision/analyze", {"data": {"cluster_context": "test"}}),
        ("POST", "/api/root-cause/hypotheses", {"json": {"reviews": []}}),
        ("POST", "/api/compare/targets", {"json": {"reviews": []}}),
        ("POST", "/api/compare/versions", {"json": {"version_a": "1", "version_b": "2"}}),
        ("POST", "/api/compare/products", {"json": {"product_a": "A", "product_b": "B"}}),
        ("GET", "/api/ml/metrics", {}),
        ("POST", "/api/ml/predict", {"json": {"text": "Great product"}}),
    ],
)
def test_all_protected_routes_reject_unauthenticated(
    unauthenticated_client, method, path, kwargs
):
    """Every protected endpoint must return 401 when called without Authorization."""
    if method == "POST":
        resp = unauthenticated_client.post(path, **kwargs)
    elif method == "GET":
        resp = unauthenticated_client.get(path, **kwargs)
    else:
        raise ValueError(f"Unexpected method: {method}")

    assert resp.status_code == 401, f"Expected 401 on {method} {path}, got {resp.status_code}"
    assert resp.json()["error"]["code"] == "unauthorized"


# ==============================================================================
# 4. Data Isolation & RLS Policy Verification
# ==============================================================================

def test_migration_rls_policies_file_exists():
    """Verify that the SQL migration exists and defines strict RLS policies."""
    from pathlib import Path

    sql_path = Path(__file__).resolve().parents[2] / "supabase" / "migrations" / "20261011_create_analysis_history.sql"
    assert sql_path.exists(), "Migration file must exist"
    content = sql_path.read_text(encoding="utf-8")

    assert "ENABLE ROW LEVEL SECURITY" in content
    assert "auth.uid() = user_id" in content
    assert "WITH CHECK (auth.uid() = user_id)" in content
    assert "REVOKE ALL ON public.analysis_history FROM anon" in content
    assert "CREATE POLICY \"Users can view own analysis history\"" in content
    assert "CREATE POLICY \"Users can insert own analysis history\"" in content
    assert "CREATE POLICY \"Users can update own analysis history\"" in content
    assert "CREATE POLICY \"Users can delete own analysis history\"" in content
