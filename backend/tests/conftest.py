import pytest
from fastapi.testclient import TestClient

from auth import AuthenticatedUser, get_current_user
from main import app as global_app, create_app
from settings import Settings

ORIGIN = "http://localhost:5173"

MOCK_USER = AuthenticatedUser(
    user_id="00000000-0000-0000-0000-000000000001",
    email="testuser@example.com",
    role="authenticated",
    claims={
        "sub": "00000000-0000-0000-0000-000000000001",
        "email": "testuser@example.com",
        "role": "authenticated",
    },
)


@pytest.fixture(autouse=True)
def override_auth_default():
    """Default dependency override allowing existing functional pipeline tests to run authenticated."""
    global_app.dependency_overrides[get_current_user] = lambda: MOCK_USER
    yield
    global_app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def make_client():
    """Build an isolated app + client per call; no global state is touched."""

    def _make(settings: Settings | None = None, extra_routes=()) -> TestClient:
        app = create_app(settings or Settings())
        for path, endpoint in extra_routes:
            app.add_api_route(path, endpoint)
        app.dependency_overrides[get_current_user] = lambda: MOCK_USER
        # Return 500 responses instead of re-raising server exceptions.
        return TestClient(app, raise_server_exceptions=False)

    return _make


@pytest.fixture
def client(make_client):
    return make_client()
