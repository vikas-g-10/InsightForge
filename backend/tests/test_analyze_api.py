"""HTTP-level API tests for POST /api/analyze and POST /api/analyze/file."""

import pytest
from fastapi.testclient import TestClient

from helpers import BASIC_CSV, upload, xlsx_bytes
from main import app, create_app
from settings import Settings


@pytest.fixture
def client():
    return TestClient(app)


def test_analyze_valid_json_batch(client):
    payload = {
        "reviews": [
            {"id": "rev-1", "text": "The camera is fantastic and battery is great", "rating": 5.0},
            {"id": "rev-2", "text": "Terrible delivery and damaged box", "rating": 1.0},
        ],
        "mode": "lite",
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "reviews" in data
    assert "meta" in data
    assert len(data["reviews"]) == 2

    r1, r2 = data["reviews"][0], data["reviews"][1]
    assert r1["review_id"] == "rev-1"
    assert r1["polarity"] == "pos"
    assert r1["rating"] == 5.0
    assert r1["safety"]["tier"] == "none"

    assert r2["review_id"] == "rev-2"
    assert r2["polarity"] == "neg"
    assert r2["rating"] == 1.0
    assert "delivery" in r2["aspects"] or "packaging" in r2["aspects"]

    # Verify summary metadata
    meta = data["meta"]
    assert meta["mode"] == "lite"
    assert meta["summary"]["total_reviews"] == 2
    assert meta["summary"]["analyzed_reviews"] == 2
    assert meta["summary"]["total_clauses"] >= 2


def test_analyze_empty_batch_returns_400(client):
    resp = client.post("/api/analyze", json={"reviews": []})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "empty_batch"
    assert "empty" in data["error"]["message"].lower()


def test_analyze_missing_body_returns_422(client):
    resp = client.post("/api/analyze", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["code"] == "validation_error"


def test_analyze_oversized_batch_returns_413(make_client):
    low_limit_client = make_client(Settings(max_rows=5))
    oversized = [{"id": str(i), "text": "Good phone", "rating": 5} for i in range(10)]
    resp = low_limit_client.post("/api/analyze", json={"reviews": oversized})
    assert resp.status_code == 413
    data = resp.json()
    assert data["error"]["code"] == "payload_too_large"


def test_analyze_safety_and_sarcasm_flags(client):
    payload = {
        "reviews": [
            {"id": "safe-1", "text": "The phone battery caught fire and exploded", "rating": 1},
            {"id": "sarc-1", "text": "Great update, stopped working immediately", "rating": 1},
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    reviews = resp.json()["reviews"]

    # Safety alert verification
    fire_rev = reviews[0]
    assert fire_rev["safety"]["tier"] == "critical"
    assert fire_rev["safety"]["is_alert"] is True
    assert len(fire_rev["safety"]["matches"]) >= 1

    # Sarcasm flag verification
    sarc_rev = reviews[1]
    assert sarc_rev["sarcasm_suspect"] is True
    assert sarc_rev["polarity"] == "neg"


def test_analyze_rating_conflict_flag(client):
    payload = {
        "reviews": [
            {"id": "conf-1", "text": "This phone is brilliant, superb and amazing!", "rating": 1},
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    rev = resp.json()["reviews"][0]
    assert rev["rating_conflict"] is True


def test_analyze_multilingual_reviews(client):
    payload = {
        "reviews": [
            {"id": "hi-1", "text": "ऐप बहुत अच्छा है", "rating": 5},
            {"id": "hing-1", "text": "Yeh app bahut accha hai aur fast hai", "rating": 5},
            {"id": "kan-1", "text": "camera tumba chennagide", "rating": 5},
            {"id": "tam-1", "text": "phone romba waste", "rating": 1},
            {"id": "tel-1", "text": "camera chala bagundi", "rating": 5},
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    by_id = {r["review_id"]: r for r in resp.json()["reviews"]}

    assert by_id["hi-1"]["language"] == "hi"
    assert by_id["hi-1"]["language_badge"] == "Hindi"

    assert by_id["hing-1"]["language"] == "hinglish"
    assert by_id["hing-1"]["language_badge"] == "Hinglish"

    assert by_id["kan-1"]["language"] == "kanglish"
    assert by_id["kan-1"]["language_badge"] == "Kanglish"

    assert by_id["tam-1"]["language"] == "tanglish"
    assert by_id["tam-1"]["language_badge"] == "Tanglish"

    assert by_id["tel-1"]["language"] == "tenglish"
    assert by_id["tel-1"]["language_badge"] == "Tenglish"


def test_analyze_file_upload_csv(client):
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("reviews.csv", BASIC_CSV.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["reviews"]) == 2
    assert data["meta"]["summary"]["total_reviews"] == 2


def test_analyze_file_upload_xlsx(client):
    xlsx_data = xlsx_bytes([
        ["Review", "Rating"],
        ["Great camera and display", 5],
        ["Battery drains fast", 2],
    ])
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("reviews.xlsx", xlsx_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["reviews"]) == 2
    assert data["meta"]["summary"]["total_reviews"] == 2


def test_analyze_file_unsupported_type(client):
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("reviews.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_analyze_file_empty(client):
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("reviews.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_file"


def test_cors_headers_on_analyze_endpoint(client):
    resp = client.post(
        "/api/analyze",
        json={"reviews": [{"text": "Good app", "rating": 5}]},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --- Phase 1.1 API Correctness & PRD Compliance Tests ---


def test_analyze_api_pii_redacted_in_json(client):
    payload = {
        "reviews": [
            {
                "id": "rev-pii",
                "text": "Please reach me at support@insightforge.dev or call +1 555-123-4567. Great display!",
                "rating": 5,
            }
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    rev = data["reviews"][0]

    # Email and phone must never appear unredacted in returned JSON
    assert "support@insightforge.dev" not in rev["original_text"]
    assert "+1 555-123-4567" not in rev["original_text"]
    assert "[email]" in rev["original_text"]
    assert "[phone]" in rev["original_text"]

    assert "support@insightforge.dev" not in rev["normalized_text"]
    assert "+1 555-123-4567" not in rev["normalized_text"]
    assert "[email]" in rev["normalized_text"]
    assert "[phone]" in rev["normalized_text"]

    for c in rev["clauses"]:
        assert "support@insightforge.dev" not in c["text"]
        assert "+1 555-123-4567" not in c["text"]


def test_analyze_api_rejects_unsupported_full_mode(client):
    payload = {
        "reviews": [{"id": "r1", "text": "Good camera", "rating": 5}],
        "mode": "full",
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "unsupported_mode"
    assert "full mode" in err["message"].lower()


def test_analyze_api_missing_rating_provides_pseudo(client):
    payload = {
        "reviews": [
            {"id": "r1", "text": "The camera is fantastic and battery lasts long", "rating": None}
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    rev = resp.json()["reviews"][0]
    assert rev["rating"] is None
    assert rev["rating_source"] == "pseudo"
    assert rev["pseudo_rating"] is not None
    assert rev["pseudo_rating"] >= 3.5
    assert rev["rating_conflict"] is False


def test_analyze_file_pii_redacted_csv(client):
    csv_content = "Review,Rating\nMy email is alert@hazard.com and phone is 9988776655. Battery caught fire!,1\n"
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("safety_pii.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert resp.status_code == 200
    rev = resp.json()["reviews"][0]

    assert "alert@hazard.com" not in rev["original_text"]
    assert "9988776655" not in rev["original_text"]
    assert "[email]" in rev["original_text"]
    assert "[phone]" in rev["original_text"]

    for m in rev["safety"]["matches"]:
        assert "alert@hazard.com" not in m["excerpt"]
        assert "9988776655" not in m["excerpt"]


def test_analyze_file_duplicates_tracked(client):
    csv_content = "Review,Rating\nGreat performance and screen,5\nGreat performance and screen,5\n"
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("dup.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["reviews"]) == 2
    r1, r2 = data["reviews"][0], data["reviews"][1]

    assert r1["is_duplicate"] is False
    assert r2["is_duplicate"] is True
    assert r2["duplicate_of"] == r1["review_id"]
    assert data["meta"]["summary"]["duplicate_count"] == 1
    assert data["meta"]["summary"]["analyzed_reviews"] == 1


def test_analyze_file_rejects_unsupported_full_mode(client):
    csv_content = "Review,Rating\nGood phone,5\n"
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")},
        data={"mode": "full"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_mode"


def test_validation_error_does_not_leak_submitted_pii(client):
    # Invalid JSON payload structure with sensitive text
    resp = client.post(
        "/api/analyze",
        json={"reviews": "secret-pii-phone-1234567890"},
    )
    assert resp.status_code == 422
    err_body = resp.text
    # Ensure error response does not echo user input or stack traces
    assert "1234567890" not in err_body
    assert "traceback" not in err_body.lower()


def test_analyze_file_pii_and_duplicates_xlsx(client):
    xlsx_data = xlsx_bytes([
        ["Review", "Rating"],
        ["Reach test.author@domain.com or 9876543210 for battery replacement", 1],
        ["Reach test.author@domain.com or 9876543210 for battery replacement", 1],
    ])
    resp = client.post(
        "/api/analyze/file",
        files={"file": ("reviews.xlsx", xlsx_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["reviews"]) == 2
    r1, r2 = data["reviews"][0], data["reviews"][1]

    # Redaction verified in XLSX
    assert "test.author@domain.com" not in r1["original_text"]
    assert "9876543210" not in r1["original_text"]
    assert "[email]" in r1["original_text"]
    assert "[phone]" in r1["original_text"]

    # Duplicate tracking verified in XLSX
    assert r1["is_duplicate"] is False
    assert r2["is_duplicate"] is True
    assert r2["duplicate_of"] == r1["review_id"]
    assert data["meta"]["summary"]["duplicate_count"] == 1
    assert data["meta"]["summary"]["analyzed_reviews"] == 1


def test_analyze_api_duplicate_rules_metadata(client):
    payload = {
        "reviews": [
            {"id": "rev-1", "text": "Great screen and snappy app", "rating": 5, "version": "1.0"},
            {"id": "rev-2", "text": "Great screen and snappy app", "rating": 1, "version": "1.0"},  # Diff rating
            {"id": "rev-3", "text": "Great screen and snappy app", "rating": 5, "version": "2.0"},  # Diff version
            {"id": "rev-4", "text": "Great screen and snappy app", "rating": 5, "version": "1.0"},  # Exact duplicate of rev-1
        ]
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    reviews = data["reviews"]

    assert reviews[0]["is_duplicate"] is False
    assert reviews[1]["is_duplicate"] is False
    assert reviews[2]["is_duplicate"] is False
    assert reviews[3]["is_duplicate"] is True
    assert reviews[3]["duplicate_of"] == "rev-1"
    assert data["meta"]["summary"]["duplicate_count"] == 1
    assert data["meta"]["summary"]["analyzed_reviews"] == 3
