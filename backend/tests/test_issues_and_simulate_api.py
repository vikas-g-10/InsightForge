"""HTTP-level API tests for POST /api/issues and POST /api/simulate."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_issues_api_valid_batch(client):
    payload = {
        "reviews": [
            {"id": "rev-1", "text": "The battery drains immediately, terrible battery", "rating": 1.0},
            {"id": "rev-2", "text": "Camera quality is magnificent", "rating": 5.0},
        ],
        "mode": "lite",
    }
    resp = client.post("/api/issues", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "clusters" in data
    assert "insights" in data
    assert data["total_clusters"] >= 1
    assert len(data["clusters"]) >= 1

    # Check evidence traceability in API output
    first_cluster = data["clusters"][0]
    assert "cluster_id" in first_cluster
    assert "aspect" in first_cluster
    assert "polarity" in first_cluster
    assert "example_quote" in first_cluster
    assert len(first_cluster["evidence"]) >= 1
    evidence = first_cluster["evidence"][0]
    assert "review_id" in evidence
    assert "clause_id" in evidence
    assert "text" in evidence


def test_issues_api_empty_batch_returns_400(client):
    resp = client.post("/api/issues", json={"reviews": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_batch"


def test_issues_api_rejects_unsupported_full_mode(client):
    payload = {
        "reviews": [{"id": "r1", "text": "Good screen", "rating": 5}],
        "mode": "full",
    }
    resp = client.post("/api/issues", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_mode"


def test_simulate_api_valid_request(client):
    payload = {
        "fixes": {"battery": 0.5, "delivery": 1.0},
        "reviews": [
            {"id": "r1", "text": "Battery drains fast", "rating": 1.0},
            {"id": "r2", "text": "Battery drains fast", "rating": 2.0},
            {"id": "r3", "text": "Delivery was late", "rating": 2.0},
            {"id": "r4", "text": "Great screen and fast app", "rating": 5.0},
        ],
    }
    resp = client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "current_rating" in data
    assert "projected_rating" in data
    assert "delta" in data
    assert data["projected_rating"] >= data["current_rating"]
    assert data["projected_rating"] <= 5.0
    assert "unhappy_reviews_resolved" in data
    assert len(data["per_aspect"]) >= 1


def test_simulate_api_empty_reviews_returns_400(client):
    resp = client.post("/api/simulate", json={"fixes": {"battery": 1.0}, "reviews": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_batch"


def test_issues_api_pii_redacted_in_evidence(client):
    payload = {
        "reviews": [
            {
                "id": "pii-1",
                "text": "Email test.leak@domain.com or call 9876543210 about terrible battery!",
                "rating": 1.0,
            }
        ]
    }
    resp = client.post("/api/issues", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    for cluster in data["clusters"]:
        assert "test.leak@domain.com" not in cluster["example_quote"]
        assert "9876543210" not in cluster["example_quote"]
        for ev in cluster["evidence"]:
            assert "test.leak@domain.com" not in ev["text"]
            assert "9876543210" not in ev["text"]


def test_frontend_workflow_end_to_end(client):
    """Verifies the exact workflow executed by the React frontend:
    1. Analyze batch of reviews -> returns reviews, clusters, insights.
    2. Clusters provide traceable evidence with review IDs and clauses.
    3. Simulator computes rating uplift and resolved unhappy customers.
    """
    reviews_batch = [
        {"id": "r1", "text": "The battery caught fire and started sparking! Urgent fire hazard.", "rating": 1.0},
        {"id": "r2", "text": "Battery life is terrible, drains in two hours.", "rating": 1.0},
        {"id": "r3", "text": "Camera photos are extremely blurry in normal light.", "rating": 2.0},
        {"id": "r4", "text": "Super fast app and snappy screen refresh rate.", "rating": 5.0},
    ]

    # Step 1: Analyze
    analyze_resp = client.post("/api/analyze", json={"reviews": reviews_batch, "mode": "lite"})
    assert analyze_resp.status_code == 200
    analyze_data = analyze_resp.json()

    assert len(analyze_data["reviews"]) == 4
    assert len(analyze_data["clusters"]) >= 1
    assert len(analyze_data["insights"]) >= 1

    # Check evidence on clusters
    for cluster in analyze_data["clusters"]:
        assert cluster["example_quote"]
        assert len(cluster["evidence"]) >= 1
        for ev in cluster["evidence"]:
            assert ev["review_id"] in ["r1", "r2", "r3", "r4"]

    # Step 2: Simulate Fixes (Fix Top complaints: battery 100%, camera 50%)
    sim_payload = {
        "fixes": {"battery": 1.0, "camera": 0.5},
        "reviews": [
            {"id": r["review_id"], "text": r["original_text"], "rating": r["rating"]}
            for r in analyze_data["reviews"]
        ],
    }
    sim_resp = client.post("/api/simulate", json=sim_payload)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()

    assert sim_data["current_rating"] < sim_data["projected_rating"]
    assert sim_data["delta"] > 0
    assert sim_data["unhappy_reviews_resolved"] >= 1
    assert any(p["aspect"] == "battery" for p in sim_data["per_aspect"])

