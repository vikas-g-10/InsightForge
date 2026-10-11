"""Deterministic tests for Milestone 1: Emerging-Issue Detection."""

import pytest
from fastapi.testclient import TestClient

from main import app
from pipeline.analyze import analyze
from pipeline.emerging import detect_emerging_issues, parse_date
from schemas import ReviewInputItem


@pytest.fixture
def client():
    return TestClient(app)


def test_parse_date_formats():
    assert str(parse_date("2026-10-01")) == "2026-10-01"
    assert str(parse_date("2026/10/01")) == "2026-10-01"
    assert str(parse_date("2026-10-01T14:30:00Z")) == "2026-10-01"
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("invalid-date") is None


def test_new_issue_detection():
    """An issue cluster with 0 occurrences in baseline and >= 2 in recent window is 'new'."""
    reviews = [
        # Baseline reviews (2026-09-01 to 2026-09-10): screen complaints and compliments
        {"id": "b1", "text": "Screen glass is cracked and scratched easily", "rating": 2.0, "date": "2026-09-01"},
        {"id": "b2", "text": "Screen display flickers sometimes", "rating": 2.0, "date": "2026-09-03"},
        {"id": "b3", "text": "Love the phone, good build", "rating": 5.0, "date": "2026-09-05"},
        {"id": "b4", "text": "Decent phone, nice speaker", "rating": 4.0, "date": "2026-09-08"},
        {"id": "b5", "text": "Very sleek design and light", "rating": 5.0, "date": "2026-09-10"},
        # Recent reviews (2026-09-25 to 2026-09-30): BRAND NEW camera failure complaints!
        {"id": "r1", "text": "Camera app crashes every time I open it", "rating": 1.0, "date": "2026-09-26"},
        {"id": "r2", "text": "Camera crashes when opening photo gallery", "rating": 1.0, "date": "2026-09-28"},
        {"id": "r3", "text": "Camera app crashes repeatedly on video", "rating": 1.0, "date": "2026-09-29"},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(
        analyzed.reviews,
        recent_days=7,
        baseline_days=25,
        min_recent_count=2,
    )

    assert res.total_emerging_issues >= 1
    assert res.new_issues_count >= 1

    # Find the camera finding
    camera_findings = [f for f in res.findings if f.aspect == "camera"]
    assert len(camera_findings) >= 1
    cam = camera_findings[0]
    assert cam.status == "new"
    assert cam.baseline_count == 0
    assert cam.recent_count >= 2
    assert cam.change_count >= 2
    assert cam.first_observed_date is not None
    assert len(cam.evidence) >= 1


def test_accelerating_issue_detection():
    """An issue with low baseline frequency that spikes in recent window is 'accelerating'."""
    reviews = [
        # Baseline window: 1 battery drain complaint out of 6 reviews
        {"id": "b1", "text": "Battery drain is faster than expected", "rating": 2.0, "date": "2026-09-01"},
        {"id": "b2", "text": "Nice screen display", "rating": 5.0, "date": "2026-09-02"},
        {"id": "b3", "text": "Good audio speakers", "rating": 4.0, "date": "2026-09-03"},
        {"id": "b4", "text": "Fast shipping and delivery", "rating": 5.0, "date": "2026-09-04"},
        {"id": "b5", "text": "Sleek body design", "rating": 5.0, "date": "2026-09-05"},
        {"id": "b6", "text": "Nice clean software UI", "rating": 4.0, "date": "2026-09-06"},
        # Recent window: multiple battery drain complaints
        {"id": "r1", "text": "Terrible battery drain after the latest update", "rating": 1.0, "date": "2026-09-25"},
        {"id": "r2", "text": "Battery drain is unbearable, drains 40% an hour", "rating": 1.0, "date": "2026-09-26"},
        {"id": "r3", "text": "Severe battery drain, dies in two hours", "rating": 1.0, "date": "2026-09-27"},
        {"id": "r4", "text": "Phone looks good", "rating": 4.0, "date": "2026-09-29"},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(
        analyzed.reviews,
        recent_days=7,
        baseline_days=25,
        min_recent_count=2,
    )

    assert res.accelerating_issues_count >= 1
    battery_findings = [f for f in res.findings if f.aspect == "battery"]
    assert len(battery_findings) >= 1
    bat = battery_findings[0]
    assert bat.status == "accelerating"
    assert bat.recent_count >= 2
    assert bat.baseline_count == 1
    assert bat.change_count >= 1
    assert bat.recent_share > bat.baseline_share
    assert bat.velocity_ratio is not None and bat.velocity_ratio > 1.4


def test_single_complaint_not_labeled_as_trend():
    """Requirement: Do not label a single complaint as a confirmed trend."""
    reviews = [
        # Baseline reviews
        {"id": "b1", "text": "Awesome camera", "rating": 5.0, "date": "2026-09-01"},
        {"id": "b2", "text": "Great screen", "rating": 5.0, "date": "2026-09-02"},
        {"id": "b3", "text": "Fast performance", "rating": 5.0, "date": "2026-09-03"},
        {"id": "b4", "text": "Nice speaker", "rating": 5.0, "date": "2026-09-04"},
        {"id": "b5", "text": "Good phone", "rating": 5.0, "date": "2026-09-05"},
        # Recent window: exactly 1 non-safety complaint
        {"id": "r1", "text": "Volume button feels a bit stiff", "rating": 3.0, "date": "2026-09-25"},
        {"id": "r2", "text": "Great battery life", "rating": 5.0, "date": "2026-09-26"},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(
        analyzed.reviews,
        recent_days=7,
        baseline_days=25,
        min_recent_count=2,
    )

    for f in res.findings:
        if not f.is_safety_escalated and f.recent_count == 1 and f.baseline_count == 0:
            assert f.status == "insufficient_data"
            assert f.confidence == "low"
            assert any("insufficient sample size" in lim.lower() for lim in f.limitations)


def test_safety_escalation_independent_of_frequency():
    """Requirement: Prioritize potential safety issues independently of sentiment and frequency."""
    reviews = [
        # Baseline
        {"id": "b1", "text": "Phone is decent", "rating": 4.0, "date": "2026-09-01"},
        {"id": "b2", "text": "Nice camera photos", "rating": 5.0, "date": "2026-09-02"},
        {"id": "b3", "text": "Good battery life", "rating": 4.0, "date": "2026-09-03"},
        {"id": "b4", "text": "Screen is bright", "rating": 4.0, "date": "2026-09-04"},
        {"id": "b5", "text": "Fast delivery", "rating": 5.0, "date": "2026-09-05"},
        # Recent: Exactly 1 safety hazard (battery sparking / fire)
        {"id": "r1", "text": "The battery caught fire and started sparking while plugged in! Fire hazard!", "rating": 1.0, "date": "2026-09-26"},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(
        analyzed.reviews,
        recent_days=7,
        baseline_days=25,
    )

    assert res.safety_escalations_count >= 1
    # Check that safety issue is top priority
    first_finding = res.findings[0]
    assert first_finding.is_safety_escalated is True
    assert first_finding.severity in ("critical", "high")
    assert first_finding.priority == 1
    assert any("fire" in ev.text.lower() or "spark" in ev.text.lower() for ev in first_finding.evidence)


def test_missing_timestamps_never_fabricates_trends():
    """Requirement: Never fabricate historical trends when timestamps or baseline data are unavailable."""
    reviews = [
        {"id": "r1", "text": "Battery drains fast", "rating": 1.0, "date": None},
        {"id": "r2", "text": "Battery drains quickly", "rating": 1.0, "date": None},
        {"id": "r3", "text": "Camera is blurry", "rating": 2.0, "date": None},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(analyzed.reviews)

    assert res.time_windows.has_sufficient_timeline is False
    assert res.time_windows.undated_reviews == 3
    assert len(res.warnings) >= 1
    assert any("missing or insufficient" in w.lower() for w in res.warnings)

    # All non-safety findings should be marked as insufficient_data
    for f in res.findings:
        if not f.is_safety_escalated:
            assert f.status == "insufficient_data"
            assert f.confidence == "low"
            assert any("timestamp" in lim.lower() for lim in f.limitations)


def test_sparse_baseline_sample_size_flag():
    """Baseline with very few reviews flags low confidence and notes limitation."""
    reviews = [
        # Baseline has only 1 review
        {"id": "b1", "text": "Good screen", "rating": 5.0, "date": "2026-09-01"},
        # Recent has 3 reviews
        {"id": "r1", "text": "Battery drains immediately", "rating": 1.0, "date": "2026-09-25"},
        {"id": "r2", "text": "Battery drain is terrible", "rating": 1.0, "date": "2026-09-26"},
        {"id": "r3", "text": "Battery dying quickly", "rating": 1.0, "date": "2026-09-27"},
    ]
    analyzed = analyze(reviews)
    res = detect_emerging_issues(
        analyzed.reviews,
        recent_days=7,
        baseline_days=25,
        min_baseline_reviews=5,
    )

    battery_findings = [f for f in res.findings if f.aspect == "battery"]
    assert len(battery_findings) >= 1
    bat = battery_findings[0]
    # Sample size limitation should be reported
    assert bat.confidence == "low"
    assert any("small sample size" in lim.lower() for lim in bat.limitations)


def test_api_emerging_issues_endpoint_success(client):
    """Test POST /api/emerging-issues HTTP endpoint."""
    payload = {
        "reviews": [
            {"id": "b1", "text": "Screen is great", "rating": 5.0, "date": "2026-09-01"},
            {"id": "b2", "text": "Sound is good", "rating": 4.0, "date": "2026-09-02"},
            {"id": "b3", "text": "Phone is okay", "rating": 3.0, "date": "2026-09-03"},
            {"id": "b4", "text": "Nice body", "rating": 4.0, "date": "2026-09-04"},
            {"id": "b5", "text": "Fast app", "rating": 5.0, "date": "2026-09-05"},
            {"id": "r1", "text": "Battery caught fire and sparked! Hazard!", "rating": 1.0, "date": "2026-09-25"},
            {"id": "r2", "text": "Camera crashes on startup", "rating": 1.0, "date": "2026-09-26"},
            {"id": "r3", "text": "Camera app completely freezes", "rating": 1.0, "date": "2026-09-27"},
        ],
        "mode": "lite",
        "recent_days": 7,
        "baseline_days": 25,
        "min_recent_count": 2,
    }
    resp = client.post("/api/emerging-issues", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "total_emerging_issues" in data
    assert "new_issues_count" in data
    assert "safety_escalations_count" in data
    assert "time_windows" in data
    assert len(data["findings"]) >= 1

    # Verify response schema fields on findings
    first = data["findings"][0]
    assert "issue_id" in first
    assert "label" in first
    assert "category" in first
    assert "status" in first
    assert "severity" in first
    assert "priority" in first
    assert "confidence" in first
    assert "evidence" in first


def test_api_emerging_issues_empty_batch(client):
    resp = client.post("/api/emerging-issues", json={"reviews": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_batch"


def test_api_emerging_issues_unsupported_mode(client):
    resp = client.post(
        "/api/emerging-issues",
        json={"reviews": [{"id": "1", "text": "Sample"}], "mode": "full"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_mode"
