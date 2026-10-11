"""Tests for root-cause hypothesis generation engine and API endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import create_app
from pipeline.analyze import analyze
from pipeline.root_cause import batch_generate_hypotheses, generate_root_cause_hypotheses
from schemas import (
    ImageDefectAnalysis,
    IssueCluster,
    IssueClusterClauseEvidence,
    ReviewAnalysisResult,
    ReviewInputItem,
)




def test_root_cause_evidence_traceability():
    """Verify that all supporting evidence references point to actual real review IDs."""
    reviews = [
        ReviewInputItem(id="rev_battery_1", text="The battery drains in 2 hours and gets burning hot and is terrible.", rating=1),
        ReviewInputItem(id="rev_battery_2", text="The battery drains in 2 hours and gets burning hot and is terrible.", rating=1),
        ReviewInputItem(id="rev_battery_3", text="The battery drains in 2 hours and gets burning hot and is terrible.", rating=1),
    ]
    analysis = analyze(reviews, mode="lite")
    clusters = analysis.clusters
    assert len(clusters) > 0

    batch = batch_generate_hypotheses(clusters, analysis.reviews)
    assert batch.total_clusters_analyzed > 0
    assert batch.clusters_with_hypotheses > 0

    res = batch.results[0]
    assert res.has_sufficient_evidence is True
    assert len(res.hypotheses) > 0

    known_review_ids = {r.review_id for r in analysis.reviews}
    for hyp in res.hypotheses:
        assert hyp.is_hypothesis_only is True
        assert len(hyp.supporting_evidence) > 0
        for ev in hyp.supporting_evidence:
            if ev.source_type == "review":
                assert ev.reference_id in known_review_ids


def test_root_cause_contradictory_evidence_detection():
    """Verify that positive reviews on the same aspect are flagged as contradictory evidence."""
    reviews = [
        # Negative complaints in same cluster
        ReviewInputItem(id="rev_neg_1", text="Battery life is horrible and drains in 3 hours.", rating=1),
        ReviewInputItem(id="rev_neg_2", text="Battery life is horrible and drains in 3 hours.", rating=1),
        # Contradicting positive praise on same aspect
        ReviewInputItem(id="rev_pos_1", text="The battery life is amazing and lasts 2 full days!", rating=5),
        ReviewInputItem(id="rev_pos_2", text="The battery life is amazing and lasts 2 full days!", rating=5),
    ]
    analysis = analyze(reviews, mode="lite")
    batch = batch_generate_hypotheses(analysis.clusters, analysis.reviews)

    battery_res = next((r for r in batch.results if r.aspect == "battery"), None)
    assert battery_res is not None
    assert battery_res.has_sufficient_evidence is True
    assert battery_res.total_contradicting_reviews > 0

    hyp = battery_res.hypotheses[0]
    assert len(hyp.contradicting_evidence) > 0
    contradicting_ids = {c.reference_id for c in hyp.contradicting_evidence}
    assert "rev_pos_1" in contradicting_ids or "rev_pos_2" in contradicting_ids


def test_root_cause_insufficient_evidence_single_review():
    """Verify that a cluster with only 1 complaint and no photo returns insufficient_evidence."""
    cluster = IssueCluster(
        cluster_id="cl_single",
        aspect="screen",
        polarity="neg",
        label="Single flickering screen",
        example_clause_id=1,
        example_quote="Screen flickered once on Tuesday",
        size=1,
        review_count=1,
        review_ids=["rev_lonely"],
        evidence=[
            IssueClusterClauseEvidence(
                clause_id=1,
                review_id="rev_lonely",
                text="Screen flickered once on Tuesday",
                polarity="neg",
                score=-0.6,
            )
        ],
    )
    result = generate_root_cause_hypotheses(cluster, all_reviews=[])
    assert result.status == "insufficient_evidence"
    assert result.has_sufficient_evidence is False
    assert len(result.hypotheses) == 0
    assert "Insufficient evidence" in result.insufficient_evidence_reason
    assert result.total_supporting_reviews == 1


def test_root_cause_insufficient_evidence_generic_complaints():
    """Verify that purely generic non-symptom complaints return insufficient_evidence."""
    cluster = IssueCluster(
        cluster_id="cl_generic",
        aspect="hardware",
        polarity="neg",
        label="Terrible bad awful",
        example_clause_id=1,
        example_quote="bad product, terrible, worst item",
        size=3,
        review_count=3,
        review_ids=["r1", "r2", "r3"],
        evidence=[
            IssueClusterClauseEvidence(clause_id=1, review_id="r1", text="bad product, terrible", polarity="neg", score=-0.8),
            IssueClusterClauseEvidence(clause_id=2, review_id="r2", text="worst item ever", polarity="neg", score=-0.8),
            IssueClusterClauseEvidence(clause_id=3, review_id="r3", text="hate it, awful", polarity="neg", score=-0.8),
        ],
    )
    result = generate_root_cause_hypotheses(cluster, all_reviews=[])
    assert result.status == "insufficient_evidence"
    assert result.has_sufficient_evidence is False
    assert "lack descriptive failure symptoms" in result.insufficient_evidence_reason


def test_root_cause_incorporates_image_findings():
    """Verify that photographic defect observations are integrated into root-cause hypotheses."""
    reviews = [
        ReviewInputItem(id="r_casing_1", text="Defective broken casing, cracked seam.", rating=1),
        ReviewInputItem(id="r_casing_2", text="Defective broken casing, cracked seam.", rating=1),
    ]
    analysis = analyze(reviews, mode="lite")

    image_evidence = [
        ImageDefectAnalysis(
            image_id="img_crack_01",
            filename="crack.jpg",
            mime_type="image/jpeg",
            file_size_bytes=50000,
            visual_description="Longitudinal crack measuring 2cm along lower casing seam.",
            potential_defect_category="casing_crack",
            visual_observations=["Longitudinal crack measuring 2cm along lower casing seam"],
            severity="high",
            confidence=0.92,
            relevant_aspect="quality",
            relevant_issue_cluster="Defective broken casing, cracked seam.",
            limitations=[],
            suggested_manual_checks=["Check weld line"],
            is_defect_observed=True,
        )
    ]

    batch = batch_generate_hypotheses(
        analysis.clusters,
        analysis.reviews,
        image_findings=image_evidence,
    )
    assert batch.clusters_with_hypotheses > 0
    casing_res = batch.results[0]
    assert casing_res.associated_images_count == 1

    hyp = casing_res.hypotheses[0]
    # Check that image reference was attached
    image_refs = [ev for ev in hyp.supporting_evidence if ev.source_type == "image"]
    assert len(image_refs) > 0
    assert image_refs[0].reference_id == "img_crack_01"


def test_root_cause_falsification_criteria_and_disclaimer():
    """Verify that every hypothesis contains concrete verification and falsification criteria."""
    reviews = [
        ReviewInputItem(id="r_conn_1", text="Speaker sound is terrible, crackling static noise.", rating=1),
        ReviewInputItem(id="r_conn_2", text="Speaker sound is terrible, crackling static noise.", rating=1),
    ]
    analysis = analyze(reviews, mode="lite")
    batch = batch_generate_hypotheses(analysis.clusters, analysis.reviews)

    assert len(batch.results) > 0
    res = batch.results[0]
    assert len(res.hypotheses) > 0
    assert "HYPOTHESIS ONLY" in res.disclaimer

    for hyp in res.hypotheses:
        assert len(hyp.verification_steps) > 0
        assert len(hyp.falsification_criteria) > 0
        assert "reject" in hyp.falsification_criteria.lower()


def test_api_root_cause_endpoint_success(client):
    """Integration test for POST /api/root-cause/hypotheses endpoint."""
    payload = {
        "reviews": [
            {"id": "r1", "text": "The battery drains super fast and is terrible, gets scorching hot.", "rating": 1},
            {"id": "r2", "text": "The battery drains super fast and is terrible, gets scorching hot.", "rating": 1},
        ],
        "image_findings": [],
        "mode": "lite",
    }
    resp = client.post("/api/root-cause/hypotheses", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_clusters_analyzed"] > 0
    assert data["clusters_with_hypotheses"] > 0
    assert "results" in data
    assert len(data["results"][0]["hypotheses"]) > 0


def test_api_root_cause_endpoint_empty_batch(client):
    """Verify that empty reviews array returns HTTP 400."""
    payload = {"reviews": [], "image_findings": []}
    resp = client.post("/api/root-cause/hypotheses", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_batch"


def test_api_root_cause_nonexistent_cluster_filter(client):
    """Verify filtering by non-existent cluster_id gracefully returns 0 matches."""
    payload = {
        "reviews": [
            {"id": "r1", "text": "Battery drains instantly and overheats.", "rating": 1},
            {"id": "r2", "text": "Battery drains instantly and overheats.", "rating": 1},
        ],
        "cluster_id": "non_existent_cluster_9999",
    }
    resp = client.post("/api/root-cause/hypotheses", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters_with_hypotheses"] == 0
    assert len(data["results"]) == 0


def test_api_root_cause_malformed_payload(client):
    """Verify that malformed request payload (e.g. reviews not a list) triggers validation error."""
    resp = client.post("/api/root-cause/hypotheses", json={"reviews": "not-a-list"})
    assert resp.status_code in (400, 422)

