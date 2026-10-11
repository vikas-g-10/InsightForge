"""Unit and integration tests for Phase 2: Issue Clustering, Prioritization, and Fix-This-First Simulator."""

import pytest
import pandas as pd

from pipeline.analyze import analyze, analyze_single_review
from pipeline.clustering import cluster_clauses
from pipeline.prioritization import prioritize_issues, simulate_fixes
from schemas import (
    AnalyzeRequest,
    ClauseAnalysisResult,
    IssueCluster,
    IssuePriorityInsight,
    ReviewAnalysisResult,
    SafetyAnalysisResult,
)


def test_empty_inputs_clustering_and_prioritization():
    # Empty review list returns empty clusters and insights
    assert cluster_clauses([]) == []
    assert prioritize_issues([]) == []

    sim = simulate_fixes([], [], {"battery": 0.5})
    assert sim.projected_rating == 3.0
    assert sim.delta == 0.0
    assert sim.unhappy_reviews_resolved == 0


def test_repeated_reviews_clustering_and_prioritization():
    # 5 identical reviews with same rating
    batch = [
        {"id": f"dup-{i}", "text": "Terrible battery life and slow charging", "rating": 1}
        for i in range(5)
    ]
    resp = analyze(batch)
    assert resp.meta.summary.duplicate_count == 4
    assert resp.meta.summary.analyzed_reviews == 1

    # Clusters exist and link to canonical review
    assert len(resp.clusters) >= 1
    battery_cluster = next((c for c in resp.clusters if c.aspect == "battery"), None)
    assert battery_cluster is not None
    assert battery_cluster.example_clause_id is not None
    assert "dup-0" in battery_cluster.review_ids

    # Prioritization insight
    assert len(resp.insights) >= 1
    battery_ins = resp.insights[0]
    assert battery_ins.aspect == "battery"
    assert battery_ins.complaint_count == 1  # Not inflated by duplicates!


def test_similar_but_distinct_issues_clustered_separately():
    batch = [
        {"id": "r1", "text": "The battery drains way too fast", "rating": 1},
        {"id": "r2", "text": "Battery life is awful and runs out quickly", "rating": 1},
        {"id": "r3", "text": "Camera photos are blurry in low light", "rating": 2},
        {"id": "r4", "text": "Camera lens focus is completely broken", "rating": 2},
        {"id": "r5", "text": "Delivery was late and box was damaged", "rating": 1},
    ]
    resp = analyze(batch)
    clusters = resp.clusters
    assert len(clusters) >= 3

    # Check aspects are distinct
    aspects = {c.aspect for c in clusters}
    assert "battery" in aspects
    assert "camera" in aspects
    assert "delivery" in aspects or "packaging" in aspects

    # Check evidence traceability
    for cl in clusters:
        assert cl.size >= 1
        assert cl.example_clause_id is not None
        assert len(cl.example_quote) > 0
        assert len(cl.clause_ids) == cl.size
        assert len(cl.evidence) == cl.size
        for ev in cl.evidence:
            assert ev.review_id in cl.review_ids
            assert ev.clause_id in cl.clause_ids


def test_mixed_sentiment_separation_in_clusters():
    # Review with positive camera and negative battery
    batch = [
        {"id": "mix-1", "text": "The camera is fantastic, but the battery drains immediately", "rating": 3},
        {"id": "mix-2", "text": "Superb camera quality, however battery life is terrible", "rating": 3},
    ]
    resp = analyze(batch)
    clusters = resp.clusters

    # Must have both positive and negative clusters, separated by polarity
    polarities = {c.polarity for c in clusters}
    assert "pos" in polarities
    assert "neg" in polarities

    camera_clusters = [c for c in clusters if c.aspect == "camera"]
    battery_clusters = [c for c in clusters if c.aspect == "battery"]

    assert any(c.polarity == "pos" for c in camera_clusters)
    assert any(c.polarity == "neg" for c in battery_clusters)


def test_safety_related_issue_p0_priority():
    batch = [
        {"id": "safe-1", "text": "The battery caught fire and exploded with sparks!", "rating": 1},
        {"id": "perf-1", "text": "App is a bit slow to load", "rating": 3},
        {"id": "perf-2", "text": "App lags on startup", "rating": 3},
    ]
    resp = analyze(batch)
    clusters = resp.clusters
    insights = resp.insights

    # Critical safety cluster
    safety_clusters = [c for c in clusters if c.is_safety_alert]
    assert len(safety_clusters) >= 1
    crit_cluster = safety_clusters[0]
    assert crit_cluster.safety_tier == "critical"
    assert crit_cluster.priority == "P0"
    assert crit_cluster.team == "QA/Compliance"

    # Priority rank: Safety issue must be priority rank 1 (P0)
    assert len(insights) >= 1
    top_insight = insights[0]
    assert top_insight.safety_tier == "critical"
    assert top_insight.priority_rank == 1
    assert top_insight.priority_code == "P0"
    assert top_insight.team == "QA/Compliance"


def test_transparent_priority_scoring_and_damage():
    # Battery has 4 complaints, Sound has 1 complaint
    batch = [
        {"id": "b1", "text": "Terrible battery life", "rating": 1},
        {"id": "b2", "text": "Battery drains fast", "rating": 1},
        {"id": "b3", "text": "Battery gets hot", "rating": 2},
        {"id": "b4", "text": "Battery percentage drops fast", "rating": 1},
        {"id": "s1", "text": "Sound is a little muffled", "rating": 3},
        {"id": "ok1", "text": "Great screen and brilliant display", "rating": 5},
    ]
    resp = analyze(batch)
    insights = resp.insights

    battery_ins = next((ins for ins in insights if ins.aspect == "battery"), None)
    sound_ins = next((ins for ins in insights if ins.aspect == "sound"), None)

    assert battery_ins is not None
    assert battery_ins.complaint_count >= 3
    assert battery_ins.damage > 0
    assert battery_ins.confidence_label == "low"  # < 30 mentions

    if sound_ins:
        # Battery has more complaints and damage than sound, so battery ranks higher
        assert battery_ins.damage > sound_ins.damage
        assert battery_ins.priority_rank < sound_ins.priority_rank


def test_single_cause_unhappy_reviews_tracking():
    # Review with only 1 negative aspect rated <= 2 stars
    batch = [
        {"id": "unhappy-1", "text": "Terrible battery, cannot last 2 hours", "rating": 1},
        {"id": "multi-1", "text": "Terrible battery and awful camera and bad sound", "rating": 1},
    ]
    resp = analyze(batch)
    insights = resp.insights
    battery_ins = next((ins for ins in insights if ins.aspect == "battery"), None)

    assert battery_ins is not None
    # unhappy-1 has only battery as negative aspect; multi-1 has multiple
    assert battery_ins.single_cause_unhappy >= 1


def test_fix_this_first_simulator_calculations():
    batch = [
        {"id": "r1", "text": "Terrible battery life", "rating": 1.0},
        {"id": "r2", "text": "Terrible battery life", "rating": 2.0},
        {"id": "r3", "text": "Slow delivery", "rating": 2.0},
        {"id": "r4", "text": "Great camera and sound", "rating": 5.0},
    ]
    resp = analyze(batch)
    reviews = resp.reviews
    insights = resp.insights

    # Current average: (1 + 2 + 2 + 5) / 4 = 2.5
    # Simulate fixing 100% of battery issues
    sim = simulate_fixes(reviews, insights, {"battery": 1.0})

    assert sim.current_rating == 2.5
    assert sim.projected_rating > 2.5
    assert sim.delta > 0.0
    assert sim.projected_rating <= 5.0
    assert len(sim.per_aspect) >= 1
    assert sim.per_aspect[0].aspect == "battery"
    assert sim.per_aspect[0].fraction_fixed == 1.0


def test_simulator_honesty_pseudo_rating_label():
    # All reviews have missing ratings -> pseudo ratings used
    batch = [
        {"id": "r1", "text": "Terrible battery life", "rating": None},
        {"id": "r2", "text": "Awful battery drain", "rating": None},
    ]
    resp = analyze(batch)
    sim = simulate_fixes(resp.reviews, resp.insights, {"battery": 0.5})

    assert sim.is_sentiment_estimate is True
    assert any("pseudo-ratings" in n.lower() for n in sim.notes)


def test_data_contract_dataframe_exports():
    batch = [
        {"id": "r1", "text": "Great screen, terrible battery", "rating": 3.0},
        {"id": "r2", "text": "Battery caught fire and exploded!", "rating": 1.0},
    ]
    resp = analyze(batch)

    # PRD Table 'clusters' DataFrame export
    cl_df = resp.to_clusters_frame()
    assert isinstance(cl_df, pd.DataFrame)
    assert set(["cluster_id", "aspect", "polarity", "label", "example_clause_id", "size"]).issubset(set(cl_df.columns))
    assert len(cl_df) >= 1

    # PRD Table 'insights' DataFrame export
    ins_df = resp.to_insights_frame()
    assert isinstance(ins_df, pd.DataFrame)
    assert set(["aspect", "polarity", "mentions", "pct_negative", "beta", "ci_low", "ci_high", "damage", "priority_rank", "confidence_label", "single_cause_unhappy"]).issubset(set(ins_df.columns))
    assert len(ins_df) >= 1

    # PRD Table 'clauses' DataFrame export with cluster_id
    clause_df = resp.to_clauses_frame()
    assert isinstance(clause_df, pd.DataFrame)
    assert "cluster_id" in clause_df.columns
    assert any(cid is not None for cid in clause_df["cluster_id"])
