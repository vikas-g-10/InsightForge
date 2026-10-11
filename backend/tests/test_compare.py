"""Comprehensive deterministic tests for Product and Version Comparison (Features 1 & 2)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import create_app
from pipeline.analyze import analyze
from pipeline.compare import compare_products, compare_versions, get_available_targets
from schemas import ReviewInputItem




def test_available_targets_extraction():
    """Verify get_available_targets extracts products and versions accurately."""
    reviews = [
        ReviewInputItem(id="1", text="Good", product="ProPhone", version="v1.0"),
        ReviewInputItem(id="2", text="Okay", product="ProPhone", version="v1.0"),
        ReviewInputItem(id="3", text="Better", product="ProPhone", version="v2.0"),
        ReviewInputItem(id="4", text="Fine", product="LitePhone", version="v1.0"),
        ReviewInputItem(id="5", text="Unlabeled", product=None, version=None),
    ]
    targets = get_available_targets(reviews)
    assert len(targets.products) == 2
    pro_phone = next(p for p in targets.products if p.product == "ProPhone")
    assert pro_phone.count == 3
    assert len(pro_phone.versions) == 2
    assert targets.unlabeled_product_count == 1
    assert targets.unlabeled_version_count == 1


def test_version_compare_success():
    """Verify version comparison calculates metrics, sentiment deltas, and evidence."""
    reviews = [
        # Version 1.0: Battery issues and slow performance
        ReviewInputItem(id="v1_1", text="Battery drains in 2 hours, terrible battery life.", rating=1, version="1.0", product="PhoneX"),
        ReviewInputItem(id="v1_2", text="Battery drains in 2 hours, terrible battery life.", rating=1, version="1.0", product="PhoneX"),
        ReviewInputItem(id="v1_3", text="Very slow and laggy performance.", rating=2, version="1.0", product="PhoneX"),
        ReviewInputItem(id="v1_4", text="Nice camera photos.", rating=4, version="1.0", product="PhoneX"),
        ReviewInputItem(id="v1_5", text="Good screen quality.", rating=4, version="1.0", product="PhoneX"),
        # Version 2.0: Battery improved, but new audio crackling issue
        ReviewInputItem(id="v2_1", text="Battery life is great now, lasts all day!", rating=5, version="2.0", product="PhoneX"),
        ReviewInputItem(id="v2_2", text="Performance is fast and snappy.", rating=5, version="2.0", product="PhoneX"),
        ReviewInputItem(id="v2_3", text="Terrible audio speaker with horrible crackling noise.", rating=1, version="2.0", product="PhoneX"),
        ReviewInputItem(id="v2_4", text="Terrible audio speaker with horrible crackling noise.", rating=1, version="2.0", product="PhoneX"),
        ReviewInputItem(id="v2_5", text="Super fast charging and great screen.", rating=5, version="2.0", product="PhoneX"),
    ]

    res = compare_versions(reviews, version_a="1.0", version_b="2.0", product="PhoneX")

    assert res.product == "PhoneX"
    assert res.version_a.version == "1.0"
    assert res.version_a.review_count == 5
    assert res.version_b.version == "2.0"
    assert res.version_b.review_count == 5
    assert res.version_b.average_rating > res.version_a.average_rating

    # Battery aspect should be improved
    battery_asp = next((a for a in res.aspect_comparisons if a.aspect == "battery"), None)
    assert battery_asp is not None
    assert battery_asp.complaint_count_a >= 2
    assert battery_asp.complaint_count_b == 0
    assert battery_asp.status in ("improved", "not_observed")

    # Sound aspect should be newly observed in 2.0
    sound_asp = next((a for a in res.aspect_comparisons if a.aspect == "sound"), None)
    assert sound_asp is not None
    assert sound_asp.complaint_count_b >= 2
    assert sound_asp.status == "newly_observed"

    # Evidence traceability
    for cl in res.cluster_comparisons:
        for ev in cl.evidence_a:
            assert ev.source_type == "review"
            assert ev.reference_id.startswith("v1_")
        for ev in cl.evidence_b:
            assert ev.source_type == "review"
            assert ev.reference_id.startswith("v2_")

    assert "OBSERVATIONAL" in res.disclaimer


def test_version_compare_not_observed_vs_resolved():
    """Verify small sample labels absent issues as not_observed rather than resolved."""
    # Small sample in Version B (5 reviews): absent battery complaint is not_observed
    reviews_small_b = [
        ReviewInputItem(id="a1", text="Battery is terrible and dead.", rating=1, version="1.0"),
        ReviewInputItem(id="a2", text="Battery is terrible and dead.", rating=1, version="1.0"),
        ReviewInputItem(id="a3", text="Battery is terrible and dead.", rating=1, version="1.0"),
        ReviewInputItem(id="a4", text="Phone is okay.", rating=3, version="1.0"),
        ReviewInputItem(id="a5", text="Phone is okay.", rating=3, version="1.0"),
        # Version B has only 5 reviews, no battery mentions
        ReviewInputItem(id="b1", text="Screen is bright.", rating=4, version="2.0"),
        ReviewInputItem(id="b2", text="Camera is great.", rating=5, version="2.0"),
        ReviewInputItem(id="b3", text="Nice design.", rating=4, version="2.0"),
        ReviewInputItem(id="b4", text="Great look.", rating=4, version="2.0"),
        ReviewInputItem(id="b5", text="Good feel.", rating=4, version="2.0"),
    ]
    res_small = compare_versions(reviews_small_b, "1.0", "2.0")
    battery_small = next(a for a in res_small.aspect_comparisons if a.aspect == "battery")
    assert battery_small.status == "not_observed"
    assert "insufficient to declare the issue resolved" in (battery_small.sample_note or "")


def test_version_compare_mixed_products_error():
    """Verify error when versions belong to distinct products."""
    reviews = [
        ReviewInputItem(id="1", text="Good phone", product="BrandA", version="v1"),
        ReviewInputItem(id="2", text="Good tablet", product="BrandB", version="v2"),
    ]
    with pytest.raises(Exception) as exc:
        compare_versions(reviews, version_a="v1", version_b="v2")
    assert getattr(exc.value, "code", "") == "mixed_products" or "requires both versions to belong to the same product" in str(exc.value)


def test_version_compare_same_version_error():
    """Verify error when comparing a version with itself."""
    reviews = [ReviewInputItem(id="1", text="Fine", version="v1")]
    with pytest.raises(Exception) as exc:
        compare_versions(reviews, version_a="v1", version_b="v1")
    assert "same_version" in str(exc.value) or "Cannot compare a version with itself" in str(exc.value)


def test_version_compare_unmatched_version_error():
    """Verify 404/not_found error when version doesn't exist."""
    reviews = [ReviewInputItem(id="1", text="Fine", version="v1")]
    with pytest.raises(Exception) as exc:
        compare_versions(reviews, version_a="v1", version_b="v999")
    assert "version_not_found" in str(exc.value) or "No reviews found for Version B" in str(exc.value)


def test_version_compare_sparse_warning():
    """Verify warning when versions have fewer than 5 reviews."""
    reviews = [
        ReviewInputItem(id="1", text="Fine", version="v1"),
        ReviewInputItem(id="2", text="Good", version="v2"),
    ]
    res = compare_versions(reviews, "v1", "v2")
    assert res.is_sparse_sample is True
    assert len(res.warnings) > 0
    assert "Sparse sample warning" in res.warnings[0]


def test_version_compare_safety_escalation():
    """Verify safety alert escalation is prominently tracked across versions."""
    reviews = [
        ReviewInputItem(id="v1_1", text="Phone gets slightly warm.", rating=3, version="1.0"),
        ReviewInputItem(id="v1_2", text="Phone is normal.", rating=4, version="1.0"),
        ReviewInputItem(id="v2_1", text="The charger caught fire and started smoking electric shock!", rating=1, version="2.0"),
        ReviewInputItem(id="v2_2", text="Phone exploded and burned my hand!", rating=1, version="2.0"),
    ]
    res = compare_versions(reviews, "1.0", "2.0")
    assert res.version_b.safety_alert_count >= 1
    assert len(res.safety_changes) > 0
    assert any("Safety alert volume increased" in sc or "surfaced" in sc for sc in res.safety_changes)


def test_product_compare_success():
    """Verify product comparison across strengths, weaknesses, common themes, and trade-offs."""
    reviews = [
        # Product Alpha: Great battery, poor build quality
        ReviewInputItem(id="pa_1", text="Battery lasts 3 days, amazing battery life!", rating=5, product="Alpha"),
        ReviewInputItem(id="pa_2", text="Battery life is incredible.", rating=5, product="Alpha"),
        ReviewInputItem(id="pa_3", text="Build quality is flimsy, cheap plastic broke.", rating=1, product="Alpha"),
        ReviewInputItem(id="pa_4", text="Damaged packaging and terrible box.", rating=2, product="Alpha"),
        ReviewInputItem(id="pa_5", text="Good performance and fast.", rating=4, product="Alpha"),
        # Product Beta: Sturdy quality, poor battery
        ReviewInputItem(id="pb_1", text="Build quality is rock solid, premium metal.", rating=5, product="Beta"),
        ReviewInputItem(id="pb_2", text="Super durable quality, excellent finish.", rating=5, product="Beta"),
        ReviewInputItem(id="pb_3", text="Battery drains in 2 hours, terrible battery.", rating=1, product="Beta"),
        ReviewInputItem(id="pb_4", text="Packaging was bad and damaged.", rating=2, product="Beta"),
        ReviewInputItem(id="pb_5", text="Good performance and fast.", rating=4, product="Beta"),
    ]

    res = compare_products(reviews, product_a="Alpha", product_b="Beta")

    assert res.product_a.product == "Alpha"
    assert res.product_b.product == "Beta"

    # Battery leader should be Alpha
    battery_strength = next(s for s in res.aspect_strengths if s.aspect == "battery")
    assert battery_strength.leader == "product_a"

    # Quality leader should be Beta
    quality_strength = next(s for s in res.aspect_strengths if s.aspect == "quality")
    assert quality_strength.leader == "product_b"

    # Common complaint theme should include packaging
    assert any("Packaging" in theme for theme in res.common_complaint_themes)

    # Balanced summary without single universal winner
    assert "Alpha" in res.tradeoff_summary
    assert "Beta" in res.tradeoff_summary
    assert "COMPARATIVE FEEDBACK ONLY" in res.disclaimer


def test_product_compare_focus_aspect():
    """Verify tailored recommendations when a focus aspect is specified."""
    reviews = [
        ReviewInputItem(id="1", text="Battery lasts 3 days!", rating=5, product="PhoneA"),
        ReviewInputItem(id="2", text="Battery is awesome!", rating=5, product="PhoneA"),
        ReviewInputItem(id="3", text="Battery dies instantly.", rating=1, product="PhoneB"),
        ReviewInputItem(id="4", text="Horrible battery life.", rating=1, product="PhoneB"),
    ]
    res = compare_products(reviews, "PhoneA", "PhoneB", focus_aspect="battery")
    assert res.focus_aspect == "battery"
    assert res.aspect_strengths[0].aspect == "battery"
    assert "PhoneA is favored" in res.recommended_fit


def test_product_compare_unbalanced_warning():
    """Verify unbalanced sample size warning when ratio exceeds 4:1."""
    reviews = [
        *[ReviewInputItem(id=f"a_{i}", text="Phone A review text", rating=4, product="Alpha") for i in range(25)],
        *[ReviewInputItem(id=f"b_{i}", text="Phone B review text", rating=3, product="Beta") for i in range(3)],
    ]
    res = compare_products(reviews, "Alpha", "Beta")
    assert res.is_unbalanced is True
    assert any("imbalance" in w for w in res.warnings)


def test_product_compare_same_product_error():
    """Verify error when comparing a product with itself."""
    reviews = [ReviewInputItem(id="1", text="Good", product="P1")]
    with pytest.raises(Exception) as exc:
        compare_products(reviews, "P1", "P1")
    assert "same_product" in str(exc.value) or "Cannot compare a product with itself" in str(exc.value)


def test_product_compare_evidence_traceability():
    """Verify that all evidence references map to actual review IDs."""
    reviews = [
        ReviewInputItem(id="prod_a_rev_100", text="Camera is broken and blurry.", rating=1, product="ProdA"),
        ReviewInputItem(id="prod_b_rev_200", text="Speaker sound is noisy and muffled.", rating=1, product="ProdB"),
    ]
    res = compare_products(reviews, "ProdA", "ProdB")
    assert len(res.key_evidence_a) > 0
    assert res.key_evidence_a[0].reference_id == "prod_a_rev_100"
    assert len(res.key_evidence_b) > 0
    assert res.key_evidence_b[0].reference_id == "prod_b_rev_200"


def test_api_version_compare_endpoint(client):
    """API endpoint test for POST /api/compare/versions."""
    payload = {
        "reviews": [
            {"id": "1", "text": "Battery drains fast.", "rating": 1, "version": "v1"},
            {"id": "2", "text": "Battery drains fast.", "rating": 1, "version": "v1"},
            {"id": "3", "text": "Battery lasts 2 days!", "rating": 5, "version": "v2"},
            {"id": "4", "text": "Battery lasts 2 days!", "rating": 5, "version": "v2"},
        ],
        "version_a": "v1",
        "version_b": "v2",
    }
    resp = client.post("/api/compare/versions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["version_a"]["version"] == "v1"
    assert data["version_b"]["version"] == "v2"
    assert len(data["aspect_comparisons"]) > 0


def test_api_product_compare_endpoint(client):
    """API endpoint test for POST /api/compare/products."""
    payload = {
        "reviews": [
            {"id": "1", "text": "Great battery.", "rating": 5, "product": "PhoneX"},
            {"id": "2", "text": "Terrible battery.", "rating": 1, "product": "PhoneY"},
        ],
        "product_a": "PhoneX",
        "product_b": "PhoneY",
        "focus_aspect": "battery",
    }
    resp = client.post("/api/compare/products", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_a"]["product"] == "PhoneX"
    assert data["product_b"]["product"] == "PhoneY"
    assert data["focus_aspect"] == "battery"


def test_api_compare_targets_endpoint(client):
    """API endpoint test for POST /api/compare/targets."""
    payload = {
        "reviews": [
            {"id": "1", "text": "Test review", "product": "Watch1", "version": "1.0"},
            {"id": "2", "text": "Test review", "product": "Watch1", "version": "2.0"},
        ]
    }
    resp = client.post("/api/compare/targets", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["products"]) == 1
    assert data["products"][0]["product"] == "Watch1"
    assert len(data["products"][0]["versions"]) == 2


def test_api_two_batches_input_mode(client):
    """Verify submitting separate review batches for version and product comparison."""
    payload_ver = {
        "reviews_a": [{"id": "a1", "text": "Battery dies in 1 hr.", "rating": 1}],
        "reviews_b": [{"id": "b1", "text": "Battery lasts 24 hrs.", "rating": 5}],
        "version_a": "alpha",
        "version_b": "beta",
    }
    resp = client.post("/api/compare/versions", json=payload_ver)
    assert resp.status_code == 200
    assert resp.json()["version_a"]["review_count"] == 1
    assert resp.json()["version_b"]["review_count"] == 1

    payload_prod = {
        "reviews_a": [{"id": "p1", "text": "Solid build quality.", "rating": 5}],
        "reviews_b": [{"id": "p2", "text": "Flimsy plastic casing.", "rating": 1}],
        "product_a": "LaptopA",
        "product_b": "LaptopB",
    }
    resp_prod = client.post("/api/compare/products", json=payload_prod)
    assert resp_prod.status_code == 200
    assert resp_prod.json()["product_a"]["product"] == "LaptopA"
    assert resp_prod.json()["product_b"]["product"] == "LaptopB"
