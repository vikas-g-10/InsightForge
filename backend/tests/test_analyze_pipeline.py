"""Unit and orchestration tests for the core review analysis pipeline."""

import pandas as pd
import pytest

from ingest import IngestError
from pipeline.analyze import analyze, analyze_reviews, analyze_single_review
from schemas import AnalyzeRequest, ReviewInputItem


def test_valid_positive_review():
    res, warns = analyze_single_review("The camera quality is excellent and battery lasts all day", raw_rating=5)
    assert res.status == "analyzed"
    assert res.polarity == "pos"
    assert res.sentiment_score > 0
    assert "camera" in res.aspects or "battery" in res.aspects
    assert res.rating == 5.0
    assert res.rating_source == "given"
    assert res.safety.tier == "none"
    assert not res.safety.is_alert


def test_valid_negative_review():
    res, warns = analyze_single_review("Terrible product, battery drains immediately and app crashes", raw_rating=1)
    assert res.status == "analyzed"
    assert res.polarity == "neg"
    assert res.sentiment_score < 0
    assert "battery" in res.aspects or "performance" in res.aspects
    assert res.rating == 1.0
    assert res.safety.tier == "none"


def test_neutral_or_ambiguous_review():
    res, warns = analyze_single_review("It is okay, nothing special", raw_rating=3)
    assert res.status == "analyzed"
    assert res.polarity == "neu"
    assert res.rating == 3.0


def test_missing_optional_rating():
    res, warns = analyze_single_review("Great headphones and clear sound", raw_rating=None)
    assert res.status == "analyzed"
    assert res.rating is None
    assert res.rating_source == "pseudo"
    assert res.pseudo_rating is not None
    assert res.polarity == "pos"
    assert res.rating_conflict is False


def test_invalid_rating_handled_safely():
    res, warns = analyze_single_review("Good phone", raw_rating="five stars")
    assert res.status == "analyzed"
    assert res.rating is None
    assert any("dropped invalid" in w.lower() for w in warns)


def test_out_of_range_rating_coerced_safely():
    res, warns = analyze_single_review("Average phone", raw_rating=10)
    assert res.status == "analyzed"
    # 10 is coerced to 5 or dropped per ingest validation
    assert res.rating is not None or any("coerced" in w.lower() for w in warns)


def test_fractional_rating():
    res, warns = analyze_single_review("Nice display", raw_rating="4/5")
    assert res.status == "analyzed"
    assert res.rating == 4.0
    assert res.rating_source == "given"


def test_empty_review_text_handled_gracefully():
    res, warns = analyze_single_review("   ", review_id="empty-1", raw_rating=4)
    assert res.status == "empty"
    assert res.review_id == "empty-1"
    assert res.normalized_text == ""
    assert res.polarity == "neu"
    assert len(res.clauses) == 0


def test_multi_clause_aspect_sentiment_separation():
    res, warns = analyze_single_review("Great camera, terrible battery", raw_rating=3)
    assert res.status == "analyzed"
    assert len(res.clauses) == 2
    c1, c2 = res.clauses[0], res.clauses[1]
    assert "camera" in c1.aspects or c1.aspect == "camera"
    assert c1.polarity == "pos"
    assert "battery" in c2.aspects or c2.aspect == "battery"
    assert c2.polarity == "neg"


def test_safety_positive_critical_hazard():
    res, warns = analyze_single_review("The battery caught fire and exploded!", raw_rating=1)
    assert res.safety.tier == "critical"
    assert res.safety.is_alert is True
    assert len(res.safety.matches) >= 1
    assert res.safety.suggested_action is not None


def test_safety_positive_medium_hazard():
    res, warns = analyze_single_review("The phone has severe overheating while charging", raw_rating=2)
    assert res.safety.tier == "medium"
    assert res.safety.is_alert is True


def test_safety_negation_guard():
    res, warns = analyze_single_review("There is no overheating and no fire hazard", raw_rating=5)
    assert res.safety.tier == "none"
    assert res.safety.is_alert is False
    assert len(res.safety.matches) == 0


def test_safety_context_false_positive():
    res, warns = analyze_single_review("The company burn rate is too fast", raw_rating=3)
    assert res.safety.tier == "none"
    assert res.safety.is_alert is False


def test_sarcasm_heuristic_rule_1():
    res, warns = analyze_single_review("Great update, phone stopped working immediately", raw_rating=1)
    assert res.sarcasm_suspect is True
    assert res.polarity == "neg"


def test_rating_text_conflict_rule_2():
    # Strongly positive text with 1 star rating
    res, warns = analyze_single_review("This phone is amazing, incredible and fantastic!", raw_rating=1)
    assert res.rating_conflict is True


def test_multilingual_hinglish():
    res, warns = analyze_single_review("Yeh phone bahut accha hai aur battery mast hai", raw_rating=5)
    assert res.language == "hinglish"
    assert res.language_badge == "Hinglish"
    assert res.polarity == "pos"


def test_multilingual_devanagari_hindi():
    res, warns = analyze_single_review("ऐप बहुत अच्छा है और उपयोग में आसान है", raw_rating=5)
    assert res.language == "hi"
    assert res.language_badge == "Hindi"
    assert res.polarity == "pos"


def test_multilingual_kanglish():
    res, warns = analyze_single_review("camera tumba chennagide", raw_rating=5)
    assert res.language == "kanglish"
    assert res.language_badge == "Kanglish"
    assert res.polarity == "pos"


def test_multilingual_tanglish():
    res, warns = analyze_single_review("phone romba waste", raw_rating=1)
    assert res.language == "tanglish"
    assert res.language_badge == "Tanglish"
    assert res.polarity == "neg"


def test_multilingual_tenglish():
    res, warns = analyze_single_review("camera chala bagundi", raw_rating=5)
    assert res.language == "tenglish"
    assert res.language_badge == "Tenglish"
    assert res.polarity == "pos"


def test_1000_char_truncation():
    long_text = "Good phone. " * 150  # ~1800 characters
    res, warns = analyze_single_review(long_text, raw_rating=4)
    assert res.truncated is True
    assert len(res.normalized_text) <= 1000


def test_batch_duplicate_detection_and_aggregation():
    batch = [
        {"id": "r1", "text": "Great camera", "rating": 5},
        {"id": "r2", "text": "Great camera", "rating": 5},  # exact duplicate
        {"id": "r3", "text": "Terrible battery", "rating": 1},
        {"id": "r4", "text": "", "rating": None},  # empty text
    ]
    resp = analyze(batch)
    assert len(resp.reviews) == 4
    assert resp.reviews[0].is_duplicate is False
    assert resp.reviews[1].is_duplicate is True
    assert resp.reviews[1].duplicate_of == "r1"
    assert resp.reviews[3].status == "empty"
    assert resp.meta.summary.total_reviews == 4
    assert resp.meta.summary.analyzed_reviews == 2
    assert resp.meta.summary.duplicate_count == 1
    assert resp.meta.summary.unique_reviews == 3
    assert any("duplicate" in w.lower() for w in resp.meta.summary.warnings)


def test_empty_batch_raises_error():
    with pytest.raises(IngestError) as exc_info:
        analyze([])
    assert exc_info.value.code == "empty_batch"
    assert exc_info.value.status_code == 400


def test_oversized_batch_raises_error():
    oversized = [{"text": "Good phone", "rating": 5} for _ in range(15)]
    with pytest.raises(IngestError) as exc_info:
        analyze(oversized, max_rows=10)
    assert exc_info.value.code == "payload_too_large"
    assert exc_info.value.status_code == 413


def test_dataframe_input_and_frame_exports():
    df = pd.DataFrame([
        {"source_row": 2, "text": "Great sound, terrible battery", "rating": 3.0, "date": "2024-01-01"},
        {"source_row": 3, "text": "App is fast", "rating": 5.0, "date": "2024-01-02"},
    ])
    resp = analyze(df)
    assert len(resp.reviews) == 2

    # Verify export to PRD Table 'reviews' DataFrame
    rev_df = resp.to_reviews_frame()
    assert isinstance(rev_df, pd.DataFrame)
    assert set(["review_id", "text", "rating", "lang"]).issubset(set(rev_df.columns))
    assert len(rev_df) == 2

    # Verify export to PRD Table 'clauses' DataFrame
    clause_df = resp.to_clauses_frame()
    assert isinstance(clause_df, pd.DataFrame)
    assert set(["review_id", "clause_id", "clause_text", "aspect", "score", "polarity"]).issubset(set(clause_df.columns))
    assert len(clause_df) >= 3  # review 1 has 2 clauses, review 2 has 1 clause


# --- Phase 1.1 Correctness & PRD Compliance Tests ---


def test_pii_redaction_in_all_output_fields():
    raw = "Contact test.user@company.com or call 9876543210. Phone battery caught fire and exploded!"
    res, warns = analyze_single_review(raw, review_id="pii-1", raw_rating=1)

    # 1. Unredacted email and phone must NOT appear in output fields
    assert "test.user@company.com" not in res.original_text
    assert "9876543210" not in res.original_text
    assert "[email]" in res.original_text
    assert "[phone]" in res.original_text

    assert "test.user@company.com" not in res.normalized_text
    assert "9876543210" not in res.normalized_text
    assert "[email]" in res.normalized_text
    assert "[phone]" in res.normalized_text

    for c in res.clauses:
        assert "test.user@company.com" not in c.text
        assert "9876543210" not in c.text
        # Verify offsets cleanly slice normalized_text
        assert res.normalized_text[c.start : c.end] == c.text

    assert res.safety.tier == "critical"
    for m in res.safety.matches:
        assert "test.user@company.com" not in m.excerpt
        assert "9876543210" not in m.excerpt


def test_duplicate_whitespace_and_case_variants():
    batch = [
        {"id": "r1", "text": "Superb battery life and fast charging", "rating": 5},
        {"id": "r2", "text": "   Superb   battery life   and fast charging   ", "rating": 5},
        {"id": "r3", "text": "superb battery life and fast charging", "rating": 5},
        {"id": "r4", "text": "Completely different text about camera", "rating": 4},
    ]
    resp = analyze(batch)
    assert len(resp.reviews) == 4

    assert resp.reviews[0].is_duplicate is False
    assert resp.reviews[0].duplicate_of is None

    assert resp.reviews[1].is_duplicate is True
    assert resp.reviews[1].duplicate_of == "r1"

    assert resp.reviews[2].is_duplicate is True
    assert resp.reviews[2].duplicate_of == "r1"

    assert resp.reviews[3].is_duplicate is False
    assert resp.reviews[3].duplicate_of is None

    assert resp.meta.summary.total_reviews == 4
    assert resp.meta.summary.analyzed_reviews == 2
    assert resp.meta.summary.duplicate_count == 2
    assert resp.meta.summary.unique_reviews == 2


def test_duplicate_rules_preserve_different_ratings_and_versions():
    # Differing rating, differing version, and differing product must NOT be merged as duplicates
    batch = [
        {"id": "r1", "text": "The app crashes constantly", "rating": 1, "version": "1.0.0", "product": "AppA"},
        {"id": "r2", "text": "The app crashes constantly", "rating": 5, "version": "1.0.0", "product": "AppA"},  # Differing rating
        {"id": "r3", "text": "The app crashes constantly", "rating": 1, "version": "2.0.0", "product": "AppA"},  # Differing version
        {"id": "r4", "text": "The app crashes constantly", "rating": 1, "version": "1.0.0", "product": "AppB"},  # Differing product
        {"id": "r5", "text": "The app crashes constantly", "rating": 1, "version": "1.0.0", "product": "AppA"},  # Genuine exact duplicate of r1
    ]
    resp = analyze(batch)
    assert len(resp.reviews) == 5

    assert resp.reviews[0].is_duplicate is False
    assert resp.reviews[1].is_duplicate is False  # preserved due to rating difference
    assert resp.reviews[2].is_duplicate is False  # preserved due to version difference
    assert resp.reviews[3].is_duplicate is False  # preserved due to product difference
    assert resp.reviews[4].is_duplicate is True   # genuine duplicate of r1
    assert resp.reviews[4].duplicate_of == "r1"

    assert resp.meta.summary.total_reviews == 5
    assert resp.meta.summary.analyzed_reviews == 4
    assert resp.meta.summary.duplicate_count == 1


def test_duplicate_issue_counts_not_inflated():
    # 10 identical reviews pointing out battery fire
    repeated = [
        {"id": f"dup-{i}", "text": "Terrible battery, phone caught fire!", "rating": 1}
        for i in range(10)
    ]
    resp = analyze(repeated)
    summary = resp.meta.summary

    assert summary.total_reviews == 10
    assert summary.analyzed_reviews == 1
    assert summary.duplicate_count == 9
    assert summary.unique_reviews == 1

    # Issue metrics must only count the single unique review, avoiding 10x inflation
    assert summary.safety_breakdown.get("critical") == 1
    assert summary.sentiment_breakdown.get("neg") == 1
    assert summary.aspect_mentions.get("battery") == 1


def test_missing_rating_does_not_trigger_false_conflict():
    text = "This is the best phone ever made, fantastic, amazing and brilliant!"
    res, warns = analyze_single_review(text, review_id="no-rate", raw_rating=None)

    assert res.status == "analyzed"
    assert res.rating is None
    assert res.rating_source == "pseudo"
    assert res.pseudo_rating is not None
    assert res.pseudo_rating >= 4.0
    # Must NOT flag rating conflict when rating was absent
    assert res.rating_conflict is False


def test_unsupported_full_mode_rejected():
    with pytest.raises(IngestError) as exc_info:
        analyze([{"text": "Good phone", "rating": 5}], mode="full")
    assert exc_info.value.code == "unsupported_mode"
    assert exc_info.value.status_code == 400


def test_fallback_to_lite_reports_honest_mode():
    resp = analyze(
        [{"text": "Good phone", "rating": 5}],
        mode="full",
        fallback_to_lite=True,
    )
    # Must honestly report actual mode executed as lite
    assert resp.meta.mode == "lite"
    assert resp.meta.requested_mode == "full"
    assert any("Requested mode 'full' is unavailable" in w for w in resp.meta.summary.warnings)
