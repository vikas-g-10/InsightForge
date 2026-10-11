import pytest

from pipeline._common import PipelineConfigError
from pipeline.aspects import detect_aspects, find_aspects, load_aspect_config
from pipeline.clauses import split_clauses
from pipeline.normalize import normalize_review


def detect(text, row=1):
    review = normalize_review(text, row)
    return review, [m for c in split_clauses(review) for m in detect_aspects(c)]


def aspect_names(text):
    return sorted({m.aspect for m in detect(text)[1]})


def test_single_aspect_with_evidence_span():
    review, matches = detect("The app keeps crashing on startup", row=9)
    assert [m.aspect for m in matches] == ["stability"]
    evidence = matches[0].evidence[0]
    assert evidence.text == "crashing" and evidence.term == "crash*" and evidence.strength == "strong"
    assert review.text[evidence.start : evidence.end] == evidence.text
    assert matches[0].source_row == 9 and matches[0].clause_index == 0


def test_multiple_aspects_in_one_clause():
    assert aspect_names("The app crashes and the customer care never replied") == [
        "customer_support", "stability",
    ]  # fmt: skip


def test_multiple_aspects_across_clauses_keep_clause_index():
    _, matches = detect("Fast delivery but the price is high")
    assert [(m.clause_index, m.aspect) for m in matches] == [(0, "delivery"), (1, "pricing")]


def test_no_aspect_match():
    assert aspect_names("Great app, love it") == []
    assert aspect_names("") == []
    assert aspect_names(None) == []
    assert aspect_names("\U0001F60D\U0001F60D") == []


def test_single_weak_term_is_insufficient_evidence():
    assert aspect_names("Support doesn't support dark mode") == ["ui_ux"]  # not customer_support
    assert aspect_names("Fast") == []
    assert aspect_names("The service was fine") == []


def test_two_distinct_weak_terms_are_enough():
    assert aspect_names("The loading speed is fine") == ["performance"]


def test_same_weak_term_twice_is_not_two_terms():
    assert find_aspects("service service service") == {}


def test_word_boundaries_prevent_substring_matches():
    assert aspect_names("The bugle played") == []
    assert aspect_names("Debug mode") == []
    assert aspect_names("The crashing was bad") == ["stability"]


def test_longest_overlapping_match_wins():
    hits = find_aspects("call customer care now")["customer_support"]
    assert [h.text for h in hits] == ["customer care"]


def test_hinglish_and_devanagari_terms():
    assert aspect_names("Yeh app bahut mehnga hai") == ["pricing"]
    assert aspect_names("ऐप महंगा है") == ["pricing"]
    assert "महंगा" not in find_aspects("महंगाई बहुत है")  # no match inside a longer word


def test_case_insensitive_and_apostrophe_variants():
    assert aspect_names("THE APP CRASHES") == ["stability"]
    assert aspect_names("It doesn\u2019t work at all") == ["stability"]


def test_evidence_offsets_index_into_review_for_later_clauses():
    review, matches = detect("Nice look. But the PRICE is too high")
    evidence = matches[0].evidence[0]
    assert review.text[evidence.start : evidence.end] == "PRICE"


def test_deterministic():
    text = "Crashes often, price is high, delivery late"
    assert detect(text)[1] == detect(text)[1]


def test_custom_config_and_validation(tmp_path):
    good = tmp_path / "a.yaml"
    good.write_text("aspects:\n  zed:\n    strong: [zorp]\n", encoding="utf-8")
    cfg = load_aspect_config(good)
    assert list(find_aspects("a zorp here", cfg)) == ["zed"]
    for body in (
        "aspects:\n  zed:\n    weak: [zorp]\n",  # no strong term
        "aspects:\n  zed:\n    strong: [no]\n",  # YAML boolean, must be quoted
        "aspects:\n  zed:\n    strong: [ab*]\n",  # prefix too short
        "aspects: {}\n",
        "min_weak_terms: 0\naspects:\n  zed:\n    strong: [zorp]\n",
    ):
        bad = tmp_path / "b.yaml"
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(PipelineConfigError):
            load_aspect_config(bad)


def test_missing_config_file_has_clear_error(tmp_path):
    with pytest.raises(PipelineConfigError, match="not found"):
        load_aspect_config(tmp_path / "nope.yaml")


def test_prd_great_camera_terrible_battery_clauses_and_sentiment():
    # PRD F2 & Section 12 Test 1:
    # "Great camera, terrible battery" must produce camera positive and battery negative
    from pipeline.sentiment import sentiment_for_clause

    review = normalize_review("Great camera, terrible battery", 1)
    clauses = split_clauses(review)
    assert len(clauses) == 2

    # Clause 0: camera positive
    aspects_0 = [m.aspect for m in detect_aspects(clauses[0])]
    assert aspects_0 == ["camera"]
    sent_0 = sentiment_for_clause(clauses[0])
    assert sent_0.polarity == "pos" and sent_0.score >= 0.25

    # Clause 1: battery negative
    aspects_1 = [m.aspect for m in detect_aspects(clauses[1])]
    assert aspects_1 == ["battery"]
    sent_1 = sentiment_for_clause(clauses[1])
    assert sent_1.polarity == "neg" and sent_1.score <= -0.25


def test_prd_aspect_seeds_and_separation():
    # Hinglish seeds from PRD Section 7.3: 'awaaz' for sound, 'dabba' for packaging, 'kharab hona' for quality
    assert aspect_names("awaaz achhi hai") == ["sound"]
    assert aspect_names("dabba kharab tha") == ["packaging"]
    assert aspect_names("kharab hona shuru ho gaya") == ["quality"]

    # Packaging and Delivery are distinct aspects
    assert aspect_names("Delivery fast thi lekin dabba tuta tha") == ["delivery", "packaging"]


def test_prd_aspect_routing():
    from pipeline.aspects import get_aspect_routing

    # PRD Section 7.6 routing map
    assert get_aspect_routing("battery")["team"] == "Product/Engineering"
    assert get_aspect_routing("camera")["team"] == "Product/Engineering"
    assert get_aspect_routing("performance")["team"] == "Product/Engineering"
    assert get_aspect_routing("delivery")["team"] == "Logistics"
    assert get_aspect_routing("packaging")["team"] == "Logistics"
    assert get_aspect_routing("customer_support")["team"] == "Customer Support"
    assert get_aspect_routing("pricing")["team"] == "Pricing/Marketing"
    assert get_aspect_routing("safety")["team"] == "QA/Compliance"
    assert get_aspect_routing("safety")["priority"] == "P0"
