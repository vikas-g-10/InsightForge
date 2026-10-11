import pytest

from pipeline._common import PipelineConfigError
from pipeline.normalize import normalize_review
from pipeline.safety import highest_tier, load_safety_config, scan_review


def scan(text, row=3):
    return scan_review(normalize_review(text, row))


def test_critical_tier_match_has_structured_fields():
    (match,) = scan("The charger caught fire last night", row=8)
    assert (match.category, match.tier, match.term, match.phrase) == (
        "physical_safety", "critical", "caught fire", "caught fire",
    )  # fmt: skip
    assert match.source_row == 8
    assert "caught fire" in match.excerpt
    assert "human review" in match.note and "not proof" in match.note
    assert match.suggested_action == "escalate to QA and Compliance within 24 hours"


def test_prd_mandatory_examples():
    # 1. "phone caught fire" must trigger critical
    matches = scan("phone caught fire")
    assert len(matches) == 1
    assert matches[0].tier == "critical"
    assert highest_tier(matches) == "critical"

    # 2. "no overheating" must not trigger a safety alert
    assert scan("no overheating") == []

    # 3. "koi problem nahi" must respect documented negation behavior
    assert scan("overheating ki koi problem nahi") == []
    assert scan("koi problem nahi") == []

    # 4. "burn rate" must not be mistaken for a physical injury or burn hazard
    assert scan("Our burn rate is too high") == []
    assert scan("The company has a high burn rate") == []


def test_all_prd_tiers():
    # Critical: fire, explosion, sparks, shock, burn, smoke
    assert scan("device exploded")[0].tier == "critical"
    assert scan("current laga")[0].tier == "critical"
    assert scan("burned my hand")[0].tier == "critical"
    assert scan("smoking charger")[0].tier == "critical"

    # High: allergic reaction, rash, choking, bleeding, injury, fainting
    assert scan("got a severe rash")[0].tier == "high"
    assert scan("choking hazard for kids")[0].tier == "high"
    assert scan("caused an allergic reaction")[0].tier == "high"

    # Medium: overheating, swelling battery, leakage
    assert scan("phone overheats when charging")[0].tier == "medium"
    assert scan("swollen battery")[0].tier == "medium"
    assert scan("battery leakage")[0].tier == "medium"


def test_span_indexes_into_review_text():
    review = normalize_review("It exploded. Scary", 1)
    (match,) = scan_review(review)
    assert review.text[match.start : match.end] == match.phrase == "exploded"


def test_medium_tier_and_highest_tier():
    matches = scan("It overheats and my account was hacked")
    assert {(m.category, m.tier) for m in matches} == {
        ("physical_safety", "medium"), ("privacy_security", "high"),
    }  # fmt: skip
    assert highest_tier(matches) == "high"
    assert highest_tier(scan("It overheats")) == "medium"
    assert highest_tier([]) is None
    assert highest_tier([], default="none") == "none"


@pytest.mark.parametrize(
    "text",
    [
        "Great app, love it", "", None, "The firewall settings are fine",
        "Fireworks and fun", "I scammed nothing",
        "there is no fire hazard", "no overheating", "never overheated",
        "phone does not overheat", "without any overheating",
    ],
)
def test_non_matches_and_negations(text):
    assert scan(text) == []


def test_hinglish_and_devanagari_terms():
    assert scan("Isme dhokha hua")[0].category == "fraud_financial"
    assert scan("यह धोखा है")[0].tier == "high"
    assert scan("charger mein aag lag gayi")[0].category == "physical_safety"
    assert scan("आगे बढ़ो") == []  # "आग" is not matched inside "आगे"


def test_one_alert_per_hit_in_stable_order():
    matches = scan("Scam! Total scam. Also hacked")
    assert [m.phrase for m in matches] == ["Scam", "scam", "hacked"]
    assert matches == sorted(matches, key=lambda m: (m.start, m.end, m.category, m.tier, m.term))


def test_redacted_input_gives_redacted_excerpt():
    (match,) = scan_review(normalize_review("Call [phone] it caught fire, mail [email]", 1))
    assert "[phone]" in match.excerpt and "[email]" in match.excerpt


def test_excerpt_is_redacted_even_if_raw_text_is_passed():
    raw = "jane.doe@example.com 9876543210 hacked"
    (match,) = scan_review(normalize_review(raw, 1))
    assert "jane.doe@example.com" not in match.excerpt and "9876543210" not in match.excerpt
    assert "[email]" in match.excerpt and "[phone]" in match.excerpt


def test_long_text_excerpt_is_short_and_marked():
    text = "word " * 100 + "it caught fire " + "word " * 100
    (match,) = scan_review(normalize_review(text, 1))
    assert len(match.excerpt) < 120
    assert match.excerpt.startswith("…") and match.excerpt.endswith("…")


def test_unicode_and_emoji_around_match():
    (match,) = scan("\U0001F525 it caught fire \U0001F631")
    assert "\U0001F631" in match.excerpt


def test_deterministic():
    text = "scam, hacked, overheats, caught fire"
    assert scan(text) == scan(text)


def test_suggested_action_only_when_configured(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "categories:\n  c:\n    - tier: high\n      suggested_action: Escalate to the owner\n      terms: [zorp]\n",
        encoding="utf-8",
    )
    (match,) = scan_review(normalize_review("a zorp", 1), load_safety_config(path))
    assert match.suggested_action == "Escalate to the owner"


@pytest.mark.parametrize(
    "body",
    [
        "categories: {}\n",
        "categories:\n  c:\n    - tier: urgent\n      terms: [zorp]\n",
        "categories:\n  c:\n    - tier: high\n      terms: []\n",
        "categories:\n  c:\n    - tier: high\n      terms: [no]\n",
    ],
)
def test_invalid_safety_configs_are_rejected(tmp_path, body):
    path = tmp_path / "s.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PipelineConfigError):
        load_safety_config(path)
