import csv
from pathlib import Path

import pytest

from pipeline._common import PipelineConfigError
from pipeline.clauses import split_clauses
from pipeline.normalize import normalize_review
from pipeline.sentiment import (
    MIXED, NEGATIVE, NEUTRAL, POSITIVE, SENTIMENT_LABELS, UNCERTAIN,
    load_sentiment_config, score_sentiment, sentiment_for_clause,
)  # fmt: skip

BENCHMARK = Path(__file__).parent / "fixtures" / "sentiment_benchmark.csv"


def label(text):
    return score_sentiment(text).label


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Great app, love it", POSITIVE),
        ("This is the worst app ever", NEGATIVE),
        ("Terrible, it crashes every time", NEGATIVE),
        ("It is okay", NEUTRAL),
        ("I installed it yesterday", UNCERTAIN),
    ],
)
def test_english_examples(text, expected):
    assert label(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yeh app bahut accha hai", POSITIVE),
        ("Yeh bakwas hai", NEGATIVE),
        ("Paisa vasool app", POSITIVE),
        ("ऐप बहुत बढ़िया है", POSITIVE),
        ("यह बेकार है", NEGATIVE),
        ("Delivery ekdum mast thi", POSITIVE),
    ],
)
def test_hinglish_and_hindi_examples(text, expected):
    assert label(text) == expected


def test_devanagari_precomposed_and_decomposed_forms_match():
    assert label("ऐप ब\u0922\u093c\u093fया है") == POSITIVE  # decomposed nukta
    assert label("ऐप \u095d\u093fया है") == label("ऐप ढ़िया है")  # no crash on odd forms


@pytest.mark.parametrize(
    "text,expected",
    [
        ("not good", NEGATIVE),
        ("This is not good at all", NEGATIVE),
        ("not bad", POSITIVE),
        ("never crashes", POSITIVE),
        ("no issues so far", POSITIVE),
        ("I don't love it", NEGATIVE),
        ("The app doesn't work", NEGATIVE),
        ("App accha nahi hai", NEGATIVE),  # Hindi negator after the term
        ("ऐप अच्छा नहीं है", NEGATIVE),
    ],
)
def test_negation(text, expected):
    assert label(text) == expected


def test_negation_is_dampened_not_a_full_flip():
    plain = score_sentiment("good").score
    negated = score_sentiment("not good").score
    assert negated < 0 and abs(negated) < abs(plain) + 0.01
    hit = score_sentiment("not good").hits[0]
    assert hit.negated and hit.effective_weight == -0.5


def test_negator_outside_window_or_not_only_is_ignored():
    assert not score_sentiment("not once in a long time did I say it was good").hits[0].negated
    assert not score_sentiment("It is not only good but fast").hits[0].negated


def test_mixed_and_uncertain():
    assert label("Good features but very slow") == MIXED
    assert label("Nice design but it keeps crashing") == MIXED
    assert label("Amazing app, only slightly slow") == POSITIVE  # strong positive outweighs
    assert label("") == UNCERTAIN and label(None) == UNCERTAIN
    assert label("xyz abc") == UNCERTAIN


def test_repeated_terms_are_not_double_counted():
    once, many = score_sentiment("good"), score_sentiment("good good good good good")
    assert many.score == once.score and len(many.hits) == 1


def test_one_word_cannot_reach_full_score():
    assert abs(score_sentiment("worst").score) < 1.0
    assert abs(score_sentiment("worst worst worst worst").score) < 1.0


def test_score_is_bounded_and_label_valid():
    for text in ["great " * 50, "worst terrible awful horrible scam fraud", "love hate"]:
        result = score_sentiment(text)
        assert -1.0 < result.score < 1.0 and result.label in SENTIMENT_LABELS


def test_hit_spans_index_into_text():
    text = "Really love this app"
    hit = score_sentiment(text).hits[0]
    assert text[hit.start : hit.end] == hit.text == "love"


def test_prefix_and_phrase_matching():
    assert label("I loved it") == POSITIVE
    assert label("Total waste of money") == NEGATIVE
    assert score_sentiment("Total waste of money").hits[0].term == "waste of money"


def test_emoji():
    assert label("\U0001F60D\U0001F60D") == POSITIVE
    assert label("\U0001F621") == NEGATIVE
    assert label("Love it \u2764\ufe0f") == POSITIVE


def test_clause_level_scoring_keeps_traceability():
    review = normalize_review("Great app, but terrible support", 12)
    first, second = [sentiment_for_clause(c) for c in split_clauses(review)]
    assert (first.source_row, first.clause_index, first.label) == (12, 0, POSITIVE)
    assert (second.source_row, second.clause_index, second.label) == (12, 1, NEGATIVE)


def test_deterministic():
    text = "Good but slow, accha hai lekin mehnga \U0001F60D"
    assert len({score_sentiment(text) for _ in range(5)}) == 1


def test_small_manual_benchmark_meets_baseline():
    rows = list(csv.DictReader(BENCHMARK.open(encoding="utf-8")))
    assert len(rows) >= 40
    correct = sum(score_sentiment(r["text"]).label == r["label"] for r in rows)
    # Guard against regressions only. The benchmark was written by the same
    # author as the lexicon, so this is optimistic, not an accuracy claim.
    assert correct / len(rows) >= 0.9


@pytest.mark.parametrize(
    "body",
    [
        "lexicon: {}\n",
        "negators_before: [no]\nnegators_after: []\nlexicon: {english: {good: 1}}\n",  # unquoted no
        "negators_before: []\nnegators_after: []\nlexicon: {english: {good: 9}}\n",
        "negators_before: []\nnegators_after: []\nlexicon: {english: {good: 1}, other: {good: -1}}\n",
        "negators_before: []\nnegators_after: []\nlexicon: {english: {ab*: 1}}\n",
    ],
)
def test_invalid_lexicons_are_rejected(tmp_path, body):
    path = tmp_path / "lex.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PipelineConfigError):
        load_sentiment_config(path)


def test_custom_lexicon(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text(
        "negators_before: ['not']\nnegators_after: []\nlexicon: {custom: {zorp: 3}}\n", encoding="utf-8"
    )
    cfg = load_sentiment_config(path)
    assert score_sentiment("what a zorp", cfg).label == POSITIVE
    assert score_sentiment("great", cfg).label == UNCERTAIN


def test_sarcasm_heuristic_rule_1():
    # PRD Section 7.4 Rule 1: Positive phrase followed by failure event
    # Same clause
    r1 = score_sentiment("Great, it broke on the first day")
    assert r1.sarcasm_suspect is True
    assert r1.score < 0
    assert r1.polarity == "neg"
    assert r1.label == NEGATIVE

    r2 = score_sentiment("Thanks for arriving 2 weeks late")
    assert r2.sarcasm_suspect is True
    assert r2.score < 0

    r3 = score_sentiment("Waah, stopped working already")
    assert r3.sarcasm_suspect is True
    assert r3.score < 0

    # Across clauses
    r4 = score_sentiment("Great product", next_text="stopped working after two days")
    assert r4.sarcasm_suspect is True
    assert r4.score < 0

    # Non-sarcastic positive
    r5 = score_sentiment("Great product, love it")
    assert r5.sarcasm_suspect is False
    assert r5.score >= 0.25
    assert r5.polarity == "pos"


def test_rating_conflict_rule_2():
    # PRD Section 7.4 Rule 2: Strongly positive text with 1 or 2 stars
    r_pos_1 = score_sentiment("Great app, love it", rating=1)
    assert r_pos_1.rating_conflict is True
    assert r_pos_1.polarity == "pos"

    r_pos_2 = score_sentiment("Great app, love it", rating=2)
    assert r_pos_2.rating_conflict is True

    # Strongly positive with 5 stars: NO conflict
    r_pos_5 = score_sentiment("Great app, love it", rating=5)
    assert r_pos_5.rating_conflict is False

    # Strongly negative with 5 stars: CONFLICT
    r_neg_5 = score_sentiment("Worst app ever, terrible", rating=5)
    assert r_neg_5.rating_conflict is True
    assert r_neg_5.polarity == "neg"

    # Strongly negative with 4 stars: CONFLICT
    r_neg_4 = score_sentiment("Worst app ever, terrible", rating=4)
    assert r_neg_4.rating_conflict is True

    # Strongly negative with 1 star: NO conflict
    r_neg_1 = score_sentiment("Worst app ever, terrible", rating=1)
    assert r_neg_1.rating_conflict is False

    # Neutral / missing rating: NO conflict
    assert score_sentiment("It is okay", rating=1).rating_conflict is False
    assert score_sentiment("Great app", rating=None).rating_conflict is False


def test_polarity_thresholds():
    # PRD Section 5: polarity is neg at <= -0.25, pos at >= 0.25, neu otherwise
    assert score_sentiment("great awesome").polarity == "pos"
    assert score_sentiment("terrible awful").polarity == "neg"
    assert score_sentiment("okay average").polarity == "neu"
