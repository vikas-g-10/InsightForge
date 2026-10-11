import pytest

from pipeline.clauses import Clause, split_clauses, split_spans
from pipeline.normalize import normalize_review


def clauses(text, row=1):
    return split_clauses(normalize_review(text, row))


def texts(text):
    return [c.text for c in clauses(text)]


def test_empty_and_whitespace_input():
    assert clauses("") == [] and clauses("   ") == [] and clauses(None) == []


def test_single_clause_review():
    result = clauses("Great app")
    assert [(c.text, c.start, c.end, c.index) for c in result] == [("Great app", 0, 9, 0)]


def test_multiple_sentences():
    assert texts("Great app. Slow delivery! Why?") == ["Great app.", "Slow delivery!", "Why?"]


def test_contrast_words_split_and_trailing_comma_dropped():
    assert texts("Great app, but it crashes") == ["Great app", "but it crashes"]
    assert texts("Accha hai lekin slow hai") == ["Accha hai", "lekin slow hai"]
    assert texts("सुंदर है लेकिन धीमा है") == ["सुंदर है", "लेकिन धीमा है"]


def test_contrast_at_sentence_start_or_after_nothing_does_not_split():
    assert texts("But still good") == ["But still good"]
    assert texts("Although slow, it works.") == ["Although slow, it works."]
    assert texts("Nothing but trouble.") == ["Nothing but trouble."]


def test_commas_and_and_do_not_split():
    assert texts("Fast, cheap and reliable") == ["Fast, cheap and reliable"]


def test_semicolon_splits():
    assert texts("Good value; poor support") == ["Good value;", "poor support"]


@pytest.mark.parametrize(
    "text",
    [
        "Rated 4.5 stars after v2.1.3 update",  # decimals and versions
        "See www.example.com for details",  # domain
        "Features e.g. dark mode are nice.",  # abbreviation
        "Costs Rs. 499 per month",  # abbreviation
        "Made in the U.S. by Dr. Smith",  # abbreviations
    ],
)
def test_periods_inside_common_patterns_do_not_split(text):
    assert texts(text) == [text]


def test_ellipsis_and_repeated_punctuation_are_single_boundaries():
    assert texts("Wait... what?! Really???") == ["Wait...", "what?!", "Really???"]


def test_devanagari_danda_splits():
    assert texts("ऐप अच्छा है। डिलीवरी खराब है।") == ["ऐप अच्छा है।", "डिलीवरी खराब है।"]


def test_emoji_stays_with_its_clause_and_emoji_only_text_is_kept():
    assert texts("Great app. \U0001F60D") == ["Great app. \U0001F60D"]
    assert texts("\U0001F60D Great app") == ["\U0001F60D Great app"]
    assert texts("\U0001F600\U0001F600") == ["\U0001F600\U0001F600"]
    assert texts("Love it \U0001F60D but slow \U0001F621") == ["Love it \U0001F60D", "but slow \U0001F621"]


def test_unicode_text_is_preserved():
    assert texts("Café très bon. Prix élevé") == ["Café très bon.", "Prix élevé"]


def test_traceability_offsets_row_and_language():
    review = normalize_review("The app is great. Delivery bahut late hai! ऐप बेकार है।", 42)
    result = split_clauses(review)
    assert len(result) == 3
    for i, clause in enumerate(result):
        assert isinstance(clause, Clause)
        assert clause.source_row == 42 and clause.index == i
        assert review.text[clause.start : clause.end] == clause.text
    assert [c.language for c in result] == ["en", "hinglish", "hi"]
    assert all(a.end <= b.start for a, b in zip(result, result[1:]))


def test_source_row_none_is_kept():
    assert clauses("ok", row=None)[0].source_row is None


def test_deterministic():
    text = "Great, but slow. Accha hai lekin mehnga; why?"
    assert clauses(text) == clauses(text)
    assert split_spans(text) == split_spans(text)


def test_long_text_is_linear_enough():
    big = "Good app. " * 20_000
    assert len(split_spans(big)) == 20_000


def test_pathological_inputs_stay_fast():
    # Regression: slicing/scanning the whole prefix per match was quadratic.
    for text in ("but " * 30_000, "Good; " * 30_000, "e.g. " * 30_000, "x" * 200_000 + ". ok"):
        assert split_spans(text)
