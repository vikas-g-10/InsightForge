import math

import pandas as pd
import pytest

from pipeline.normalize import (
    NormalizedReview,
    normalize_review,
    normalize_reviews,
    normalize_text,
    to_text,
)


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", "   ", "\n\t "])
def test_missing_and_blank_values_become_empty(value):
    assert normalize_text(value) == ""
    review = normalize_review(value, 7)
    assert review.text == "" and review.source_row == 7


def test_non_string_values_are_handled_safely():
    assert to_text(42) == "42"
    assert to_text(b"caf\xc3\xa9") == "café"
    assert to_text(b"\xff") == "\ufffd"
    assert to_text([1, 2]) == "[1, 2]"  # array-likes do not crash pd.isna
    assert normalize_text(3.5) == "3.5"


def test_whitespace_is_collapsed_and_trimmed():
    assert normalize_text("  great \n\n app\t\tnow  ") == "great app now"


def test_artifacts_are_normalized():
    assert normalize_text("it\u2019s \u201cgood\u201d \u2013 really\u2026") == "it's \"good\" - really..."
    assert normalize_text("Tom &amp; Jerry&nbsp;rock") == "Tom & Jerry rock"
    assert normalize_text("Tom &amp; Jerry&nbsp;rock", lowercase=True) == "tom & jerry rock"
    assert normalize_text("a\u200bb\ufeffc\x00d") == "abcd"
    assert normalize_text("sooooo goooood!!!!!!") == "soo good!!!"


def test_indic_joiners_survive():
    zwnj = "क\u200cख"
    assert normalize_text(zwnj) == zwnj


def test_unicode_is_nfc_and_emoji_survive():
    assert normalize_text("cafe\u0301") == "caf\u00e9"
    assert normalize_text("Love it \U0001F60D\U0001F44D") == "Love it \U0001F60D\U0001F44D"
    assert normalize_text("Love it \U0001F60D\U0001F44D", lowercase=True) == "love it \U0001F60D\U0001F44D"


def test_repeated_letters_collapse_and_hinglish_variants():
    # PRD explicit example: 'bahuuut' to 'bahut'
    assert normalize_text("bahuuut") == "bahut"
    assert normalize_text("Yeh bahuuut kharaab product hai, nhi chal raha", lowercase=True) == (
        "yeh bahut kharab product hai, nahi chal raha"
    )
    # PRD Section 7.3: kharab, kharaab, kharb -> kharab
    assert normalize_text("kharaab") == "kharab"
    assert normalize_text("kharb") == "kharab"
    # PRD short forms: nhi, nahi, ni -> nahi
    assert normalize_text("nhi") == "nahi"
    assert normalize_text("ni") == "nahi"
    # Other Hinglish common forms
    assert normalize_text("acha") == "accha"
    assert normalize_text("badiya") == "badhiya"
    assert normalize_text("jabardast") == "zabardast"


def test_idempotent():
    once = normalize_text("  Sooooo   good!!!!! &amp; \u201cfast\u201d ")
    assert normalize_text(once) == once


def test_prd_default_1000_char_truncation():
    # PRD Section 11: Truncate each review to 1,000 characters before analysis
    big_text = "word " * 300  # 1500 chars
    review = normalize_review(big_text, 1)
    assert review.truncated is True
    assert len(review.text) <= 1000
    assert review.original_length == len(big_text)

    # Short review is not truncated
    short_review = normalize_review("Everything is great", 1)
    assert short_review.truncated is False
    assert len(short_review.text) < 1000


def test_explicit_none_length_limit():
    review = normalize_review("a b " * 10_000, 1, max_chars=None)
    assert not review.truncated and len(review.text) == 39_999


def test_max_chars_truncates_and_flags():
    review = normalize_review("word " * 100, 4, max_chars=20)
    assert review.truncated and len(review.text) <= 20
    assert review.original_length == 500
    short = normalize_review("ok", 4, max_chars=20)
    assert not short.truncated


def test_invalid_max_chars_rejected():
    with pytest.raises(ValueError):
        normalize_review("x", 1, max_chars=0)


def test_normalize_reviews_uses_redacted_text_and_keeps_source_rows():
    frame = pd.DataFrame(
        {
            "source_row": [2, 5, 9],
            "text": ["mail me a@b.com", "ok", None],
            "text_redacted": ["mail me [email]", "ok", None],
        }
    )
    snapshot = frame.copy(deep=True)
    reviews = normalize_reviews(frame)
    assert [r.source_row for r in reviews] == [2, 5, 9]
    assert reviews[0].text == "mail me [email]"
    assert reviews[2].text == ""
    assert all(isinstance(r.source_row, int) for r in reviews)
    pd.testing.assert_frame_equal(frame, snapshot)  # originals untouched
    assert normalize_reviews(frame, text_column="text")[0].text == "mail me a@b.com"


def test_normalize_reviews_reports_missing_columns_without_text():
    with pytest.raises(ValueError, match="source_row"):
        normalize_reviews(pd.DataFrame({"text_redacted": ["secret"]}))
    with pytest.raises(ValueError, match="text_redacted"):
        normalize_reviews(pd.DataFrame({"source_row": [1]}))


def test_review_is_frozen():
    review = NormalizedReview(1, "x", 1)
    with pytest.raises(Exception):
        review.text = "y"  # type: ignore[misc]
    assert not math.isnan(review.original_length)
