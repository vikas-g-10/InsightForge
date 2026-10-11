"""Pipeline stages composed on real ingest_file() output (synthetic data only)."""

import logging

import pandas as pd

from ingest import ingest_file
from pipeline.aspects import detect_aspects
from pipeline.clauses import split_clauses
from pipeline.language import tag_language
from pipeline.normalize import normalize_reviews
from pipeline.safety import scan_review
from pipeline.sentiment import sentiment_for_clause

CSV = (
    "review_text,rating,date\n"
    '"Great app, but it crashes. Call me on 9876543210",2,2024-01-01\n'
    '"Yeh app bahut accha hai lekin delivery late hai",4,2024-01-02\n'
    ",5,2024-01-03\n"
    '"The charger caught fire! mail a@b.com",1,2024-01-04\n'
    '"😍",5,2024-01-05\n'
)
MARKER = "ZZ-UNIQUE-SENTINEL-ZZ"


def run_pipeline(frame):
    out = []
    for review in normalize_reviews(frame):
        clauses = split_clauses(review)
        out.append(
            (
                review.source_row,
                tag_language(review.text).code,
                [(c.text, c.language) for c in clauses],
                [(m.clause_index, m.aspect) for c in clauses for m in detect_aspects(c)],
                [sentiment_for_clause(c).label for c in clauses],
                [(m.category, m.tier, m.excerpt) for m in scan_review(review)],
            )
        )
    return out


def test_pipeline_runs_on_ingestion_output_and_keeps_contract():
    result = ingest_file("reviews.csv", CSV.encode(), max_rows=100)
    frame = result.reviews
    before = frame.copy(deep=True)

    rows = run_pipeline(frame)

    pd.testing.assert_frame_equal(frame, before)  # ingestion frame untouched
    assert [r[0] for r in rows] == list(frame["source_row"])  # empty row excluded upstream
    by_row = {r[0]: r for r in rows}
    first = by_row[2]
    assert first[1] == "en"
    assert ("stability" in {a for _, a in first[3]})
    assert "[phone]" in first[2][-1][0] and "9876543210" not in str(first)  # redacted text analysed
    second = by_row[3]
    assert second[1] == "hinglish"
    assert {a for _, a in second[3]} == {"delivery"}
    assert second[4][0] in ("mixed", "positive")
    fire = by_row[5]
    assert fire[5][0][:2] == ("physical_safety", "critical")
    assert "a@b.com" not in fire[5][0][2]
    assert by_row[6][4] == ["positive"]  # emoji-only review is still analysed


def test_repeated_runs_are_identical():
    frame = ingest_file("reviews.csv", CSV.encode(), max_rows=100).reviews
    assert run_pipeline(frame) == run_pipeline(frame) == run_pipeline(frame.copy())


def test_review_text_is_never_logged(caplog):
    csv = f'review_text\n"{MARKER} it caught fire, scam, not good but crashes"\n'
    with caplog.at_level(logging.DEBUG):
        frame = ingest_file("r.csv", csv.encode(), max_rows=10).reviews
        run_pipeline(frame)
    assert MARKER not in caplog.text
