"""HTTP-level tests for POST /api/ingest/validate."""

import logging

import pytest

from helpers import BASIC_CSV, SENTINEL, URL, error_of, upload, xlsx_bytes
from settings import Settings

ORIGIN = "http://localhost:5173"


def test_valid_csv_upload(client):
    r = upload(client, "reviews.csv", BASIC_CSV.encode(), "text/csv")
    assert r.status_code == 200
    assert r.json() == {
        "source_name": "reviews.csv",
        "file_type": "csv",
        "encoding": "utf-8",
        "original_columns": ["review_text", "rating", "date"],
        "text_column": "review_text",
        "rating_column": "rating",
        "date_column": "date",
        "version_column": None,
        "rows_read": 2,
        "excluded_empty": 0,
        "usable_rows": 2,
        "warnings": [],
    }


def test_valid_xlsx_upload(client):
    data = xlsx_bytes([["Review", "Rating"], ["Nice", 5], ["Meh", 3]])
    r = upload(client, "reviews.xlsx", data)
    assert r.status_code == 200
    body = r.json()
    assert (body["file_type"], body["text_column"], body["usable_rows"]) == ("xlsx", "Review", 2)


def test_response_never_contains_review_text(client):
    r = upload(client, "r.csv", f"review_text,rating\n{SENTINEL},5\n".encode())
    assert r.status_code == 200
    assert SENTINEL not in r.text


def test_content_type_and_filename_are_not_trusted(client):
    # Misleading content type is ignored; real bytes decide.
    ok = upload(client, "r.csv", BASIC_CSV.encode(), "application/pdf")
    assert ok.status_code == 200
    # An xlsx renamed to .csv is rejected despite a "text/csv" content type.
    r = upload(client, "r.csv", xlsx_bytes([["review_text"], ["x"]]), "text/csv")
    assert (r.status_code, error_of(r)["code"]) == (400, "invalid_file")


def test_explicit_text_column_form_field(client):
    data = b"product_name,notes\nWidget,looks cheap\n"
    r = upload(client, "r.csv", data)
    assert r.status_code == 422
    assert error_of(r) == {
        "code": "no_text_column",
        "message": "No review-text column was found. Choose the column that contains the reviews.",
        "columns": ["product_name", "notes"],
    }
    ok = upload(client, "r.csv", data, text_column="notes")
    assert ok.status_code == 200 and ok.json()["text_column"] == "notes"


def test_ambiguous_columns_error_has_columns(client):
    r = upload(client, "r.csv", b"review,comment\nfine,ok\n")
    assert r.status_code == 422
    assert error_of(r)["code"] == "ambiguous_text_column"
    assert error_of(r)["columns"] == ["review", "comment"]


@pytest.mark.parametrize(
    "name, data, status, code",
    [
        ("r.csv", b"", 400, "empty_file"),
        ("r.csv", b"review_text\n", 422, "empty_dataset"),
        ("r.csv", b"a\x00b\x00\x00", 400, "invalid_file"),
        ("r.xlsx", b"garbage", 400, "invalid_file"),
        ("r.txt", b"review_text\nx\n", 415, "unsupported_file_type"),
    ],
)
def test_error_codes_and_statuses(client, name, data, status, code):
    r = upload(client, name, data)
    assert (r.status_code, error_of(r)["code"]) == (status, code)
    assert set(error_of(r)) == {"code", "message"}  # no stray `columns` key


def test_missing_file_uses_validation_envelope(client):
    r = client.post(URL, data={"text_column": "x"})
    assert (r.status_code, error_of(r)["code"]) == (422, "validation_error")


def test_row_limit_via_api(make_client):
    client = make_client(Settings(max_rows=3))
    r = upload(client, "r.csv", ("review_text\n" + "ok\n" * 4).encode())
    assert (r.status_code, error_of(r)["code"]) == (422, "too_many_rows")
    assert upload(client, "r.csv", ("review_text\n" + "ok\n" * 3).encode()).status_code == 200


def test_upload_size_limit_in_handler(make_client):
    # Within the Content-Length slack, so the streaming reader must catch it.
    client = make_client(Settings(max_upload_bytes=1000))
    data = ("review_text\n" + "x" * 2000 + "\n").encode()
    r = upload(client, "r.csv", data)
    assert (r.status_code, error_of(r)["code"]) == (413, "file_too_large")
    small = upload(client, "r.csv", b"review_text\nfine\n")
    assert small.status_code == 200


def test_oversized_content_length_rejected_early_with_cors(make_client):
    client = make_client(Settings(max_upload_bytes=1000))
    big = b"review_text\n" + b"x" * 200_000
    r = client.post(
        URL,
        files={"file": ("r.csv", big, "text/csv")},
        headers={"Origin": ORIGIN},
    )
    assert (r.status_code, error_of(r)["code"]) == (413, "file_too_large")
    assert r.headers["access-control-allow-origin"] == ORIGIN  # CORS stays outermost


def test_env_configured_limits(make_client):
    s = Settings.from_env({"MAX_ROWS": "1", "MAX_UPLOAD_BYTES": "5000"})
    client = make_client(s)
    r = upload(client, "r.csv", b"review_text\na\nb\n")
    assert error_of(r)["code"] == "too_many_rows"


def test_no_review_text_in_error_responses_or_logs(make_client, caplog):
    client = make_client()
    cases = [
        ("r.csv", f"review,comment\n{SENTINEL},{SENTINEL}\n".encode(), {}),
        ("r.csv", f"id,notes\n1,{SENTINEL}\n".encode(), {}),
        ("r.csv", f'review_text\n"{SENTINEL}\n'.encode(), {}),
        ("r.csv", f"review_text\n{SENTINEL},extra\n".encode(), {}),
        ("r.csv", f"notes,rating\n{SENTINEL},2\n".encode(), {"text_column": "rating"}),
        ("r.csv", f"review_text,rating\n{SENTINEL},5\n".encode(), {}),  # success path
    ]
    with caplog.at_level(logging.DEBUG):
        for name, data, form in cases:
            r = upload(client, name, data, **form)
            assert SENTINEL not in r.text
            assert "Traceback" not in r.text
    assert SENTINEL not in caplog.text
    assert SENTINEL not in " ".join(str(rec.args) for rec in caplog.records)


def test_filename_is_sanitized_in_metadata(client):
    r = upload(client, "..\\..\\evil/../my reviews.csv", BASIC_CSV.encode())
    assert r.status_code == 200
    assert r.json()["source_name"] == "my reviews.csv"


def test_health_unchanged(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "mode": "lite",
        "db_connected": False,
    }


def test_valid_csv_below_5mb_accepted(client):
    """Verify that a valid CSV below the previous 5 MB limit is accepted."""
    row = "Great battery life and clear screen,5,2024-01-01\n"
    data = ("review_text,rating,date\n" + row * 100).encode("utf-8")
    assert len(data) < 5 * 1024 * 1024
    r = upload(client, "small.csv", data)
    assert r.status_code == 200
    assert r.json()["usable_rows"] == 100
    assert r.json()["file_type"] == "csv"


def test_valid_csv_above_5mb_and_up_to_50mb_accepted(client):
    """Verify that a valid CSV between 5 MB and 50 MB is accepted."""
    # 1,000 rows of ~6 KB each = ~6 MB (within max_rows=2000 and <= 50 MB)
    row_text = "Detailed feedback note: " + ("helpful device feature " * 280) + ",5,2024-01-01\n"
    data = ("review_text,rating,date\n" + row_text * 1000).encode("utf-8")
    assert 5 * 1024 * 1024 < len(data) <= 50 * 1024 * 1024
    r = upload(client, "large_valid.csv", data)
    assert r.status_code == 200
    assert r.json()["usable_rows"] == 1000
    assert r.json()["file_type"] == "csv"


def test_csv_exceeding_50mb_rejected_with_413(client):
    """Verify that a CSV exceeding the 50 MB limit is rejected with 413 file_too_large."""
    oversized = b"review_text,rating\n" + b"x" * (50 * 1024 * 1024 + 1024)
    r = upload(client, "oversized.csv", oversized)
    assert r.status_code == 413
    assert error_of(r)["code"] == "file_too_large"
    assert "50.0 MB" in error_of(r)["message"]

