"""Unit tests for ingest_file (no HTTP)."""

import pandas as pd
import pytest

import ingest
from helpers import BASIC_CSV, SENTINEL, xlsx_bytes, zip_bytes
from ingest import IngestError, ingest_file

MAX = 2000


def run(name, data, **kw):
    return ingest_file(name, data, max_rows=kw.pop("max_rows", MAX), **kw)


def code_of(name, data, **kw):
    with pytest.raises(IngestError) as exc:
        run(name, data, **kw)
    return exc.value


def test_valid_csv():
    result = run("reviews.csv", BASIC_CSV.encode())
    m = result.metadata
    assert (m.file_type, m.rows_read, m.excluded_empty, m.usable_rows) == ("csv", 2, 0, 2)
    assert m.text_column == "review_text"
    assert (m.rating_column, m.date_column, m.version_column) == ("rating", "date", None)
    assert m.original_columns == ["review_text", "rating", "date"]
    df = result.reviews
    assert list(df.columns) == ["source_row", "text", "text_redacted", "is_duplicate", "rating_source", "rating", "date"]
    assert df["is_duplicate"].tolist() == [False, False]
    assert df["rating_source"].tolist() == ["given", "given"]
    assert df["rating"].tolist() == [5.0, 1.0]
    assert df["text"].tolist() == ["Great battery life", "Terrible fit and finish"]
    assert df["source_row"].tolist() == [2, 3]


def test_valid_xlsx_reads_first_sheet_and_warns():
    data = xlsx_bytes(
        [["Review", "Stars"], ["Good value", 4], ["Bad strap", 2.0]], extra_sheet=True
    )
    m = run("r.xlsx", data).metadata
    assert (m.file_type, m.text_column, m.rating_column, m.usable_rows) == (
        "xlsx", "Review", "Stars", 2,
    )  # fmt: skip
    assert m.encoding is None
    assert any("first worksheet" in w for w in m.warnings)
    assert run("r.xlsx", data).reviews["rating"].tolist() == [4.0, 2.0]


def test_duplicate_review_detection():
    # PRD Section 4 Step 1 & Section 5: Exact duplicate detection and is_duplicate flag
    csv = (
        "review_text,rating\n"
        "Great phone,5\n"
        "Terrible battery,1\n"
        "Great phone,5\n"  # Exact duplicate
        "Great phone,4\n"  # Another exact duplicate of text
    )
    result = run("r.csv", csv.encode())
    df = result.reviews
    assert df["is_duplicate"].tolist() == [False, False, True, True]
    assert any("exact duplicate" in w for w in result.metadata.warnings)


def test_rating_coercion_and_validation():
    # PRD Section 11: Coerce to 1 to 5, drop invalid, warn with count
    csv = (
        "review_text,rating\n"
        "Row one,5/5\n"       # 5/5 -> 5.0
        "Row two,4.5\n"       # decimal -> 4.5
        "Row three,8/10\n"     # 8/10 -> 4.0
        "Row four,0\n"        # < 1 -> coerced to 1.0
        "Row five,6\n"        # > 5 -> coerced to 5.0
        "Row six,invalid\n"   # non-numeric -> dropped to None
        "Row seven,\n"        # empty -> None
    )
    result = run("r.csv", csv.encode())
    df = result.reviews
    assert df["rating"].tolist()[:5] == [5.0, 4.5, 4.0, 1.0, 5.0]
    assert pd.isna(df["rating"].iloc[5])
    assert pd.isna(df["rating"].iloc[6])
    assert df["rating_source"].tolist()[:5] == ["given", "given", "given", "given", "given"]
    assert pd.isna(df["rating_source"].iloc[5])
    assert pd.isna(df["rating_source"].iloc[6])

    # Warnings recorded
    assert any("Coerced" in w and "1-5" in w for w in result.metadata.warnings)
    assert any("Dropped" in w and "invalid" in w for w in result.metadata.warnings)


def test_missing_optional_ratings():
    # No rating column in upload
    csv = "review_text\nNice app\n"
    result = run("r.csv", csv.encode())
    df = result.reviews
    assert result.metadata.rating_column is None
    assert "rating" not in df.columns
    assert df["rating_source"].iloc[0] is None


def test_utf8_bom_is_stripped():
    data = ("\ufeff" + BASIC_CSV).encode("utf-8")
    result = run("r.csv", data)
    assert result.metadata.encoding == "utf-8-sig"
    assert result.metadata.original_columns[0] == "review_text"


def test_cp1252_fallback_with_warning():
    data = "review_text\nCaf\u00e9 was \u201cgreat\u201d\n".encode("cp1252")
    result = run("r.csv", data)
    assert result.metadata.encoding == "cp1252"
    assert result.reviews["text"].iloc[0] == "Caf\u00e9 was \u201cgreat\u201d"
    assert any("not valid UTF-8" in w for w in result.metadata.warnings)


def test_utf16_with_bom():
    data = BASIC_CSV.encode("utf-16")
    assert run("r.csv", data).metadata.encoding == "utf-16"


def test_semicolon_delimiter_and_devanagari_text_preserved():
    data = "review;rating\nबहुत अच्छा\u200d phone;5\nokay;3\n".encode()
    df = run("r.csv", data).reviews
    assert df["text"].iloc[0] == "बहुत अच्छा\u200d phone"  # ZWJ kept


def test_multiline_quoted_field_and_row_numbers():
    data = 'review_text,rating\n"line one\nline two",5\nnext,4\n'.encode()
    df = run("r.csv", data).reviews
    assert df["text"].tolist() == ["line one line two", "next"]
    assert df["source_row"].tolist() == [3, 4]


@pytest.mark.parametrize("data", [b"", b"   \n\n  ", b"\xef\xbb\xbf"])
def test_empty_csv_files(data):
    assert code_of("r.csv", data).code == "empty_file"


def test_empty_xlsx_bytes():
    assert code_of("r.xlsx", b"").code == "empty_file"


def test_header_only_files_are_empty_datasets():
    assert code_of("r.csv", b"review_text,rating\n").code == "empty_dataset"
    data = xlsx_bytes([["review_text", "rating"]])
    assert code_of("r.xlsx", data).code == "empty_dataset"


def test_all_blank_text_is_empty_dataset():
    err = code_of("r.csv", b"review_text,rating\n ,5\nN/A,4\n")
    assert err.code == "empty_dataset"


@pytest.mark.parametrize(
    "data",
    [
        b"review_text\nab\x00cd\n",  # NUL bytes: binary content
        b'review_text,rating\n"unterminated,5\n',  # bad quoting
        b"review_text,rating\nok,5,surplus\n",  # more fields than header
    ],
)
def test_invalid_csv(data):
    assert code_of("r.csv", data).code == "invalid_file"


def test_invalid_xlsx_variants():
    good = xlsx_bytes([["review_text"], ["fine"]])
    cases = {
        "random bytes": b"not a workbook at all",
        "truncated zip": good[: len(good) // 2],
        "zip without workbook": zip_bytes({"hello.txt": "hi"}),
        "broken workbook xml": zip_bytes({"xl/workbook.xml": "<oops", "[Content_Types].xml": "<x/>"}),
        "csv text named xlsx": BASIC_CSV.encode(),
    }
    for label, data in cases.items():
        assert code_of("r.xlsx", data).code == "invalid_file", label


def test_xlsx_bytes_named_csv_are_rejected():
    data = xlsx_bytes([["review_text"], ["fine"]])
    assert code_of("r.csv", data).code == "invalid_file"


@pytest.mark.parametrize("name", ["notes.txt", "old.xls", "reviews", "reviews.csv.exe", ""])
def test_unsupported_extension(name):
    err = code_of(name, BASIC_CSV.encode())
    assert (err.code, err.status_code) == ("unsupported_file_type", 415)


def test_row_limit_csv_and_xlsx():
    body = "review_text\n" + "ok\n" * 6
    err = code_of("r.csv", body.encode(), max_rows=5)
    assert (err.code, err.status_code) == ("too_many_rows", 422)
    exact = "review_text\n" + "ok\n" * 5
    assert run("r.csv", exact.encode(), max_rows=5).metadata.rows_read == 5
    data = xlsx_bytes([["review_text"]] + [["ok"]] * 6)
    assert code_of("r.xlsx", data, max_rows=5).code == "too_many_rows"


def test_row_limit_is_enforced_before_redaction(monkeypatch):
    def boom(_text):
        raise AssertionError("redaction ran before the row cap")

    monkeypatch.setattr(ingest, "redact_pii", boom)
    body = "review_text\n" + "ok\n" * 50
    assert code_of("r.csv", body.encode(), max_rows=10).code == "too_many_rows"


def test_blank_rows_are_not_counted_toward_row_limit():
    body = "review_text\n\nok\n\n,\nok\n"
    assert run("r.csv", body.encode(), max_rows=2).metadata.rows_read == 2


@pytest.mark.parametrize(
    "header",
    ["review_text", "Review Text", "REVIEW", "Customer-Feedback", "Comments", "Review_Body"],
)
def test_text_column_aliases(header):
    data = f"id,{header}\n1,works fine\n2,broke fast\n".encode()
    assert run("r.csv", data).metadata.text_column == header


def test_explicit_text_column_exact_and_normalized():
    data = b"product_name,notes\nWidget,looks cheap\nGadget,solid build\n"
    assert code_of("r.csv", data).code == "no_text_column"
    assert run("r.csv", data, text_column="notes").metadata.text_column == "notes"
    assert run("r.csv", data, text_column=" NOTES ").metadata.text_column == "notes"


def test_explicit_column_missing_or_numeric_is_rejected():
    data = b"notes,rating\nlooks cheap,2\nsolid build,5\n"
    missing = code_of("r.csv", data, text_column="nope")
    assert (missing.code, missing.columns) == ("invalid_text_column", ["notes"])
    numeric = code_of("r.csv", data, text_column="rating")
    assert numeric.code == "invalid_text_column"


def test_ambiguous_columns_list_candidates():
    data = b"review,comment\nfine,ok\nbad,meh\n"
    err = code_of("r.csv", data)
    assert (err.code, err.columns) == ("ambiguous_text_column", ["review", "comment"])
    assert run("r.csv", data, text_column="comment").metadata.text_column == "comment"


def test_missing_text_column_offers_text_like_candidates():
    data = b"id,rating,notes\n1,5,hello there\n2,3,so so\n"
    err = code_of("r.csv", data)
    assert (err.code, err.columns) == ("no_text_column", ["notes"])


def test_numeric_alias_column_is_never_selected_as_text():
    data = b"review,feedback\n5,loved it\n3,meh\n"
    assert run("r.csv", data).metadata.text_column == "feedback"
    only_numeric = b"review,id\n5,1\n3,2\n"
    err = code_of("r.csv", only_numeric)
    assert (err.code, err.columns) == ("no_text_column", [])


def test_single_column_file_uses_that_column():
    assert run("r.csv", b"x\nhello\nworld\n").metadata.text_column == "x"


def test_duplicate_headers_are_renamed_not_merged():
    data = b"review,review\na,b\nc,d\n"
    err = code_of("r.csv", data)
    assert err.code == "ambiguous_text_column"
    assert err.columns == ["review", "review (2)"]


def test_empty_rows_are_excluded_and_counted():
    body = (
        "review_text,rating\n"
        "good,5\n"
        ",4\n"          # empty text
        "   ,3\n"       # whitespace only
        "NaN,2\n"       # missing token
        "N/A,1\n"
        ",\n"           # fully blank: ignored, not counted
        "fine,4\n"
    )
    m = run("r.csv", body.encode()).metadata
    assert (m.rows_read, m.excluded_empty, m.usable_rows) == (6, 4, 2)


def test_whitespace_and_control_characters_are_normalized():
    data = "review_text\n  too\t\tmuch \u00a0 space\u200b\x07 \n".encode()
    assert run("r.csv", data).reviews["text"].iloc[0] == "too much space"


def test_redacted_text_column_and_original_kept_in_memory_only():
    data = b"review_text\nCall me on +91 98765 43210 or mail a@b.com\n"
    row = run("r.csv", data).reviews.iloc[0]
    assert "98765" in row["text"] and "a@b.com" in row["text"]
    assert row["text_redacted"] == "Call me on [phone] or mail [email]"


def test_unneeded_columns_are_not_retained():
    data = b"review_text,customer_email\nfine,someone@example.com\n"
    result = run("r.csv", data)
    assert "customer_email" not in result.reviews.columns
    assert "someone@example.com" not in result.reviews.to_csv()


def test_csv_formula_text_is_kept_as_literal_text():
    data = b"review_text\n=1+1\n@cmd\n"
    assert run("r.csv", data).reviews["text"].tolist() == ["=1+1", "@cmd"]


def test_xlsx_formulas_are_never_evaluated():
    data = xlsx_bytes([["review_text", "rating"], ["=1+1", 5], ["real text", 4]])
    result = run("r.xlsx", data)
    # openpyxl wrote no cached value, so the formula cell reads as empty.
    assert result.metadata.excluded_empty == 1
    assert result.reviews["text"].tolist() == ["real text"]


def test_xml_parsing_is_hardened():
    from openpyxl.xml import DEFUSEDXML

    assert DEFUSEDXML is True


def test_error_messages_never_contain_review_text():
    bodies = {
        "ambiguous": f"review,comment\n{SENTINEL},{SENTINEL}\n",
        "no_column": f"id,notes\n1,{SENTINEL}\n",
        "ragged": f"review_text\n{SENTINEL},{SENTINEL}\n",
        "quote": f'review_text\n"{SENTINEL}\n',
    }
    for label, body in bodies.items():
        err = code_of("r.csv", body.encode(), max_rows=MAX)
        text = f"{err.code} {err.message} {err.columns}"
        assert SENTINEL not in text, label
