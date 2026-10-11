# InsightForge backend (Milestones 1-3: foundation, ingestion, Lite NLP core)

FastAPI service, Lite mode only. No PyTorch, transformers, model downloads,
database, or API keys are needed. So far: `GET /health`, CORS, a consistent
error format, and CSV/XLSX upload validation (`POST /api/ingest/validate`).
The Lite NLP core lives in `pipeline/` (library code only; no analysis endpoint yet).

Requires Python 3.11+ (developed and tested here on 3.12).

## Setup

```bash
cd backend
python3.11 -m venv .venv          # any Python >= 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # runtime + test deps
# runtime only: pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
```

Optional `.env` (see `../.env.example`):

```bash
cp ../.env.example ../.env
uvicorn main:app --reload --env-file ../.env
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `APP_MODE` | `lite` | `lite` or `full` (case-insensitive). Anything else fails startup with a clear error. `full` only changes the reported mode for now. |
| `MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Positive integer. Larger uploads get `413 file_too_large`. |
| `MAX_ROWS` | `2000` | Positive integer. Data rows above this get `422 too_many_rows`. |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated explicit origins. `*` and malformed values fail startup. Credentials are disabled. |

## `GET /health`

```json
{"status": "ok", "mode": "lite", "db_connected": false}
```

`db_connected` is `true` only when a database connection has actually been
checked and confirmed. No database check exists yet, so it is always `false`.
The endpoint exposes no secrets, env values, paths, or stack traces.

## `POST /api/ingest/validate`

Multipart upload that validates a review file and returns metadata only
(never review text). Fields: `file` (required, `.csv` or `.xlsx`) and
`text_column` (optional, a column name to use as the review text).

```bash
curl -F "file=@reviews.csv" http://localhost:8000/api/ingest/validate
curl -F "file=@reviews.xlsx" -F "text_column=Comments" http://localhost:8000/api/ingest/validate
```

```json
{"source_name": "reviews.csv", "file_type": "csv", "encoding": "utf-8",
 "original_columns": ["Review Text", "Rating", "Date"],
 "text_column": "Review Text", "rating_column": "Rating",
 "date_column": "Date", "version_column": null,
 "rows_read": 2, "excluded_empty": 0, "usable_rows": 2, "warnings": []}
```

Behavior:
- **Type checks** use the extension *and* file signature; the client content type
  is ignored. `.xls`, `.xlsm`, `.txt`, etc. are unsupported.
- **Encodings:** UTF-8 (with/without BOM) and UTF-16 with BOM; otherwise falls
  back to Windows-1252 and finally Latin-1, with a warning. Delimiters `, ; tab |`
  are auto-detected.
- **XLSX:** first worksheet only, read in read-only mode using cached values.
  Formulas are never evaluated. Archives are checked for zip-bomb size.
- **Rows** are data rows below the header; completely blank rows are ignored and
  not counted. The cap is enforced while reading, before any cleaning.
- **Text column:** matched by normalized header against a documented alias list
  (`TEXT_ALIASES` in `ingest.py`: `review_text`, `review`, `comment(s)`,
  `feedback`, ...). A single-column file uses that column. Numeric-looking
  columns (e.g. ratings) are never chosen. Otherwise the API returns
  `ambiguous_text_column` or `no_text_column` with a `columns` list for a picker;
  resend with `text_column` to resolve.
- **Optional columns** detected by alias: rating, date, version (names only; values
  are not parsed yet).
- **Cleaning:** whitespace collapsed, control chars removed, `""`, `nan`, `none`,
  `null`, `n/a`, `#n/a` treated as missing. Rows with empty review text are
  excluded and counted in `excluded_empty`.
- **Privacy:** `redaction.redact_pii` replaces common emails and phone numbers
  with `[email]`/`[phone]` (best effort, not guaranteed). Original text stays in
  process memory only; nothing is stored, and review text is never logged.
  Starlette may spool large multipart bodies to a temp file until the request
  ends.

## Errors

Every error uses one envelope:

```json
{"error": {"code": "not_found", "message": "Not Found"}}
```

When a column picker is needed, the error also carries `"columns": [...]`
(omitted otherwise).

| Code | HTTP | Meaning |
|---|---|---|
| `unsupported_file_type` | 415 | Extension is not `.csv`/`.xlsx` |
| `file_too_large` | 413 | Upload exceeds `MAX_UPLOAD_BYTES` |
| `too_many_rows` | 422 | More than `MAX_ROWS` data rows |
| `empty_file` | 400 | No bytes / no content |
| `invalid_file` | 400 | Corrupt, binary, malformed or mismatched content |
| `empty_dataset` | 422 | Header only, or every review text is blank |
| `no_text_column` | 422 | No usable review-text column (`columns` = picker candidates) |
| `ambiguous_text_column` | 422 | Several candidate columns (`columns` lists them) |
| `invalid_text_column` | 422 | `text_column` not found or not text-like (`columns` = candidates) |
| `validation_error` | 422 | Malformed request, e.g. missing `file` |
| `not_found`, `method_not_allowed`, `internal_error` | 404/405/500 | Generic |

Also `bad_request`, `payload_too_large` and `http_error` for other statuses. Messages are generic; validation details
and exception text are never returned. Unexpected errors are logged server-side
with a traceback.

## Test

```bash
cd backend
pytest
```

Tests need no network, Supabase, or models, and build an isolated app per test.
You may see a Starlette deprecation warning about `httpx`; it is harmless.

## Lite NLP core (`pipeline/`, Milestone 3)

Rule-based, CPU-only, deterministic. No models, no downloads, no review text in
logs. Only new dependency: `PyYAML`. There is no endpoint yet; stages are plain
functions that work on the DataFrame from `ingest_file()`.

| Module | Main functions | Notes |
|---|---|---|
| `normalize.py` | `normalize_reviews(frame)`, `normalize_review`, `normalize_text` | Analyses `text_redacted` by default. Never mutates the frame. Optional `max_chars`. |
| `language.py` | `tag_language`, `language_code` | `en`, `hi`, `hinglish`, `unknown`. Script ratio plus marker words. |
| `clauses.py` | `split_clauses(review)` | Sentence ends, `;`, and contrast words (but, lekin...). Offsets index `review.text`. |
| `aspects.py` | `detect_aspects(clause)` | Strong/weak keyword rules with exact evidence spans. |
| `sentiment.py` | `score_sentiment`, `sentiment_for_clause` | Lexicon baseline: score in (-1, 1), label `positive/negative/neutral/mixed/uncertain`. |
| `safety.py` | `scan_review(review)`, `highest_tier` | Keyword alerts with tier, category and a redacted excerpt. |

Editable rules are in `pipeline/config/*.yaml` (aspects, sentiment lexicon,
safety terms, language markers). Quote YAML words like `"no"`/`"on"`; invalid
config raises `PipelineConfigError` at load time.

**Proposed defaults, not PRD-derived.** The PRD was not available when this
milestone was built. The aspect taxonomy, sentiment labels and thresholds, the
Hindi/Hinglish term lists, safety categories/tiers/terms, and "no suggested
action" are all proposals. Reconcile them with the PRD before relying on them.

Known limitations
- **Review length:** no limit is applied (the PRD value is unknown; ingestion
  enforces only upload and row caps). `max_chars` exists but is unset.
- **Offsets** index the normalized text (`NormalizedReview.text`), not the raw upload.
- **Language:** heuristic. Short text is noisy; other Latin-script languages that
  share English words can read as `en`; Kannada/Tamil/Telugu are not supported (`unknown`).
- **Sentiment:** a lexicon, not a trained model. No sarcasm, no intensifiers,
  no double negation. The 40-line benchmark (`tests/fixtures/sentiment_benchmark.csv`)
  scores 38/40 but was written by the same author as the lexicon, so treat it as
  optimistic. Known misses: sarcasm ("Oh great another crash") and phrases not in
  the lexicon ("Not worth it").
- **Clauses:** no split on "and" or commas; abbreviations are a short list.
- **Aspects:** presence only, polarity is ignored ("no crashes" still hits `stability`).
- **Safety:** keyword alerts for human review, not findings. No negation handling.
  Excerpts are re-redacted, but a spaced phone number cut at an excerpt edge could
  leave a fragment; scan `text_redacted` (the default) as the primary protection.
