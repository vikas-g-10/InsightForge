"""Text normalization for the NLP pipeline (PRD v4 Sections 4, 7.3, 11).

Builds ``NormalizedReview`` objects from raw values or from the DataFrame that
``ingest.ingest_file()`` returns. Original fields are never modified.

Pipeline steps:
  1. NFC normalization and whitespace collapse.
  2. Artifact and entity cleaning.
  3. Lowercasing (for consistent token matching).
  4. Repeated-character and elongation collapse ("bahuuut" -> "bahut").
  5. Hinglish spelling variants and short forms canonicalization.
  6. Review length truncation at PRD's 1,000-character limit with explicit flag.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_MAX_CHARS = 1000

_WHITESPACE_RE = re.compile(r"\s+")
# Control characters, zero-width space, BOM and soft hyphen. ZWNJ/ZWJ
# (U+200C/U+200D) are kept because Indic scripts rely on them.
_DROP_CHARS = {
    ord(c): None
    for c in [*map(chr, range(0, 32)), *map(chr, range(0x7F, 0xA0)), "\u200b", "\ufeff", "\u00ad"]
}
_CHAR_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2026": "...",
    }
)  # fmt: skip
_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|apos|#39|nbsp);", re.IGNORECASE)
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "#39": "'", "nbsp": " "}
_ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}")  # soooo -> soo, goooood -> good
_PUNCT_RUN_RE = re.compile(r"([!?.])\1{3,}")  # !!!!!! -> !!!
_WORD_RE = re.compile(r"\b[A-Za-z]+(?:\'[A-Za-z]+)?\b")

# Hinglish spelling variants and short forms specified in PRD v4 Section 7.3
HINGLISH_SPELLING_VARIANTS: dict[str, str] = {
    # kharab, kharaab, kharb -> kharab
    "kharaab": "kharab",
    "kharb": "kharab",
    # short forms: nhi, nahi, ni -> nahi
    "nhi": "nahi",
    "ni": "nahi",
    "nahin": "nahi",
    # bahuut, bohot, bohat, bhot -> bahut
    "bohot": "bahut",
    "bohat": "bahut",
    "bahot": "bahut",
    "bhot": "bahut",
    # accha, acha, achha, achchha -> accha
    "acha": "accha",
    "achha": "accha",
    "achchha": "accha",
    # badhiya, badiya -> badhiya
    "badiya": "badhiya",
    # bekar, bekaar -> bekar
    "bekaar": "bekar",
    # bakwas, bakwaas -> bakwas
    "bakwaas": "bakwas",
    # zabardast, jabardast -> zabardast
    "jabardast": "zabardast",
    # mehnga, mahanga -> mehnga
    "mahanga": "mehnga",
    # shandar, shaandaar -> shandar
    "shaandaar": "shandar",
    # kyu -> kyun
    "kyu": "kyun",
    # ye -> yeh
    "ye": "yeh",
    # wo -> woh
    "wo": "woh",
}


def _collapse_elongation(match: re.Match[str]) -> str:
    # Keep "www." in URLs intact; collapse real elongation (sooo -> soo).
    if match.group(0).lower() == "www" and match.string[match.end() : match.end() + 1] == ".":
        return match.group(0)
    return match.group(1) * 2


def _map_hinglish_variants(text: str) -> str:
    # Specific regex to collapse repeated letters in bahut variants ('bahuuut' -> 'bahut')
    text = re.sub(r"\bb+a+h+u+t+\b", "bahut", text, flags=re.IGNORECASE)
    text = re.sub(r"\bb+o+h+[ao]+t+\b", "bahut", text, flags=re.IGNORECASE)
    text = re.sub(r"\bb+h+o+t+\b", "bahut", text, flags=re.IGNORECASE)

    def _replace_word(m: re.Match[str]) -> str:
        word_lower = m.group(0).lower()
        if word_lower in HINGLISH_SPELLING_VARIANTS:
            replacement = HINGLISH_SPELLING_VARIANTS[word_lower]
            return replacement
        return m.group(0)

    return _WORD_RE.sub(_replace_word, text)


@dataclass(frozen=True)
class NormalizedReview:
    source_row: int | None  # stable reference to the ingestion row
    text: str  # normalized text; all later spans index into this
    original_length: int  # characters before normalization and truncation
    truncated: bool = False


def to_text(value: Any) -> str:
    """Convert any value to a string; None, NaN and pandas NA become ''."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass  # array-likes and other non-scalars fall through to str()
    return str(value)


def normalize_text(
    value: Any, *, lowercase: bool = False, map_variants: bool = True
) -> str:
    """Normalize one value: NFC, whitespace, artifacts, elongation, Hinglish variants.
    Optional lowercasing. Never raises.
    """
    raw = to_text(value)
    if not raw.strip():
        return ""
    text = unicodedata.normalize("NFC", raw)
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.translate(_DROP_CHARS).translate(_CHAR_MAP)
    text = _ENTITY_RE.sub(lambda m: _ENTITIES[m.group(1).lower()], text)
    text = _WHITESPACE_RE.sub(" ", text)  # entities (&nbsp;) may add spaces
    text = _ELONGATION_RE.sub(_collapse_elongation, text)
    text = _PUNCT_RUN_RE.sub(r"\1\1\1", text)
    if lowercase:
        text = text.lower()
    if map_variants:
        text = _map_hinglish_variants(text)
    return text.strip()


def normalize_review(
    value: Any,
    source_row: int | None = None,
    *,
    max_chars: int | None = DEFAULT_MAX_CHARS,
    lowercase: bool = False,
    map_variants: bool = True,
) -> NormalizedReview:
    """Normalize one review value, enforcing PRD 1,000-character limit by default."""
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be a positive integer or None.")
    raw = to_text(value)
    text = normalize_text(raw, lowercase=lowercase, map_variants=map_variants)
    truncated = max_chars is not None and len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()
    row = None if source_row is None else int(source_row)
    return NormalizedReview(row, text, len(raw), truncated)


def normalize_reviews(
    frame: pd.DataFrame,
    *,
    text_column: str = "text_redacted",
    max_chars: int | None = DEFAULT_MAX_CHARS,
    lowercase: bool = False,
    map_variants: bool = True,
) -> list[NormalizedReview]:
    """Normalize every row of an ingestion DataFrame, in row order.

    Rows with empty text are kept (with ``text == ""``) so ``source_row``
    traceability is never lost; later stages return nothing for them.
    """
    missing = [c for c in ("source_row", text_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {', '.join(missing)}")
    return [
        normalize_review(
            value, row, max_chars=max_chars, lowercase=lowercase, map_variants=map_variants
        )
        for row, value in zip(frame["source_row"], frame[text_column])
    ]
