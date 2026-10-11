"""Deterministic clause splitting with traceable offsets.

Boundaries
  1. Sentence terminators ``. ! ? ... । ॥`` (closing quotes/brackets stay with
     the sentence). A period or ``!?`` only splits when followed by whitespace
     or end of text, so decimals (4.5), versions (v2.1.3) and domains are safe.
     Known abbreviations (e.g., i.e., vs., Mr., Dr., U.S. ...) do not split.
  2. Semicolons.
  3. Contrast words that start a new clause: but, however, although, lekin,
     magar, parantu, and their Devanagari forms. A contrast word at the very
     start of a sentence does not split, and "nothing but"-style phrases are skipped.
Commas and "and" never split on their own ("fast and reliable" stays whole).
Segments with no letters or digits (an emoji after a period) are merged into
the neighbouring clause. Non-empty text always yields at least one clause.

Offsets ``start``/``end`` index into ``NormalizedReview.text`` and satisfy
``review.text[start:end] == clause.text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline._common import WORD_CLASS
from pipeline.language import language_code
from pipeline.normalize import NormalizedReview

_TERMINATOR_RE = re.compile(r"(?:\.{2,}|[.!?]+|[।॥]+)[\"')\]]*")
_DANDA_RE = re.compile(r"[।॥]")
_CONTRAST_RE = re.compile(
    rf"(?<![{WORD_CLASS}])(?:but|however|although|lekin|magar|par|parantu|परंतु|लेकिन|मगर|पर)(?![{WORD_CLASS}])",
    re.IGNORECASE,
)
_PREV_WORD_RE = re.compile(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?)$")
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "vs", "mr", "mrs", "ms", "dr", "approx", "inc", "ltd", "rs", "ver", "fig",
     "pvt", "sr", "jr", "u.s", "u.k"}
)  # fmt: skip
_NOT_CONTRAST_AFTER = frozenset({"nothing", "anything", "everything", "all", "cannot", "anyone"})
_PREV_TOKEN_RE = re.compile(r"([^\W\d_]+)\W*$")

_OPINION_CLAUSE_START = re.compile(
    r"^(?:great|good|nice|excellent|amazing|superb|poor|bad|terrible|awful|horrible|slow|fast|worst|best|accha|acha|kharab|bekar|bakwas|ghatiya)\s+[a-zA-Z\u0900-\u097F]+",
    re.IGNORECASE,
)
_ASPECT_CLAUSE_START = re.compile(
    r"^(?:battery|camera|delivery|performance|screen|display|sound|audio|support|service|quality|design|price|packaging)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Clause:
    source_row: int | None
    index: int  # 0-based position within the review
    text: str
    start: int
    end: int
    language: str


_LOOKBACK = 12  # longer than any listed abbreviation; keeps matching linear overall


def _is_abbreviation(text: str, dot_index: int) -> bool:
    lo = max(0, dot_index - _LOOKBACK)
    match = _PREV_WORD_RE.search(text[lo:dot_index])
    if not match or (match.start() == 0 and lo > 0):
        return False  # word is longer than the lookback window
    return match.group(1).lower() in _ABBREVIATIONS


def _sentence_ends(text: str) -> list[int]:
    """Exclusive end offsets of sentences (each includes its terminator)."""
    ends: list[int] = []
    for match in _TERMINATOR_RE.finditer(text):
        end = match.end()
        run = match.group(0)
        at_edge = end >= len(text) or text[end].isspace()
        if not (at_edge or _DANDA_RE.search(run)):
            continue
        if run.startswith(".") and not run.startswith("..") and run.rstrip("\"')]") == ".":
            if _is_abbreviation(text, match.start()):
                continue
        ends.append(end)
    if not ends or ends[-1] < len(text):
        ends.append(len(text))
    return ends


_FIRST_ALNUM_RE = re.compile(r"[^\W_]")


def _contrast_cuts(text: str, lo: int, hi: int) -> list[int]:
    first = _FIRST_ALNUM_RE.search(text, lo, hi)
    if first is None:
        return []
    cuts: list[int] = []
    for match in _CONTRAST_RE.finditer(text, lo, hi):
        if match.start() <= first.start():
            continue  # nothing before it in this sentence
        window = text[max(lo, match.start() - 30) : match.start()]
        prev = _PREV_TOKEN_RE.search(window)
        if prev and prev.group(1).lower() in _NOT_CONTRAST_AFTER:
            continue
        cuts.append(match.start())
    return cuts


def _comma_cuts(text: str, lo: int, hi: int) -> list[int]:
    """Split on commas that separate independent aspect/opinion clauses."""
    cuts: list[int] = []
    for i in range(lo, hi):
        if text[i] == ",":
            after = text[i + 1 : hi].lstrip()
            # Do not split if following word is "and" or "or" (e.g. series list)
            if re.match(r"^(?:and|or)\b", after, re.IGNORECASE):
                continue
            if _OPINION_CLAUSE_START.match(after) or _ASPECT_CLAUSE_START.match(after):
                before = text[lo:i].strip()
                before_tokens = [w for w in before.split() if w.isalnum()]
                if len(before_tokens) >= 2 or any(
                    w in before.lower()
                    for w in ("camera", "battery", "delivery", "quality", "sound", "price", "support", "design")
                ):
                    cuts.append(i + 1)
    return cuts


def split_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans for the clauses of already-normalized text."""
    if not text.strip():
        return []
    raw: list[tuple[int, int]] = []
    lo = 0
    for sentence_end in _sentence_ends(text):
        cuts = [lo]
        cuts += _contrast_cuts(text, lo, sentence_end)
        cuts += _comma_cuts(text, lo, sentence_end)
        for i in range(lo, sentence_end):
            if text[i] == ";":
                cuts.append(i + 1)
        cuts = sorted(set(cuts)) + [sentence_end]
        raw.extend(zip(cuts, cuts[1:]))
        lo = sentence_end

    spans: list[tuple[int, int]] = []
    pending_start: int | None = None
    for start, end in raw:
        piece = text[start:end]
        stripped = piece.strip()
        if not stripped:
            continue
        start += len(piece) - len(piece.lstrip())
        end = start + len(stripped.rstrip(", "))  # drop a trailing comma before "but"
        if not any(ch.isalnum() for ch in text[start:end]):
            if spans:  # symbols/emoji only: attach to the previous clause
                spans[-1] = (spans[-1][0], end)
            elif pending_start is None:
                pending_start = start
            continue
        if pending_start is not None:
            start, pending_start = pending_start, None
        spans.append((start, end))
    if not spans:  # no letters or digits anywhere: keep the whole text as one clause
        spans.append((len(text) - len(text.lstrip()), len(text.rstrip())))
    return spans


def split_clauses(review: NormalizedReview) -> list[Clause]:
    """Split a normalized review into clauses tagged with language and source row."""
    return [
        Clause(review.source_row, i, review.text[s:e], s, e, language_code(review.text[s:e]))
        for i, (s, e) in enumerate(split_spans(review.text))
    ]
