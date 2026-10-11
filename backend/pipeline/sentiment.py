"""Transparent lexicon-based sentiment baseline with sarcasm & conflict detection (PRD v4).

Scope:
  - Lexicon-based sentiment scoring (-1.0 to 1.0) with Hindi/Hinglish support.
  - Conservative negation handling (before & after windows).
  - Sarcasm heuristic (PRD Section 7.4 Rule 1): positive phrase followed by failure
    event in same or next clause flags sarcasm and scores negative.
  - Rating conflict check (PRD Section 7.4 Rule 2): strong sentiment opposite to star rating.
  - Polarity classification aligned with PRD Data Contract: pos (>= 0.25), neg (<= -0.25), neu.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline._common import PipelineConfigError, clean_term, load_yaml, string_list
from pipeline.clauses import Clause

POSITIVE, NEGATIVE, NEUTRAL, MIXED, UNCERTAIN = "positive", "negative", "neutral", "mixed", "uncertain"
SENTIMENT_LABELS = (POSITIVE, NEGATIVE, NEUTRAL, MIXED, UNCERTAIN)

POS, NEU, NEG = "pos", "neu", "neg"
POLARITY_LABELS = (POS, NEU, NEG)

LABEL_THRESHOLD = 0.25
MIXED_MIN_MASS = 1.0
MIXED_MIN_RATIO = 0.5
NEGATION_FACTOR = -0.5
NEGATOR_WINDOW_BEFORE = 3
NEGATOR_WINDOW_AFTER = 2
MAX_WEIGHT = 3

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+(?:'[\w]+)?|[^\w\s]")

# PRD Section 7.4 Rule 1: Sarcasm positive phrase + failure event patterns
_POSITIVE_SARCASM_RE = re.compile(
    r"\b(?:great|wow|thanks|thank you|waah|wah|superb|awesome|amazing|love|loved|perfect|brilliant|kamaal|shandar|zabardast)\b",
    re.IGNORECASE,
)
_FAILURE_EVENT_RE = re.compile(
    r"\b(?:broke|broken|stopped working|stop working|stops working|late|toot gaya|toot gayi|kharab ho gaya|kharab ho gayi|crashed|crash|crashes|fails|failed|band ho gaya|band ho jata|never arrived|didn't work|didnt work|does not work|doesn't work|doesnt work|not working)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SentimentConfig:
    exact: dict[str, int]
    phrases: dict[tuple[str, ...], int]
    prefixes: tuple[tuple[str, int], ...]  # longest prefix first
    negators_before: frozenset[str]
    negators_after: frozenset[str]
    max_phrase_len: int


@dataclass(frozen=True)
class SentimentHit:
    term: str  # lexicon entry key (e.g. "love*")
    text: str  # text as it appeared
    weight: int
    effective_weight: float
    negated: bool
    start: int
    end: int


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    positive_mass: float
    negative_mass: float
    hits: tuple[SentimentHit, ...]  # counted hits (repeats removed)
    source_row: int | None = None
    clause_index: int | None = None
    polarity: str = NEU  # "pos", "neu", "neg" (PRD Section 5)
    confidence: float = 1.0
    sarcasm_suspect: bool = False
    rating_conflict: bool = False


def load_sentiment_config(path: Path | str | None = None) -> SentimentConfig:
    data = load_yaml(path, "sentiment_lexicon.yaml")
    sections = data.get("lexicon")
    if not isinstance(sections, dict) or not sections:
        raise PipelineConfigError("sentiment_lexicon: 'lexicon' must be a non-empty mapping.")
    entries: dict[str, int] = {}
    for section, body in sections.items():
        if not isinstance(body, dict):
            raise PipelineConfigError(f"sentiment_lexicon.{section}: must be a mapping.")
        for raw_term, weight in body.items():
            term = clean_term(raw_term, f"sentiment_lexicon.{section}")
            if isinstance(weight, bool) or not isinstance(weight, int) or abs(weight) > MAX_WEIGHT:
                raise PipelineConfigError(
                    f"sentiment_lexicon.{section}: weight for {term!r} must be an integer in "
                    f"[-{MAX_WEIGHT}, {MAX_WEIGHT}]."
                )
            if entries.get(term, weight) != weight:
                raise PipelineConfigError(f"sentiment_lexicon: conflicting weights for {term!r}.")
            entries[term] = weight
    exact, phrases, prefixes = {}, {}, []
    for term, weight in entries.items():
        if term.endswith("*"):
            prefixes.append((term[:-1], weight))
        elif " " in term:
            phrases[tuple(term.split(" "))] = weight
        else:
            exact[term] = weight
    prefixes.sort(key=lambda item: (-len(item[0]), item[0]))
    before = frozenset(string_list(data, "negators_before", "sentiment_lexicon"))
    after = frozenset(string_list(data, "negators_after", "sentiment_lexicon"))
    longest = max((len(p) for p in phrases), default=1)
    return SentimentConfig(exact, phrases, tuple(prefixes), before, after, longest)


@lru_cache(maxsize=1)
def _default_config() -> SentimentConfig:
    return load_sentiment_config()


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) == 1 and not token.isalnum() and unicodedata.category(token) != "So":
            continue  # punctuation and variation selectors are not tokens
        tokens.append((unicodedata.normalize("NFC", token.lower()).replace("\u2019", "'"), match.start(), match.end()))
    return tokens


def _lookup(tokens: list[str], i: int, cfg: SentimentConfig) -> tuple[str, int, int] | None:
    """Return (entry key, weight, tokens consumed) for the entry starting at ``i``."""
    for n in range(min(cfg.max_phrase_len, len(tokens) - i), 1, -1):
        key = tuple(tokens[i : i + n])
        if key in cfg.phrases:
            return " ".join(key), cfg.phrases[key], n
    token = tokens[i]
    if token in cfg.exact:
        return token, cfg.exact[token], 1
    for prefix, weight in cfg.prefixes:
        if token.startswith(prefix):
            return prefix + "*", weight, 1
    return None


def _negated_before(words: list[str], first: int, cfg: SentimentConfig) -> bool:
    """True if a negator sits in the window before ``first``. "not only" is not negation."""
    for j in range(max(0, first - NEGATOR_WINDOW_BEFORE), first):
        word = words[j]
        if word in cfg.negators_before or word.endswith("n't"):
            if word == "not" and j + 1 < len(words) and words[j + 1] == "only":
                continue
            return True
    return False


def detect_sarcasm(text: str, next_text: str | None = None) -> bool:
    """PRD Section 7.4 Rule 1: Positive phrase followed by failure event."""
    if not text:
        return False
    pos_match = _POSITIVE_SARCASM_RE.search(text)
    if pos_match:
        if _FAILURE_EVENT_RE.search(text, pos_match.end()):
            return True
        if next_text and _FAILURE_EVENT_RE.search(next_text):
            return True
    return False


def detect_rating_conflict(score: float, rating: float | int | None) -> bool:
    """PRD Section 7.4 Rule 2: Strong text sentiment conflicting with star rating."""
    if rating is None:
        return False
    try:
        r = float(rating)
        if score >= LABEL_THRESHOLD and r <= 2.0:
            return True
        if score <= -LABEL_THRESHOLD and r >= 4.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def score_sentiment(
    text: str | None,
    config: SentimentConfig | None = None,
    *,
    rating: float | int | None = None,
    next_text: str | None = None,
) -> SentimentResult:
    """Score ``text``. Applies lexicon, sarcasm heuristic, and rating conflict checks."""
    cfg = config or _default_config()
    clean_str = text or ""
    spans = _tokenize(clean_str)
    words = [t for t, _, _ in spans]
    hits: list[SentimentHit] = []
    seen: set[str] = set()
    neutral_terms = 0
    i = 0
    while i < len(words):
        found = _lookup(words, i, cfg)
        if found is None:
            i += 1
            continue
        key, weight, used = found
        first, last = i, i + used
        i = last
        if key in seen:
            continue  # repeated term: counted once
        seen.add(key)
        if weight == 0:
            neutral_terms += 1
            continue
        negated = _negated_before(words, first, cfg) or any(
            w in cfg.negators_after for w in words[last : last + NEGATOR_WINDOW_AFTER]
        )
        effective = weight * NEGATION_FACTOR if negated else float(weight)
        start, end = spans[first][1], spans[last - 1][2]
        hits.append(SentimentHit(key, clean_str[start:end], weight, effective, negated, start, end))

    pos = sum(h.effective_weight for h in hits if h.effective_weight > 0)
    neg = -sum(h.effective_weight for h in hits if h.effective_weight < 0)
    score = round((pos - neg) / (pos + neg + 1.0), 4)

    # Base label
    if not hits and not neutral_terms:
        label = UNCERTAIN
        confidence = 0.0
    elif not hits:
        label = NEUTRAL
        confidence = 0.5
    elif min(pos, neg) >= MIXED_MIN_MASS and min(pos, neg) / max(pos, neg) >= MIXED_MIN_RATIO:
        label = MIXED
        confidence = 0.7
    elif score >= LABEL_THRESHOLD:
        label = POSITIVE
        confidence = 1.0
    elif score <= -LABEL_THRESHOLD:
        label = NEGATIVE
        confidence = 1.0
    else:
        label = NEUTRAL
        confidence = 0.7

    # Sarcasm heuristic (Rule 1)
    sarcasm_suspect = detect_sarcasm(clean_str, next_text)
    if sarcasm_suspect:
        score = -max(abs(score), 0.5)
        label = NEGATIVE
        confidence = 0.85

    # Polarity threshold check (-0.25 to 0.25 per PRD Section 5)
    if score >= LABEL_THRESHOLD:
        polarity = POS
    elif score <= -LABEL_THRESHOLD:
        polarity = NEG
    else:
        polarity = NEU

    # Rating conflict check (Rule 2)
    rating_conflict = detect_rating_conflict(score, rating)

    return SentimentResult(
        score, label, round(pos, 4), round(neg, 4), tuple(hits),
        polarity=polarity, confidence=round(confidence, 2),
        sarcasm_suspect=sarcasm_suspect, rating_conflict=rating_conflict,
    )


def sentiment_for_clause(
    clause: Clause,
    config: SentimentConfig | None = None,
    *,
    rating: float | int | None = None,
    next_clause: Clause | None = None,
) -> SentimentResult:
    """Score one clause, carrying its source row, clause index, sarcasm and conflict checks."""
    next_text = next_clause.text if next_clause else None
    base = score_sentiment(clause.text, config, rating=rating, next_text=next_text)
    return SentimentResult(
        base.score, base.label, base.positive_mass, base.negative_mass, base.hits,
        clause.source_row, clause.index,
        polarity=base.polarity, confidence=base.confidence,
        sarcasm_suspect=base.sarcasm_suspect, rating_conflict=base.rating_conflict,
    )
