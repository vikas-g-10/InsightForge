"""Keyword safety scan with negation and context guards (PRD v4 Section 7.2).

Tiers:
  - critical: fire, explosion, sparks, shock, burn, smoke
  - high: allergic reaction, rash, choking, bleeding, injury, fainting
  - medium: overheating, swelling battery, leakage
  - none: no alert or negated hazard

Guards:
  - Negation guard: looks before and after the hit ("no overheating", "koi problem nahi",
    "there is no fire hazard").
  - Context guard: excludes non-hazard idioms such as "burn rate" or "shockingly good".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline._common import PipelineConfigError, clean_term, compile_terms, load_yaml
from pipeline.normalize import NormalizedReview
from redaction import redact_pii

TIER_RANK = {"none": 0, "medium": 1, "high": 2, "critical": 3}
EXCERPT_CONTEXT_CHARS = 40
DISCLAIMER = "Keyword alert: needs human review; not proof of a safety incident."
DEFAULT_ACTION = "escalate to QA and Compliance within 24 hours"

_WORD_SPLIT_RE = re.compile(r"[\w\u0900-\u097F]+|[.!?।॥;]")

_NEGATORS_BEFORE = frozenset(
    {
        "no", "not", "never", "without", "neither", "nor", "hardly", "barely",
        "doesn't", "doesnt", "don't", "dont", "didn't", "didnt", "isn't", "isnt",
        "wasn't", "wasnt", "aren't", "arent", "weren't", "werent",
        "nahi", "nahin", "nhi", "ni", "na", "mat", "bilkul", "koi",
        "न", "नहीं", "नही", "ना", "मत", "बिना", "इल्ला", "illa", "ledu", "alla",
    }
)

_NEGATORS_AFTER = frozenset(
    {
        "nahi", "nahin", "nhi", "ni", "na", "न", "नहीं", "नही", "illa", "ledu", "alla",
    }
)

_NEGATION_PHRASES_AFTER = (
    "koi problem nahi",
    "koi issue nahi",
    "koi dikkat nahi",
    "problem nahi",
    "issue nahi",
    "dikkat nahi",
    "nahi hai",
    "nahi hua",
    "nahi hui",
    "nahi laga",
    "nahi aayi",
    "नहीं है",
    "नहीं हुआ",
    "नहीं हुई",
    "नहीं लगा",
)

# Context false positives that look like hazard words but are non-hazard idioms
_BURN_NON_HAZARD_RE = re.compile(
    r"\bburn(?:ing|ed|s)?\s*(?:[-/]?\s*rate|calories|fat|cd|dvd|money|cash|through)\b|"
    r"\b(?:money|cash)\s+to\s+burn\b",
    re.IGNORECASE,
)
_SHOCK_NON_HAZARD_RE = re.compile(
    r"\bshocking(?:ly)?\s+(?:good|great|fast|cheap|amazing|awesome|nice|impressive|well)\b",
    re.IGNORECASE,
)
_SMOKE_NON_HAZARD_RE = re.compile(
    r"\bsmoke\s+test(?:ing|s)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyGroup:
    category: str
    tier: str
    suggested_action: str | None
    pattern: re.Pattern[str]
    ordered_terms: tuple[str, ...]


@dataclass(frozen=True)
class SafetyConfig:
    groups: tuple[SafetyGroup, ...]


@dataclass(frozen=True)
class SafetyMatch:
    source_row: int | None
    category: str
    tier: str
    term: str  # configured term that triggered the alert
    phrase: str  # exact matched text
    start: int
    end: int
    excerpt: str  # short redacted context around the phrase
    suggested_action: str | None = None
    note: str = DISCLAIMER


def load_safety_config(path: Path | str | None = None) -> SafetyConfig:
    data = load_yaml(path, "safety_terms.yaml")
    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise PipelineConfigError("safety_terms: 'categories' must be a non-empty mapping.")
    groups = []
    for category, tiers in categories.items():
        if not isinstance(tiers, list) or not tiers:
            raise PipelineConfigError(f"safety_terms.{category}: must be a non-empty list.")
        for entry in tiers:
            where = f"safety_terms.{category}"
            if not isinstance(entry, dict) or entry.get("tier") not in TIER_RANK:
                raise PipelineConfigError(f"{where}: each entry needs tier: {' | '.join(TIER_RANK)}.")
            terms = [clean_term(t, where) for t in entry.get("terms") or []]
            if not terms:
                raise PipelineConfigError(f"{where}: entry has no terms.")
            action = entry.get("suggested_action", DEFAULT_ACTION)
            if action is not None and not isinstance(action, str):
                raise PipelineConfigError(f"{where}: suggested_action must be text.")
            pattern, ordered = compile_terms(terms)
            groups.append(SafetyGroup(str(category), entry["tier"], action, pattern, ordered))
    return SafetyConfig(tuple(groups))


@lru_cache(maxsize=1)
def _default_config() -> SafetyConfig:
    return load_safety_config()


def _excerpt(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - EXCERPT_CONTEXT_CHARS), min(len(text), end + EXCERPT_CONTEXT_CHARS)
    if lo > 0:  # start at a word boundary instead of mid-word
        space = text.find(" ", lo, start)
        lo = space + 1 if space != -1 else lo
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        hi = space if space != -1 else hi
    snippet = redact_pii(text[lo:hi].strip())
    return ("…" if lo > 0 else "") + snippet + ("…" if hi < len(text) else "")


def _is_negated(text: str, start: int, end: int) -> bool:
    """Check whether a hazard match is negated in its immediate clause context."""
    # Look back up to 4 words before the hit, stopping at punctuation
    prefix = text[max(0, start - 50) : start]
    tokens_before: list[str] = []
    for m in _WORD_SPLIT_RE.finditer(prefix):
        t = m.group(0).lower()
        if t in {".", "!", "?", "।", "॥", ";"}:
            tokens_before.clear()
        else:
            tokens_before.append(t)
    window_before = tokens_before[-4:]
    if any(t in _NEGATORS_BEFORE for t in window_before):
        return True

    # Look ahead up to 50 chars or 4 words after the hit
    suffix = text[end : min(len(text), end + 50)]
    suffix_lower = suffix.lower()
    for phrase in _NEGATION_PHRASES_AFTER:
        if phrase in suffix_lower:
            # Verify the phrase is within immediate reach
            pos = suffix_lower.find(phrase)
            if pos <= 25:
                return True

    tokens_after: list[str] = []
    for m in _WORD_SPLIT_RE.finditer(suffix):
        t = m.group(0).lower()
        if t in {".", "!", "?", "।", "॥", ";"}:
            break
        tokens_after.append(t)
        if len(tokens_after) >= 4:
            break
    if any(t in _NEGATORS_AFTER for t in tokens_after):
        return True

    return False


def _is_context_false_positive(text: str, start: int, end: int, term: str) -> bool:
    """Check whether a hit is a non-hazard figurative or technical idiom."""
    snippet = text[max(0, start - 30) : min(len(text), end + 40)]
    term_lower = term.lower()

    if term_lower.startswith("burn"):
        if _BURN_NON_HAZARD_RE.search(snippet):
            return True

    if "shock" in term_lower:
        if _SHOCK_NON_HAZARD_RE.search(snippet):
            return True

    if "smoke" in term_lower:
        if _SMOKE_NON_HAZARD_RE.search(snippet):
            return True

    return False


def scan_review(review: NormalizedReview, config: SafetyConfig | None = None) -> list[SafetyMatch]:
    """Return safety alerts for one review, ordered by position then category.

    Applies negation guards and context false-positive guards.
    """
    cfg = config or _default_config()
    text = review.text
    matches: list[SafetyMatch] = []
    for group in cfg.groups:
        for found in group.pattern.finditer(text):
            term = group.ordered_terms[int(found.lastgroup[1:])]  # type: ignore[index]
            start, end = found.start(), found.end()

            if _is_negated(text, start, end):
                continue
            if _is_context_false_positive(text, start, end, term):
                continue

            matches.append(
                SafetyMatch(
                    review.source_row, group.category, group.tier, term, found.group(0),
                    start, end, _excerpt(text, start, end),
                    group.suggested_action,
                )  # fmt: skip
            )
    return sorted(matches, key=lambda m: (m.start, m.end, m.category, m.tier, m.term))


def highest_tier(matches: list[SafetyMatch], default: str | None = None) -> str | None:
    """The most severe tier among ``matches``, or ``default`` when there are none."""
    if not matches:
        return default
    return max((m.tier for m in matches), key=TIER_RANK.__getitem__)
