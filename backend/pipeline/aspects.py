"""Transparent keyword/phrase aspect detection.

Aspect definitions live in ``config/aspects.yaml`` (proposed defaults; the PRD
taxonomy was not available). Each aspect has ``strong`` and ``weak`` terms:

  - one strong match is enough evidence;
  - weak terms (ambiguous words such as "service" or "fast") need at least
    ``min_weak_terms`` DISTINCT weak terms in the same clause;
  - otherwise the aspect is not assigned.

Several aspects can match one clause. Within one aspect, overlapping matches
keep the longest ("customer care" beats "care"). The evidence spans index into
the review text (``review.text[start:end] == evidence.text``). No ML is used.
Aspect detection ignores polarity; "no crashes" still matches ``stability``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline._common import PipelineConfigError, clean_term, compile_terms, load_yaml
from pipeline.clauses import Clause

STRONG, WEAK = "strong", "weak"


@dataclass(frozen=True)
class AspectDefinition:
    key: str
    label: str
    pattern: re.Pattern[str]
    strengths: dict[str, str]  # configured term -> "strong" | "weak"
    ordered_terms: tuple[str, ...]  # index i matches regex group t<i>


@dataclass(frozen=True)
class AspectConfig:
    aspects: tuple[AspectDefinition, ...]
    min_weak_terms: int


@dataclass(frozen=True)
class Evidence:
    start: int
    end: int
    text: str  # exact text that matched
    term: str  # configured term that triggered it
    strength: str


@dataclass(frozen=True)
class AspectMatch:
    source_row: int | None
    clause_index: int
    aspect: str
    evidence: tuple[Evidence, ...]


def load_aspect_config(path: Path | str | None = None) -> AspectConfig:
    data = load_yaml(path, "aspects.yaml")
    min_weak = data.get("min_weak_terms", 2)
    if not isinstance(min_weak, int) or isinstance(min_weak, bool) or min_weak < 1:
        raise PipelineConfigError("aspects: min_weak_terms must be a positive integer.")
    raw = data.get("aspects")
    if not isinstance(raw, dict) or not raw:
        raise PipelineConfigError("aspects: 'aspects' must be a non-empty mapping.")
    definitions = []
    for key, body in raw.items():
        where = f"aspects.{key}"
        if not isinstance(body, dict):
            raise PipelineConfigError(f"{where}: must be a mapping.")
        strengths: dict[str, str] = {}
        for level in (WEAK, STRONG):  # strong written last so it wins on duplicates
            for term in body.get(level) or []:
                strengths[clean_term(term, f"{where}.{level}")] = level
        if not any(v == STRONG for v in strengths.values()):
            raise PipelineConfigError(f"{where}: needs at least one strong term.")
        pattern, ordered = compile_terms(list(strengths))
        definitions.append(
            AspectDefinition(str(key), str(body.get("label", key)), pattern, strengths, ordered)
        )
    return AspectConfig(tuple(definitions), min_weak)


@lru_cache(maxsize=1)
def _default_config() -> AspectConfig:
    return load_aspect_config()


def _non_overlapping(items: list[Evidence]) -> list[Evidence]:
    """Keep the longest evidence when spans overlap; return in position order."""
    kept: list[Evidence] = []
    for item in sorted(items, key=lambda e: (-(e.end - e.start), e.start)):
        if all(item.end <= k.start or item.start >= k.end for k in kept):
            kept.append(item)
    return sorted(kept, key=lambda e: (e.start, e.end))


def find_aspects(text: str, config: AspectConfig | None = None) -> dict[str, list[Evidence]]:
    """Aspect -> evidence for ``text`` (offsets relative to ``text``). Empty if none."""
    cfg = config or _default_config()
    found: dict[str, list[Evidence]] = {}
    for definition in cfg.aspects:
        hits = []
        for match in definition.pattern.finditer(text):
            term = definition.ordered_terms[int(match.lastgroup[1:])]  # type: ignore[index]
            hits.append(
                Evidence(match.start(), match.end(), match.group(0), term, definition.strengths[term])
            )
        hits = _non_overlapping(hits)
        has_strong = any(h.strength == STRONG for h in hits)
        weak_terms = {h.term for h in hits if h.strength == WEAK}
        if has_strong or len(weak_terms) >= cfg.min_weak_terms:
            found[definition.key] = hits
    return found


def detect_aspects(clause: Clause, config: AspectConfig | None = None) -> list[AspectMatch]:
    """Detect aspects in one clause. Offsets in the evidence index into the review."""
    matches = []
    for aspect, hits in find_aspects(clause.text, config).items():
        shifted = tuple(
            Evidence(h.start + clause.start, h.end + clause.start, h.text, h.term, h.strength)
            for h in hits
        )
        matches.append(AspectMatch(clause.source_row, clause.index, aspect, shifted))
    return sorted(matches, key=lambda m: (m.evidence[0].start, m.aspect))


@lru_cache(maxsize=1)
def load_routing_config(path: Path | str | None = None) -> dict[str, dict[str, str]]:
    data = load_yaml(path, "routing.yaml")
    return data.get("routing", {})


def get_aspect_routing(aspect: str, path: Path | str | None = None) -> dict[str, str]:
    """Return team and priority routing for ``aspect`` per PRD Section 7.6."""
    routes = load_routing_config(path)
    return routes.get(aspect.lower(), {"team": "Product/Engineering", "priority": "P2"})

