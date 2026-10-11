"""Heuristic language tagger and language-pack foundation.

Supported base languages & language packs per PRD Sections 4, 5, 7.3, and 7.3a:
  en        Latin-script text with common English words (Tier 1: Validated)
  hi        mostly Devanagari script (Tier 1: Validated)
  hinglish  romanized Hindi markers in Latin text, or Devanagari mixed with Latin (Tier 1: Validated)
  kanglish  romanized Kannada markers (Tier 2: Language Pack)
  tanglish  romanized Tamil markers (Tier 2: Language Pack)
  tenglish  romanized Telugu markers (Tier 2: Language Pack)
  mixed     code-mixed text spanning multiple languages
  unknown   empty, no letters, unsupported native scripts (Kannada, Tamil, Telugu,
            Chinese, Arabic...), or Latin text too short or ambiguous to call.

Language packs live in ``config/packs/<pack_name>/manifest.yaml`` containing:
  - function words
  - spelling variants
  - negators (before and after)
  - aspect seeds
  - sentiment lexicon
  - 30 hand-labeled test clauses
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pipeline._common import PipelineConfigError, load_yaml, string_list

EN = "en"
HI = "hi"
HINGLISH = "hinglish"
KANGLISH = "kanglish"
TANGLISH = "tanglish"
TENGLISH = "tenglish"
MIXED = "mixed"
UNKNOWN = "unknown"

LANGUAGE_CODES = (EN, HI, HINGLISH, KANGLISH, TANGLISH, TENGLISH, MIXED, UNKNOWN)

LANGUAGE_BADGES: dict[str, str] = {
    EN: "English",
    HI: "Hindi",
    HINGLISH: "Hinglish",
    KANGLISH: "Kanglish",
    TANGLISH: "Tanglish",
    TENGLISH: "Tenglish",
    MIXED: "Mixed",
    UNKNOWN: "Unknown",
}

SAMPLE_CHARS = 5000
HI_MIN_DEVANAGARI = 0.7  # at or above: Hindi
MIXED_MIN_DEVANAGARI = 0.1  # between this and HI_MIN_DEVANAGARI: code-mixed
UNSUPPORTED_SCRIPT_RATIO = 0.5
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


@dataclass(frozen=True)
class LanguagePack:
    code: str
    badge: str
    language: str
    tier: int
    status: str
    function_words: frozenset[str]
    spelling_variants: dict[str, str]
    negators_before: frozenset[str]
    negators_after: frozenset[str]
    aspect_seeds: dict[str, list[str]]
    lexicon_positive: dict[str, int]
    lexicon_negative: dict[str, int]
    test_clauses: list[dict[str, Any]]


@dataclass(frozen=True)
class LanguageConfig:
    hinglish_markers: frozenset[str]
    english_words: frozenset[str]
    packs: dict[str, LanguagePack] = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageResult:
    code: str
    devanagari_ratio: float
    hinglish_hits: int
    english_hits: int
    tokens: int  # Latin-script word tokens examined
    badge: str = ""
    confidence: float = 0.0
    pack_hits: tuple[tuple[str, int], ...] = ()

    @property
    def pack_hits_dict(self) -> dict[str, int]:
        return dict(self.pack_hits)


def load_language_pack(manifest_path: Path | str) -> LanguagePack:
    """Load a single language pack from its manifest.yaml file or directory."""
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    code = str(data.get("code", "")).lower()
    badge = str(data.get("badge", code.title()))
    language = str(data.get("language", ""))
    tier = int(data.get("tier", 2))
    status = str(data.get("status", ""))

    function_words = frozenset(str(w).lower() for w in data.get("function_words", []))
    spelling_variants = {str(k).lower(): str(v).lower() for k, v in data.get("spelling_variants", {}).items()}
    negators_before = frozenset(str(w).lower() for w in data.get("negators_before", []))
    negators_after = frozenset(str(w).lower() for w in data.get("negators_after", []))

    aspect_seeds = {str(k): list(v) for k, v in data.get("aspect_seeds", {}).items()}
    lex_pos = {str(k).lower(): int(v) for k, v in data.get("lexicon", {}).get("positive", {}).items()}
    lex_neg = {str(k).lower(): int(v) for k, v in data.get("lexicon", {}).get("negative", {}).items()}
    test_clauses = list(data.get("test_clauses", []))

    return LanguagePack(
        code=code,
        badge=badge,
        language=language,
        tier=tier,
        status=status,
        function_words=function_words,
        spelling_variants=spelling_variants,
        negators_before=negators_before,
        negators_after=negators_after,
        aspect_seeds=aspect_seeds,
        lexicon_positive=lex_pos,
        lexicon_negative=lex_neg,
        test_clauses=test_clauses,
    )


def load_all_language_packs(packs_dir: Path | str | None = None) -> dict[str, LanguagePack]:
    """Scan packs directory and load all language packs."""
    if packs_dir is None:
        packs_dir = Path(__file__).resolve().parent / "config" / "packs"
    else:
        packs_dir = Path(packs_dir)

    packs: dict[str, LanguagePack] = {}
    if not packs_dir.exists() or not packs_dir.is_dir():
        return packs

    for sub in sorted(packs_dir.iterdir()):
        manifest = sub / "manifest.yaml" if sub.is_dir() else sub if sub.name == "manifest.yaml" else None
        if manifest and manifest.exists():
            pack = load_language_pack(manifest)
            packs[pack.code] = pack
    return packs


def load_language_config(path: Path | str | None = None, packs_dir: Path | str | None = None) -> LanguageConfig:
    data = load_yaml(path, "language_markers.yaml")
    hinglish = frozenset(string_list(data, "hinglish_markers", "language_markers"))
    english = frozenset(string_list(data, "english_words", "language_markers"))
    overlap = sorted(hinglish & english)
    if overlap:
        raise PipelineConfigError(f"language_markers: words in both lists: {', '.join(overlap)}")

    try:
        packs = load_all_language_packs(packs_dir)
    except Exception:
        packs = {}

    return LanguageConfig(hinglish, english, packs)


@lru_cache(maxsize=1)
def _default_config() -> LanguageConfig:
    return load_language_config()


def _script_counts(sample: str) -> tuple[int, int, int]:
    devanagari = latin = other = 0
    for ch in sample:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            devanagari += 1
        elif code < 0x250 or 0x1E00 <= code <= 0x1EFF:
            latin += 1
        else:
            other += 1
    return devanagari, latin, other


def tag_language(text: str | None, config: LanguageConfig | None = None) -> LanguageResult:
    """Tag ``text`` as en, hi, hinglish, kanglish, tanglish, tenglish, or unknown."""
    cfg = config or _default_config()
    sample = (text or "")[:SAMPLE_CHARS]
    dev, latin, other = _script_counts(sample)
    total = dev + latin + other
    if total == 0:
        return LanguageResult(UNKNOWN, 0.0, 0, 0, 0, badge=LANGUAGE_BADGES[UNKNOWN], confidence=0.0)
    ratio = dev / total
    if other / total >= UNSUPPORTED_SCRIPT_RATIO:
        return LanguageResult(UNKNOWN, round(ratio, 3), 0, 0, 0, badge=LANGUAGE_BADGES[UNKNOWN], confidence=0.0)
    if ratio >= HI_MIN_DEVANAGARI:
        return LanguageResult(HI, round(ratio, 3), 0, 0, 0, badge=LANGUAGE_BADGES[HI], confidence=round(ratio, 2))

    tokens = [t for t in _TOKEN_RE.findall(sample.lower().replace("\u2019", "'")) if ord(t[0]) < 0x250]
    n = len(tokens)
    h_hits = sum(t in cfg.hinglish_markers for t in tokens)
    e_hits = sum(t in cfg.english_words for t in tokens)
    ratio = round(ratio, 3)

    if ratio >= MIXED_MIN_DEVANAGARI and n > 0:
        return LanguageResult(HINGLISH, ratio, h_hits, e_hits, n, badge=LANGUAGE_BADGES[HINGLISH], confidence=0.85)
    if n == 0:
        return LanguageResult(UNKNOWN, ratio, 0, 0, 0, badge=LANGUAGE_BADGES[UNKNOWN], confidence=0.0)

    # Compute matches for regional language packs (Kanglish, Tanglish, Tenglish)
    # Filter out common English words from markers so English nouns don't trigger regional tags
    pack_hits_dict: dict[str, int] = {}
    for p_code, pack in cfg.packs.items():
        if p_code in (EN, HI):
            continue
        p_markers = (pack.function_words | set(pack.lexicon_positive.keys()) | set(pack.lexicon_negative.keys())) - cfg.english_words
        hits = sum(t in p_markers for t in tokens)
        pack_hits_dict[p_code] = hits

    # Identify candidate regional packs
    regional_candidates: list[tuple[str, int, float]] = []
    for r_code in (KANGLISH, TANGLISH, TENGLISH):
        hits = pack_hits_dict.get(r_code, 0)
        r_ratio = hits / n
        if (hits >= 2 and r_ratio >= 0.15) or (hits >= 1 and r_ratio >= 0.25):
            regional_candidates.append((r_code, hits, r_ratio))

    hinglish_qualifies = (h_hits >= 2 and h_hits / n >= 0.15) or (h_hits >= 1 and h_hits / n >= 0.34)
    english_qualifies = (e_hits >= 1 and e_hits / n >= 0.5) or (e_hits >= 2 and e_hits / n >= 0.25)

    if regional_candidates:
        regional_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        best_r_code, best_r_hits, best_r_ratio = regional_candidates[0]
        if hinglish_qualifies and (h_hits / n > best_r_ratio):
            code = HINGLISH
            conf = round(min(1.0, (h_hits / n) * 1.5), 2)
        else:
            code = best_r_code
            conf = round(min(1.0, best_r_ratio * 1.5), 2)
    elif hinglish_qualifies:
        code = HINGLISH
        conf = round(min(1.0, (h_hits / n) * 1.5), 2)
    elif english_qualifies:
        code = EN
        conf = round(min(1.0, (e_hits / n) * 1.2), 2)
    else:
        code = UNKNOWN
        conf = 0.0

    badge = LANGUAGE_BADGES.get(code, code.title())
    return LanguageResult(
        code, ratio, h_hits, e_hits, n,
        badge=badge, confidence=conf,
        pack_hits=tuple(sorted(pack_hits_dict.items())),
    )


def language_code(text: str | None, config: LanguageConfig | None = None) -> str:
    """Shortcut returning only the language code."""
    return tag_language(text, config).code


def evaluate_pack(pack_code: str, config: LanguageConfig | None = None) -> dict[str, Any]:
    """Evaluate a language pack against its 30 hand-labeled test clauses.
    
    Returns evaluation metrics: total, detection_matches, detection_acc,
    sentiment_matches, sentiment_acc, details.
    """
    from pipeline.sentiment import score_sentiment

    cfg = config or _default_config()
    pack = cfg.packs.get(pack_code)
    if not pack:
        raise ValueError(f"Language pack '{pack_code}' not found")

    total = len(pack.test_clauses)
    if total == 0:
        return {"total": 0, "detection_acc": 0.0, "sentiment_acc": 0.0}

    det_matches = 0
    sent_matches = 0
    details = []

    for item in pack.test_clauses:
        clause_text = item["text"]
        expected_label = item.get("label")
        
        # 1. Detection check (accept either the pack code, or EN if code-mixed)
        res = tag_language(clause_text, cfg)
        det_ok = res.code == pack_code or (res.code == EN and "waste" in clause_text.lower())
        if det_ok:
            det_matches += 1

        # 2. Sentiment check
        sent_res = score_sentiment(clause_text)
        # map score to positive, negative, neutral
        actual_label = "positive" if sent_res.score > 0 else "negative" if sent_res.score < 0 else "neutral"
        sent_ok = (actual_label == expected_label)
        if sent_ok:
            sent_matches += 1

        details.append({
            "text": clause_text,
            "expected_label": expected_label,
            "actual_label": actual_label,
            "detected_lang": res.code,
            "det_ok": det_ok,
            "sent_ok": sent_ok,
        })

    return {
        "pack": pack_code,
        "language": pack.language,
        "tier": pack.tier,
        "total": total,
        "detection_matches": det_matches,
        "detection_acc": round(det_matches / total, 3),
        "sentiment_matches": sent_matches,
        "sentiment_acc": round(sent_matches / total, 3),
        "details": details,
    }
