"""Shared helpers: YAML config loading and term-to-regex compilation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Word characters for boundary checks. Python's \w misses Devanagari vowel
# signs (combining marks), so the Devanagari block is added explicitly.
WORD_CLASS = r"\w\u0900-\u097F"
_SUFFIX = r"(?:[^\W\d_]|[\u0900-\u097F])*"
MIN_PREFIX_CHARS = 4


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration file is missing or malformed."""


def load_yaml(path: Path | str | None, default_name: str) -> dict:
    """Load a YAML mapping from ``path`` or the bundled ``config/<default_name>``."""
    target = Path(path) if path is not None else CONFIG_DIR / default_name
    try:
        with open(target, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        raise PipelineConfigError(f"Config file not found: {target.name}") from None
    except yaml.YAMLError as exc:
        raise PipelineConfigError(f"Invalid YAML in {target.name}: {exc}") from None
    if not isinstance(data, dict):
        raise PipelineConfigError(f"{target.name} must contain a mapping at the top level.")
    return data


def clean_term(value: object, where: str) -> str:
    """Validate and NFC-normalize one configured term.

    YAML turns bare ``no``/``yes``/``on`` into booleans, so non-strings are
    rejected with a hint to quote them.
    """
    if not isinstance(value, str):
        raise PipelineConfigError(
            f"{where}: expected a string term, got {value!r} (quote YAML words like 'no')."
        )
    term = unicodedata.normalize("NFC", " ".join(value.split())).lower()
    if not term:
        raise PipelineConfigError(f"{where}: empty term.")
    if "*" in term[:-1]:
        raise PipelineConfigError(f"{where}: '*' is only allowed at the end of {term!r}.")
    if term.endswith("*") and len(term) - 1 < MIN_PREFIX_CHARS:
        raise PipelineConfigError(
            f"{where}: prefix {term!r} is shorter than {MIN_PREFIX_CHARS} characters."
        )
    return term


def string_list(config: dict, key: str, where: str) -> list[str]:
    values = config.get(key)
    if not isinstance(values, list):
        raise PipelineConfigError(f"{where}: '{key}' must be a list.")
    return [clean_term(v, f"{where}.{key}") for v in values]


def _term_pattern(term: str) -> str:
    wildcard = term.endswith("*")
    body = term[:-1] if wildcard else term
    parts = [re.escape(piece).replace("'", "['\u2019]") for piece in body.split(" ")]
    pattern = r"\s+".join(parts)
    return pattern + (_SUFFIX if wildcard else "")


def compile_terms(terms: Sequence[str]) -> tuple[re.Pattern[str], tuple[str, ...]]:
    """Compile terms into one case-insensitive, boundary-aware alternation.

    A trailing ``*`` allows any letter suffix (``crash*`` -> crashes, crashing).
    Longer terms are tried first so ``customer care`` wins over ``care``.
    The matched term is recovered from ``match.lastgroup`` (``t<index>``).
    """
    ordered = tuple(sorted(dict.fromkeys(terms), key=lambda t: (-len(t), t)))
    if not ordered:
        return re.compile(r"(?!x)x"), ordered  # never matches
    alternation = "|".join(f"(?P<t{i}>{_term_pattern(t)})" for i, t in enumerate(ordered))
    pattern = re.compile(
        rf"(?<![{WORD_CLASS}])(?:{alternation})(?![{WORD_CLASS}])", re.IGNORECASE
    )
    return pattern, ordered
