"""Best-effort PII redaction for review text.

This is a conservative, pattern-based utility. It catches common email and
phone formats only. It does NOT guarantee that all personal data is removed
(names, addresses, order IDs, obfuscated emails, etc. are left untouched), and
it can occasionally redact a long digit sequence that is not a phone number.
"""

from __future__ import annotations

import re

EMAIL_PLACEHOLDER = "[email]"
PHONE_PLACEHOLDER = "[phone]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# A run of digits and common separators that starts and ends with a digit.
# Candidates are then filtered by digit count and shape in _redact_phone.
_PHONE_RE = re.compile(r"(?<![\w.])[+(]?\d[\d\s().\-]{8,20}\d(?!\w)")
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_DATE_RE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}\b")


def _redact_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = sum(ch.isdigit() for ch in candidate)
    if not 10 <= digits <= 15:
        return candidate
    if _DECIMAL_RE.fullmatch(candidate) or _DATE_RE.search(candidate):
        return candidate
    return PHONE_PLACEHOLDER


def redact_pii(text: str) -> str:
    """Replace common emails and phone numbers with placeholders."""
    text = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    return _PHONE_RE.sub(_redact_phone, text)
