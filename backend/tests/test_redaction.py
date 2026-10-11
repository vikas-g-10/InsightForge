import pytest

from redaction import redact_pii


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at jane.doe+x@example.co.in now", "mail me at [email] now"),
        ("call +91 98765 43210 today", "call [phone] today"),
        ("call 9876543210.", "call [phone]."),
        ("phone (555) 123-4567 please", "phone [phone] please"),
        ("try 555-123-4567", "try [phone]"),
        ("+1-555-123-4567", "[phone]"),
    ],
)
def test_emails_and_phones_are_redacted(text, expected):
    assert redact_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "rated 4.5 out of 5",
        "bought on 2024-01-15 for 12,999",
        "01-02-2024 10:30 delivered",
        "order 12345 arrived",
        "version 3.14159265358979",
        "battery lasts 2 hours",
    ],
)
def test_non_pii_numbers_are_left_alone(text):
    assert redact_pii(text) == text


def test_redaction_is_best_effort_not_exhaustive():
    # Documented limitation: names and obfuscated emails are not detected.
    assert redact_pii("I am Jane Doe, jane at example dot com") == (
        "I am Jane Doe, jane at example dot com"
    )
