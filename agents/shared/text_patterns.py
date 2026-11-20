"""Reusable text-pattern helpers for structured observation parsing."""

import re


FULL_DATE_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
MONTH_DAY_TEXT_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SCORELINE_PATTERN = re.compile(r"\b\d{1,3}\s*[-–]\s*\d{1,3}\b")
QUOTED_TEXT_PATTERN = re.compile(r"[\"\u201c\u201d']([^\"\u201c\u201d'\n]{4,140})[\"\u201c\u201d']")


MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BLOB_URL_PATTERN = re.compile(r"blob:https?://\S+")
_WS_RUN_PATTERN = re.compile(r"[ \t]{2,}")
_BLANK_RUN_PATTERN = re.compile(r"\n{3,}")


def delink_markdown(text: str) -> str:
    """Strip markdown link plumbing, keeping every word of prose.

    Images, ``blob:`` URLs, and the URL half of ``[text](url)`` go; the anchor
    text stays and no line is dropped. Deliberately does not filter by line
    length or link count — that is what makes chrome-stripping able to lose a
    short table row, whereas rewriting link syntax cannot remove prose.

    Idempotent, and a no-op on text that carries no markdown links.
    """
    if not text:
        return ""
    cleaned = MD_IMAGE_PATTERN.sub(" ", text)
    cleaned = BLOB_URL_PATTERN.sub(" ", cleaned)
    cleaned = MD_LINK_PATTERN.sub(r"\1", cleaned)
    cleaned = _WS_RUN_PATTERN.sub(" ", cleaned)
    return _BLANK_RUN_PATTERN.sub("\n\n", cleaned).strip()
