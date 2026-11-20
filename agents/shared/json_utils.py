"""Shared JSON parsing helpers for OmniVQA."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_DECODER = json.JSONDecoder()


def safe_load_json_value(content: str | None) -> Any | None:
    """Parse one JSON value from plain or lightly wrapped model output.

    Exact JSON is preferred, followed by fenced JSON and then the first
    decodable value embedded in explanatory text. Invalid input returns
    ``None``; consequently, JSON ``null`` is intentionally indistinguishable
    from a parse failure in this compatibility-oriented interface.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        pass

    for match in _FENCED_JSON_RE.finditer(content):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    for start, character in enumerate(content):
        if character not in "[{":
            continue
        try:
            value, _ = _JSON_DECODER.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        return value

    for start, character in enumerate(content):
        if character not in '"-0123456789tfn':
            continue
        if start and (content[start - 1].isalnum() or content[start - 1] in "._"):
            continue
        try:
            value, end = _JSON_DECODER.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        if end < len(content) and (content[end].isalnum() or content[end] == "_"):
            continue
        return value
    return None


def safe_load_json_object(content: str | None) -> dict | None:
    """Return a parsed JSON object while rejecting every other top-level type."""
    payload = safe_load_json_value(content)
    if isinstance(payload, dict):
        return payload
    return None


def dumps_llm_payload(value: Any) -> str:
    """Serialize an LLM-visible JSON payload without indentation padding."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
