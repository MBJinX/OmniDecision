from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Literal


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_ANSI_RESET = "\x1b[0m"
_ANSI_DIM = "\x1b[2m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_BOLD_CYAN = "\x1b[1;36m"
_ANSI_BOLD_YELLOW = "\x1b[1;33m"
_ANSI_BOLD_GREEN = "\x1b[1;32m"
_ANSI_BOLD_MAGENTA = "\x1b[1;35m"
_ANSI_RED = "\x1b[31m"

_FULL_SECTION_CAP = 262144


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", str(text))


def _truncate(text: str, limit: int = 240) -> str:
    stripped = str(text).strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _truncate_middle(text: str, head: int = 120, tail: int = 120) -> str:
    stripped = str(text).strip()
    if len(stripped) <= head + tail + 5:
        return stripped
    return stripped[:head] + " ... " + stripped[-tail:]


def resolve_ansi_enabled(mode: Literal["auto", "always", "never"]) -> bool:
    """Resolve the StepLogger ANSI mode once at construction time."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    no_color = os.environ.get("NO_COLOR")
    if no_color is not None and no_color != "":
        return False
    return sys.stdout.isatty()


def _colors_for_record(record: dict[str, Any], *, ansi_enabled: bool) -> dict[str, str]:
    if not ansi_enabled:
        return {
            "reset": "",
            "dim": "",
            "step": "",
            "role": "",
            "status": "",
        }
    role = str(record.get("role") or "")
    role_color = _ANSI_BOLD
    if role == "critic.check":
        role_color = _ANSI_BOLD_YELLOW
    elif role.startswith("tool:"):
        role_color = _ANSI_BOLD_GREEN
    elif role == "repair.attempt":
        role_color = _ANSI_BOLD_MAGENTA
    return {
        "reset": _ANSI_RESET,
        "dim": _ANSI_DIM,
        "step": _ANSI_BOLD_CYAN,
        "role": role_color,
        "status": _ANSI_RED if record.get("status") == "error" else "",
    }


def _one_line(text: str) -> str:
    """Collapse internal newlines and whitespace runs to a single space."""
    return " ".join(str(text).split())


def _section_cap(text: str, limit: int = _FULL_SECTION_CAP) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]\n"


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "".join(f"{prefix}{line}\n" if line else "\n" for line in str(text).splitlines())


def _format_json_if_possible(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    text = str(value)
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _header_for_record(record: dict[str, Any], *, colors: dict[str, str], ansi_enabled: bool) -> str:
    rule = "═" if ansi_enabled else "="
    if record.get("role") == "case":
        title = f"Case {record.get('event')} / {record.get('case_id')}"
    else:
        title = f"Step {record.get('step')} / {record.get('role')}"
    header_text = f" {title} "
    width = max(60, len(header_text) + 6)
    left = rule * 3
    right = rule * max(3, width - len(left) - len(header_text))
    return (
        f"{colors['dim']}{left}{colors['step']}{header_text}"
        f"{colors['dim']}{right}{colors['reset']}\n"
    )


def _append_core_fields(body: list[str], record: dict[str, Any], *, colors: dict[str, str]) -> None:
    body.append(f"  status   : {colors['status']}{record.get('status')}{colors['reset']}\n")
    model = record.get("model")
    if model:
        body.append(f"  model    : {model}\n")
    if record.get("tokens_in") is not None or record.get("tokens_out") is not None:
        body.append(f"  tokens   : {record.get('tokens_in')} -> {record.get('tokens_out')}\n")
    if record.get("latency_ms") is not None:
        body.append(f"  latency  : {record.get('latency_ms')}ms\n")


def _append_compact_body(body: list[str], record: dict[str, Any], *, colors: dict[str, str]) -> None:
    _append_core_fields(body, record, colors=colors)
    if record.get("input_preview"):
        body.append(f"  input    : {_truncate(_one_line(record.get('input_preview', '')), limit=200)}\n")
    if record.get("output_preview"):
        body.append(f"  output   : {_truncate(_one_line(record.get('output_preview', '')), limit=200)}\n")
    extra = record.get("extra") or {}
    verdict = extra.get("verdict")
    if verdict:
        body.append(f"  verdict  : {_truncate(str(verdict), 200)}\n")
    evidence_delta = extra.get("evidence_delta")
    evidence_digest = extra.get("evidence_state_digest")
    if evidence_delta or evidence_digest:
        if evidence_delta and isinstance(evidence_delta, dict):
            parts = []
            for k, v in evidence_delta.items():
                if isinstance(v, (list, dict)):
                    parts.append(f"{k}={len(v)}")
                else:
                    parts.append(f"{k}={v}")
            summary = " ".join(parts)[:200]
        elif evidence_digest:
            summary = _truncate(str(evidence_digest), 200)
        else:
            summary = ""
        if summary:
            body.append(f"  evidence : {summary}\n")
    if extra.get("sub_calls"):
        body.append(f"  subcalls : {len(extra['sub_calls'])}\n")
    body.append("\n")


def _append_section(body: list[str], title: str, content: Any) -> None:
    text = _section_cap(_format_json_if_possible(content))
    if not text:
        return
    body.append(f"\n  [{title}]\n")
    body.append(_indent(text, 4))


def _append_full_body(body: list[str], record: dict[str, Any], *, colors: dict[str, str]) -> None:
    _append_core_fields(body, record, colors=colors)
    extra = record.get("extra") or {}

    if record.get("role") == "case" and record.get("event") == "start":
        _append_section(body, "QUESTION", record.get("input_preview", ""))
    elif record.get("input_preview"):
        _append_section(body, "INPUT", record.get("input_preview", ""))

    if record.get("role") == "case" and record.get("event") == "end":
        _append_section(body, "FINAL_ANSWER", record.get("output_preview", ""))
    elif record.get("output_preview"):
        _append_section(body, "OUTPUT", record.get("output_preview", ""))

    verdict = extra.get("verdict")
    if verdict:
        _append_section(body, "VERDICT", verdict)
    gap_diagnosis = extra.get("gap_diagnosis")
    if gap_diagnosis:
        _append_section(body, "GAP_DIAGNOSIS", gap_diagnosis)

    evidence_delta = extra.get("evidence_delta")
    if isinstance(evidence_delta, dict) and evidence_delta:
        body.append("\n  [EVIDENCE_DELTA]\n")
        for key, value in evidence_delta.items():
            rendered = _section_cap(_format_json_if_possible(value))
            if "\n" in rendered:
                body.append(f"    evidence.{key}:\n")
                body.append(_indent(rendered, 6))
            else:
                body.append(f"    evidence.{key}: {rendered}\n")

    sub_calls = extra.get("sub_calls") or []
    if sub_calls:
        body.append("\n  [SUB_CALLS]\n")
        for index, sub_call in enumerate(sub_calls, start=1):
            caller = sub_call.get("caller") or "unknown"
            model = sub_call.get("model") or "unknown"
            prompt_tokens = sub_call.get("prompt_tokens")
            completion_tokens = sub_call.get("completion_tokens")
            body.append(
                f"    #{index} caller={caller} model={model} "
                f"tokens={prompt_tokens}->{completion_tokens}\n"
            )
            if sub_call.get("tool_calls"):
                tool_calls = json.dumps(sub_call.get("tool_calls"), ensure_ascii=False, separators=(",", ":"))
                body.append(f"      [TOOL_CALLS] {_section_cap(tool_calls)}\n")
            if sub_call.get("finish_reason"):
                body.append(f"      finish_reason: {sub_call.get('finish_reason')}\n")
            for message in sub_call.get("messages") or []:
                role = str(message.get("role") or "message").upper()
                content = message.get("content", "")
                body.append(f"      [{role}]\n")
                body.append(_indent(_section_cap(_format_json_if_possible(content)), 8))
            if sub_call.get("response_text"):
                body.append("      [ASSISTANT]\n")
                body.append(_indent(_section_cap(_format_json_if_possible(sub_call.get("response_text"))), 8))

    if extra.get("error_type") or extra.get("error_msg"):
        body.append("\n  [ERROR]\n")
        if extra.get("error_type"):
            body.append(f"    type: {extra.get('error_type')}\n")
        if extra.get("error_msg"):
            body.append(f"    message: {extra.get('error_msg')}\n")

    if extra.get("token_snapshot") is not None:
        _append_section(body, "TOKEN_SNAPSHOT", extra.get("token_snapshot"))

    body.append("\n")


def render_record(
    record: dict[str, Any],
    *,
    full: bool = False,
    color: Literal["auto", "always", "never"] = "auto",
) -> str:
    ansi_enabled = resolve_ansi_enabled(color)
    colors = _colors_for_record(record, ansi_enabled=ansi_enabled)
    body = [_header_for_record(record, colors=colors, ansi_enabled=ansi_enabled)]
    if full:
        _append_full_body(body, record, colors=colors)
    else:
        _append_compact_body(body, record, colors=colors)
    return "".join(body)
