from __future__ import annotations

from agents.logging.trace_render import _truncate
from agents.shared.json_utils import safe_load_json_object


_CRITIC_PREFIX = "Internal critic checkpoint result for OmniVQA "
_ANSWER_PREFIX = "Internal answer checkpoint result for OmniVQA "


def _extract_checkpoint_payload(content: str) -> dict | None:
    if not content:
        return None
    json_start = content.find("{")
    if json_start == -1:
        return None
    policy_start = content.find("\nAction policy:")
    if policy_start != -1 and policy_start > json_start:
        content = content[json_start:policy_start].strip()
    else:
        content = content[json_start:].strip()
    return safe_load_json_object(content)


def format_stream_event(event: dict) -> str | None:
    role = event.get("role", "")

    if role == "tool_call":
        name = str(event.get("name") or "unknown")
        arguments = _truncate(str(event.get("arguments") or "{}"), limit=320)
        return f"[tool_call] {name}: {arguments}"

    if role == "tool":
        name = str(event.get("name") or "unknown")
        payload = safe_load_json_object(event.get("content", ""))
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").strip()
            summary = _truncate(str(payload.get("summary") or ""), limit=320)
            fallback_hint = _truncate(str(payload.get("fallback_hint") or ""), limit=240)
            parts = [f"[tool] {name}"]
            if status:
                parts.append(f"status={status}")
            if summary:
                parts.append(f"summary={summary}")
            if fallback_hint:
                parts.append(f"hint={fallback_hint}")
            return " | ".join(parts)

        content = _truncate(str(event.get("content") or ""), limit=320)
        if content:
            return f"[tool] {name}: {content}"
        return f"[tool] {name}"

    if role == "system":
        content = str(event.get("content") or "").strip()
        if content.startswith(_CRITIC_PREFIX):
            payload = _extract_checkpoint_payload(content)
            if isinstance(payload, dict):
                decision = str(payload.get("decision") or "").strip()
                verdict = str(payload.get("verdict") or "").strip()
                reason = _truncate(str(payload.get("reason") or ""), limit=240)
                hint = _truncate(str(payload.get("next_action_hint") or ""), limit=240)
                parts = ["[critic]"]
                if decision:
                    parts.append(f"decision={decision}")
                if verdict:
                    parts.append(f"verdict={verdict}")
                if reason:
                    parts.append(f"reason={reason}")
                if hint:
                    parts.append(f"next={hint}")
                return " | ".join(parts)
        if content.startswith(_ANSWER_PREFIX):
            payload = _extract_checkpoint_payload(content)
            if isinstance(payload, dict):
                decision = str(payload.get("decision") or "").strip()
                verdict = str(payload.get("verdict") or "").strip()
                reason = _truncate(str(payload.get("reason") or ""), limit=240)
                hint = _truncate(str(payload.get("next_action_hint") or ""), limit=240)
                parts = ["[answer_critic]"]
                if decision:
                    parts.append(f"decision={decision}")
                if verdict:
                    parts.append(f"verdict={verdict}")
                if reason:
                    parts.append(f"reason={reason}")
                if hint:
                    parts.append(f"next={hint}")
                return " | ".join(parts)
        return None

    if role == "assistant":
        content = _truncate(str(event.get("content") or ""), limit=320)
        if content:
            return f"[assistant] {content}"
        return None

    return None


def run_agent_with_optional_live_trace(agent, question: str, show_live_trace: bool):
    if not show_live_trace:
        return agent.run(question)

    iterator = agent.stream_run(question)
    try:
        while True:
            event = next(iterator)
            formatted = format_stream_event(event)
            if formatted:
                print(formatted, flush=True)
    except StopIteration as stop:
        return stop.value
