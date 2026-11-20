"""Observation summarization, text extraction, and evidence preview helpers."""

import re
from typing import Any

from agents.shared.text_patterns import (
    FULL_DATE_PATTERN,
    ISO_DATE_PATTERN,
    MD_LINK_PATTERN,
    MONTH_DAY_TEXT_PATTERN,
    QUOTED_TEXT_PATTERN,
    delink_markdown,
)
from agents.shared.constants import (
    COMPACTED_OBSERVATION_MARKER,
    TRUNC_KEY_FINDINGS_PREVIEW,
)
from agents.shared.json_utils import safe_load_json_object


class ObservationUtils:
    """Provides observation condensation, text compaction, evidence preview,
    and haystack-building helpers."""

    # ------------------------------------------------------------------
    # Primitive text helpers
    # ------------------------------------------------------------------

    def _compact_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Evidence focus / preview
    # ------------------------------------------------------------------

    def _excerpt_around_span(
        self,
        text: str,
        span: tuple[int, int],
        *,
        max_chars: int = 320,
    ) -> str:
        normalized = self._compact_text(text)
        if not normalized or len(normalized) <= max_chars:
            return normalized

        start, end = span
        start = max(0, min(start, len(normalized)))
        end = max(start, min(end, len(normalized)))
        if end - start >= max_chars:
            excerpt = normalized[start:end][:max_chars]
            if start > 0:
                excerpt = f"...{excerpt}"
            if end < len(normalized):
                excerpt = f"{excerpt}..."
            return excerpt

        remaining = max_chars - (end - start)
        left_budget = remaining // 2
        right_budget = remaining - left_budget
        window_start = max(0, start - left_budget)
        window_end = min(len(normalized), end + right_budget)
        if window_end - window_start < max_chars:
            deficit = max_chars - (window_end - window_start)
            if window_start == 0:
                window_end = min(len(normalized), window_end + deficit)
            elif window_end == len(normalized):
                window_start = max(0, window_start - deficit)
        excerpt = normalized[window_start:window_end].strip()
        if window_start > 0:
            excerpt = f"...{excerpt}"
        if window_end < len(normalized):
            excerpt = f"{excerpt}..."
        return excerpt

    def _evidence_focus_span(self, text: Any, source_modality: Any) -> tuple[int, int] | None:
        normalized = self._compact_text(text)
        if not normalized:
            return None

        for pattern in (FULL_DATE_PATTERN, MONTH_DAY_TEXT_PATTERN, ISO_DATE_PATTERN):
            match = pattern.search(normalized)
            if match:
                return match.span()

        quoted_match = QUOTED_TEXT_PATTERN.search(normalized)
        if quoted_match:
            return quoted_match.span(1)
        return None

    def _build_evidence_preview_text(
        self,
        text: Any,
        source_modality: Any,
        *,
        max_chars: int = 320,
    ) -> str:
        """Excerpt the most likely answer-bearing span of one evidence text."""
        normalized = self._compact_text(text)
        if not normalized:
            return ""
        if MD_LINK_PATTERN.search(normalized):
            delinked = self._compact_text(delink_markdown(normalized))
            if delinked:
                normalized = delinked
        focus_span = self._evidence_focus_span(normalized, source_modality)
        if focus_span is None:
            return normalized[:max_chars]
        return self._excerpt_around_span(
            normalized,
            focus_span,
            max_chars=max_chars,
        )

    # ------------------------------------------------------------------
    # Pattern counting / verdict haystack
    # ------------------------------------------------------------------

    def _count_pattern_hits(self, pattern: re.Pattern[str], text: str) -> int:
        normalized = self._compact_text(text)
        if not normalized:
            return 0
        return len({match.group(0).lower() for match in pattern.finditer(normalized)})

    # ------------------------------------------------------------------
    # Recursive string collection
    # ------------------------------------------------------------------

    def _collect_string_leaves(self, value: Any, sink: list[str]) -> None:
        if isinstance(value, str):
            normalized = self._compact_text(value)
            if normalized:
                sink.append(normalized)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._collect_string_leaves(item, sink)
            return
        if isinstance(value, list):
            for item in value:
                self._collect_string_leaves(item, sink)

    def _extract_date_mentions_from_observation(
        self,
        observation: dict[str, Any] | None,
    ) -> list[str]:
        fragments: list[str] = []
        self._collect_string_leaves(observation or {}, fragments)
        return self._extract_date_mentions(" ".join(fragments))

    # ------------------------------------------------------------------
    # Observation summarization and condensation
    # ------------------------------------------------------------------

    def _summarize_observation(self, tool_name: str, payload: dict) -> dict:
        if payload.get(COMPACTED_OBSERVATION_MARKER):
            return {
                "tool_name": payload.get("tool_name") or tool_name,
                "status": str(payload.get("status") or "unknown"),
                "summary": self._compact_text(payload.get("summary")),
                "key_findings": list(payload.get("key_findings") or [])[:3],
            }

        summary_text = self._compact_text(payload.get("summary"))
        if not summary_text:
            summary_text = self._compact_text(payload.get("status") or "unknown")

        key_findings: list[str] = []

        def add_finding(value: Any) -> None:
            normalized = self._compact_text(value)
            if not normalized or normalized in key_findings:
                return
            key_findings.append(normalized)

        for item in (payload.get("candidate_ranges") or [])[:2]:
            if not isinstance(item, dict):
                continue
            start_time = item.get("start_time") or item.get("start") or "?"
            end_time = item.get("end_time") or item.get("end") or "?"
            add_finding(f"Range: {start_time}-{end_time}s")

        for item in (payload.get("time_ranges") or [])[:2]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                add_finding(f"Range: {item[0]}-{item[1]}")

        source_modality = payload.get("source_modality")
        for item in (payload.get("evidence") or [])[:2]:
            if not isinstance(item, dict):
                continue
            add_finding(
                self._build_evidence_preview_text(item.get("text"), source_modality, max_chars=TRUNC_KEY_FINDINGS_PREVIEW)
            )

        for key in ("matched_entities", "exact_date", "result"):
            value = payload.get(key)
            if value:
                label = key.replace("_", " ").title()
                add_finding(f"{label}: {value}")

        if payload.get("stdout"):
            add_finding(f"Output: {str(payload.get('stdout'))[:TRUNC_KEY_FINDINGS_PREVIEW]}")

        for key in ("observations", "range_observations"):
            for obs in (payload.get(key) or [])[:2]:
                if isinstance(obs, dict):
                    add_finding(obs.get("observation") or obs.get("text") or obs.get("summary"))
                else:
                    add_finding(obs)

        if payload.get("source_modality") == "web":
            web_profile = {}
            if hasattr(self, "_build_web_evidence_profile"):
                web_profile = self._build_web_evidence_profile(payload)
            elif isinstance(payload.get("evidence_profile"), dict):
                web_profile = dict(payload.get("evidence_profile") or {})
            if web_profile.get("has_terminal_fact_candidate"):
                add_finding("Terminal fact candidate found")
            search_feedback = payload.get("search_feedback") or {}
            if isinstance(search_feedback, dict) and search_feedback.get("status"):
                add_finding(f"Search status: {search_feedback['status']}")

        return {
            "tool_name": tool_name,
            "status": str(payload.get("status") or "unknown"),
            "summary": summary_text,
            "key_findings": key_findings[:3],
        }

    def _find_latest_tool_payload(
        self,
        msgs: list[dict],
        tool_names: set[str],
        *,
        successful_only: bool = False,
    ) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        for message in reversed(msgs):
            if message.get("role") != "tool":
                continue
            if message.get("name") not in tool_names:
                continue
            payload = safe_load_json_object(message.get("content", ""))
            if payload is None:
                continue
            if fallback is None:
                fallback = payload
            if not successful_only or str(payload.get("status") or "").strip().lower() == "ok":
                return payload
        return fallback if successful_only else None

    def _collect_condensed_observations(self, msgs: list[dict]) -> list[dict]:
        observations: list[dict] = []
        for message in msgs:
            if message.get("role") != "tool":
                continue
            payload = safe_load_json_object(message.get("content", ""))
            if payload is None:
                continue
            observations.append(
                self._summarize_observation(message.get("name", "unknown"), payload)
            )
        return observations

    def _find_latest_condensed_observation(
        self, msgs: list[dict], tool_names: set[str]
    ) -> dict[str, Any] | None:
        for message in reversed(msgs):
            if message.get("role") != "tool":
                continue
            if message.get("name") not in tool_names:
                continue
            payload = safe_load_json_object(message.get("content", ""))
            if payload is None:
                continue
            return self._summarize_observation(message.get("name", "unknown"), payload)
        return None

    def _find_latest_successful_condensed_observation(
        self, msgs: list[dict], tool_names: set[str]
    ) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        for message in reversed(msgs):
            if message.get("role") != "tool":
                continue
            if message.get("name") not in tool_names:
                continue
            payload = safe_load_json_object(message.get("content", ""))
            if payload is None:
                continue
            summary = self._summarize_observation(message.get("name", "unknown"), payload)
            if fallback is None:
                fallback = summary
            if str(summary.get("status") or "").strip().lower() == "ok":
                return summary
        return fallback

    # ------------------------------------------------------------------
    # Observation haystack
    # ------------------------------------------------------------------

    def _build_observation_haystack(self, msgs: list[dict]) -> str:
        observations = self._collect_condensed_observations(msgs)[-5:]
        fragments: list[str] = []
        self._collect_string_leaves(observations, fragments)
        return " ".join(fragments)

    def _summarize_tool_content(self, tool_name: str, content: str) -> str:
        payload = safe_load_json_object(content)
        if isinstance(payload, dict):
            summary_payload = self._summarize_observation(tool_name, payload)
            summary = summary_payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        stripped = content.strip()
        return stripped[:240]
