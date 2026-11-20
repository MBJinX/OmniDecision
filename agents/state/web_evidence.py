"""Web evidence profiling and source quality assessment."""

import re
from typing import Any

from agents.shared.constants import (
    TERMINAL_FACT_MAX_SCORE,
    TRUNC_EVIDENCE_BUNDLE_TEXT,
    TRUNC_EVIDENCE_BUNDLE_TITLE,
)
from agents.shared.text_patterns import (
    FULL_DATE_PATTERN,
    ISO_DATE_PATTERN,
    MONTH_DAY_TEXT_PATTERN,
    SCORELINE_PATTERN,
)


class WebEvidence:
    """Provides web-evidence summarization and date extraction."""

    def _build_web_evidence_profile(self, payload: dict | None) -> dict[str, Any]:
        payload = payload or {}
        evidence = payload.get("evidence") or []
        evidence_bundle: list[dict[str, Any]] = []
        explicit_full_date_count = 0
        iso_date_count = 0
        scoreline_count = 0
        official_like_source_count = 0
        max_score = 0.0
        unique_sources: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            text = self._compact_text(item.get("text"))
            title = self._compact_text(metadata.get("title"))
            url = self._compact_text(metadata.get("url"))
            query_used = self._compact_text(metadata.get("query_used"))
            provider = self._compact_text(metadata.get("provider"))
            score = self._safe_float(item.get("score"))
            max_score = max(max_score, score)
            explicit_full_date_count += self._count_pattern_hits(FULL_DATE_PATTERN, text)
            iso_date_count += self._count_pattern_hits(ISO_DATE_PATTERN, text)
            scoreline_count += self._count_pattern_hits(SCORELINE_PATTERN, text)
            source_key = url or title or query_used
            if source_key:
                unique_sources.add(source_key.lower())
            if metadata.get("llm_is_authoritative") or metadata.get("page_family") in {"detail", "authority"}:
                official_like_source_count += 1
            if len(evidence_bundle) < 3:
                bundle_item: dict[str, Any] = {
                    "text": text[:TRUNC_EVIDENCE_BUNDLE_TEXT],
                    "score": round(score, 4),
                }
                if title:
                    bundle_item["title"] = title[:TRUNC_EVIDENCE_BUNDLE_TITLE]
                if url:
                    bundle_item["url"] = url[:240]
                if query_used:
                    bundle_item["query_used"] = query_used[:240]
                if provider:
                    bundle_item["provider"] = provider[:80]
                evidence_bundle.append(bundle_item)

        precise_temporal_fact_count = explicit_full_date_count + iso_date_count
        has_terminal_fact_candidate = (
            precise_temporal_fact_count > 0
            and bool(unique_sources)
            and (
                official_like_source_count > 0
                or len(unique_sources) > 1
                or max_score >= TERMINAL_FACT_MAX_SCORE
            )
        )
        has_detail_or_authority = any(
            (item.get("metadata") or {}).get("page_family") in {"detail", "authority"}
            for item in evidence
            if isinstance(item, dict)
        )
        if has_terminal_fact_candidate and not has_detail_or_authority:
            has_terminal_fact_candidate = False
        fact_tags: list[str] = []
        if precise_temporal_fact_count > 0:
            fact_tags.append("precise_temporal_fact")
        if scoreline_count > 0:
            fact_tags.append("numeric_result")
        if official_like_source_count > 0:
            fact_tags.append("official_like_source")
        if has_terminal_fact_candidate:
            fact_tags.append("terminal_external_fact_candidate")

        return {
            "evidence_count": len(evidence_bundle),
            "precise_temporal_fact_count": precise_temporal_fact_count,
            "scoreline_count": scoreline_count,
            "official_like_source_count": official_like_source_count,
            "unique_source_count": len(unique_sources),
            "max_score": round(max_score, 4),
            "has_terminal_fact_candidate": has_terminal_fact_candidate,
            "fact_tags": fact_tags,
            "evidence_bundle": evidence_bundle,
        }

    def _extract_date_mentions(self, text: Any) -> list[str]:
        normalized = self._compact_text(text)
        if not normalized:
            return []
        mentions: list[str] = []
        for pattern in (FULL_DATE_PATTERN, MONTH_DAY_TEXT_PATTERN, ISO_DATE_PATTERN):
            for match in pattern.findall(normalized):
                token = self._compact_text(match)
                if token:
                    mentions.append(token)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", normalized):
            mentions.append(year)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in mentions:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:8]
