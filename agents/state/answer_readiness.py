"""Answer readiness and risk profiling."""

from typing import Any

from agents.planning.question_profile import (
    _evidence_needs_require_entity_bridge,
    _evidence_needs_require_temporal,
)
from agents.shared.constants import (
    EXTERNAL_LOOKUP_TOOLS,
    VIDEO_GROUNDING_TOOLS,
    VISUAL_CONFIRM_TOOLS,
)


class AnswerReadiness:
    """Provides evidence-integrity and answer-risk helpers."""

    def _build_evidence_integrity_profile(
        self,
        question: str,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        constraints = self._build_question_constraint_profile(question, run_report)
        observation_haystack = self._build_observation_haystack(msgs)
        grounded_date_mentions = self._extract_date_mentions(observation_haystack)
        latest_web_feedback = self._latest_web_search_feedback(run_report) or {}
        missing_entity_slots = list(latest_web_feedback.get("missing_entity_slots") or [])

        has_unverified_date_gap = bool(
            _evidence_needs_require_temporal(constraints)
            and "exact_date_or_record" in missing_entity_slots
        )
        has_entity_bridge_gap = bool(
            _evidence_needs_require_entity_bridge(constraints)
            and bool(missing_entity_slots)
        )

        return {
            "question_constraints": constraints,
            "grounded_date_mentions": grounded_date_mentions,
            "has_unverified_date_gap": has_unverified_date_gap,
            "has_entity_bridge_gap": has_entity_bridge_gap,
            "latest_missing_entity_slots": missing_entity_slots[:6],
        }

    def _build_answer_risk_profile(
        self,
        proposed_answer: str,
        question: str,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_profile = self._build_evidence_integrity_profile(question, msgs, run_report)
        latest_video_grounding = self._find_latest_condensed_observation(
            msgs, VIDEO_GROUNDING_TOOLS
        )
        latest_visual = self._find_latest_condensed_observation(msgs, VISUAL_CONFIRM_TOOLS)
        latest_external = self._find_latest_condensed_observation(msgs, EXTERNAL_LOOKUP_TOOLS)
        video_grounded_dates = self._extract_date_mentions_from_observation(
            latest_video_grounding
        ) + self._extract_date_mentions_from_observation(latest_visual)
        externally_verified_dates = self._extract_date_mentions_from_observation(
            latest_external
        )
        grounded_date_keys = {
            self._compact_text(item).lower()
            for item in evidence_profile.get("grounded_date_mentions") or []
        }
        video_grounded_keys = {
            self._compact_text(item).lower() for item in video_grounded_dates
        }
        externally_verified_keys = {
            self._compact_text(item).lower() for item in externally_verified_dates
        }
        answer_dates = self._extract_date_mentions(proposed_answer)
        unsupported_answer_dates = [
            item
            for item in answer_dates
            if self._compact_text(item).lower()
            not in grounded_date_keys | video_grounded_keys | externally_verified_keys
        ]
        requires_explicit_uncertainty = bool(
            evidence_profile.get("has_unverified_date_gap")
            and unsupported_answer_dates
        )
        return {
            "answer_date_mentions": answer_dates,
            "video_grounded_date_mentions": video_grounded_dates[:8],
            "externally_verified_date_mentions": externally_verified_dates[:8],
            "unsupported_answer_dates": unsupported_answer_dates[:6],
            "unverified_answer_dates": unsupported_answer_dates[:6],
            "requires_explicit_uncertainty": requires_explicit_uncertainty,
            "has_entity_bridge_gap": evidence_profile.get("has_entity_bridge_gap", False),
        }
