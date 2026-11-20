"""Evidence state initialization, normalization, ingestion, and delta/digest building."""

import copy
import hashlib
from typing import Any

from agents.planning.question_profile import (
    _evidence_needs_prefer_source,
    _evidence_needs_require_entity_bridge,
    _evidence_needs_require_numeric,
    _evidence_needs_require_temporal,
)
from agents.shared.constants import (
    COMPUTATION_TOOLS,
    EVIDENCE_ATOM_COUNT_THRESHOLD,
    EXTERNAL_LOOKUP_TOOLS,
    TOTAL_ATOM_BUDGET,
    TEMPORAL_ANCHOR_COUNT_THRESHOLD,
    TOOL_COUNT_EVIDENCE_COLLECTED,
    VISUAL_CONFIRM_TOOLS,
    WEAK_GROUNDING_CONFIDENCE,
    WINDOW_ATOMS_PER_OBSERVATION,
    WINDOW_CANDIDATE_RANGES_CAP,
    WINDOW_EVIDENCE_ATOMS_CAP,
    WINDOW_NORMALIZE_ITEMS,
)
from agents.shared.types import (
    EvidenceAtom,
    EvidenceState,
    EvidenceStateDigest,
    EstimatedCost,
    NormalizedToolObservation,
)
from agents.shared.json_utils import safe_load_json_object
from agents.shared.text_patterns import (
    FULL_DATE_PATTERN,
    ISO_DATE_PATTERN,
    MONTH_DAY_TEXT_PATTERN,
)

# Rank of each need status, for deciding whether a need moved forward.
# "unfillable" ranks with "filled": both are terminal, and a critic declaring a
# need dead has resolved it as surely as filling it would have.
NEED_STATUS_ORDER = {"unfilled": 0, "partial": 1, "filled": 2, "unfillable": 2}


def need_statuses_improved(before: list[str], after: list[str]) -> bool:
    """Return whether any need advanced between two positional status lists.

    Sole definition of "a need improved". Both the observation reducer and the
    verdict reducer read it, because the progress guard treats need closure as
    the strongest form of progress and two copies of this comparison would let
    one path drift into a laxer reading of the same word.

    Needs are matched by position, and a missing index counts as "unfilled", so
    a profile that grew a need mid-run does not register the new entry as
    progress on its own.
    """
    for idx in range(max(len(before), len(after))):
        before_status = before[idx] if idx < len(before) else "unfilled"
        after_status = after[idx] if idx < len(after) else "unfilled"
        if NEED_STATUS_ORDER.get(after_status, 0) > NEED_STATUS_ORDER.get(
            before_status, 0
        ):
            return True
    return False


def need_status_list(needs: Any) -> list[str]:
    """Project needs onto the positional status list the comparison expects."""
    return [
        str(need.get("status") or "unfilled")
        for need in (needs or [])
        if isinstance(need, dict)
    ]


class EvidenceStateMachine:
    """Provides evidence state lifecycle: init, normalize, ingest, delta, digest."""

    def _init_evidence_state(self, question: str, run_report: dict[str, Any] | None = None) -> EvidenceState:
        profile = self._build_question_constraint_profile(question, run_report)
        return {
            "query_id": self._build_query_id(question),
            "video_id": self._current_video_id(),
            "current_stage": "start",
            "candidate_ranges": [],
            "evidence_atoms": [],
            "conflicts": [],
            "unresolved_questions": self._initial_unresolved_questions(question, run_report),
            "uncertainty_summary": {
                "temporal_gap": False,
                "entity_gap": False,
                "cross_modal_conflict": False,
                "external_fact_gap": False,
                "visual_gap": False,
            },
            "bridge_status": {
                "has_video_bridge": False,
                "has_external_lookup": False,
                "has_external_exact_fact": False,
                "entity_bridge_ready": False,
                "visual_identity_ready": False,
                "computation_ready": False,
                "entity_cards_extracted": False,
                "fact_bridge_populated": False,
                "entity_lookup_pending": False,
            },
            "budget_state": {
                "high_cost_call_count": 0,
                "reground_count": 0,
                "blocked_finish_count": 0,
                "answer_revision_count": 0,
            },
            "sufficiency_status": "seeking_grounding",
            "stop_ready": False,
            "decision_state": {
                "stop_insufficient_ready": False,
                "computation_ready": False,
                "estimated_cost_to_continue": 0.25,
                "expected_value_of_next_step": 0.9,
            },
            "entity_cards": [],
            "fact_bridge_records": [],
            "pending_entity_lookups": [],
            "evidence_needs": list(profile.get("evidence_needs") or []),
            "_latest_gap_diagnosis": {
                "gap_type": "insufficient",
                "description": "No evidence collected yet",
            },
        }

    def _build_estimated_cost(self, tool_name: str, payload: dict[str, Any]) -> EstimatedCost:
        if tool_name in VISUAL_CONFIRM_TOOLS:
            return {
                "tier": "high",
                "score": 1.0,
                "high_cost_units": 1,
                "rationale": "Visual verification is the dominant high-cost action.",
            }
        if tool_name in EXTERNAL_LOOKUP_TOOLS:
            return {
                "tier": "medium",
                "score": 0.6,
                "high_cost_units": 0,
                "rationale": "External lookup is moderate-cost and may require follow-up verification.",
            }
        if tool_name in COMPUTATION_TOOLS:
            return {
                "tier": "low",
                "score": 0.2,
                "high_cost_units": 0,
                "rationale": "Deterministic computation is low-cost once evidence is sufficient.",
            }
        if payload.get("status") in {"guardrail", "error", "unavailable"}:
            return {
                "tier": "low",
                "score": 0.1,
                "high_cost_units": 0,
                "rationale": "Guardrail or failure observation did not consume a substantive evidence budget.",
            }
        return {
            "tier": "low",
            "score": 0.25,
            "high_cost_units": 0,
            "rationale": "Grounding and bookkeeping observations are treated as low-cost.",
        }

    def _build_time_anchor(
        self,
        item: dict[str, Any] | list[str] | tuple[str, str] | None,
        *,
        source_tool: str,
        source_modality: str,
        fallback_confidence: float = 0.0,
        fallback_rationale: str = "",
    ) -> dict[str, Any] | None:
        if item is None:
            return None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start_time = str(item[0] or "").strip()
            end_time = str(item[1] or "").strip()
            if not start_time or not end_time:
                return None
            return {
                "start_time": start_time,
                "end_time": end_time,
                "confidence": round(float(fallback_confidence), 4),
                "source_tool": source_tool,
                "source_modality": source_modality,
                "rationale": fallback_rationale,
            }
        if not isinstance(item, dict):
            return None
        start_time = str(item.get("start_time") or item.get("start") or "").strip()
        end_time = str(item.get("end_time") or item.get("end") or "").strip()
        if not start_time or not end_time:
            return None
        confidence = self._safe_float(item.get("confidence") or fallback_confidence)
        rationale = self._compact_text(item.get("rationale") or fallback_rationale)
        return {
            "start_time": start_time,
            "end_time": end_time,
            "confidence": round(confidence, 4),
            "source_tool": source_tool,
            "source_modality": source_modality,
            "rationale": rationale,
        }

    def _source_uncertainty_from_payload(
        self,
        payload: dict[str, Any],
        question: str,
    ) -> list[str]:
        uncertainty: list[str] = []
        status = str(payload.get("status") or "")
        if status in {"no_match", "unavailable", "error"}:
            uncertainty.append(status)
        confidence = self._safe_float(payload.get("confidence"))
        if payload.get("source_modality") in {"audio", "subtitle", "clip"} and confidence < WEAK_GROUNDING_CONFIDENCE:
            uncertainty.append("weak_grounding")
        if payload.get("source_modality") == "web":
            feedback = payload.get("search_feedback") or {}
            missing_entity_slots = list(feedback.get("missing_entity_slots") or [])
            if missing_entity_slots:
                uncertainty.append("entity_bridge_gap")
            if str(feedback.get("progress_status") or "") == "no_progress":
                uncertainty.append("stale_external_lookup")
        # Entity gap: clip/web returned no matched_entities (purely payload-driven).
        if (
            payload.get("source_modality") in {"clip", "web"}
            and not payload.get("matched_entities")
            and payload.get("status") == "ok"
        ):
            uncertainty.append("entity_gap")
        if payload.get("source_modality") == "visual" and str(payload.get("status") or "") != "ok":
            uncertainty.append("visual_gap")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in uncertainty:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _make_evidence_atom(
        self,
        *,
        source_tool: str,
        source_modality: str,
        text: str,
        confidence: float,
        evidence_kind: str,
        anchor: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        atom_index: int = 0,
        asset_id: str | None = None,
    ) -> EvidenceAtom:
        fingerprint = hashlib.sha1(
            f"{source_tool}|{source_modality}|{evidence_kind}|{text}|{anchor}|{asset_id or ''}".encode("utf-8")
        ).hexdigest()[:12]
        atom: EvidenceAtom = {
            "evidence_id": f"{source_tool}_{atom_index}_{fingerprint}",
            "source_tool": source_tool,
            "source_modality": source_modality,
            "text": text,
            "confidence": round(float(confidence), 4),
            "evidence_kind": evidence_kind,
            "metadata": metadata or {},
        }
        if anchor:
            atom["anchor"] = anchor
        if asset_id:
            atom["asset_id"] = asset_id
        return atom

    def _normalize_tool_observation(
        self,
        tool_name: str,
        raw_observation: str | dict[str, Any],
        *,
        question: str,
        asset_id: str | None = None,
    ) -> NormalizedToolObservation:
        payload = (
            raw_observation
            if isinstance(raw_observation, dict)
            else safe_load_json_object(str(raw_observation))
        )
        if not isinstance(payload, dict):
            payload = {
                "status": "ok",
                "source_modality": "unknown",
                "summary": self._compact_text(raw_observation),
                "fallback_hint": "",
            }
        source_modality = str(payload.get("source_modality") or "unknown")
        confidence = self._safe_float(payload.get("confidence"))
        candidate_ranges: list[dict[str, Any]] = []
        for item in payload.get("candidate_ranges") or []:
            anchor = self._build_time_anchor(
                item,
                source_tool=tool_name,
                source_modality=source_modality,
                fallback_confidence=confidence,
            )
            if anchor:
                candidate_ranges.append(anchor)
        for item in payload.get("time_ranges") or []:
            anchor = self._build_time_anchor(
                item,
                source_tool=tool_name,
                source_modality=source_modality,
                fallback_confidence=confidence,
            )
            if anchor:
                candidate_ranges.append(anchor)

        evidence_atoms: list[EvidenceAtom] = []
        evidence_index = 1
        for item in payload.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            atom_text = self._compact_text(item.get("text"))
            if not atom_text:
                continue
            metadata = dict(item.get("metadata") or {})
            anchor = self._build_time_anchor(
                {
                    "start_time": metadata.get("start_time") or metadata.get("start_hhmmss"),
                    "end_time": metadata.get("end_time") or metadata.get("end_hhmmss"),
                    "confidence": item.get("score") or confidence,
                    "rationale": metadata.get("rationale") or "",
                },
                source_tool=tool_name,
                source_modality=source_modality,
                fallback_confidence=item.get("score") or confidence,
            )
            evidence_atoms.append(
                self._make_evidence_atom(
                    source_tool=tool_name,
                    source_modality=source_modality,
                    text=atom_text,
                    confidence=self._safe_float(item.get("score") or confidence),
                    evidence_kind="web_fact" if source_modality == "web" else "text",
                    anchor=anchor,
                    metadata=metadata,
                    atom_index=evidence_index,
                    asset_id=asset_id,
                )
            )
            evidence_index += 1

        observations = payload.get("observations") or []
        range_observations = payload.get("range_observations") or []
        for idx, text in enumerate(observations, start=1):
            atom_text = self._compact_text(text)
            if not atom_text:
                continue
            anchor = None
            if idx - 1 < len(range_observations):
                range_payload = range_observations[idx - 1]
                if isinstance(range_payload, dict):
                    anchor = self._build_time_anchor(
                        range_payload.get("time_range"),
                        source_tool=tool_name,
                        source_modality=source_modality,
                        fallback_confidence=confidence,
                    )
            evidence_atoms.append(
                self._make_evidence_atom(
                    source_tool=tool_name,
                    source_modality=source_modality,
                    text=atom_text,
                    confidence=confidence,
                    evidence_kind="visual_observation",
                    anchor=anchor,
                    atom_index=evidence_index,
                    asset_id=asset_id,
                )
            )
            evidence_index += 1

        computation_text = self._compact_text(
            payload.get("result") or payload.get("stdout") or payload.get("summary")
        )
        if source_modality == "computation" and computation_text:
            evidence_atoms.append(
                self._make_evidence_atom(
                    source_tool=tool_name,
                    source_modality=source_modality,
                    text=computation_text,
                    confidence=max(confidence, 0.95),
                    evidence_kind="computation_result",
                    metadata={
                        "stderr": self._compact_text(payload.get("stderr")),
                        "code_preview": self._compact_text(payload.get("code_preview")),
                    },
                    atom_index=evidence_index,
                    asset_id=asset_id,
                )
            )
            evidence_index += 1

        summary_text = self._compact_text(payload.get("summary"))
        if summary_text and not evidence_atoms:
            evidence_atoms.append(
                self._make_evidence_atom(
                    source_tool=tool_name,
                    source_modality=source_modality,
                    text=summary_text,
                    confidence=confidence,
                    evidence_kind="guardrail"
                    if str(payload.get("status") or "") == "guardrail"
                    else "summary",
                    atom_index=evidence_index,
                    asset_id=asset_id,
                )
            )

        normalized: NormalizedToolObservation = {
            "status": str(payload.get("status") or "ok"),
            "source_tool": tool_name,
            "source_modality": source_modality,
            "tool_role": self._tool_role(tool_name),
            "summary": summary_text,
            "confidence": round(confidence, 4),
            "estimated_cost": self._build_estimated_cost(tool_name, payload),
            "fallback_hint": str(payload.get("fallback_hint") or ""),
            "candidate_ranges": candidate_ranges[:5],
            "evidence_atoms": evidence_atoms[:WINDOW_ATOMS_PER_OBSERVATION],
            "source_uncertainty": self._source_uncertainty_from_payload(payload, question),
            "raw_payload": payload,
        }
        return normalized

    def _range_signature(self, item: dict[str, Any]) -> str:
        return (
            f"{item.get('start_time')}|{item.get('end_time')}|"
            f"{item.get('source_tool')}|{item.get('source_modality')}"
        )

    def _merge_candidate_ranges(
        self,
        existing_ranges: list[dict[str, Any]],
        new_ranges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in existing_ranges + new_ranges:
            if not isinstance(item, dict):
                continue
            signature = self._range_signature(item)
            if signature not in merged or self._safe_float(item.get("confidence")) > self._safe_float(
                merged[signature].get("confidence")
            ):
                merged[signature] = dict(item)
        ranked = sorted(
            merged.values(),
            key=lambda item: self._safe_float(item.get("confidence")),
            reverse=True,
        )
        return ranked[:WINDOW_CANDIDATE_RANGES_CAP]

    def _build_conflicts(
        self,
        latest_visual: dict[str, Any] | None,
        latest_external: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        visual_dates = {
            self._compact_text(item).lower()
            for item in self._extract_date_mentions_from_observation(latest_visual)
        }
        external_dates = {
            self._compact_text(item).lower()
            for item in self._extract_date_mentions_from_observation(latest_external)
        }
        if visual_dates and external_dates and visual_dates.isdisjoint(external_dates):
            conflicts.append(
                {
                    "conflict_id": hashlib.sha1(
                        "|".join(sorted(visual_dates | external_dates)).encode("utf-8")
                    ).hexdigest()[:10],
                    "conflict_type": "temporal_mismatch",
                    "severity": "medium",
                    "description": "Visual and external evidence mention different date anchors.",
                    "evidence_ids": [],
                }
            )
        return conflicts

    def _recompute_unresolved_questions(
        self,
        question: str,
        run_report: dict[str, Any],
        state: EvidenceState,
    ) -> list[str]:
        unresolved = set(self._initial_unresolved_questions(question, run_report))
        candidate_ranges = state.get("candidate_ranges") or []
        if candidate_ranges:
            unresolved.discard("ground_relevant_video_moment")

        constraints = self._build_question_constraint_profile(question, run_report)
        tool_sequence = run_report.get("tool_sequence") or []
        if _evidence_needs_prefer_source(constraints, "subtitle") and any(
            tool_name == "subtitle_grounding_tool" for tool_name in tool_sequence
        ):
            unresolved.discard("recover_audio_or_lyric_anchor")
        if _evidence_needs_prefer_source(constraints, "audio") and any(
            tool_name in {"audio_scout_tool", "audio_event_locator_tool"}
            for tool_name in tool_sequence
        ):
            unresolved.discard("recover_audio_or_lyric_anchor")
        evidence_atoms = list(state.get("evidence_atoms") or [])
        if any(
            str(atom.get("source_tool") or "") in {
                "subtitle_grounding_tool",
                "audio_scout_tool",
                "audio_event_locator_tool",
            }
            and self._compact_text(atom.get("text"))
            for atom in evidence_atoms
            if isinstance(atom, dict)
        ):
            unresolved.discard("recover_audio_or_lyric_anchor")
        latest_computation = state.get("bridge_status", {}).get("computation_ready")
        state_temporal_anchors = self._collect_state_temporal_anchors(state)
        if _evidence_needs_require_temporal(constraints) and (
            len(state_temporal_anchors) >= TEMPORAL_ANCHOR_COUNT_THRESHOLD or latest_computation
        ):
            unresolved.discard("recover_exact_temporal_anchor")
        if _evidence_needs_require_entity_bridge(constraints):
            bridge_status = state.get("bridge_status", {}) or {}
            if bridge_status.get("entity_bridge_ready") or (
                bridge_status.get("visual_identity_ready")
                and not bridge_status.get("entity_lookup_pending")
            ):
                unresolved.discard("resolve_entity_identity_or_model")
        if not self._state_requires_external_lookup(question, run_report, state):
            unresolved.discard("recover_external_public_fact")
        if self._state_requires_deterministic_compute(run_report, state):
            unresolved.add("derive_exact_numeric_result")
        elif (
            _evidence_needs_require_numeric(constraints)
            and state.get("bridge_status", {}).get("computation_ready")
        ):
            unresolved.discard("derive_exact_numeric_result")

        latest_feedback = self._latest_web_search_feedback(run_report) or {}
        if _evidence_needs_require_entity_bridge(constraints):
            for slot in list(latest_feedback.get("missing_entity_slots") or [])[:4]:
                unresolved.add(f"missing_external_fact_bridge:{slot}")
        if state.get("bridge_status", {}).get("entity_bridge_ready"):
            unresolved = {item for item in unresolved if not item.startswith("missing_external_fact_bridge:")}
        return sorted(unresolved)

    def _collect_state_temporal_anchors(self, state: EvidenceState) -> list[str]:
        fragments: list[str] = []
        for atom in state.get("evidence_atoms") or []:
            if not isinstance(atom, dict):
                continue
            atom_text = self._compact_text(atom.get("text"))
            if atom_text:
                fragments.append(atom_text)
        latest_observation = state.get("latest_observation") or {}
        latest_summary = self._compact_text(latest_observation.get("summary"))
        if latest_summary:
            fragments.append(latest_summary)
        for atom in latest_observation.get("evidence_atoms") or []:
            if not isinstance(atom, dict):
                continue
            atom_text = self._compact_text(atom.get("text"))
            if atom_text:
                fragments.append(atom_text)
        normalized = " ".join(fragments)
        anchors: list[str] = []
        full_dates = [
            self._compact_text(match)
            for match in FULL_DATE_PATTERN.findall(normalized)
            if self._compact_text(match)
        ]
        iso_dates = [
            self._compact_text(match)
            for match in ISO_DATE_PATTERN.findall(normalized)
            if self._compact_text(match)
        ]
        month_day_mentions = [
            self._compact_text(match)
            for match in MONTH_DAY_TEXT_PATTERN.findall(normalized)
            if self._compact_text(match)
        ]
        anchors.extend(full_dates)
        anchors.extend(iso_dates)
        full_date_prefixes = {item.rsplit(",", 1)[0].strip().lower() for item in full_dates}
        for item in month_day_mentions:
            if item.lower() in full_date_prefixes:
                continue
            anchors.append(item)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in anchors:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _recompute_state_fields(
        self,
        state: EvidenceState,
        *,
        question: str,
        run_report: dict[str, Any],
    ) -> None:
        current_messages = self._current_messages_from_state(run_report)
        latest_visual = self._find_latest_successful_condensed_observation(
            current_messages, VISUAL_CONFIRM_TOOLS
        )
        latest_external = self._find_latest_successful_condensed_observation(
            current_messages, EXTERNAL_LOOKUP_TOOLS
        )
        latest_external_payload = self._find_latest_tool_payload(
            current_messages,
            EXTERNAL_LOOKUP_TOOLS,
            successful_only=True,
        ) or {}
        latest_computation = self._find_latest_successful_condensed_observation(
            current_messages, COMPUTATION_TOOLS
        )
        latest_computation_payload = self._find_latest_tool_payload(
            current_messages,
            COMPUTATION_TOOLS,
            successful_only=True,
        ) or {}
        latest_feedback = self._latest_web_search_feedback(run_report) or {}
        question_constraints = self._build_question_constraint_profile(question, run_report)
        resolved_entity_cards = [
            card
            for card in (state.get("entity_cards") or [])
            if isinstance(card, dict)
            and str(card.get("lookup_status") or "") == "resolved"
            and (
                self._compact_text(card.get("resolved_fact"))
                or self._compact_text(card.get("canonical_name"))
            )
        ]
        latest_external_feedback = dict((latest_external_payload or {}).get("search_feedback") or {})
        resolved_fact_bridge = any(
            str(record.get("status") or "") == "ok" and self._compact_text(record.get("extracted_value"))
            for record in (state.get("fact_bridge_records") or [])
            if isinstance(record, dict)
        )

        latest_external_profile = self._build_web_evidence_profile(latest_external_payload)
        has_external_exact_fact = bool(
            resolved_fact_bridge
            or (
                latest_external_payload
                and (
                    latest_external_payload.get("exact_date")
                    or latest_external_payload.get("matched_entities")
                    or latest_external_profile.get("has_terminal_fact_candidate")
                    or (
                        latest_external_feedback.get("progress_status") == "resolved_gap"
                        and not list(latest_external_feedback.get("missing_entity_slots") or [])
                    )
                )
            )
        )
        bridge_status = {
            "has_video_bridge": self._has_video_grounding_attempt(run_report),
            "has_external_lookup": self._has_external_lookup_attempt(run_report),
            "has_external_exact_fact": has_external_exact_fact,
            "entity_bridge_ready": (
                not _evidence_needs_require_entity_bridge(question_constraints)
                or has_external_exact_fact
                or bool(resolved_entity_cards)
                or (
                    latest_external_payload
                    and (
                        latest_external_payload.get("matched_entities")
                        or latest_external_payload.get("exact_date")
                        or (
                            str(latest_external_feedback.get("progress_status") or "")
                            == "resolved_gap"
                            and not list(latest_external_feedback.get("missing_entity_slots") or [])
                            and not list(latest_feedback.get("missing_entity_slots") or [])
                        )
                    )
                )
            ),
            "visual_identity_ready": bool(latest_visual and latest_visual.get("status") == "ok"),
            "computation_ready": bool(
                latest_computation
                and latest_computation.get("status") == "ok"
                and self._compact_text(
                    latest_computation_payload.get("result")
                    or latest_computation_payload.get("stdout")
                )
            ),
        }
        state["bridge_status"] = bridge_status
        self._refresh_entity_bridge(state, question=question, run_report=run_report)
        bridge_status = state.get("bridge_status") or {}
        state["budget_state"] = {
            "high_cost_call_count": int(run_report.get("high_cost_call_count", 0)),
            "reground_count": int(run_report.get("reground_count", 0)),
            "blocked_finish_count": int(run_report.get("blocked_finish_count", 0)),
            "answer_revision_count": int(run_report.get("answer_revision_count", 0)),
        }
        state["current_stage"] = self._infer_protocol_stage(run_report)
        state["conflicts"] = self._build_conflicts(latest_visual, latest_external)
        unresolved_questions = self._recompute_unresolved_questions(question, run_report, state)
        state["unresolved_questions"] = unresolved_questions
        state_temporal_anchors = self._collect_state_temporal_anchors(state)
        needs_visual_temporal_verification = bool(
            _evidence_needs_require_temporal(question_constraints)
            and state.get("candidate_ranges")
            and len(state_temporal_anchors) < 2
            and not bridge_status["visual_identity_ready"]
        )
        state["uncertainty_summary"] = {
            "temporal_gap": "recover_exact_temporal_anchor" in unresolved_questions,
            "entity_gap": "resolve_entity_identity_or_model" in unresolved_questions
            or any(item.startswith("missing_external_fact_bridge:") for item in unresolved_questions),
            "cross_modal_conflict": bool(state.get("conflicts")),
            "external_fact_gap": any(
                item.startswith("missing_external_fact_bridge:") for item in unresolved_questions
            ) or "recover_external_public_fact" in unresolved_questions,
            "visual_gap": bool(
                (
                    _evidence_needs_require_entity_bridge(question_constraints)
                    or needs_visual_temporal_verification
                )
                and not bridge_status["visual_identity_ready"]
            ),
        }
        latest_critic = run_report.get("latest_critic_verdict") or {}
        unresolved_set = set(unresolved_questions)

        if run_report.get("final_response_kind") == "answer":
            sufficiency_status = "answered"
        elif not unresolved_set:
            sufficiency_status = "ready_to_finish"
        elif unresolved_set <= {"derive_exact_numeric_result"}:
            sufficiency_status = "needs_computation"
        else:
            critic_verdict = str((latest_critic or {}).get("verdict") or "")
            if critic_verdict == "SUFFICIENT":
                sufficiency_status = "ready_to_finish"
            elif critic_verdict == "CONFLICTING":
                sufficiency_status = "needs_reground"
            else:
                # If the critic keeps saying INSUFFICIENT but we have evidence
                # atoms and have done multiple tool calls, promote to
                # "evidence_collected" so the answer checkpoint can judge the
                # answer on its own merits rather than being biased by
                # "seeking_grounding".
                evidence_atom_count = len(state.get("evidence_atoms") or [])
                tool_count = len(run_report.get("tool_sequence") or [])
                if evidence_atom_count >= EVIDENCE_ATOM_COUNT_THRESHOLD and tool_count >= TOOL_COUNT_EVIDENCE_COLLECTED:
                    sufficiency_status = "evidence_collected"
                else:
                    sufficiency_status = "seeking_grounding"
        state["sufficiency_status"] = sufficiency_status
        state["stop_ready"] = sufficiency_status in {"ready_to_finish", "answered"}
        state["decision_state"] = self._build_decision_state(
            state,
            question=question,
            run_report=run_report,
        )

        self._update_evidence_graph(state, question=question, run_report=run_report)

    def _ingest_normalized_observation(
        self,
        run_report: dict[str, Any],
        normalized_observation: NormalizedToolObservation,
        *,
        question: str,
    ) -> tuple[EvidenceState, dict[str, Any]]:
        state_before = copy.deepcopy(run_report.get("evidence_state") or self._init_evidence_state(question))
        state_after = copy.deepcopy(state_before)
        state_after["latest_observation"] = copy.deepcopy(normalized_observation)
        if normalized_observation.get("candidate_ranges"):
            state_after["candidate_ranges"] = self._merge_candidate_ranges(
                list(state_after.get("candidate_ranges") or []),
                list(normalized_observation.get("candidate_ranges") or []),
            )
        atoms = list(state_after.get("evidence_atoms") or [])
        existing_atom_ids = {str(item.get("evidence_id") or "") for item in atoms}
        for atom in normalized_observation.get("evidence_atoms") or []:
            atom_id = str(atom.get("evidence_id") or "")
            if not atom_id or atom_id in existing_atom_ids:
                continue
            existing_atom_ids.add(atom_id)
            atoms.append(copy.deepcopy(atom))
        state_after["evidence_atoms"] = atoms[-WINDOW_EVIDENCE_ATOMS_CAP:]
        self._recompute_state_fields(
            state_after,
            question=question,
            run_report=run_report,
        )
        step_index = len(run_report.get("step_records") or []) + 1
        # The return value is unused; the call is kept for its mutation of
        # state_after, and the recompute below runs either way (see comment).
        self._try_populate_fact_bridge(
            state_after,
            normalized_observation,
            question=question,
            step_index=step_index,
        )
        # Always recompute after fact bridge attempt — even when bridge didn't
        # change, other state fields (sufficiency_status, stop_ready) may need
        # updating based on accumulated evidence atoms from this step.
        self._recompute_state_fields(
            state_after,
            question=question,
            run_report=run_report,
        )
        delta = self._build_state_delta(state_before, state_after)
        state_after["latest_state_delta"] = copy.deepcopy(delta)
        run_report["evidence_state"] = state_after
        run_report["latest_state_delta"] = delta
        digest = self._build_evidence_state_digest(state_after)
        run_report["latest_state_digest"] = digest
        return state_after, delta

    def _build_state_delta(
        self,
        state_before: EvidenceState,
        state_after: EvidenceState,
    ) -> dict[str, Any]:
        before_ranges = {
            self._range_signature(item)
            for item in (state_before.get("candidate_ranges") or [])
            if isinstance(item, dict)
        }
        after_ranges = {
            self._range_signature(item)
            for item in (state_after.get("candidate_ranges") or [])
            if isinstance(item, dict)
        }
        before_atoms = {
            str(item.get("evidence_id") or "")
            for item in (state_before.get("evidence_atoms") or [])
            if isinstance(item, dict)
        }
        after_atoms = {
            str(item.get("evidence_id") or "")
            for item in (state_after.get("evidence_atoms") or [])
            if isinstance(item, dict)
        }
        before_unresolved = set(state_before.get("unresolved_questions") or [])
        after_unresolved = set(state_after.get("unresolved_questions") or [])
        before_budget = state_before.get("budget_state") or {}
        after_budget = state_after.get("budget_state") or {}

        before_fb = [
            (str(r.get("entity_id", "")), str(r.get("status", "")))
            for r in (state_before.get("fact_bridge_records") or [])
            if isinstance(r, dict)
        ]
        after_fb = [
            (str(r.get("entity_id", "")), str(r.get("status", "")))
            for r in (state_after.get("fact_bridge_records") or [])
            if isinstance(r, dict)
        ]
        fact_bridge_changed = set(after_fb) != set(before_fb)

        needs_improved = need_statuses_improved(
            need_status_list(state_before.get("evidence_needs")),
            need_status_list(state_after.get("evidence_needs")),
        )

        before_bridge = state_before.get("bridge_status") or {}
        after_bridge = state_after.get("bridge_status") or {}
        bridge_improved = False
        # entity_bridge_ready advanced
        if after_bridge.get("entity_bridge_ready") and not before_bridge.get("entity_bridge_ready"):
            bridge_improved = True
        # visual_identity_ready advanced
        if after_bridge.get("visual_identity_ready") and not before_bridge.get("visual_identity_ready"):
            bridge_improved = True
        # has_external_exact_fact advanced
        if after_bridge.get("has_external_exact_fact") and not before_bridge.get("has_external_exact_fact"):
            bridge_improved = True
        # has_video_bridge advanced
        if after_bridge.get("has_video_bridge") and not before_bridge.get("has_video_bridge"):
            bridge_improved = True

        range_narrowed = bool(
            before_ranges
            and after_ranges
            and len(after_ranges) < len(before_ranges)
        )

        delta = {
            "added_candidate_ranges": max(0, len(after_ranges - before_ranges)),
            "added_evidence_atoms": max(0, len(after_atoms - before_atoms)),
            "resolved_questions": sorted(before_unresolved - after_unresolved),
            "new_questions": sorted(after_unresolved - before_unresolved),
            "sufficiency_transition": {
                "before": state_before.get("sufficiency_status"),
                "after": state_after.get("sufficiency_status"),
            },
            "stop_ready_changed": bool(state_before.get("stop_ready")) != bool(
                state_after.get("stop_ready")
            ),
            "budget_delta": {
                key: int(after_budget.get(key) or 0) - int(before_budget.get(key) or 0)
                for key in (
                    "high_cost_call_count",
                    "reground_count",
                    "blocked_finish_count",
                    "answer_revision_count",
                )
            },
            "fact_bridge_changed": fact_bridge_changed,
            "needs_improved": needs_improved,
            "bridge_improved": bridge_improved,
            "range_narrowed": range_narrowed,
        }
        delta["state_changed"] = bool(
            delta["added_candidate_ranges"]
            or delta["added_evidence_atoms"]
            or delta["resolved_questions"]
            or delta["new_questions"]
            or delta["sufficiency_transition"]["before"]
            != delta["sufficiency_transition"]["after"]
            or delta["stop_ready_changed"]
            or any(delta["budget_delta"].values())
            or delta["fact_bridge_changed"]
            or delta["needs_improved"]
            or delta["bridge_improved"]
        )
        return delta

    def _refresh_delta_after_verdict(
        self,
        run_report: dict[str, Any],
        before_need_statuses: list[str],
    ) -> None:
        """Fold the verdict's need closures into the step's delta."""
        evidence_state = run_report.get("evidence_state") or {}
        if not need_statuses_improved(
            list(before_need_statuses or []),
            need_status_list(evidence_state.get("evidence_needs")),
        ):
            # Still rebuild the digest: statuses may have moved without improving
            # (a re-open), and the planner reads the digest, not the delta.
            run_report["latest_state_digest"] = self._build_evidence_state_digest(
                evidence_state
            )
            return

        # A new dict rather than an in-place edit: the trace layer attaches the
        # delta to a span handle by reference (agent.py _attach_evidence_state),
        # so mutating it would retroactively rewrite an already-logged step.
        previous = run_report.get("latest_state_delta") or {}
        refreshed = {**previous, "needs_improved": True, "state_changed": True}
        run_report["latest_state_delta"] = refreshed
        # evidence_state carries its own copy (see _ingest_normalized_observation).
        # Leaving it stale would put two contradicting values for one field into
        # the step record.
        evidence_state["latest_state_delta"] = copy.deepcopy(refreshed)
        run_report["evidence_state"] = evidence_state
        run_report["latest_state_digest"] = self._build_evidence_state_digest(
            evidence_state
        )

    @staticmethod
    def critical_open_needs(state: EvidenceState) -> list[str]:
        """Return all critical non-computation needs still open."""
        open_needs: list[str] = []

        # Path 1: evidence_needs (structured, preferred)
        evidence_needs = list(state.get("evidence_needs") or [])
        if evidence_needs:
            for need in evidence_needs:
                if not isinstance(need, dict):
                    continue
                status = str(need.get("status") or "unfilled")
                preferred_source = str(need.get("preferred_source") or "")
                if status in {"filled", "unfillable"}:
                    continue
                if preferred_source == "computation":
                    continue
                need_text = str(need.get("need") or "unknown")
                open_needs.append(need_text)

        # Path 2: unresolved_questions fallback (for non-evidence-needs mode)
        if not evidence_needs:
            COMPUTATION_QUESTIONS = {"derive_exact_numeric_result"}
            for q in (state.get("unresolved_questions") or []):
                if q in COMPUTATION_QUESTIONS:
                    continue
                open_needs.append(str(q))

        return sorted(set(open_needs))

    def _build_evidence_state_digest(
        self,
        state: EvidenceState,
    ) -> EvidenceStateDigest:
        atoms = list(state.get("evidence_atoms") or [])
        unique_asset_ids = {
            str(atom.get("asset_id") or "")
            for atom in atoms
            if isinstance(atom, dict)
        } - {""}

        requested_budget = getattr(self, "atoms_budget_override", None)
        normalized_requested_budget = None
        try:
            if requested_budget is not None:
                normalized_requested_budget = int(requested_budget)
        except (TypeError, ValueError):
            normalized_requested_budget = None

        budget_getter = getattr(self, "_effective_atoms_budget", None)
        if callable(budget_getter):
            budget_limit = int(budget_getter())
        else:
            budget_limit = normalized_requested_budget or TOTAL_ATOM_BUDGET
            if budget_limit < 1:
                budget_limit = TOTAL_ATOM_BUDGET

        if len(unique_asset_ids) > 1:
            # Multi-asset: keep chronological order while honoring the per-run
            # ablation atom budget override.
            valid_atoms = [a for a in atoms if isinstance(a, dict)]
            recent_evidence = []
            for atom in valid_atoms[-budget_limit:]:
                recent_evidence.append(
                    {
                        "asset_id": atom.get("asset_id"),
                        "evidence_id": atom.get("evidence_id"),
                        "kind": atom.get("evidence_kind"),
                        "source_tool": atom.get("source_tool"),
                        "text": self._build_evidence_preview_text(
                            atom.get("text"),
                            atom.get("source_modality"),
                        ),
                        "confidence": atom.get("confidence"),
                    }
                )
        else:
            recent_evidence = []
            single_asset_limit = min(4, budget_limit)
            for atom in atoms[-single_asset_limit:]:
                if not isinstance(atom, dict):
                    continue
                recent_evidence.append(
                    {
                        "evidence_id": atom.get("evidence_id"),
                        "kind": atom.get("evidence_kind"),
                        "source_tool": atom.get("source_tool"),
                        "text": self._build_evidence_preview_text(
                            atom.get("text"),
                            atom.get("source_modality"),
                        ),
                        "confidence": atom.get("confidence"),
                    }
                )

        budget_state = dict(state.get("budget_state") or {})
        if normalized_requested_budget is not None:
            budget_state["atoms_budget_requested"] = normalized_requested_budget
            budget_state["atoms_budget_effective"] = int(budget_limit)

        result: EvidenceStateDigest = {
            "query_id": str(state.get("query_id") or ""),
            "video_id": str(state.get("video_id") or ""),
            "current_stage": state.get("current_stage", "start"),
            "sufficiency_status": state.get("sufficiency_status", "seeking_grounding"),
            "unresolved_questions": list(state.get("unresolved_questions") or [])[:8],
            "uncertainty_summary": dict(state.get("uncertainty_summary") or {}),
            "bridge_status": dict(state.get("bridge_status") or {}),
            "budget_state": budget_state,
            "candidate_range_preview": list(state.get("candidate_ranges") or [])[:WINDOW_NORMALIZE_ITEMS],
            "recent_evidence": recent_evidence,
            "conflict_count": len(state.get("conflicts") or []),
            "evidence_atom_count": len(state.get("evidence_atoms") or []),
            "stop_ready": bool(state.get("stop_ready")),
            "decision_state": copy.deepcopy(state.get("decision_state") or {}),
            "gap_diagnosis": dict(state.get("_latest_gap_diagnosis") or {}),
        }
        raw_needs = list(state.get("evidence_needs") or [])
        enriched_needs: list[dict[str, Any]] = []
        for need in raw_needs:
            if not isinstance(need, dict):
                continue
            depends_on = [
                dep
                for dep in list(need.get("depends_on") or [])
                if isinstance(dep, int) and not isinstance(dep, bool)
            ]
            enriched_needs.append(
                {
                    "need": need.get("need", ""),
                    "preferred_source": need.get("preferred_source", ""),
                    "closure_level": need.get("closure_level", "blocking"),
                    "status": need.get("status", "unfilled"),
                    "filled_by": need.get("filled_by"),
                    "need_role": need.get("need_role", "anchor"),
                    "depends_on": depends_on,
                    "query_focus": need.get("query_focus"),
                }
            )

        terminal_statuses = {"filled", "unfillable"}
        actionable_indices: list[int] = []
        blocked_indices: list[int] = []
        for idx, need in enumerate(enriched_needs):
            if str(need.get("status") or "unfilled") in terminal_statuses:
                continue
            deps = list(need.get("depends_on") or [])
            deps_terminal = all(
                enriched_needs[dep]["status"] in terminal_statuses
                for dep in deps
                if 0 <= dep < len(enriched_needs)
            )
            if not deps or deps_terminal:
                actionable_indices.append(idx)
            else:
                blocked_indices.append(idx)

        result["evidence_needs"] = enriched_needs
        result["actionable_need_indices"] = actionable_indices
        result["blocked_need_indices"] = blocked_indices
        # Include entity-card and fact-bridge summaries.
        if state.get("entity_cards") or state.get("fact_bridge_records"):
            result["entity_cards"] = [  # type: ignore[typeddict-unknown-key]
                {
                    "name": c.get("canonical_name"),
                    "type": c.get("entity_type"),
                    "status": c.get("lookup_status"),
                    "lookup_slot": c.get("lookup_slot"),
                    "resolved_fact": c.get("resolved_fact", ""),
                }
                for c in (state.get("entity_cards") or [])[:4]
            ]
            result["fact_bridge_records"] = [  # type: ignore[typeddict-unknown-key]
                {
                    "entity_id": r.get("entity_id"),
                    "query_type": r.get("query_type"),
                    "extracted_value": r.get("extracted_value"),
                    "status": r.get("status"),
                }
                for r in (state.get("fact_bridge_records") or [])[-4:]
            ]
        # Include graph summary and working-memory slot count in the digest.
        evidence_graph = state.get("evidence_graph") or {}
        graph_summary = str(evidence_graph.get("graph_summary") or "")
        conflict_paths = list(evidence_graph.get("conflict_paths") or [])
        expose_graph_summary = bool(graph_summary and conflict_paths)
        if expose_graph_summary:
            result["graph_summary"] = graph_summary  # type: ignore[typeddict-unknown-key]
        memory_slots = state.get("memory_slots") or []
        if memory_slots:
            last_slot = memory_slots[-1] if memory_slots else {}
            result["bridge_status"] = {  # type: ignore[typeddict-unknown-key]
                **dict(result.get("bridge_status") or {}),
                "memory_hit": bool(int(last_slot.get("hit_count") or 0) > 0),
                "memory_slot_count": len(memory_slots),
            }
        return result
