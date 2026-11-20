"""Critic and answer checkpoint pipeline."""
import copy
import hashlib
import json
import logging
import sys
from typing import Any
import agents.config as config
from agents.planning.question_profile import VALID_NEED_ROLES, _clear_cycle_participants
from agents.planning.prompts import (
    get_answer_critic_prompt_template,
    get_critic_prompt_template,
)
from agents.shared.constants import (
    ANSWER_CHECK_VERDICTS,
    COMPUTATION_TOOLS,
    CRITIC_VERDICTS,
    EXTERNAL_LOOKUP_TOOLS,
    TRUNC_PROPOSED_ANSWER_PREVIEW,
    TRUNC_TOOL_CONTENT_SUMMARY,
    VIDEO_GROUNDING_TOOLS,
    VISUAL_CONFIRM_TOOLS,
    WINDOW_ACCUMULATED_OBS,
    WINDOW_RECENT_FEEDBACK,
    build_tool_call_counts,
)
from agents.shared.json_utils import dumps_llm_payload, safe_load_json_object
from agents.shared.utils import LLMCallFailure, call_openai_model_with_tools
from agents.state.state_machine import need_status_list

REDUNDANT_OBS_STEP_KEYS = {"raw_observation", "normalized_observation"}
HISTORICAL_DIGEST_STEP_KEYS = {"evidence_state_digest"}
STEP_CRITIC_VERDICT_KEYS = {"critic_verdict"}
CONDENSE_STATE_STEP_KEYS = (
    {"state_after"}
    | REDUNDANT_OBS_STEP_KEYS
    | HISTORICAL_DIGEST_STEP_KEYS
    | STEP_CRITIC_VERDICT_KEYS
)

REDUNDANT_OBS_PAYLOAD_KEYS = {"raw_payload"}


def _condense_step_records(records: Any, drop_keys: set) -> list[dict]:
    """Return step records with ``drop_keys`` removed from each, order-preserving.

    ``drop_keys`` naming a field nested under ``action`` (as
    ``evidence_state_digest`` is) removes it there too, without disturbing the
    rest of the action.
    """
    condensed: list[dict] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        entry = {k: v for k, v in record.items() if k not in drop_keys}
        action = entry.get("action")
        if isinstance(action, dict) and any(k in action for k in drop_keys):
            entry["action"] = {
                k: v for k, v in action.items() if k not in drop_keys
            }
        condensed.append(entry)
    return condensed


def _condense_observations(observations: Any, drop_keys: set) -> list[dict]:
    """Return observations with ``drop_keys`` projected out of each.

    Builds new dicts rather than mutating: the entries are shared references
    into ``run_report["normalized_observations"]``, which the state machine and
    the entity bridge still read in full.
    """
    return [
        {k: v for k, v in observation.items() if k not in drop_keys}
        for observation in (observations or [])
        if isinstance(observation, dict)
    ]


class CriticPipeline:
    """Provides critic checkpoint, answer checkpoint, reground abort,
    and stale web-search loop abort logic."""

    def _build_shared_critic_context(
        self,
        question: str,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build context fields shared between evidence critic and answer critic.

        Centralizes the fields that both payloads need, eliminating copy-paste
        divergence risk and making payload changes a single-point edit.
        """
        payload = {
            "question": question,
            "normalized_observations": _condense_observations(
                (run_report.get("normalized_observations") or [])[-WINDOW_ACCUMULATED_OBS:],
                REDUNDANT_OBS_PAYLOAD_KEYS,
            ),
            "high_cost_call_count": run_report.get("high_cost_call_count", 0),
            "reground_count": run_report.get("reground_count", 0),
            "blocked_finish_count": run_report.get("blocked_finish_count", 0),
            "answer_revision_count": run_report.get("answer_revision_count", 0),
            "tool_sequence": run_report.get("tool_sequence", []),
            "tool_call_counts": build_tool_call_counts(
                (run_report or {}).get("tool_sequence")
            ),
            "evidence_needs": list(
                (run_report.get("evidence_state") or {}).get("evidence_needs") or []
            ),
            "latest_video_grounding_observation": self._find_latest_condensed_observation(
                msgs, VIDEO_GROUNDING_TOOLS
            ),
            "latest_external_observation": self._find_latest_condensed_observation(
                msgs, EXTERNAL_LOOKUP_TOOLS
            ),
            "latest_visual_observation": self._find_latest_condensed_observation(
                msgs, VISUAL_CONFIRM_TOOLS
            ),
            "latest_web_search_feedback": self._latest_web_search_feedback(run_report),
            "recent_web_search_feedback": (run_report.get("web_search_feedback") or [])[-WINDOW_RECENT_FEEDBACK:],
            "latest_answer_checkpoint": (
                self._summarize_recent_answer_checkpoints(run_report, limit=1) or [None]
            )[-1],
            "recent_critic_checkpoints": self._summarize_recent_critic_checkpoints(run_report),
            "recent_answer_checkpoints": self._summarize_recent_answer_checkpoints(run_report),
            "accumulated_observations": self._collect_condensed_observations(msgs)[-WINDOW_ACCUMULATED_OBS:],
            "recent_step_records": (run_report.get("step_records") or [])[-WINDOW_RECENT_FEEDBACK:],
            "evidence_state_digest": run_report.get("latest_state_digest") or {},
            "latest_state_delta": run_report.get("latest_state_delta") or {},
        }

        if self._is_history_only_mode():
            window = int(getattr(self, "history_only_message_window", 15) or 15)
            if window < 1:
                window = 15
            payload["message_history"] = copy.deepcopy(msgs[-window:])
            payload["history_truncated"] = len(msgs) > window
            payload["history_window_steps"] = window
            for _leaky in (
                "evidence_needs",
                "evidence_state_digest",
                "latest_state_delta",
                "recent_critic_checkpoints",
            ):
                payload.pop(_leaky, None)
            _LEAKY_STEP_KEYS = {
                "action",
                "state_after",
                "critic_verdict",
                "answer_verdict",
                "evidence_delta",
            }
            payload["recent_step_records"] = _condense_step_records(
                payload.get("recent_step_records"),
                _LEAKY_STEP_KEYS | REDUNDANT_OBS_STEP_KEYS,
            )
            return payload

        payload["recent_step_records"] = _condense_step_records(
            payload.get("recent_step_records"), CONDENSE_STATE_STEP_KEYS
        )
        return payload
    def _call_critic_model_with_tools(self, *args, **kwargs):
        """Issue a critic LLM call, labelled ``caller="critic"`` for accounting.

        The name is mixin-specific on purpose. CoreAgent inherits both this
        mixin and QuestionProfile, which had a same-named helper; MRO resolved
        every critic call to QuestionProfile's version, so critic tokens were
        booked under ``question_profile`` and the ``critic`` bucket never
        appeared in any run's by_caller breakdown.
        """
        core_agent_module = sys.modules.get("agents.core.agent") or sys.modules.get(
            "agents.core_agent"
        )
        try:
            if core_agent_module is not None:
                patched_caller = getattr(core_agent_module, "call_openai_model_with_tools", None)
                if callable(patched_caller):
                    return patched_caller(*args, caller="critic", **kwargs)
            return call_openai_model_with_tools(*args, caller="critic", **kwargs)
        except LLMCallFailure:
            # The provider will refuse an identical request again, so the critic
            # falls back to its deterministic verdict rather than propagating a
            # transport failure into the decision loop. The failure is already
            # recorded in the shared ledger.
            return None
    def _should_run_critic_after_tool(self, name: str) -> bool:
        return True
    def _should_abort_after_stale_web_search_loop(
        self,
        run_report: dict[str, Any],
    ) -> bool:
        """Compatibility hook."""
        return False
    def _build_stale_web_search_abort_message(
        self,
        run_report: dict[str, Any],
    ) -> str:
        """Compatibility hook."""
        return ""
    def _abort_after_stale_web_search_loop(
        self,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> None:
        del msgs, run_report
        return
    def _normalize_gap_diagnosis(
        self,
        raw_gap: Any,
        verdict: str,
    ) -> dict[str, str]:
        if not isinstance(raw_gap, dict):
            return {"gap_type": verdict.lower(), "description": ""}
        gap_type = str(raw_gap.get("gap_type") or "").strip().lower()
        if gap_type not in {"sufficient", "insufficient", "conflicting"}:
            gap_type = verdict.lower()
        return {
            "gap_type": gap_type,
            "description": str(raw_gap.get("description") or "").strip(),
        }
    def _fallback_critic_verdict(
        self,
        run_report: dict[str, Any],
        state_digest: dict[str, Any] | None = None,
        after_tool: str = "",
    ) -> dict[str, Any]:
        checkpoint_index = len(run_report.get("critic_checkpoints") or []) + 1
        return {
            "verdict": "INSUFFICIENT",
            "gap_diagnosis": {
                "gap_type": "insufficient",
                "description": "critic unavailable",
            },
            "stage": "after_tool",
            "after_tool": after_tool,
            "checkpoint_index": checkpoint_index,
            "source": "fallback",
        }
    def _normalize_critic_verdict(
        self,
        raw: dict[str, Any],
        run_report: dict[str, Any],
        state_digest: dict[str, Any] | None = None,
        after_tool: str = "",
    ) -> dict[str, Any]:
        verdict_str = str(raw.get("verdict") or "").strip().upper()
        if verdict_str not in CRITIC_VERDICTS:
            verdict_str = "INSUFFICIENT"
        checkpoint_index = len(run_report.get("critic_checkpoints") or []) + 1
        raw_updates = raw.get("updated_need_statuses")
        if not isinstance(raw_updates, list):
            raw_updates = []
        normalized_updates = []
        valid_statuses = {"unfilled", "partial", "filled", "unfillable"}
        filtered_count = 0
        for entry in raw_updates:
            if not isinstance(entry, dict):
                filtered_count += 1
                continue
            need_index = entry.get("need_index")
            status = str(entry.get("status") or "").strip().lower()
            if (
                isinstance(need_index, bool)
                or not isinstance(need_index, int)
                or need_index < 0
                or status not in valid_statuses
            ):
                filtered_count += 1
                continue
            normalized_updates.append(
                {
                    "need_index": need_index,
                    "status": status,
                    "filled_by": str(entry.get("filled_by") or "").strip() or None,
                    "revised_query_focus": entry.get("revised_query_focus"),
                    "revised_need_role": entry.get("revised_need_role"),
                    "add_depends_on": entry.get("add_depends_on"),
                }
            )
        if filtered_count > 0:
            logging.getLogger("omnivqa.critic").debug(
                "updated_need_statuses: filtered %d malformed entries out of %d total",
                filtered_count,
                len(raw_updates),
            )
        return {
            "verdict": verdict_str,
            "gap_diagnosis": self._normalize_gap_diagnosis(
                raw.get("gap_diagnosis"),
                verdict_str,
            ),
            "stage": "after_tool",
            "after_tool": after_tool,
            "checkpoint_index": checkpoint_index,
            "source": "llm",
            "updated_need_statuses": normalized_updates,
        }

    def _sync_critic_need_updates(
        self,
        checkpoint: dict[str, Any],
        run_report: dict[str, Any],
    ) -> None:
        updated_statuses = checkpoint.get("updated_need_statuses") or []
        if not updated_statuses:
            return

        evidence_state = run_report.get("evidence_state") or {}
        current_needs = list(evidence_state.get("evidence_needs") or [])
        if not current_needs:
            return

        index_map = {
            int(status["need_index"]): status
            for status in updated_statuses
            if (
                isinstance(status, dict)
                and isinstance(status.get("need_index"), int)
                and not isinstance(status.get("need_index"), bool)
            )
        }
        if not index_map:
            return

        # Snapshot before the mutation loop: the reducer folds these closures into
        # the step's delta afterwards and cannot recover the prior statuses once
        # they are overwritten.
        before_need_statuses = need_status_list(current_needs)

        for idx, need in enumerate(current_needs):
            if not isinstance(need, dict):
                continue
            match = index_map.get(idx)
            if not match:
                continue
            need["status"] = match["status"]
            if "filled_by" in match:
                need["filled_by"] = match.get("filled_by")
            if match.get("revised_query_focus"):
                query_focus = str(match.get("revised_query_focus") or "").strip()[:100]
                if query_focus:
                    need["query_focus"] = query_focus
            if match.get("revised_need_role"):
                need_role = str(match.get("revised_need_role") or "").strip().lower()
                if need_role in VALID_NEED_ROLES:
                    need["need_role"] = need_role

        any_dep_changed = False
        for idx, need in enumerate(current_needs):
            if not isinstance(need, dict):
                continue
            match = index_map.get(idx)
            if not match or not isinstance(match.get("add_depends_on"), list):
                continue
            existing = {
                dep
                for dep in list(need.get("depends_on") or [])
                if isinstance(dep, int) and not isinstance(dep, bool)
            }
            new_deps = {
                dep
                for dep in match["add_depends_on"]
                if isinstance(dep, int) and not isinstance(dep, bool) and 0 <= dep < len(current_needs) and dep != idx
            }
            if new_deps - existing:
                need["depends_on"] = sorted(existing | new_deps)
                any_dep_changed = True

        if any_dep_changed:
            _clear_cycle_participants(current_needs)

        evidence_state["evidence_needs"] = current_needs
        run_report["evidence_state"] = evidence_state

        # Hand off to the reducer: it folds these closures into the step's delta
        # and rebuilds the digest. The getattr guard exists because the critic
        # pipeline is exercised standalone in scripts/flow_test, without the state
        # machine mixin bound. Failures are not swallowed — a silently skipped
        # refresh leaves needs_improved False, which is the defect this repairs.
        refresh_delta = getattr(self, "_refresh_delta_after_verdict", None)
        if callable(refresh_delta):
            refresh_delta(run_report, before_need_statuses)
            return

        build_digest = getattr(self, "_build_evidence_state_digest", None)
        if callable(build_digest):
            try:
                run_report["latest_state_digest"] = build_digest(evidence_state)
            except Exception:
                pass

    def _fallback_answer_verdict(
        self,
        proposed_answer: str,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Deterministic safety nets only: empty answer and repeated answer."""
        checkpoint_index = len(run_report.get("answer_checkpoints") or []) + 1
        base = {
            "stage": "before_finish",
            "checkpoint_index": checkpoint_index,
            "proposed_answer_preview": (proposed_answer or "")[:TRUNC_PROPOSED_ANSWER_PREVIEW],
            "evidence_signature": self._build_evidence_signature(msgs),
            "source": "fallback",
        }
        if not (proposed_answer or "").strip():
            return {
                **base,
                "verdict": "BLOCK",
                "revision_hint": "Answer is empty.",
            }
        recent_blocks = [
            checkpoint
            for checkpoint in (run_report.get("answer_checkpoints") or [])[-WINDOW_RECENT_FEEDBACK:]
            if checkpoint.get("verdict") == "BLOCK"
        ]
        if len(recent_blocks) >= 2:
            prev_answers = [
                str(checkpoint.get("proposed_answer_preview") or "").strip().lower()
                for checkpoint in recent_blocks
            ]
            current = (proposed_answer or "").strip().lower()[:TRUNC_PROPOSED_ANSWER_PREVIEW]
            if current and any(current == prev_answer for prev_answer in prev_answers):
                return {
                    **base,
                    "verdict": "PASS",
                    "revision_hint": "",
                }
        return {
            **base,
            "verdict": "BLOCK",
            "revision_hint": "Answer critic unavailable -- cannot verify answer quality.",
        }
    def _normalize_answer_verdict(
        self,
        raw: dict[str, Any],
        proposed_answer: str,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        verdict_str = str(raw.get("verdict") or "").strip().upper()
        if verdict_str not in ANSWER_CHECK_VERDICTS:
            verdict_str = "BLOCK"
        revision_hint = ""
        if verdict_str == "BLOCK":
            revision_hint = str(
                raw.get("revision_hint")
                or raw.get("reason")
                or ""
            ).strip()
        checkpoint_index = len(run_report.get("answer_checkpoints") or []) + 1
        return {
            "verdict": verdict_str,
            "revision_hint": revision_hint,
            "stage": "before_finish",
            "checkpoint_index": checkpoint_index,
            "proposed_answer_preview": (proposed_answer or "")[:TRUNC_PROPOSED_ANSWER_PREVIEW],
            "evidence_signature": self._build_evidence_signature(msgs),
            "source": "llm",
        }
    def _should_abort_after_reground(
        self,
        checkpoint: dict[str, Any],
        run_report: dict[str, Any],
    ) -> bool:
        """Compatibility hook."""
        return False
    def _build_reground_abort_message(self, checkpoint: dict[str, Any]) -> str:
        """Compatibility hook."""
        return ""
    def _abort_after_reground(
        self,
        checkpoint: dict[str, Any],
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> None:
        del checkpoint, msgs, run_report
        return
    def _run_critic_checkpoint(
        self,
        stage: str,
        tool_name: str,
        tool_content: str,
        msgs: list[dict],
        question: str,
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_no_state_react_mode():
            return {"tool": tool_name}
        latest_payload = safe_load_json_object(tool_content)
        run_report["_message_snapshot"] = copy.deepcopy(msgs)
        core_agent_module = sys.modules.get("agents.core.agent") or sys.modules.get(
            "agents.core_agent"
        )
        critic_prompt_getter = None
        if core_agent_module is not None:
            critic_prompt_getter = getattr(core_agent_module, "get_critic_prompt_template", None)
        critic_prompt_template = (
            critic_prompt_getter()
            if callable(critic_prompt_getter)
            else get_critic_prompt_template()
        )
        evidence_digest = run_report.get("latest_state_digest") or {}
        recent_evidence = evidence_digest.get("recent_evidence") or []
        unique_assets = {
            str(item.get("asset_id") or "")
            for item in recent_evidence
            if isinstance(item, dict)
        } - {""}
        if len(unique_assets) > 1 and critic_prompt_template:
            critic_prompt_template += (
                "\n\nMultiple assets are involved. Check whether evidence from "
                "different assets (e.g., audio_0 vs image_1) is consistent and "
                "whether cross-asset entities have been correctly identified and linked."
            )
        raw_payload = self._build_shared_critic_context(question, msgs, run_report)
        raw_payload["stage"] = stage
        raw_payload["latest_observation"] = self._summarize_observation(
            tool_name,
            latest_payload or {"summary": tool_content[:TRUNC_TOOL_CONTENT_SUMMARY]},
        )
        parsed_verdict = None
        response = None
        checkpoint_messages = [
            {"role": "system", "content": critic_prompt_template},
            {
                "role": "user",
                "content": dumps_llm_payload(raw_payload),
            },
        ]
        if critic_prompt_template:
            response = self._call_critic_model_with_tools(
                checkpoint_messages,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                return_json=True,
            )
            if response is not None:
                parsed_verdict = safe_load_json_object(response.get("content", ""))
        if isinstance(parsed_verdict, dict):
            checkpoint = self._normalize_critic_verdict(
                parsed_verdict,
                run_report,
                state_digest=run_report.get("latest_state_digest") or {},
                after_tool=tool_name,
            )
        else:
            checkpoint = self._fallback_critic_verdict(
                run_report,
                state_digest=run_report.get("latest_state_digest") or {},
                after_tool=tool_name,
            )
        run_report["critic_checkpoints"].append(checkpoint)
        if getattr(self, "_dump_checkpoints", False) and getattr(self, "_dump_checkpoint_sink", None):
            try:
                state_keys = [
                    "latest_state_digest",
                    "latest_state_delta",
                    "evidence_state",
                    "step_records",
                    "tool_sequence",
                    "high_cost_call_count",
                    "reground_count",
                    "normalized_observations",
                ]
                rec = {
                    "case_id": getattr(self, "_current_case_id", None),
                    "step": len(run_report.get("step_records") or []),
                    "stage": stage,
                    "role": "critic",
                    "msgs_snapshot": copy.deepcopy(msgs),
                    "checkpoint_messages": copy.deepcopy(checkpoint_messages),
                    "raw_payload": copy.deepcopy(raw_payload),
                    "prompt_hash": (
                        hashlib.sha256(critic_prompt_template.encode()).hexdigest()
                        if critic_prompt_template
                        else ""
                    ),
                    "tool_name": tool_name,
                    "tool_content": str(raw_payload.get("latest_observation") or ""),
                    "response_content_raw": str((response or {}).get("content") or ""),
                    "run_report_state_slice": copy.deepcopy(
                        {key: run_report.get(key) for key in state_keys}
                    ),
                    "verdict": copy.deepcopy(checkpoint),
                }
                self._dump_checkpoint_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._dump_checkpoint_sink.flush()
            except Exception as dump_exc:  # noqa: BLE001 - never fail a case on dump errors
                logging.getLogger(__name__).warning("dump_checkpoint failed: %s", dump_exc)
        run_report["latest_critic_verdict"] = checkpoint
        self._sync_critic_need_updates(checkpoint, run_report)
        evidence_state = run_report.get("evidence_state") or {}
        evidence_state["_latest_gap_diagnosis"] = checkpoint.get("gap_diagnosis") or {
            "gap_type": "insufficient",
            "description": "Critic did not provide gap diagnosis",
        }
        run_report["evidence_state"] = evidence_state
        if checkpoint.get("verdict") in {"INSUFFICIENT", "CONFLICTING"}:
            run_report["reground_count"] += 1
            # Only count as a repair attempt when there has been at least one
            # prior finish attempt or the agent is in answer_revision stage,
            # meaning the agent is actually trying to fix/re-gather evidence
            # after a blocked answer. Initial evidence gathering (before any
            # finish attempt) is normal operation, not repair.
            if (
                (run_report.get("last_finish_attempt") or run_report.get("blocked_finish_count", 0) > 0)
                and not self._is_no_repair_mode()
            ):
                run_report["repair_attempt_count"] += 1
        if checkpoint.get("verdict") == "SUFFICIENT":
            last_finish_attempt = str(run_report.get("last_finish_attempt") or "").strip()
            if last_finish_attempt:
                current_step = len(run_report.get("tool_sequence") or [])
                existing = run_report.get("best_candidate_answer")
                existing_step = (
                    int((existing or {}).get("cached_at_step") or 0)
                    if isinstance(existing, dict)
                    else 0
                )
                if current_step >= existing_step:
                    run_report["best_candidate_answer"] = {
                        "answer_text": last_finish_attempt,
                        "sufficiency_status": str(
                            (run_report.get("evidence_state") or {}).get(
                                "sufficiency_status"
                            )
                            or ""
                        ),
                        "cached_at_step": current_step,
                        "source": "critic_answer_verdict",
                    }
        return checkpoint
    def _run_answer_checkpoint(
        self,
        proposed_answer: str,
        msgs: list[dict],
        question: str,
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_answer = proposed_answer.strip()
        previous_answer = str(run_report.get("last_finish_attempt") or "").strip()
        run_report["last_finish_attempt"] = normalized_answer
        run_report["_message_snapshot"] = copy.deepcopy(msgs)
        answer_critic_prompt_template = get_answer_critic_prompt_template()
        current_evidence_signature = self._build_evidence_signature(msgs)
        raw_payload = self._build_shared_critic_context(question, msgs, run_report)
        raw_payload["stage"] = "before_finish"
        raw_payload["proposed_answer"] = proposed_answer
        raw_payload["previous_proposed_answer"] = previous_answer
        if not self._is_history_only_mode():
            raw_payload["latest_evidence_verdict"] = run_report.get("latest_critic_verdict")
        raw_payload["latest_computation_observation"] = self._find_latest_condensed_observation(
            msgs, COMPUTATION_TOOLS
        )
        parsed_verdict = None
        response = None
        checkpoint_messages = [
            {"role": "system", "content": answer_critic_prompt_template},
            {
                "role": "user",
                "content": dumps_llm_payload(raw_payload),
            },
        ]
        if answer_critic_prompt_template:
            response = self._call_critic_model_with_tools(
                checkpoint_messages,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                return_json=True,
            )
            if response is not None:
                parsed_verdict = safe_load_json_object(response.get("content", ""))
        if isinstance(parsed_verdict, dict):
            checkpoint = self._normalize_answer_verdict(
                parsed_verdict,
                proposed_answer,
                msgs,
                run_report,
            )
        else:
            checkpoint = self._fallback_answer_verdict(
                proposed_answer,
                msgs,
                run_report,
            )
        checkpoint["evidence_signature"] = current_evidence_signature
        checkpoint["proposed_answer_preview"] = proposed_answer[:TRUNC_PROPOSED_ANSWER_PREVIEW]
        run_report["answer_checkpoints"].append(checkpoint)
        if getattr(self, "_dump_checkpoints", False) and getattr(self, "_dump_checkpoint_sink", None):
            try:
                state_keys = [
                    "latest_state_digest",
                    "latest_state_delta",
                    "evidence_state",
                    "step_records",
                    "tool_sequence",
                    "high_cost_call_count",
                    "reground_count",
                    "normalized_observations",
                ]
                rec = {
                    "case_id": getattr(self, "_current_case_id", None),
                    "step": len(run_report.get("step_records") or []),
                    "stage": "before_finish",
                    "role": "answer",
                    "msgs_snapshot": copy.deepcopy(msgs),
                    "checkpoint_messages": copy.deepcopy(checkpoint_messages),
                    "raw_payload": copy.deepcopy(raw_payload),
                    "prompt_hash": (
                        hashlib.sha256(answer_critic_prompt_template.encode()).hexdigest()
                        if answer_critic_prompt_template
                        else ""
                    ),
                    "tool_name": "",
                    "tool_content": str(raw_payload.get("latest_observation") or ""),
                    "response_content_raw": str((response or {}).get("content") or ""),
                    "run_report_state_slice": copy.deepcopy(
                        {key: run_report.get(key) for key in state_keys}
                    ),
                    "verdict": copy.deepcopy(checkpoint),
                    "proposed_answer": proposed_answer,
                }
                self._dump_checkpoint_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._dump_checkpoint_sink.flush()
            except Exception as dump_exc:  # noqa: BLE001 - never fail a case on dump errors
                logging.getLogger(__name__).warning("dump_checkpoint failed: %s", dump_exc)
        run_report["latest_answer_verdict"] = checkpoint
        if checkpoint.get("verdict") in {"PASS", "BLOCK"}:
            current_step = len(run_report.get("tool_sequence") or [])
            existing = run_report.get("best_candidate_answer")
            existing_step = (
                int((existing or {}).get("cached_at_step") or 0)
                if isinstance(existing, dict)
                else 0
            )
            if normalized_answer and current_step >= existing_step:
                run_report["best_candidate_answer"] = {
                    "answer_text": normalized_answer,
                    "sufficiency_status": str(
                        (run_report.get("evidence_state") or {}).get(
                            "sufficiency_status"
                        )
                        or ""
                    ),
                    "cached_at_step": current_step,
                    "source": "answer_checkpoint",
                }
        return checkpoint
