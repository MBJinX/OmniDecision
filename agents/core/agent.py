import copy
import hashlib
import inspect
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Annotated as A, Any

import agents.config as config
from agents.asset_runtime import consume_runtime_cache_summary, reset_runtime_cache_summary
from agents.build_database import init_single_video_db, init_audio_only_db
from agents.shared.func_call_schema import as_json_schema
from agents.shared.func_call_schema import doc as D
from agents.shared.json_utils import dumps_llm_payload, safe_load_json_object
from agents.planning.prompts import (
    build_omni_messages,
    get_tools_for_asset_type,
)
from agents.shared.types import (
    ActionTraceItem,
    DecisionRecord,
    EvidenceState,
    NormalizedToolObservation,
    ProtocolStage,
    StepRecord,
    ToolRole,
)
from agents.tools.audio_scout_tool import audio_scout_tool
from agents.tools.clip_grounding_tool import clip_grounding_tool
from agents.tools.frame_confirm_tool import frame_confirm_tool
from agents.tools.subtitle_grounding_tool import subtitle_grounding_tool
from agents.tools.web_search_tool import web_search_tool
from agents.shared.utils import (
    LLMCallFailure,
    call_openai_model_with_tools,
    provider_failure_ledger,
    token_tracker,
)
from agents.shared.constants import (
    BATCH_MAX_WIDTH,
    COMPACTED_OBSERVATION_MARKER,
    COMPUTATION_TOOLS,
    CONSECUTIVE_STALE_STEP_THRESHOLD,
    EXTERNAL_LOOKUP_TOOLS,
    GROUNDING_TOOLS,
    MIN_TOOL_CALLS_BEFORE_FINISH,
    NO_VERIFIED_VALUE_ANSWER,
    REPEAT_GUARD_MAX,
    TOTAL_ATOM_BUDGET,
    TRUNC_BEST_EFFORT_ANSWER_SPAN,
    TRUNC_MEMORY_RESULT_TEXT,
    TRUNC_PROPOSED_ANSWER_PREVIEW,
    TRUNC_RESULT_SUMMARY,
    VIDEO_GROUNDING_TOOLS,
    VISUAL_CONFIRM_TOOLS,
    WINDOW_EVIDENCE_SIGNATURE_OBS,
    build_tool_call_counts,
)
from agents.state.answer_readiness import AnswerReadiness
from agents.planning.critic import CriticPipeline
from agents.state.entity_bridge import EntityBridge
from agents.state.evidence_graph import EvidenceGraphMixin
from agents.state.state_machine import EvidenceStateMachine
from agents.state.memory import MemorySlots
from agents.observation_utils import ObservationUtils
from agents.planning.question_profile import (
    QuestionProfile,
    _evidence_needs_prefer_source,
    _evidence_needs_require_entity_bridge,
    _evidence_needs_require_numeric,
    _evidence_needs_require_temporal,
)
from agents.planning.synthetic_tools import SyntheticTools
from agents.tool_guards import MAX_ANSWER_REVISION_REOPENS, ToolGuards
from agents.state.web_evidence import WebEvidence

if TYPE_CHECKING:
    from agents.logging.step_logger import StepLogger


class StopException(Exception):
    """Stop execution by raising this exception when the task is finished."""


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer and finish the conversation."""
    raise StopException(answer)


MAX_EMPTY_RETRIES = 4  # was 3, raised per D-02 / GATE-02
MAX_ANSWER_REVISIONS = 1  # Maximum answer revision attempts before force-accept (D-04 / STOP-04)

PLANNER_RAW_OBSERVATION_WINDOW = 2

STATE_DIGEST_MESSAGE_PREFIX = "Evidence State Digest for OmniVQA"

NAIVE_MEMORY_MESSAGE_PREFIX = "Rolling memory for OmniVQA"

# These are internal schema labels, not task evidence. The naive-memory arm must
# not expose them to the planner, even when the summarizer copies a label from
# its source despite the prompt.
NAIVE_MEMORY_SCHEMA_TERMS = {
    "evidence_needs": "open evidence requirements",
    "candidate_range_preview": "candidate time ranges",
    "actionable_need_indices": "requirements that can be pursued now",
    "blocked_need_indices": "requirements that are currently blocked",
    "tool_call_counts": "tools already used",
    "gap_diagnosis": "remaining gap",
    "entity_cards": "known entities",
    "fact_bridge_records": "resolved external facts",
    "finish_available": "whether an answer may be submitted",
    "remaining_rounds": "search rounds left",
    "unresolved_questions": "unresolved questions",
    "current_stage": "current stage",
    "sufficiency_status": "evidence sufficiency",
    "decision_state": "decision status",
    "bridge_status": "cross-source resolution status",
    "memory_hit": "a repeated lookup served from memory",
}

# Tail of the answer-phase evidence summary. It closes the summary the answer
# phase sends to the tool-calling planner, and _best_effort_finalize strips it
# before reusing that summary for the plain-text finalizer, which cannot call
# finish. One constant so the strip cannot drift out of sync with the text.
ANSWER_PHASE_FINISH_INSTRUCTION = (
    "\n\nEvidence gathering is complete. You MUST call finish with your best answer now."
    " If the question asks for an identity, a name, a number, or a date, commit to a"
    " specific value; a description or category of the target is not an answer."
)
VALID_ABLATION_MODES = {
    "full",
    "history_only",
    "naive_memory",
    "no_repair",
    "no_stop_insufficient",
    "no_state_react",
}


class CoreAgent(
    MemorySlots,
    QuestionProfile,
    EvidenceGraphMixin,
    WebEvidence,
    AnswerReadiness,
    EntityBridge,
    EvidenceStateMachine,
    CriticPipeline,
    SyntheticTools,
    ObservationUtils,
    ToolGuards,
):
    """OmniVQA core runtime for the current baseline tool loop."""

    def __init__(
        self,
        video_db_path=None,
        video_caption_path=None,
        max_iterations=15,
        agent_name: str = "omnivqa",
        debug_trace: bool = False,
        *,
        asset_descriptors: list[dict] | None = None,
        ablation_mode: str = "full",
        atoms_budget_override: int | None = None,
        disabled_tools: set[str] | None = None,
        step_logger: "StepLogger | None" = None,
    ):
        self.agent_name = agent_name
        self.debug_trace = debug_trace
        self._step_logger = step_logger
        self._pending_case_id: str | None = None
        self.disabled_tools = {str(name) for name in (disabled_tools or set())}
        self.disabled_tools.add("code_executor_tool")
        if not config.TAVILY_API_KEY:
            self.disabled_tools.add("web_search_tool")
        self.tools = self._build_tools()
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.max_iterations = max_iterations

        normalized_ablation_mode = str(ablation_mode or "full").strip().lower() or "full"
        if normalized_ablation_mode not in VALID_ABLATION_MODES:
            raise ValueError(
                f"Unsupported ablation_mode={ablation_mode!r}; expected one of {sorted(VALID_ABLATION_MODES)}"
            )
        normalized_atoms_budget = None
        if atoms_budget_override is not None:
            normalized_atoms_budget = int(atoms_budget_override)
            if normalized_atoms_budget < 1:
                raise ValueError("atoms_budget_override must be >= 1")

        self.ablation_mode = normalized_ablation_mode
        self.atoms_budget_override = normalized_atoms_budget
        self.history_only_message_window = 15

        # Backward compat shim: wrap old-style args into asset_descriptors
        if asset_descriptors is None and video_db_path is not None:
            asset_descriptors = [{
                "asset_id": Path(video_db_path).parent.name,
                "type": "video",
                "db_path": str(video_db_path),
                "caption_path": str(video_caption_path) if video_caption_path else None,
            }]

        # Build asset registry and per-asset NanoVectorDBs
        self.asset_registry: dict[str, dict] = {}
        self.asset_dbs: dict[str, Any] = {}

        if asset_descriptors:
            seen_ids: set[str] = set()
            for desc in asset_descriptors:
                aid = desc.get("asset_id", "")
                if not aid or not isinstance(aid, str):
                    raise ValueError(f"Invalid asset_id: {aid!r}")
                if aid in seen_ids:
                    raise ValueError(f"Duplicate asset_id: {aid!r}")
                seen_ids.add(aid)

        for desc in (asset_descriptors or []):
            asset_id = desc["asset_id"]
            asset_type = desc.get("type", "video")

            if asset_type == "video":
                vdb = init_single_video_db(
                    desc.get("caption_path"),
                    desc.get("db_path"),
                    config.AOAI_EMBEDDING_LARGE_DIM,
                )
                self.asset_dbs[asset_id] = vdb
                try:
                    desc["duration_hhmmss"] = vdb.get_additional_data()["video_length"]
                except Exception:
                    desc["duration_hhmmss"] = "unknown"

            elif asset_type == "audio":
                vdb = init_audio_only_db(
                    desc["path"],
                    desc.get("db_path", ""),
                    config.AOAI_EMBEDDING_LARGE_DIM,
                    t_start=desc.get("t_start"),
                    t_end=desc.get("t_end"),
                )
                self.asset_dbs[asset_id] = vdb
                try:
                    desc["duration_hhmmss"] = vdb.get_additional_data()["video_length"]
                except Exception:
                    desc["duration_hhmmss"] = "unknown"

            elif asset_type == "image":
                try:
                    from PIL import Image
                    img = Image.open(desc["path"])
                    desc["dimensions"] = img.size
                    img.close()
                except Exception:
                    desc["dimensions"] = None

            self.asset_registry[asset_id] = desc

        # Primary DB: first available DB for legacy code paths
        self.primary_db = next(iter(self.asset_dbs.values()), None)
        self.video_db = self.primary_db  # legacy alias

        # Preserve legacy attributes for backward compat
        if asset_descriptors:
            first_video = next(
                (d for d in asset_descriptors if d.get("type") == "video"),
                asset_descriptors[0] if asset_descriptors else None,
            )
            self.video_db_path = (first_video or {}).get("db_path") or video_db_path
            self.video_caption_path = (first_video or {}).get("caption_path") or video_caption_path
        else:
            self.video_db_path = video_db_path
            self.video_caption_path = video_caption_path

        self._asset_descriptors = asset_descriptors
        self._multi_asset = (
            len(self.asset_registry) > 1
            or any(d.get("type") != "video" for d in (asset_descriptors or []))
        )
        self._refresh_asset_id_mappings()

        self.function_schemas = self._planner_schemas()
        self.messages = self._construct_messages()
        self.last_run_report: dict[str, Any] = {}
        self.last_parallel_run_reports: list[dict[str, Any] | None] = []

    def _planner_schemas(self):
        """Expose user-supplied arguments and the asset IDs each tool can read."""
        schemas = []
        for tool in self.tools:
            schema = as_json_schema(tool)
            params = schema["parameters"]
            fields = params["properties"]
            if "database" in fields:
                del fields["database"]
                required = params.get("required", [])
                required.remove("database")
                eligible = [a for a in self.asset_registry.values()
                            if tool.__name__ in get_tools_for_asset_type(a["type"])]
                if not eligible:
                    continue
                if self._multi_asset:
                    fields["asset_id"] = {"type": "string", "enum": [a["simplified_id"] for a in eligible]}
                    required.append("asset_id")
                if "time_ranges_hhmmss" in fields and any(a["type"] == "image" for a in eligible):
                    required.remove("time_ranges_hhmmss")
                    fields["time_ranges_hhmmss"]["description"] = "Required for video. Omit for still images."
            schemas.append({"type": "function", "function": schema})
        return schemas

    def _refresh_asset_id_mappings(self) -> None:
        self._simplified_asset_ids: dict[str, str] = {}
        self._original_asset_ids: dict[str, str] = {}
        type_counters: dict[str, int] = {}
        for original_id, desc in self.asset_registry.items():
            asset_type = str((desc or {}).get("type") or "unknown")
            index = type_counters.get(asset_type, 0)
            type_counters[asset_type] = index + 1
            simplified = f"{asset_type}_{index}"
            self._simplified_asset_ids[original_id] = simplified
            self._original_asset_ids[simplified] = original_id
            desc["simplified_id"] = simplified

    def _debug_print(self, message: str) -> None:
        if self.debug_trace:
            print(message)

    def _span(self, role: str, **extra):
        """Open a logger span, or a no-op context manager if no logger attached."""
        logger = getattr(self, "_step_logger", None)
        if logger is None:
            from contextlib import nullcontext
            return nullcontext()
        return logger.span(role, **extra)

    def _derive_case_id(self, question: str) -> str:
        override = getattr(self, "_pending_case_id", None)
        if override:
            return str(override)

        descriptors = getattr(self, "_asset_descriptors", None) or []
        if getattr(self, "_multi_asset", False) and descriptors:
            first = descriptors[0] or {}
            aid = first.get("asset_id") or first.get("id")
            if aid:
                return str(aid)

        if getattr(self, "video_db_path", None):
            return Path(str(self.video_db_path)).name

        return "q_" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]

    def set_case_id(self, case_id: str) -> None:
        """Set the case id for the next run() invocation."""
        self._pending_case_id = str(case_id)

    def _attach_planner_span_fields(
        self,
        handle,
        response: dict | None,
        run_report: dict[str, Any] | None = None,
    ) -> None:
        if handle is None or not isinstance(response, dict):
            return
        tool_calls = []
        for tool_call in response.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                tool_calls.append({"name": name})
        finish_attempted = any(item.get("name") == "finish" for item in tool_calls)
        handle.extra["tool_calls"] = tool_calls
        handle.extra["finish_attempted"] = finish_attempted
        output_preview = str(response.get("content") or "").strip()
        if not output_preview and tool_calls:
            output_preview = "tool_calls: " + ", ".join(item["name"] for item in tool_calls)
        handle.attach_io(output_preview=output_preview)
        phase = handle.extra.get("phase")
        if not phase and run_report is not None:
            try:
                phase = self._infer_protocol_stage(run_report)
            except Exception:
                phase = None
        input_preview_payload = {
            "phase": str(phase or "unknown"),
            "planned_tools": [call["name"] for call in tool_calls[:3]],
            "finish_attempted": finish_attempted,
        }
        handle.attach_io(input_preview=json.dumps(input_preview_payload, ensure_ascii=False))

    def _attach_tool_span_fields(
        self,
        handle,
        tool_name: str,
        tool_messages: list[dict],
        run_report: dict[str, Any],
        full_args: str = "",
    ) -> None:
        if handle is None:
            return
        tool_msg = next(
            (
                message
                for message in tool_messages
                if isinstance(message, dict) and message.get("role") == "tool"
            ),
            None,
        )
        tool_content = ""
        if tool_msg is not None:
            tool_content = str(tool_msg.get("content") or "")
        # Store summarized output for optional reference; do NOT use as output_preview.
        output_summary = self._summarize_tool_content(tool_name, tool_content) if tool_content else ""
        handle.extra["output_summary"] = output_summary
        # Full raw tool content as output_preview; fall back to final_output_text if empty.
        output_preview = tool_content if tool_content else str(run_report.get("final_output_text") or "").strip()
        # Full args string as input_preview; fall back to existing arguments_preview in extra.
        input_preview = full_args if full_args else handle.extra.get("arguments_preview", "")
        handle.attach_io(input_preview=input_preview, output_preview=output_preview)
        payload = safe_load_json_object(tool_content)
        tool_status = "ok"
        if isinstance(payload, dict):
            tool_status = str(payload.get("status") or tool_status).strip() or tool_status
        handle.extra["tool_status"] = tool_status
        # Wire evidence state from run_report (populated after _exec_tool returns).
        evidence_delta = run_report.get("latest_state_delta") or {}
        evidence_digest = (
            run_report.get("latest_state_digest")
            or run_report.get("evidence_state_digest")
            or ""
        )
        handle.extra["evidence_delta"] = evidence_delta
        if evidence_digest and isinstance(evidence_digest, str):
            handle.extra["evidence_state_digest"] = evidence_digest

    def _agent_label(self) -> str:
        return self.agent_name

    def _current_ablation_mode(self) -> str:
        mode = str(getattr(self, "ablation_mode", "full") or "full").strip().lower() or "full"
        if mode not in VALID_ABLATION_MODES:
            return "full"
        return mode

    def _is_history_only_mode(self) -> bool:
        return self._current_ablation_mode() == "history_only"

    def _is_naive_memory_mode(self) -> bool:
        return self._current_ablation_mode() == "naive_memory"

    def _is_no_repair_mode(self) -> bool:
        return self._current_ablation_mode() == "no_repair"

    def _is_no_state_react_mode(self) -> bool:
        return self._current_ablation_mode() == "no_state_react"

    def _inert_evidence_state(self, question: str) -> dict[str, Any]:
        """Return a non-falsy, inert evidence-state placeholder for no_state_react."""
        return {
            "_inert": True,
            "query_id": self._build_query_id(question),
            "video_id": self._current_video_id(),
            "memory_slots": [],
        }

    def _effective_atoms_budget(self) -> int:
        override = getattr(self, "atoms_budget_override", None)
        try:
            limit = int(override)
        except (TypeError, ValueError):
            return TOTAL_ATOM_BUDGET
        return limit if limit > 0 else TOTAL_ATOM_BUDGET

    def _build_tools(self):
        tools = [
            audio_scout_tool,
            subtitle_grounding_tool,
            clip_grounding_tool,
            frame_confirm_tool,
            finish,
            web_search_tool,
        ]
        if not getattr(self, "disabled_tools", None):
            return tools
        return [tool for tool in tools if tool.__name__ not in self.disabled_tools]

    def _construct_messages(self):
        if self._multi_asset:
            from agents.planning.prompts import build_omni_messages_multi
            return build_omni_messages_multi(
                self.asset_registry,
                disabled_tools=set(self.disabled_tools or ()),
                ablation_mode=self._current_ablation_mode(),
            )
        else:
            # Single video: use original path for backward compat
            video_length = self.primary_db.get_additional_data()["video_length"] if self.primary_db else "unknown"
            return build_omni_messages(
                video_length,
                disabled_tools=set(self.disabled_tools or ()),
                ablation_mode=self._current_ablation_mode(),
            )

    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        message = {
            "role": "tool",
            "name": name,
            "content": content,
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        msgs.append(message)
        return message

    def _append_system_msg(self, content: str, msgs: list[dict]) -> dict:
        message = {"role": "system", "content": content}
        msgs.append(message)
        return message

    def _init_run_report(self, question: str) -> dict[str, Any]:
        no_state = self._is_no_state_react_mode()
        if no_state:
            question_profile: dict[str, Any] = {}
            evidence_state = self._inert_evidence_state(question)
            latest_state_digest: dict[str, Any] = {}
        else:
            # Infer question profile via LLM before building evidence state so that
            # _initial_unresolved_questions (called inside _init_evidence_state) can
            # use the cached profile without triggering an extra LLM call.
            bootstrap_report: dict[str, Any] = {"_question_profile": {}}
            bootstrap_report["_question_profile"] = self._build_question_constraint_profile(
                question, bootstrap_report
            )
            question_profile = bootstrap_report["_question_profile"]
            evidence_state = self._init_evidence_state(question, bootstrap_report)
            latest_state_digest = self._build_evidence_state_digest(evidence_state)
        return {
            "agent_name": self.agent_name,
            "question": question,
            "ablation_mode": self._current_ablation_mode(),
            "atoms_budget_requested": getattr(self, "atoms_budget_override", None),
            "atoms_budget_effective": self._effective_atoms_budget(),
            "tool_sequence": [],
            "action_trace": [],
            "step_records": [],
            "critic_checkpoints": [],
            "high_cost_call_count": 0,
            "reground_count": 0,
            "repair_attempt_count": 0,
            "best_candidate_answer": None,
            "latest_critic_verdict": None,
            "answer_checkpoints": [],
            "blocked_finish_count": 0,
            "answer_revision_count": 0,
            "_inline_answer_revision_count": 0,
            "latest_answer_verdict": None,
            "last_finish_attempt": "",
            "final_response_kind": "incomplete",
            "final_output_text": "",
            "stop_reason": "",
            "answer_phase": False,
            "web_search_feedback": [],
            "asset_cache_summary": [],
            "asset_cache_totals": {
                "hit_count": 0,
                "build_count": 0,
                "invalidate_count": 0,
            },
            "grounding_cache_totals": {
                "hit_count": 0,
                "build_count": 0,
                "invalidate_count": 0,
            },
            "latest_asset_cache_summary": None,
            "evidence_state": evidence_state,
            "latest_state_digest": latest_state_digest,
            "latest_state_delta": {},
            "normalized_observations": [],
            "strategy_switch_injections": [],
            "last_strategy_switch_step": 0,
            "last_strategy_switch_round": 0,
            "_message_snapshot": [],
            "_question_profile": question_profile,
        }

    def _reset_asset_cache_tracking(self) -> None:
        if getattr(self.primary_db, "storage_file", None) is None:
            return
        try:
            reset_runtime_cache_summary(self.primary_db)
        except Exception:
            return

    def _capture_asset_cache_summary(
        self,
        run_report: dict[str, Any],
        tool_name: str,
    ) -> None:
        if getattr(self.primary_db, "storage_file", None) is None:
            run_report["latest_asset_cache_summary"] = {
                "tool_name": tool_name,
                "has_activity": False,
            }
            return
        try:
            summary = consume_runtime_cache_summary(self.primary_db)
        except Exception as exc:
            run_report["latest_asset_cache_summary"] = {
                "tool_name": tool_name,
                "has_activity": False,
                "error": str(exc),
            }
            return

        summary["tool_name"] = tool_name
        run_report["latest_asset_cache_summary"] = summary
        if not summary.get("has_activity"):
            return

        run_report.setdefault("asset_cache_summary", []).append(summary)
        totals = run_report.setdefault(
            "asset_cache_totals",
            {"hit_count": 0, "build_count": 0, "invalidate_count": 0},
        )
        totals["hit_count"] += int(summary.get("hit_count") or 0)
        totals["build_count"] += int(summary.get("build_count") or 0)
        totals["invalidate_count"] += int(summary.get("invalidate_count") or 0)
        grounding_totals = run_report.setdefault(
            "grounding_cache_totals",
            {"hit_count": 0, "build_count": 0, "invalidate_count": 0},
        )
        grounding_summary = summary.get("grounding_cache") or {}
        grounding_totals["hit_count"] += int(grounding_summary.get("hit_count") or 0)
        grounding_totals["build_count"] += int(grounding_summary.get("build_count") or 0)
        grounding_totals["invalidate_count"] += int(grounding_summary.get("invalidate_count") or 0)

    def _current_video_id(self) -> str:
        additional_data_getter = getattr(self.primary_db, "get_additional_data", None)
        if callable(additional_data_getter):
            try:
                additional_data = additional_data_getter() or {}
            except Exception:
                additional_data = {}
            for key in ("video_id", "vid", "video_name"):
                value = str(additional_data.get(key) or "").strip()
                if value:
                    return value
        return Path(str(self.video_db_path)).resolve().parent.name

    def _build_query_id(self, question: str) -> str:
        payload = f"{self._current_video_id()}::{self._compact_text(question)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _initial_unresolved_questions(self, question: str, run_report: dict[str, Any] | None = None) -> list[str]:
        constraints = self._build_question_constraint_profile(question, run_report)
        unresolved = ["ground_relevant_video_moment"]
        if _evidence_needs_prefer_source(
            constraints, "subtitle"
        ) or _evidence_needs_prefer_source(constraints, "audio"):
            unresolved.append("recover_audio_or_lyric_anchor")
        if _evidence_needs_require_temporal(constraints):
            unresolved.append("recover_exact_temporal_anchor")
        if _evidence_needs_require_entity_bridge(constraints):
            unresolved.append("resolve_entity_identity_or_model")
        if _evidence_needs_prefer_source(constraints, "web"):
            unresolved.append("recover_external_public_fact")
        if _evidence_needs_require_numeric(constraints):
            unresolved.append("derive_exact_numeric_result")
        return unresolved

    def _build_decision_state(
        self,
        state: EvidenceState,
        *,
        question: str,
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_status = state.get("bridge_status") or {}

        consecutive_stale = self._count_consecutive_stale_steps(run_report)

        stop_insufficient_ready = consecutive_stale >= CONSECUTIVE_STALE_STEP_THRESHOLD

        graph = state.get("evidence_graph") or {}
        estimated_cost = 0.25
        expected_value = 0.9
        llm_cost = graph.get("estimated_cost_to_continue")
        llm_value = graph.get("expected_value_of_next_step")
        if isinstance(llm_cost, (int, float)) and 0.0 <= llm_cost <= 1.0:
            estimated_cost = float(llm_cost)
        if isinstance(llm_value, (int, float)) and 0.0 <= llm_value <= 1.0:
            expected_value = float(llm_value)

        raw_computation_ready = bool(bridge_status.get("computation_ready"))
        open_source_needs = self.critical_open_needs(state)
        computation_blocked_by_source = raw_computation_ready and bool(open_source_needs)

        return {
            "stop_insufficient_ready": stop_insufficient_ready,
            "computation_ready": raw_computation_ready and not computation_blocked_by_source,
            "computation_blocked_by_source": computation_blocked_by_source,
            "estimated_cost_to_continue": round(float(estimated_cost), 4),
            "expected_value_of_next_step": round(float(expected_value), 4),
        }

    @staticmethod
    def _compute_finish_available(
        run_report: dict[str, Any],
        *,
        is_last_iteration: bool = False,
        ablation_mode: str | None = None,
    ) -> bool:
        """Determine whether finish should be in the tool schema."""
        latest_critic = run_report.get("latest_critic_verdict") or {}
        critic_verdict = str(latest_critic.get("verdict") or "").upper()
        evidence_state = run_report.get("evidence_state") or {}
        latest_digest = run_report.get("latest_state_digest") or {}
        decision_state = (
            evidence_state.get("decision_state")
            or latest_digest.get("decision_state")
            or {}
        )
        stop_ready = bool(decision_state.get("stop_insufficient_ready", False))
        mode = str(ablation_mode or "full").strip().lower() or "full"
        if mode not in VALID_ABLATION_MODES:
            mode = "full"

        if mode == "no_state_react":
            return True

        if critic_verdict == "SUFFICIENT":
            return True

        if stop_ready and mode != "no_stop_insufficient":
            return True

        if is_last_iteration:
            return True

        evidence_needs = list(
            evidence_state.get("evidence_needs")
            or latest_digest.get("evidence_needs")
            or []
        )
        blocking_open = [
            need
            for need in evidence_needs
            if isinstance(need, dict)
            and str(need.get("closure_level") or "blocking") == "blocking"
            and str(need.get("status") or "unfilled") in {"unfilled", "partial"}
        ]
        if not blocking_open:
            return True

        blocking_partial = [
            n for n in blocking_open
            if str(n.get("status") or "") == "partial"
        ]
        if blocking_partial:
            return False

        tool_count = len(run_report.get("tool_sequence") or [])
        return tool_count >= MIN_TOOL_CALLS_BEFORE_FINISH

    def _count_consecutive_stale_steps(self, run_report: dict[str, Any]) -> int:
        """Count consecutive tool calls from the end that produced no real progress."""
        steps = self._step_history(run_report)
        count = 0
        reset_after_step = int(run_report.get("last_strategy_switch_step") or 0)
        for idx in range(len(steps) - 1, -1, -1):
            step = steps[idx]
            if (idx + 1) <= reset_after_step:
                break
            # Skip blocked steps
            if step.get("action", {}).get("blocked"):
                continue

            if self._step_is_productive(step):
                break

            count += 1

        return count

    @staticmethod
    def _step_is_productive(step: dict[str, Any]) -> bool:
        """PGR-04 productivity for one step, in the definition every consumer shares."""
        delta = step.get("evidence_delta") or {}
        return bool(
            list(delta.get("resolved_questions") or [])
            or delta.get("fact_bridge_changed", False)
            or delta.get("needs_improved", False)
            or delta.get("bridge_improved", False)
            or delta.get("range_narrowed", False)
        )

    def _dominant_stale_tool(
        self,
        run_report: dict[str, Any],
        stale_steps: int,
    ) -> str | None:
        """Return the tool that dominated the stale window, or None on a tie.

        The injection records what the planner was doing when advised, so a
        trace shows whether the advice was taken.
        """
        if stale_steps <= 0:
            return None
        steps = [
            step
            for step in self._step_history(run_report)
            if not (step.get("action") or {}).get("blocked")
        ]
        window = steps[-stale_steps:]
        counts = Counter(
            str((step.get("action") or {}).get("selected_tool") or "")
            for step in window
        )
        counts.pop("", None)
        if not counts:
            return None
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    @staticmethod
    def _record_llm_call_failure(
        run_report: dict[str, Any],
        failure: LLMCallFailure,
        *,
        layer: str,
    ) -> None:
        """Note which layer absorbed a deterministic model failure.

        The failure itself is already classified and latched in the shared
        provider ledger; what the run report adds is where it surfaced, so a
        trace shows whether the run lost the planner, the finalizer, or both.
        """
        record = failure.as_record()
        record["layer"] = layer
        run_report.setdefault("provider_failures", []).append(record)

    @staticmethod
    def _has_deterministic_provider_failure(run_report: dict[str, Any]) -> bool:
        """Return whether a non-retryable provider failure has already occurred."""
        return provider_failure_ledger.has_deterministic_failure()

    @staticmethod
    def _last_assistant_answer_span(msgs: list[dict] | None) -> str:
        """Return the newest assistant `content` usable as a submitted answer.

        The final terminal of the degrade chain must not fabricate, so the only
        remaining material is what the model already wrote in this run. Reading
        it back is not a new claim; it is the last committed statement the run
        produced. Refusal-shaped and empty content is skipped, and the span is
        capped so a paragraph of reasoning does not become the answer.
        """
        for message in reversed(list(msgs or [])):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "") != "assistant":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if CoreAgent._is_refusal_or_abort_answer(content):
                continue
            # Prefer the last non-empty line: a planner message that ends in a
            # conclusion carries its reasoning above it.
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            span = lines[-1] if lines else content
            if CoreAgent._is_refusal_or_abort_answer(span):
                span = content
            return span[:TRUNC_BEST_EFFORT_ANSWER_SPAN].strip()
        return ""

    @staticmethod
    def _finalize_best_effort_answer(
        run_report: dict[str, Any],
        *,
        stop_reason: str,
        computation_fallback: str = "",
        msgs: list[dict] | None = None,
    ) -> None:
        """Emit the terminal answer without fabricating and without refusing.

        The degrade chain is ordered by how directly each source commits to a
        value: the cached candidate, then a computed result, then the newest
        assistant statement in the history. The last resort used to be a static
        string beginning "I could not determine", which `_is_refusal_or_abort_answer`
        classifies as a refusal — the final line of defense emitted exactly what
        the answer policy forbids. It now states that no value was obtained,
        which scores the same zero but keeps the refusal detector honest and
        leaves `stop_reason` as the signal attribution reads.
        """
        cached_answer = run_report.get("best_candidate_answer") or {}
        answer_text = str(cached_answer.get("answer_text") or "").strip()
        if not answer_text:
            # Prefer computation fallback over generic refusal text
            if computation_fallback:
                answer_text = computation_fallback
        if not answer_text:
            answer_text = CoreAgent._last_assistant_answer_span(msgs)
            if answer_text:
                run_report["best_effort_answer_source"] = "last_assistant_content"
        if not answer_text:
            # Nothing in the run committed to a value. A guess invented here
            # would be fabrication, so state the absence in terms the refusal
            # detector does not classify as an abstention.
            answer_text = NO_VERIFIED_VALUE_ANSWER
            run_report["best_effort_answer_source"] = "no_verified_value"
        run_report["final_response_kind"] = "answer"
        run_report["final_output_text"] = answer_text
        run_report["stop_reason"] = stop_reason

    @staticmethod
    def _is_refusal_or_abort_answer(text: str) -> bool:
        """Classify an answer as a refusal that the policy forbids submitting."""
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return True
        refusal_prefixes = (
            "i cannot determine",
            "i can't determine",
            "i could not determine",
            "i couldn't determine",
            "cannot determine",
            "cannot be determined",
            "unable to determine",
            "could not determine",
            "i am unable to determine",
            "i'm unable to determine",
            "not enough evidence",
            "insufficient evidence",
            "there is insufficient evidence",
            "there is not enough evidence",
            "i cannot answer",
            "i can't answer",
            "i could not answer",
            "i couldn't answer",
            "cannot answer",
            "unable to answer",
            "i am unable to answer",
            "i'm unable to answer",
            "i do not know",
            "i don't know",
        )
        return any(
            normalized == prefix
            or normalized.startswith(prefix + ".")
            or normalized.startswith(prefix + ",")
            or normalized.startswith(prefix + " ")
            for prefix in refusal_prefixes
        )

    def _best_effort_finalize(
        self,
        msgs: list[dict],
        question: str,
        run_report: dict[str, Any],
    ) -> None:
        """Route answerless termination paths to a best-effort answer output."""
        existing_text = str(run_report.get("final_output_text") or "").strip()
        existing_kind = str(run_report.get("final_response_kind") or "").strip()
        if existing_text and existing_kind in {"answer", "answer_forced"}:
            run_report["final_response_kind"] = "answer"
            return

        cached_answer = run_report.get("best_candidate_answer") or {}
        cached_text = str(cached_answer.get("answer_text") or "").strip()
        if cached_text:
            run_report["final_response_kind"] = "answer"
            run_report["final_output_text"] = cached_text
            run_report["stop_reason"] = (
                run_report.get("stop_reason") or "best_effort_cached_answer"
            )
            return

        # Prefer the raw computed value over wrapper text when a successful
        # computation observation exists.
        computation_fallback = ""
        for obs in reversed(run_report.get("normalized_observations") or []):
            if (
                isinstance(obs, dict)
                and obs.get("source_modality") == "computation"
                and obs.get("status") == "ok"
            ):
                computation_fallback = (
                    str(obs.get("result") or "").strip()
                    or str(obs.get("stdout") or "").strip()
                    or str(obs.get("summary") or "").strip()
                )
                break

        if self._is_no_state_react_mode():
            finalize_msgs = copy.deepcopy(msgs)
            # Defense-in-depth: the raw history may end with an assistant tool_call
            # whose tool response was never appended (e.g. forced-finish nudge);
            # repair ordering/orphans before sending so the API accepts the batch.
            self._validate_message_history(finalize_msgs)
            finalize_msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"Give your best short answer to the question now, based on the "
                        f"tool observations above. Question: {question}\n"
                        "Respond with the answer only. Never refuse and never say you "
                        "cannot determine or lack sufficient evidence."
                    ),
                }
            )
        else:
            # The tail is a finish instruction addressed to the answer phase's
            # tool-calling planner; the finalizer answers in plain text and
            # cannot call finish, so it is stripped here. Both sites read the
            # same constant — a literal copy would drift the moment either is
            # reworded, and the strip would then silently stop matching.
            evidence_summary = self._build_answer_phase_evidence_summary(run_report).replace(
                ANSWER_PHASE_FINISH_INSTRUCTION,
                "",
            ).strip()
            if computation_fallback:
                evidence_summary += f"\n\nPrevious computation result: {computation_fallback}"
            finalize_msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise QA assistant. Based on the available evidence, "
                        "give the best direct answer to the question. "
                        "If a computation result exists in the evidence, use it directly. "
                        "Commit to a specific entity, name, number, or date when the question "
                        "asks for one; a description of the target is not an answer. "
                        "Never refuse to answer. Respond with the answer only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Evidence summary:\n{evidence_summary or 'No structured evidence collected.'}\n\n"
                        "Give the best short answer supported by this evidence."
                    ),
                },
            ]

        response = None
        # A provider that already rejected this run's requests deterministically
        # will reject the finalizer's too. Skip straight to the best-effort
        # answer instead of paying the same rejection again; the answer policy
        # still holds, because _finalize_best_effort_answer never emits Unknown.
        if not self._has_deterministic_provider_failure(run_report):
            try:
                with self._span("finalizer") as _h_fin:
                    try:
                        response = call_openai_model_with_tools(
                            finalize_msgs,
                            endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                            model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                            temperature=0.0,
                            api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                            base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                            caller="finalizer",
                        )
                    except LLMCallFailure as failure:
                        self._record_llm_call_failure(
                            run_report, failure, layer="finalizer"
                        )
                        response = None
                    if _h_fin is not None and isinstance(response, dict):
                        _h_fin.attach_io(output_preview=str(response.get("content") or ""))
            except Exception:
                response = None

        if response is not None:
            answer_text = str(response.get("content") or "").strip()
            if answer_text and not self._is_refusal_or_abort_answer(answer_text):
                run_report["final_response_kind"] = "answer"
                run_report["final_output_text"] = answer_text
                run_report["stop_reason"] = (
                    run_report.get("stop_reason") or "best_effort_llm_finalize"
                )
                return

        self._finalize_best_effort_answer(
            run_report,
            stop_reason=run_report.get("stop_reason") or "best_effort_no_evidence",
            computation_fallback=computation_fallback,
            msgs=msgs,
        )

    def _inject_strategy_switch(
        self,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> bool:
        if self._is_history_only_mode() or self._is_no_state_react_mode():
            return False

        current_step = len(run_report.get("step_records") or [])
        if current_step <= 0:
            return False
        if current_step <= int(run_report.get("last_strategy_switch_step") or 0):
            return False
        # One injection per planner round (item ③). The step-based guard above is
        # not sufficient once a round carries several actions: after injecting at
        # step 3 the stale counter restarts from there, so a fully-stale 4-wide
        # batch re-crosses the 3-step threshold on its own last action and would
        # be advised twice within the round the advice was meant to redirect.
        # Rounds are what the guard budgets in, so it declines the second.
        current_round = int(run_report.get("_iteration_index") or 0)
        if current_round and current_round == int(
            run_report.get("last_strategy_switch_round") or 0
        ):
            return False

        consecutive_stale = self._count_consecutive_stale_steps(run_report)
        if consecutive_stale < CONSECUTIVE_STALE_STEP_THRESHOLD:
            return False

        latest_gap = (run_report.get("latest_critic_verdict") or {}).get("gap_diagnosis") or {}
        gap_desc = str(latest_gap.get("description") or "").strip()
        tool_counts = build_tool_call_counts(run_report.get("tool_sequence"))
        unexplored = [name for name, count in tool_counts.items() if count == 0]
        prior_switches = len(run_report.get("strategy_switch_injections") or [])

        guidance_parts = [
            (
                "Strategy switch required: the last "
                f"{consecutive_stale} tool calls produced no meaningful progress."
            )
        ]
        if gap_desc:
            guidance_parts.append(f"Current gap: {gap_desc}.")
        if prior_switches >= 1:
            guidance_parts.append(
                "A previous strategy switch did not restore progress, so further "
                "acquisition attempts are unlikely to close this gap. Stop "
                "expanding the search: consolidate the evidence already "
                "collected and call `finish` now with the best-supported "
                "answer, committing to a specific value."
            )
        elif unexplored:
            guidance_parts.append(
                "Try a different evidence route, especially: "
                + ", ".join(unexplored[:3])
                + "."
            )
        else:
            guidance_parts.append(
                "Do not repeat the same search pattern. Change modality, narrow the time range, or verify a concrete candidate."
            )
        self._append_system_msg(" ".join(guidance_parts), msgs)

        run_report.setdefault("strategy_switch_injections", []).append(
            {
                "step_index": current_step,
                "round_index": current_round,
                "stale_steps": consecutive_stale,
                # Record which tool dominated the window that triggered this
                # injection, so a trace shows whether the planner actually
                # changed route after being told to.
                "stalled_tool": self._dominant_stale_tool(run_report, consecutive_stale),
                "escalated_to_commit": prior_switches >= 1,
            }
        )
        run_report["last_strategy_switch_round"] = current_round

        # First switch: reset the stale window (via last_strategy_switch_step)
        # and the stop-ready flag, so the fresh route gets its own stale window
        # before stop_insufficient can re-arm. On escalation the message just
        # told the planner to call finish — resetting either one would flip
        # stop_insufficient_ready back off on the next recompute and pull
        # finish out of the tool schema mid-advice. Leaving the step marker
        # alone keeps the stale count (and with it finish availability) armed;
        # the round guard above still bounds the advice to once per round.
        if prior_switches == 0:
            run_report["last_strategy_switch_step"] = current_step
            for state_key in ("evidence_state", "latest_state_digest"):
                state_obj = run_report.get(state_key)
                if isinstance(state_obj, dict):
                    decision_state = state_obj.setdefault("decision_state", {})
                    decision_state["stop_insufficient_ready"] = False

        run_report["_message_snapshot"] = copy.deepcopy(msgs)
        return True

    def _step_history(self, run_report: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the recorded step history using the current schema."""
        return list(run_report.get("step_records") or [])

    def _current_messages_from_state(self, run_report: dict[str, Any]) -> list[dict]:
        return list(run_report.get("_message_snapshot") or [])

    def _build_state_digest_control_message(
        self,
        run_report: dict[str, Any],
        *,
        is_last_iteration: bool = False,
    ) -> str:
        digest = run_report.get("latest_state_digest") or {}
        latest_critic = run_report.get("latest_critic_verdict") or {}
        gap_diagnosis = latest_critic.get("gap_diagnosis") or digest.get("gap_diagnosis")
        evidence_needs = list(
            digest.get("evidence_needs")
            or (run_report.get("evidence_state") or {}).get("evidence_needs")
            or []
        )
        use_evidence_needs = bool(evidence_needs) and not config.DISABLE_EVIDENCE_NEEDS
        finish_available = self._compute_finish_available(
            run_report,
            is_last_iteration=is_last_iteration,
            ablation_mode=self._current_ablation_mode(),
        )
        tool_call_counts = build_tool_call_counts((run_report or {}).get("tool_sequence"))

        weak_cap = (
            int(getattr(config, "WEAK_LLM_DIGEST_LIST_CAP", 5))
            if getattr(config, "WEAK_LLM_MODE", False)
            else None
        )

        def _cap(seq):
            if weak_cap is None:
                return list(seq or [])
            return list(seq or [])[:weak_cap]

        # Item ③: how many planner rounds are left, so a planner deciding how
        # wide to batch can see the budget it is spending. `_iteration_index` is
        # 1-based and stamped at the top of each round, so it is the count of
        # rounds already entered; before the loop it is unset and the whole
        # budget remains. `max_iterations` is read via getattr because this
        # builder is also exercised against partially-constructed agents (the
        # digest-contract tests under scripts/flow_test), and a digest is not
        # worth raising AttributeError over.
        remaining_rounds = max(
            0,
            int(getattr(self, "max_iterations", 0) or 0)
            - int(run_report.get("_iteration_index") or 0),
        )

        if use_evidence_needs:
            planner_digest: dict[str, Any] = {
                "evidence_needs": _cap(evidence_needs),
                "actionable_need_indices": _cap(digest.get("actionable_need_indices")),
                "blocked_need_indices": _cap(digest.get("blocked_need_indices")),
                "tool_call_counts": tool_call_counts,
                "gap_diagnosis": gap_diagnosis if gap_diagnosis else None,
                "entity_cards": _cap(digest.get("entity_cards")) or None,
                "fact_bridge_records": _cap(digest.get("fact_bridge_records")) or None,
                "finish_available": finish_available,
                "remaining_rounds": remaining_rounds,
            }
        else:
            planner_digest = {
                "unresolved_questions": _cap(digest.get("unresolved_questions")),
                "tool_call_counts": tool_call_counts,
                "gap_diagnosis": gap_diagnosis if gap_diagnosis else None,
                "entity_cards": _cap(digest.get("entity_cards")) or None,
                "fact_bridge_records": _cap(digest.get("fact_bridge_records")) or None,
                "finish_available": finish_available,
                "remaining_rounds": remaining_rounds,
            }

        digest_json = dumps_llm_payload(planner_digest)
        base = (
            f"Evidence State Digest for OmniVQA {self._agent_label()}.\n"
            f"{digest_json}\n"
            "Use this structured digest as the authoritative state summary for the next action. "
            "Prefer resolving the listed evidence needs or remaining gaps instead of reacting to raw history alone."
        )
        last_finish_block_reason = str(run_report.get("last_finish_block_reason") or "").strip()
        if last_finish_block_reason:
            base = (
                base
                + "\n[Last Finish Block Reason]\n"
                + last_finish_block_reason
            )
        fact_records = list(digest.get("fact_bridge_records") or [])
        entity_cards = list(digest.get("entity_cards") or [])
        hints: list[str] = []
        for rec in fact_records:
            ev = str(rec.get("extracted_value") or "").strip()
            status = str(rec.get("status") or "")
            eid = str(rec.get("entity_id") or "")
            entity_name = next(
                (c.get("name") or "" for c in entity_cards
                 if str(c.get("entity_id") or "") == eid),
                eid,
            )
            if status == "ok" and ev:
                hints.append(
                    f'"{ev}" is a verified external fact for entity "{entity_name}". '
                    "Feed it directly to code_executor_tool. Do NOT re-search."
                )
            elif status == "no_match":
                hints.append(
                    f'Entity lookup for "{entity_name}" returned no match. '
                    "Reformulate the retrieval target once from a different angle, then accept failure if it still returns no result."
                )
        fb_count = len(fact_records)
        if fb_count > 0 or entity_cards:
            hint_block = (
                f"\n[Fact Bridge Hint]\nResolved fact bridge records: {fb_count} record(s)."
            )
            if hints:
                hint_block += "\n" + "\n".join(f"  → {h}" for h in hints)
            base = base + hint_block
        bridge = dict(digest.get("bridge_status") or {})
        if bridge.get("memory_hit"):
            base = base + (
                "\n[Working Memory] Last tool call was served from cache (memory_hit=true). "
                "Do NOT repeat the same tool call. Proceed to the next logical action."
            )
        return base

    def _build_answer_phase_evidence_summary(self, run_report: dict[str, Any]) -> str:
        """Build a concise evidence summary for the 1-step answer phase."""
        digest = run_report.get("latest_state_digest") or {}
        evidence_state = run_report.get("evidence_state") or {}
        parts: list[str] = []

        needs = list(digest.get("evidence_needs") or evidence_state.get("evidence_needs") or [])[:5]
        if needs:
            need_lines: list[str] = []
            for need in needs:
                if not isinstance(need, dict):
                    continue
                status = str(need.get("status") or "unfilled").strip() or "unfilled"
                need_text = str(need.get("need") or "").strip()
                filled_by = str(need.get("filled_by") or "").strip()
                if not need_text:
                    continue
                line = f"- [{status}] {need_text}"
                if filled_by:
                    line += f" (filled by {filled_by})"
                need_lines.append(line)
            if need_lines:
                parts.append("Evidence needs:\n" + "\n".join(need_lines))

        entity_cards = list(digest.get("entity_cards") or [])[:4]
        if entity_cards:
            card_lines: list[str] = []
            for card in entity_cards:
                if not isinstance(card, dict):
                    continue
                name = str(card.get("name") or card.get("canonical_name") or "?").strip() or "?"
                resolved_fact = str(card.get("resolved_fact") or "").strip()
                card_lines.append(
                    f"- {name}: {resolved_fact}" if resolved_fact else f"- {name}: (unresolved)"
                )
            if card_lines:
                parts.append("Entity cards:\n" + "\n".join(card_lines))

        fact_bridge_records = list(digest.get("fact_bridge_records") or [])[:4]
        if fact_bridge_records:
            bridge_lines: list[str] = []
            for record in fact_bridge_records:
                if not isinstance(record, dict):
                    continue
                entity_id = str(record.get("entity_id") or "?").strip() or "?"
                extracted_value = str(record.get("extracted_value") or "").strip()
                bridge_lines.append(
                    f"- entity:{entity_id}: {extracted_value}"
                    if extracted_value
                    else f"- entity:{entity_id}: (no value)"
                )
            if bridge_lines:
                parts.append("Fact bridge records:\n" + "\n".join(bridge_lines))

        gap = digest.get("gap_diagnosis") or {}
        gap_desc = str(gap.get("description") or "").strip()
        if gap_desc:
            parts.append(f"Gap diagnosis: {gap_desc}")

        summary = "\n\n".join(parts) if parts else "No structured evidence collected."
        return summary + ANSWER_PHASE_FINISH_INSTRUCTION

    def _run_answer_phase(
        self,
        msgs: list[dict],
        tool_history: dict[str, int],
        question: str,
        run_report: dict[str, Any],
    ) -> list[dict]:
        """Run a final single-step answer synthesis phase after loop exhaustion."""
        run_report["answer_phase"] = True

        evidence_summary = self._build_answer_phase_evidence_summary(run_report)
        msgs.append({"role": "user", "content": evidence_summary})
        run_report["_message_snapshot"] = copy.deepcopy(msgs)

        answer_phase_schemas = [
            schema
            for schema in self.function_schemas
            if schema.get("function", {}).get("name") == "finish"
        ]
        emitted_messages: list[dict] = []

        try:
            with self._span("planner.decide", phase="answer") as _h_answer_plan:
                try:
                    response = call_openai_model_with_tools(
                        msgs,
                        endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                        model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                        tools=answer_phase_schemas,
                        temperature=0.0,
                        api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                        base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                        caller="orchestrator",
                    )
                except LLMCallFailure as failure:
                    self._record_llm_call_failure(
                        run_report, failure, layer="answer_phase_planner"
                    )
                    response = None
                self._attach_planner_span_fields(_h_answer_plan, response)
        except Exception:
            response = None

        if response is None:
            run_report["stop_reason"] = run_report.get("stop_reason") or "answer_phase_model_unavailable"
            return emitted_messages

        response.setdefault("role", "assistant")
        msgs.append(response)
        emitted_messages.append(response)
        run_report["_message_snapshot"] = copy.deepcopy(msgs)

        pending_tool_calls = list(response.get("tool_calls") or [])
        finish_calls = [
            tool_call
            for tool_call in pending_tool_calls
            if tool_call.get("function", {}).get("name") == "finish"
        ]
        if finish_calls:
            for tool_call in finish_calls:
                emitted_messages.append(
                    {
                        "role": "tool_call",
                        "name": tool_call.get("function", {}).get("name", "unknown"),
                        "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                    }
                )
                previous_len = len(msgs)
                try:
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    full_args_str = str(
                        tool_call.get("function", {}).get("arguments") or ""
                    )
                    args_preview = full_args_str[:240]
                    stop_exc = None
                    with self._span(f"tool:{tool_name}") as _h_answer_tool:
                        if _h_answer_tool is not None:
                            _h_answer_tool.extra["arguments_preview"] = args_preview
                        try:
                            self._exec_tool(tool_call, msgs, tool_history, question, run_report)
                        except StopException as exc:
                            stop_exc = exc
                        self._attach_tool_span_fields(
                            _h_answer_tool,
                            tool_name,
                            msgs[previous_len:],
                            run_report,
                            full_args=full_args_str,
                        )
                    if stop_exc is not None:
                        raise stop_exc
                except StopException:
                    emitted_messages.extend(msgs[previous_len:])
                    if run_report.get("final_response_kind") == "answer":
                        return emitted_messages
                    run_report["stop_reason"] = (
                        run_report.get("stop_reason") or "answer_phase_finish_parse_error"
                    )
                    return emitted_messages
                emitted_messages.extend(msgs[previous_len:])
                if run_report.get("final_response_kind") == "answer":
                    return emitted_messages

            run_report["stop_reason"] = run_report.get("stop_reason") or "answer_phase_finish_incomplete"
            return emitted_messages

        text_answer = str(response.get("content") or "").strip()
        if (
            text_answer
            and len(text_answer) > 3
            and not self._is_refusal_or_abort_answer(text_answer)
        ):
            run_report["final_response_kind"] = "answer"
            run_report["final_output_text"] = text_answer
            run_report["stop_reason"] = "answer_phase_text_response"
            run_report["last_finish_attempt"] = text_answer
            checkpoint_index = len(run_report.get("answer_checkpoints") or []) + 1
            answer_checkpoint = {
                "verdict": "PASS",
                "revision_hint": "",
                "stage": "before_finish",
                "checkpoint_index": checkpoint_index,
                "proposed_answer_preview": text_answer[:TRUNC_PROPOSED_ANSWER_PREVIEW],
                "evidence_signature": self._build_evidence_signature(msgs),
                "source": "answer_phase_text_capture",
            }
            run_report.setdefault("answer_checkpoints", []).append(answer_checkpoint)
            run_report["latest_answer_verdict"] = answer_checkpoint
            current_step = len(run_report.get("tool_sequence") or [])
            run_report["best_candidate_answer"] = {
                "answer_text": text_answer,
                "sufficiency_status": str(
                    (run_report.get("evidence_state") or {}).get("sufficiency_status") or ""
                ),
                "cached_at_step": current_step,
                "source": "answer_phase_text_capture",
            }
            return emitted_messages

        run_report["stop_reason"] = run_report.get("stop_reason") or "answer_phase_no_finish"
        return emitted_messages

    def _prune_stale_digest_messages(self, msgs: list[dict]) -> int:
        """Drop superseded state-view injections from the planner's view.

        The digest is a full snapshot of the current evidence state, rebuilt
        and re-injected after every tool step, so each copy strictly supersedes
        every earlier one — the planner is even instructed to treat it as "the
        authoritative state summary". Keeping the stale copies cost one digest
        per round compounding over the run (6 stale copies / 10.6K chars in a
        measured 8-round case) and buys nothing: nothing downstream reads them
        (the critic and evidence signature read only tool observations, the
        state itself lives in run_report).

        Destructive to ``msgs`` by design, like ``_compact_planner_history``:
        this is the planner's view and the loop keeps every original elsewhere.
        Called before appending the fresh digest, so the newest copy survives.

        Returns the number of messages removed or stripped.
        """
        removed = 0
        prefixes = (STATE_DIGEST_MESSAGE_PREFIX, NAIVE_MEMORY_MESSAGE_PREFIX)
        hint_markers = tuple(f"<system-hint>\n{prefix}" for prefix in prefixes)
        folded_markers = tuple(f"\n\n{marker}" for marker in hint_markers)
        for index in range(len(msgs) - 1, -1, -1):
            message = msgs[index]
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            if role == "system" and content.startswith(prefixes):
                del msgs[index]
                removed += 1
            elif role == "user" and content.startswith(hint_markers):
                del msgs[index]
                removed += 1
            elif role == "tool" and any(marker in content for marker in folded_markers):
                # fold digest role: the digest rides on the tool observation;
                # strip the hint, keep the observation.
                marker = next(marker for marker in folded_markers if marker in content)
                message["content"] = content.split(marker, 1)[0]
                removed += 1
        return removed

    @staticmethod
    def _naive_memory_encoder():
        """Return the tokenizer used for matched planner-context accounting.

        A provider's model alias may be absent from tiktoken's registry.
        We therefore pin the current OpenAI o200k vocabulary family explicitly
        and record it with every injected memory. Failing loudly is preferable
        to silently changing the matching unit during an experiment.
        """
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - formal env preflight
            raise RuntimeError(
                "naive_memory requires tiktoken for matched context accounting"
            ) from exc
        return tiktoken.get_encoding("o200k_base")

    @classmethod
    def _count_naive_memory_tokens(cls, text: str) -> int:
        return len(cls._naive_memory_encoder().encode(str(text or "")))

    @classmethod
    def _trim_naive_memory_tokens(cls, text: str, limit: int) -> str:
        encoder = cls._naive_memory_encoder()
        token_ids = encoder.encode(str(text or ""))
        return encoder.decode(token_ids[: max(1, int(limit))]).strip()

    @staticmethod
    def _sanitize_naive_memory_text(text: str) -> str:
        """Turn a summarizer response into one schema-free prose paragraph."""
        fence = chr(96) * 3
        cleaned = str(text or "").replace(fence + "json", " ").replace(fence, " ")
        cleaned = re.sub(
            r"Evidence\s+State\s+Digest(?:\s+for\s+OmniVQA)?",
            "current memory",
            cleaned,
            flags=re.IGNORECASE,
        )
        for field_name, prose in NAIVE_MEMORY_SCHEMA_TERMS.items():
            cleaned = re.sub(
                rf"\b{re.escape(field_name)}\b",
                prose,
                cleaned,
                flags=re.IGNORECASE,
            )
        # Ordinary prose can name tools, but it must not preserve machine-shaped
        # snake_case labels or JSON containers copied from the typed source.
        cleaned = cleaned.replace("_", " ")
        cleaned = cleaned.translate(
            str.maketrans({"{": " ", "}": " ", "[": " ", "]": " "})
        )
        return " ".join(cleaned.split()).strip()

    def _fallback_naive_memory_text(
        self,
        run_report: dict[str, Any],
        *,
        is_last_iteration: bool,
    ) -> str:
        """Build a schema-free fallback and make its use auditable."""
        digest = run_report.get("latest_state_digest") or {}
        state = run_report.get("evidence_state") or {}
        latest_critic = run_report.get("latest_critic_verdict") or {}
        needs = list(digest.get("evidence_needs") or state.get("evidence_needs") or [])
        open_needs = [
            str(need.get("need") or "").strip()
            for need in needs
            if isinstance(need, dict)
            and str(need.get("status") or "unfilled") in {"unfilled", "partial"}
            and str(need.get("need") or "").strip()
        ]
        gap = latest_critic.get("gap_diagnosis") or digest.get("gap_diagnosis") or {}
        gap_text = (
            str(gap.get("description") or "").strip()
            if isinstance(gap, dict)
            else str(gap or "").strip()
        )
        parts = [
            "Keep a running account of what has been established and what still needs checking."
        ]
        if open_needs:
            parts.append("The next search should resolve " + "; ".join(open_needs) + ".")
        if gap_text:
            parts.append("The main remaining gap is " + gap_text.rstrip(".") + ".")
        if self._compute_finish_available(
            run_report,
            is_last_iteration=is_last_iteration,
            ablation_mode=self._current_ablation_mode(),
        ):
            parts.append("The current evidence permits submitting the best-supported answer.")
        return " ".join(parts)

    def _build_naive_memory_control_message(
        self,
        run_report: dict[str, Any],
        *,
        is_last_iteration: bool,
    ) -> str:
        """Generate a token-ceiling-matched free-text view of the typed state."""
        typed_content = self._build_state_digest_control_message(
            run_report,
            is_last_iteration=is_last_iteration,
        )
        source_tokens = max(1, self._count_naive_memory_tokens(typed_content))
        prompt = (
            "Rewrite the controller snapshot below as one concise natural-language rolling "
            "memory paragraph for the next planner action. Preserve concrete facts, unresolved "
            "goals, remaining gaps, and whether the planner may answer. Do not add information. "
            "Do not use a heading, list, JSON, XML, code, field name, schema label, or snake_case "
            "token. Do not mention an evidence state, digest, controller, or this instruction.\n\n"
            + typed_content
        )
        response = None
        fallback_reason = ""
        try:
            response = call_openai_model_with_tools(
                [{"role": "user", "content": prompt}],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                # The injected planner text is trimmed to the exact typed-source
                # ceiling below. The derived 2x generation ceiling leaves room
                # for provider-side reasoning without adding a tuned constant.
                max_tokens=source_tokens * 2,
                tool_choice="none",
                caller="naive_memory",
            )
        except LLMCallFailure as failure:
            fallback_reason = f"provider_failure:{failure.category}"
            recorder = getattr(self, "_record_llm_call_failure", None)
            if callable(recorder):
                recorder(run_report, failure, layer="naive_memory")
        except Exception as exc:  # one failed summary must not abort the case
            fallback_reason = f"unexpected:{type(exc).__name__}"

        raw_text = str((response or {}).get("content") or "").strip()
        prose = self._sanitize_naive_memory_text(raw_text)
        if not prose:
            fallback_reason = fallback_reason or "empty_response"
            prose = self._sanitize_naive_memory_text(
                self._fallback_naive_memory_text(
                    run_report,
                    is_last_iteration=is_last_iteration,
                )
            )

        rendered = f"{NAIVE_MEMORY_MESSAGE_PREFIX} {self._agent_label()}. {prose}"
        rendered = self._trim_naive_memory_tokens(rendered, source_tokens)
        injected_tokens = self._count_naive_memory_tokens(rendered)
        run_report.setdefault("naive_memory_events", []).append(
            {
                "step_index": len(run_report.get("step_records") or []),
                "round_index": int(run_report.get("_iteration_index") or 0),
                "source_typed_tokens": source_tokens,
                "injected_tokens": injected_tokens,
                "tokenizer": "tiktoken:o200k_base",
                "fallback_reason": fallback_reason or None,
            }
        )
        if fallback_reason:
            run_report["naive_memory_fallback_count"] = int(
                run_report.get("naive_memory_fallback_count") or 0
            ) + 1
        return rendered

    def _collect_primed_transcripts(self) -> list[dict[str, Any]]:
        """Load the full dialogue track for every asset small enough to prime.

        Returns one entry per distinct eligible track. Assets sharing an
        identical dialogue track (a video and its extracted audio register as
        two assets over one soundtrack) collapse into a single entry listing
        both asset ids. TRANSCRIPT_PRIMING_MAX_CHARS is a per-case total:
        distinct tracks are admitted smallest-first until the budget is spent,
        so one long track cannot crowd out several short complete ones. Assets
        left out keep top_k retrieval as their only access path.
        """
        from agents.asset_runtime import load_dialogue_segments, resolve_video_asset_runtime
        from agents.tools.common import seconds_to_hhmmss

        candidates: list[dict[str, Any]] = []
        by_track: dict[tuple[str, ...], dict[str, Any]] = {}
        for asset_id, database in (self.asset_dbs or {}).items():
            if database is None:
                continue
            try:
                runtime = resolve_video_asset_runtime(database)
                segments, source, _asset_key = load_dialogue_segments(
                    runtime,
                    allow_asr_fallback=True,
                )
            except Exception:
                continue
            if not segments:
                continue
            if len(segments) > config.TRANSCRIPT_PRIMING_MAX_SEGMENTS:
                continue

            lines: list[str] = []
            total_chars = 0
            for segment in segments:
                text = str(
                    segment.get("clean_text") or segment.get("spoken_text") or ""
                ).strip()
                if not text:
                    continue
                try:
                    start = seconds_to_hhmmss(float(segment["start_secs"]))
                    end = seconds_to_hhmmss(float(segment["end_secs"]))
                except (KeyError, TypeError, ValueError):
                    continue
                lines.append(f"[{start} - {end}] {text}")
                total_chars += len(lines[-1])
            if not lines or total_chars > config.TRANSCRIPT_PRIMING_MAX_CHARS:
                continue

            track_key = tuple(lines)
            existing = by_track.get(track_key)
            if existing is not None:
                existing["asset_ids"].append(asset_id)
                continue
            entry = {
                "asset_ids": [asset_id],
                "source": source,
                "segment_count": len(lines),
                "chars": total_chars,
                "lines": lines,
            }
            by_track[track_key] = entry
            candidates.append(entry)

        candidates.sort(key=lambda entry: entry["chars"])
        primed: list[dict[str, Any]] = []
        budget = config.TRANSCRIPT_PRIMING_MAX_CHARS
        for entry in candidates:
            if entry["chars"] > budget:
                break
            budget -= entry["chars"]
            primed.append(entry)
        return primed

    def _append_transcript_priming_message(
        self,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> None:
        """Seed complete short dialogue tracks into the opening planner context.

        Recorded in the run report so a run can be told apart from one where the
        planner only ever saw retrieved excerpts.
        """
        if not config.TRANSCRIPT_PRIMING_ENABLED:
            return
        if self._is_no_state_react_mode() or self._is_history_only_mode():
            return

        primed = self._collect_primed_transcripts()
        if not primed:
            return

        blocks: list[str] = []
        for entry in primed:
            body = "\n".join(entry["lines"])
            label = ", ".join(f"`{asset_id}`" for asset_id in entry["asset_ids"])
            noun = "Assets" if len(entry["asset_ids"]) > 1 else "Asset"
            blocks.append(
                f"{noun} {label} "
                f"(complete {entry['source']} track, {entry['segment_count']} segments):\n"
                f"{body}"
            )
        self._append_system_msg(
            "Complete dialogue transcripts for the assets below are already "
            "available in full, so no subtitle retrieval is needed to read them. "
            "Timestamps are usable directly as visual-confirmation ranges. "
            "Absence of a statement in a complete track is itself evidence.\n\n"
            + "\n\n".join(blocks),
            msgs,
        )
        run_report["transcript_priming"] = [
            {
                "asset_ids": list(entry["asset_ids"]),
                "source": entry["source"],
                "segment_count": entry["segment_count"],
                "chars": entry["chars"],
            }
            for entry in primed
        ]

    def _append_state_digest_message(
        self,
        msgs: list[dict],
        run_report: dict[str, Any],
        *,
        is_last_iteration: bool | None = None,
    ) -> None:
        if self._is_history_only_mode() or self._is_no_state_react_mode():
            return
        self._prune_stale_digest_messages(msgs)
        if is_last_iteration is None:
            next_iteration_index = len(run_report.get("step_records") or [])
            is_last_iteration = next_iteration_index >= self.max_iterations - 1
        content = (
            self._build_naive_memory_control_message(
                run_report,
                is_last_iteration=is_last_iteration,
            )
            if self._is_naive_memory_mode()
            else self._build_state_digest_control_message(
                run_report,
                is_last_iteration=is_last_iteration,
            )
        )
        _digest_role = os.environ.get("OMNIVQA_DIGEST_ROLE", "system")
        if _digest_role == "fold":
            hint = f"<system-hint>\n{content}\n</system-hint>"
            if msgs and msgs[-1].get("role") == "tool":
                prev = str(msgs[-1].get("content") or "")
                msgs[-1]["content"] = f"{prev}\n\n{hint}" if prev else hint
                return
            # No preceding tool obs (e.g. the pre-loop injection): mirror the
            # training converter, which only folds a user turn that immediately
            # follows a tool and otherwise leaves it standalone.
            msgs.append({"role": "user", "content": hint})
            return
        if _digest_role == "user":
            msgs.append(
                {
                    "role": "user",
                    "content": f"<system-hint>\n{content}\n</system-hint>",
                }
            )
            return
        self._append_system_msg(
            content,
            msgs,
        )

    def _record_protocol_step(
        self,
        run_report: dict[str, Any],
        *,
        selected_tool: str,
        tool_arguments: dict[str, Any],
        stage_before: ProtocolStage,
        stage_after: ProtocolStage,
        reason_source: str,
        reason_text: str,
        blocked: bool,
        raw_observation: str | dict[str, Any],
        normalized_observation: NormalizedToolObservation | None,
        evidence_delta: dict[str, Any] | None,
        critic_verdict: dict[str, Any] | None,
        answer_verdict: dict[str, Any] | None,
        step_elapsed_ms: int = 0,
    ) -> None:
        digest = copy.deepcopy(run_report.get("latest_state_digest") or {})
        state_after = copy.deepcopy(run_report.get("evidence_state") or {})
        step_cost = (
            copy.deepcopy(normalized_observation.get("estimated_cost") or {})
            if normalized_observation
            else {}
        )
        action: DecisionRecord = {
            "step_index": len(run_report.get("step_records") or []) + 1,
            "selected_tool": selected_tool,
            "tool_role": self._tool_role(selected_tool),
            "stage_before": stage_before,
            "stage_after": stage_after,
            "reason_source": reason_source,
            "reason_text": reason_text,
            "blocked": blocked,
            "critic_decision": (
                str((answer_verdict or {}).get("verdict") or "")
                or str((critic_verdict or {}).get("verdict") or "")
            ),
            "evidence_state_digest": digest,
        }
        step_record: StepRecord = {
            "step_index": action["step_index"],
            "action": action,
            "tool_arguments": copy.deepcopy(tool_arguments),
            "raw_observation": copy.deepcopy(raw_observation),
            "normalized_observation": copy.deepcopy(normalized_observation or {}),
            "evidence_delta": copy.deepcopy(evidence_delta or {}),
            "critic_verdict": copy.deepcopy(critic_verdict) if critic_verdict else None,
            "answer_verdict": copy.deepcopy(answer_verdict) if answer_verdict else None,
            "state_after": state_after,
            "step_cost": step_cost,
            "step_elapsed_ms": int(step_elapsed_ms or 0),
            "state_changed": bool((evidence_delta or {}).get("state_changed")),
            # Which planner round produced this step. Under item ③ a round can
            # carry up to BATCH_MAX_WIDTH actions, so this is no longer redundant
            # with step_index: it is what lets a consumer that budgets in rounds
            # rather than steps count one batch once.
            "round_index": int(run_report.get("_iteration_index") or 0),
        }
        run_report.setdefault("step_records", []).append(step_record)
        if normalized_observation is not None:
            run_report.setdefault("normalized_observations", []).append(
                copy.deepcopy(normalized_observation)
            )

    def _tool_role(self, name: str) -> ToolRole:
        if name in VIDEO_GROUNDING_TOOLS:
            return "grounding_video"
        if name in EXTERNAL_LOOKUP_TOOLS:
            return "grounding_external"
        if name in VISUAL_CONFIRM_TOOLS:
            return "visual_verify"
        if name in COMPUTATION_TOOLS:
            return "computation"
        if name == "finish":
            return "finish"
        if name in {"frame_inspect_tool", "clip_search_tool", "global_browse_tool"}:
            return "unknown"
        return "unknown"

    def _infer_protocol_stage(self, run_report: dict[str, Any]) -> ProtocolStage:
        if run_report.get("final_response_kind") in {"answer", "answer_forced"}:
            return "stopped"

        latest_answer_verdict = run_report.get("latest_answer_verdict") or {}
        if latest_answer_verdict.get("verdict") == "BLOCK":
            return "answer_revision"

        latest_critic_verdict = run_report.get("latest_critic_verdict") or {}
        if latest_critic_verdict.get("verdict") == "SUFFICIENT":
            return "ready_to_finish"

        tool_sequence = run_report.get("tool_sequence") or []
        if any(tool_name in VISUAL_CONFIRM_TOOLS for tool_name in tool_sequence):
            return "after_visual_verification"
        if any(tool_name in GROUNDING_TOOLS for tool_name in tool_sequence):
            return "after_grounding"
        return "start"

    def _infer_action_reason(
        self,
        tool_name: str,
        run_report: dict[str, Any],
    ) -> tuple[str, str]:
        latest_answer_verdict = run_report.get("latest_answer_verdict") or {}
        latest_critic_verdict = run_report.get("latest_critic_verdict") or {}
        current_stage = self._infer_protocol_stage(run_report)

        if tool_name == "finish":
            if latest_answer_verdict.get("verdict") == "PASS":
                return (
                    "answer_critic",
                    "The latest answer checkpoint approved the current final answer.",
                )
            if latest_critic_verdict.get("verdict") == "SUFFICIENT":
                return (
                    "critic",
                    "The latest critic checkpoint says the current evidence is sufficient.",
                )

        if latest_critic_verdict.get("verdict") == "CONFLICTING" and tool_name in GROUNDING_TOOLS:
            return (
                "critic",
                "The latest critic checkpoint reported conflicting evidence, so the agent is returning to evidence collection.",
            )

        if (
            latest_critic_verdict.get("verdict") == "INSUFFICIENT"
            and tool_name in (GROUNDING_TOOLS | VISUAL_CONFIRM_TOOLS)
        ):
            return (
                "critic",
                "The latest critic checkpoint reported an unresolved evidence gap, so the agent is gathering more evidence.",
            )

        if current_stage == "start":
            return (
                "policy",
                "The run is starting, so the orchestrator is selecting an initial tool under the prompt policy.",
            )

        return (
            "orchestrator",
            "The orchestrator selected the next tool without a stronger explicit control message.",
        )

    def _repair_tool_arguments(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        question: str,
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        func = self.name_to_function_map.get(tool_name)
        if func is None:
            return args

        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return args

        repaired = dict(args)
        evidence_state = run_report.get("evidence_state") or {}
        latest_digest = run_report.get("latest_state_digest") or {}
        if not hasattr(self, "_simplified_asset_ids") or not hasattr(self, "_original_asset_ids"):
            self._refresh_asset_id_mappings()

        asset_id = repaired.pop("asset_id", None)

        # Resolve target asset descriptor
        target_asset = None
        resolved_asset_id = asset_id
        if asset_id and asset_id in self._original_asset_ids:
            resolved_asset_id = self._original_asset_ids[asset_id]
            target_asset = self.asset_registry.get(resolved_asset_id)
            run_report["_current_asset_id"] = asset_id
        elif asset_id and asset_id in self.asset_registry:
            target_asset = self.asset_registry[asset_id]
            run_report["_current_asset_id"] = self._simplified_asset_ids.get(asset_id, asset_id)
        elif asset_id and self._multi_asset:
            from agents.tools.common import build_failure_observation
            known_ids = sorted(self._original_asset_ids.keys())
            result = build_failure_observation(
                "unknown",
                f"Unknown asset_id '{asset_id}'. Available assets: {', '.join(known_ids)}.",
                f"Use one of the available asset_ids: {', '.join(known_ids)}.",
                status="invalid_asset",
            )
            repaired["__precomputed_result__"] = result
            run_report["_current_asset_id"] = None
            return repaired
        else:
            run_report["_current_asset_id"] = None

        # Image asset + frame_confirm_tool: intercept and swap to image_inspect_asset
        if target_asset and target_asset.get("type") == "image" and tool_name == "frame_confirm_tool":
            from agents.tools.frame_confirm_tool import image_inspect_asset
            result = image_inspect_asset(target_asset["path"], repaired.get("query") or question)
            repaired["__precomputed_result__"] = result
            return repaired

        # Unsupported tool call on wrong asset type: return structured failure
        if target_asset:
            _asset_type = target_asset.get("type")
            _unsupported = False
            if _asset_type == "image" and tool_name in ("clip_grounding_tool", "subtitle_grounding_tool", "audio_scout_tool"):
                _unsupported = True
            elif _asset_type == "audio" and tool_name in ("frame_confirm_tool", "clip_grounding_tool"):
                _unsupported = True
            if _unsupported:
                from agents.tools.common import build_failure_observation
                result = build_failure_observation(
                    "unknown",
                    f"Tool '{tool_name}' is not available for {_asset_type} asset '{asset_id}'. Check the asset manifest for available tools.",
                    f"Use a tool that supports {_asset_type} assets.",
                    status="unavailable",
                )
                repaired["__precomputed_result__"] = result
                return repaired

        # Normal DB routing
        if "database" in parameters:
            db_key = resolved_asset_id
            if db_key and db_key in self.asset_dbs:
                repaired["database"] = self.asset_dbs[db_key]
            elif len(self.asset_dbs) == 1:
                repaired["database"] = next(iter(self.asset_dbs.values()))
            elif self.asset_dbs:
                # Multiple DBs, no asset_id specified -- use primary
                repaired["database"] = self.primary_db

        if "query" in parameters and not str(repaired.get("query") or "").strip():
            fallback_query = ""
            if (
                tool_name == "web_search_tool"
                and not self._is_history_only_mode()
                and not self._is_no_state_react_mode()
            ):
                pending_entity_lookups = list(evidence_state.get("pending_entity_lookups") or [])
                if pending_entity_lookups:
                    fallback_query = str(pending_entity_lookups[0] or "").strip()
            if not fallback_query:
                fallback_query = question.strip()
            repaired["query"] = fallback_query

        if (
            "time_ranges_hhmmss" in parameters
            and not repaired.get("time_ranges_hhmmss")
            and not self._is_history_only_mode()
            and not self._is_no_state_react_mode()
        ):
            candidate_preview = list(latest_digest.get("candidate_range_preview") or [])
            synthesized_ranges = [
                [str(item.get("start_time") or ""), str(item.get("end_time") or "")]
                for item in candidate_preview
                if item.get("start_time") and item.get("end_time")
            ]
            if synthesized_ranges:
                repaired["time_ranges_hhmmss"] = synthesized_ranges[:3]

        # Migrate time_ranges -> time_ranges_hhmmss BEFORE unknown-arg deletion
        if tool_name == "frame_confirm_tool" and "time_ranges" in repaired and "time_ranges_hhmmss" not in repaired:
            repaired["time_ranges_hhmmss"] = repaired.pop("time_ranges")

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if not has_var_keyword:
            unknown_keys = [key for key in repaired if key not in parameters]
            for key in unknown_keys:
                del repaired[key]

        return repaired

    def _missing_required_tool_args(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        """Return required (no-default) tool parameters absent from ``args``.

        Harness-injected params (``database`` / ``__precomputed_result__``) are excluded —
        they are filled by asset routing, not by the planner. Used by ``_exec_tool`` to turn a
        would-be ``TypeError`` (missing required positional argument) into a recoverable error
        observation instead of crashing the whole case. Mode-agnostic; only reports a missing
        arg when one is genuinely absent (the normal full-state path passes every required arg).
        """
        func = self.name_to_function_map.get(tool_name)
        if func is None:
            return []
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return []
        missing: list[str] = []
        for pname, p in params.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.default is not inspect.Parameter.empty:
                continue
            if pname in ("database", "__precomputed_result__"):
                continue
            if pname not in args:
                missing.append(pname)
        return missing

    def _state_requires_external_lookup(
        self,
        question: str,
        run_report: dict[str, Any],
        state: EvidenceState | None = None,
    ) -> bool:
        constraints = self._build_question_constraint_profile(question, run_report)
        resolved_state = state or run_report.get("evidence_state") or {}
        bridge_status = resolved_state.get("bridge_status") or {}
        latest_digest = run_report.get("latest_state_digest") or {}
        digest_unresolved = set(latest_digest.get("unresolved_questions") or [])
        external_gap_open = "recover_external_public_fact" in digest_unresolved

        entity_cards = list(resolved_state.get("entity_cards") or [])
        pending_entities = [
            card for card in entity_cards
            if card.get("lookup_status") in {"pending", "in_flight"}
        ]
        if pending_entities and bridge_status.get("has_video_bridge"):
            return True
        if bridge_status.get("has_external_exact_fact"):
            return False

        digest_sufficiency = str(latest_digest.get("sufficiency_status") or "")
        if latest_digest.get("stop_ready") and digest_sufficiency in {
            "ready_to_finish", "answered",
        } and not external_gap_open:
            return False

        for verdict in (
            run_report.get("latest_answer_verdict"),
            run_report.get("latest_critic_verdict"),
        ):
            if not isinstance(verdict, dict):
                continue
            if verdict.get("verdict") in {"PASS", "SUFFICIENT"} and not external_gap_open:
                return False

        return _evidence_needs_prefer_source(constraints, "web")

    def _state_requires_deterministic_compute(
        self,
        run_report: dict[str, Any],
        state: EvidenceState | None = None,
    ) -> bool:
        bridge_status = (state or run_report.get("evidence_state") or {}).get(
            "bridge_status", {}
        ) or {}
        return bool(bridge_status.get("computation_ready"))

    def _latest_web_search_feedback(
        self,
        run_report: dict[str, Any],
    ) -> dict[str, Any] | None:
        feedback_list = run_report.get("web_search_feedback") or []
        if not feedback_list:
            return None
        latest = feedback_list[-1]
        return dict(latest) if isinstance(latest, dict) else None

    def _extract_web_search_feedback(self, tool_content: str) -> dict[str, Any] | None:
        payload = safe_load_json_object(tool_content)
        if not isinstance(payload, dict):
            return None
        if payload.get("source_modality") != "web":
            return None
        feedback = payload.get("search_feedback")
        if not isinstance(feedback, dict):
            return None
        normalized = dict(feedback)
        normalized["status"] = str(payload.get("status") or "")
        normalized["summary"] = str(payload.get("summary") or "")
        return normalized

    def _record_web_search_feedback(
        self,
        tool_content: str,
        run_report: dict[str, Any],
    ) -> None:
        feedback = self._extract_web_search_feedback(tool_content)
        if feedback is None:
            return
        run_report.setdefault("web_search_feedback", []).append(feedback)

    def _build_evidence_signature(self, msgs: list[dict]) -> str:
        observations = self._collect_condensed_observations(msgs)[-WINDOW_EVIDENCE_SIGNATURE_OBS:]
        payload = json.dumps(observations, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _should_accept_stale_answer_revision_loop(
        self,
        verdict: dict[str, Any],
        run_report: dict[str, Any],
        *,
        current_evidence_signature: str,
        previous_answer: str,
        proposed_answer: str,
    ) -> bool:
        if verdict.get("verdict") != "BLOCK":
            return False
        if run_report.get("blocked_finish_count", 0) < 1:
            return False
        if not previous_answer or previous_answer == proposed_answer.strip():
            return False
        previous_checkpoint = (run_report.get("answer_checkpoints") or [])[-1:]
        if not previous_checkpoint:
            return False
        latest_checkpoint = previous_checkpoint[0]
        if not isinstance(latest_checkpoint, dict):
            return False
        if latest_checkpoint.get("verdict") != "BLOCK":
            return False
        latest_signature = str(latest_checkpoint.get("evidence_signature") or "").strip()
        if latest_signature and current_evidence_signature:
            return latest_signature == current_evidence_signature
        return True

    def _accept_stale_answer_revision_loop(
        self,
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        redirected_verdict = dict(verdict)
        redirected_verdict.update(
            {
                "verdict": "INSUFFICIENT",
                "gap_diagnosis": {
                    "gap_type": "insufficient",
                    "description": "Answer revision stale -- return to evidence collection",
                },
                "source": "fallback",
            }
        )
        redirected_verdict.pop("revision_hint", None)
        return redirected_verdict

    def _summarize_recent_critic_checkpoints(
        self,
        run_report: dict[str, Any],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        checkpoints = run_report.get("critic_checkpoints") or []
        summaries: list[dict[str, Any]] = []
        for checkpoint in checkpoints[-limit:]:
            if not isinstance(checkpoint, dict):
                continue
            gap = checkpoint.get("gap_diagnosis") or {}
            summaries.append(
                {
                    "checkpoint_index": checkpoint.get("checkpoint_index"),
                    "stage": checkpoint.get("stage"),
                    "after_tool": checkpoint.get("after_tool"),
                    "verdict": checkpoint.get("verdict"),
                    "gap_diagnosis": {
                        "gap_type": gap.get("gap_type"),
                        "description": gap.get("description"),
                    },
                }
            )
        return summaries

    def _summarize_recent_answer_checkpoints(
        self,
        run_report: dict[str, Any],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        checkpoints = run_report.get("answer_checkpoints") or []
        summaries: list[dict[str, Any]] = []
        for checkpoint in checkpoints[-limit:]:
            if not isinstance(checkpoint, dict):
                continue
            summaries.append(
                {
                    "checkpoint_index": checkpoint.get("checkpoint_index"),
                    "verdict": checkpoint.get("verdict"),
                    "revision_hint": checkpoint.get("revision_hint"),
                    "proposed_answer_preview": checkpoint.get("proposed_answer_preview"),
                }
            )
        return summaries

    def _latest_answer_checkpoint_signature(
        self,
        run_report: dict[str, Any],
    ) -> str:
        latest_answer_verdict = run_report.get("latest_answer_verdict") or {}
        signature = str(latest_answer_verdict.get("evidence_signature") or "").strip()
        if signature:
            return signature
        checkpoints = run_report.get("answer_checkpoints") or []
        if not checkpoints:
            return ""
        latest_checkpoint = checkpoints[-1]
        if not isinstance(latest_checkpoint, dict):
            return ""
        return str(latest_checkpoint.get("evidence_signature") or "").strip()

    def _record_action_trace(
        self,
        run_report: dict[str, Any],
        *,
        selected_tool: str,
        stage_before: ProtocolStage,
        stage_after: ProtocolStage,
        reason_source: str,
        reason_text: str,
        blocked: bool,
        critic_decision: str = "",
        observation_summary: str = "",
    ) -> None:
        item: ActionTraceItem = {
            "step_index": len(run_report.get("action_trace") or []) + 1,
            "selected_tool": selected_tool,
            "tool_role": self._tool_role(selected_tool),
            "stage_before": stage_before,
            "stage_after": stage_after,
            "reason_source": reason_source,
            "reason_text": reason_text,
            "blocked": blocked,
        }
        if critic_decision:
            item["critic_decision"] = critic_decision
        if observation_summary:
            item["observation_summary"] = observation_summary
        run_report.setdefault("action_trace", []).append(item)

    def _has_video_grounding_attempt(self, run_report: dict[str, Any]) -> bool:
        return any(
            tool_name in VIDEO_GROUNDING_TOOLS
            for tool_name in (run_report.get("tool_sequence") or [])
        )

    def _has_external_lookup_attempt(self, run_report: dict[str, Any]) -> bool:
        return any(
            tool_name in EXTERNAL_LOOKUP_TOOLS
            for tool_name in (run_report.get("tool_sequence") or [])
        )

    @staticmethod
    def _batch_call_signature(tool_call: dict[str, Any]) -> str:
        """Identity of a tool call for batch dedup: name plus normalized arguments.

        JSON-normalized equality only (parse, re-dump with sorted keys), so
        `{"a":1,"b":2}` and `{"b":2,"a":1}` are one call while a differing query
        string is two. Arguments that do not parse fall back to the raw string:
        two identically-malformed calls are still duplicates of each other, and a
        malformed one never collides with a well-formed one.
        """
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "unknown")
        raw_args = function.get("arguments")
        try:
            normalized = json.dumps(
                json.loads(raw_args if isinstance(raw_args, str) else "{}"),
                sort_keys=True,
                ensure_ascii=False,
            )
        except (json.JSONDecodeError, TypeError):
            normalized = str(raw_args)
        return f"{name}::{normalized}"

    def _compose_tool_call_batch(
        self,
        pending_tool_calls: list[dict[str, Any]],
        msgs: list[dict],
        run_report: dict[str, Any],
        question: str,
    ) -> list[dict[str, Any]]:
        """Item ③: admit a planner's batch of tool calls, in the locked order.

        Runs dedup, then the width cap, then finish deferral. A dropped call is
        not simply discarded: the transport rejects an assistant message whose
        `tool_calls` have no matching tool response, so each one is routed
        through `_record_blocked_step`, which appends that response and records
        the drop as a blocked step. `_validate_message_history` is the oracle for
        that invariant.

        1-wide batches (the overwhelming majority, and every synthetic promotion)
        pass through untouched — no counters move and no messages are appended.
        """
        if len(pending_tool_calls) <= 1:
            self._record_batch_width(run_report, len(pending_tool_calls))
            return list(pending_tool_calls)

        admitted: list[dict[str, Any]] = []
        # (call, counter key, observation summary, planner-facing hint)
        dropped: list[tuple[dict[str, Any], str, str, str]] = []
        seen: set[str] = set()

        for tool_call in pending_tool_calls:
            signature = self._batch_call_signature(tool_call)
            if signature in seen:
                dropped.append(
                    (
                        tool_call,
                        "batch_duplicate_actions",
                        "Duplicate call within this batch: an identical call is already running this round.",
                        "Read the identical call's observation; do not re-issue it in the same batch.",
                    )
                )
                continue
            seen.add(signature)
            admitted.append(tool_call)

        if len(admitted) > BATCH_MAX_WIDTH:
            for tool_call in admitted[BATCH_MAX_WIDTH:]:
                dropped.append(
                    (
                        tool_call,
                        "batch_width_truncations",
                        f"Batch exceeded the {BATCH_MAX_WIDTH}-call width cap; this call was not executed.",
                        f"At most {BATCH_MAX_WIDTH} calls run per round. Re-issue this call next round if the need is still open.",
                    )
                )
            admitted = admitted[:BATCH_MAX_WIDTH]

        non_finish = [
            call
            for call in admitted
            if (call.get("function") or {}).get("name") != "finish"
        ]
        if (
            non_finish
            and len(non_finish) < len(admitted)
            and not self._is_no_state_react_mode()
        ):
            for tool_call in admitted:
                if (tool_call.get("function") or {}).get("name") != "finish":
                    continue
                dropped.append(
                    (
                        tool_call,
                        "batch_finish_deferrals",
                        "finish deferred: this batch's acquisition results land next round; "
                        "call finish alone next round if evidence suffices.",
                        "Send finish as the only call in its turn.",
                    )
                )
            admitted = non_finish

        for tool_call, counter_key, summary, hint in dropped:
            self._record_batch_drop(
                tool_call, counter_key, summary, hint, msgs, run_report, question
            )

        self._record_batch_width(run_report, len(admitted))
        return admitted

    def _batch_metrics(self, run_report: dict[str, Any]) -> dict[str, Any]:
        return run_report.setdefault(
            "batch_metrics",
            {
                "batch_duplicate_actions": 0,
                "batch_width_truncations": 0,
                "batch_finish_deferrals": 0,
                "width_histogram": {},
            },
        )

    def _record_batch_width(self, run_report: dict[str, Any], width: int) -> None:
        """Histogram of executed batch widths — the adoption measurement.

        Keyed by string because the run report is serialized to JSON, where an
        int key would come back as a string anyway; recording it that way keeps
        one shape between the live report and the dumped manifest.
        """
        if width <= 0:
            return
        histogram = self._batch_metrics(run_report)["width_histogram"]
        key = str(int(width))
        histogram[key] = int(histogram.get(key) or 0) + 1

    def _record_batch_drop(
        self,
        tool_call: dict[str, Any],
        counter_key: str,
        summary: str,
        fallback_hint: str,
        msgs: list[dict],
        run_report: dict[str, Any],
        question: str,
    ) -> None:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "unknown")
        tool_call_id = (
            tool_call.get("id")
            or f"{name}_batch_{len(run_report.get('tool_sequence') or []) + 1}"
        )
        try:
            public_args = json.loads(function.get("arguments") or "{}")
            if not isinstance(public_args, dict):
                public_args = {}
        except (json.JSONDecodeError, TypeError):
            public_args = {}
        public_args = {
            key: value
            for key, value in public_args.items()
            if key not in ("database", "__precomputed_result__")
        }

        metrics = self._batch_metrics(run_report)
        metrics[counter_key] = int(metrics.get(counter_key) or 0) + 1

        self._record_blocked_step(
            tool_call_id,
            name,
            {
                "status": "guardrail",
                "source_modality": "unknown",
                "summary": summary,
                "fallback_hint": fallback_hint,
            },
            msgs,
            run_report,
            public_args,
            self._infer_protocol_stage(run_report),
            "batch_composition",
            summary,
            question,
        )

    def _record_blocked_step(
        self,
        tool_call_id: str,
        name: str,
        guard_observation: str | dict,
        msgs: list[dict],
        run_report: dict[str, Any],
        public_args: dict[str, Any],
        stage_before: str,
        reason_source: str,
        reason_text: str,
        question: str,
        *,
        critic_decision: str = "",
        answer_verdict: dict[str, Any] | None = None,
    ) -> None:
        """Record a blocked tool call: append message, trace, and protocol step."""
        obs_text = guard_observation if isinstance(guard_observation, str) else dumps_llm_payload(
            guard_observation,
        )
        self._append_tool_msg(tool_call_id, name, obs_text, msgs)
        normalized = self._normalize_tool_observation(
            name,
            guard_observation,
            question=question,
            asset_id=run_report.get("_current_asset_id"),
        )
        run_report.pop("_current_asset_id", None)
        stage_after = self._infer_protocol_stage(run_report)
        if not critic_decision:
            critic_decision = (run_report.get("latest_critic_verdict") or {}).get("verdict", "")
        self._record_action_trace(
            run_report, selected_tool=name, stage_before=stage_before,
            stage_after=stage_after, reason_source=reason_source,
            reason_text=reason_text, blocked=True,
            critic_decision=critic_decision,
            observation_summary=self._summarize_tool_content(name, obs_text),
        )
        self._record_protocol_step(
            run_report, selected_tool=name, tool_arguments=public_args,
            stage_before=stage_before, stage_after=stage_after,
            reason_source=reason_source, reason_text=reason_text, blocked=True,
            raw_observation=guard_observation, normalized_observation=normalized,
            evidence_delta={"state_changed": False},
            critic_verdict=run_report.get("latest_critic_verdict"),
            answer_verdict=answer_verdict or run_report.get("latest_answer_verdict"),
            step_elapsed_ms=0,
        )

    def _exec_tool(
        self,
        tool_call,
        msgs,
        tool_history: dict[str, int],
        question: str,
        run_report: dict[str, Any],
    ):
        name = tool_call["function"]["name"]
        tool_call_id = tool_call.get("id") or f"{name}_{len(run_report['tool_sequence']) + 1}"
        tool_sequence_recorded = False
        stage_before = self._infer_protocol_stage(run_report)
        reason_source, reason_text = self._infer_action_reason(name, run_report)

        # Guard 1: invalid function name
        if name not in self.name_to_function_map:
            self._record_blocked_step(
                tool_call_id, name,
                {"status": "guardrail", "source_modality": "unknown",
                 "summary": f"Invalid function name: {name!r}",
                 "fallback_hint": "Select a valid tool from the current tool schema."},
                msgs, run_report, {}, stage_before,
                "guardrail", f"Invalid function name: {name!r}.", question,
            )
            return

        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        args = self._repair_tool_arguments(
            name,
            args,
            question=question,
            run_report=run_report,
        )
        public_args = {k: v for k, v in args.items() if k not in ("database", "__precomputed_result__")}

        # Guard 3: answer action guard
        answer_action_guard = self._pending_answer_action_guard(msgs, run_report)
        if (
            answer_action_guard is not None
            and name not in set(answer_action_guard.get("allowed_tools") or [])
        ):
            self._record_blocked_step(
                tool_call_id, name,
                self._build_answer_action_guard_observation(answer_action_guard, name),
                msgs, run_report, public_args, stage_before,
                "guardrail",
                str(answer_action_guard.get("reason") or "").strip()
                or "The latest answer checkpoint constrains the next action more narrowly than this tool call.",
                question,
                critic_decision=(run_report.get("latest_answer_verdict") or {}).get("verdict", ""),
            )
            return

        # An evidence tool running while the latest answer verdict is a
        # missing-input BLOCK is the reopen the guard above just permitted.
        # Recorded here, after the guard cleared the call and before it runs, so
        # the budget is spent by an actual acquisition rather than by a finish
        # attempt that merely consulted it.
        self._note_answer_block_reopen(run_report, name)

        # Answer checkpoint gate (finish only)
        if name == "finish":
            no_state_direct_accept = self._is_no_state_react_mode()
            if run_report.get("answer_phase") or no_state_direct_accept:
                proposed_answer = str(args.get("answer") or "").strip()
                run_report["last_finish_attempt"] = proposed_answer
                checkpoint_index = len(run_report.get("answer_checkpoints") or []) + 1
                if not proposed_answer:
                    answer_checkpoint = {
                        "verdict": "BLOCK",
                        "revision_hint": "Provide a non-empty answer when calling finish.",
                        "stage": "before_finish",
                        "checkpoint_index": checkpoint_index,
                        "proposed_answer_preview": "",
                        "evidence_signature": self._build_evidence_signature(msgs),
                        "source": (
                            "no_state_react_empty_finish"
                            if no_state_direct_accept
                            else "answer_phase_empty_finish"
                        ),
                    }
                    run_report.setdefault("answer_checkpoints", []).append(answer_checkpoint)
                    run_report["latest_answer_verdict"] = answer_checkpoint
                    run_report["blocked_finish_count"] += 1
                    run_report["last_finish_block_reason"] = str(
                        answer_checkpoint.get("revision_hint")
                        or "finish blocked by empty answer"
                    ).strip()
                    run_report["stop_reason"] = (
                        run_report.get("stop_reason") or "answer_phase_empty_finish"
                    )
                    self._record_blocked_step(
                        tool_call_id,
                        name,
                        self._build_answer_guard_observation(answer_checkpoint),
                        msgs,
                        run_report,
                        public_args,
                        stage_before,
                        "answer_phase_validation",
                        "Answer phase finish requires a non-empty answer.",
                        question,
                        critic_decision=answer_checkpoint.get("verdict", ""),
                        answer_verdict=answer_checkpoint,
                    )
                    return
                accept_source = (
                    "no_state_react_direct_accept"
                    if no_state_direct_accept
                    else "answer_phase_bypass"
                )
                answer_checkpoint = {
                    "verdict": "PASS",
                    "revision_hint": "",
                    "stage": "before_finish",
                    "checkpoint_index": checkpoint_index,
                    "proposed_answer_preview": proposed_answer[:TRUNC_PROPOSED_ANSWER_PREVIEW],
                    "evidence_signature": self._build_evidence_signature(msgs),
                    "source": accept_source,
                }
                run_report.setdefault("answer_checkpoints", []).append(answer_checkpoint)
                run_report["latest_answer_verdict"] = answer_checkpoint
                current_step = len(run_report.get("tool_sequence") or [])
                if proposed_answer:
                    run_report["best_candidate_answer"] = {
                        "answer_text": proposed_answer,
                        "sufficiency_status": str(
                            (run_report.get("evidence_state") or {}).get("sufficiency_status") or ""
                        ),
                        "cached_at_step": current_step,
                        "source": accept_source,
                    }
                reason_source = (
                    "no_state_react_direct_accept"
                    if no_state_direct_accept
                    else "answer_phase_direct_accept"
                )
                reason_text = (
                    "no_state_react: finish accepted directly without answer critic."
                    if no_state_direct_accept
                    else "Answer phase: finish accepted without answer critic."
                )
            else:
                with self._span("answer_critic.check") as _h_ac:
                    answer_checkpoint = self._run_answer_checkpoint(
                        str(args.get("answer") or ""), msgs, question, run_report,
                    )
                    if _h_ac is not None and isinstance(answer_checkpoint, dict):
                        _h_ac.extra["verdict"] = str(answer_checkpoint.get("verdict") or "")
                        _h_ac.extra["revision_hint"] = str(
                            answer_checkpoint.get("revision_hint") or ""
                        )[:240]
                        candidate_answer = str(args.get("answer") or "").strip()
                        revision_hint = str(answer_checkpoint.get("revision_hint") or "")
                        verdict_reasoning = str(answer_checkpoint.get("verdict") or "")
                        if revision_hint:
                            verdict_reasoning = verdict_reasoning + ": " + revision_hint
                        _h_ac.attach_io(
                            input_preview=candidate_answer,
                            output_preview=verdict_reasoning,
                        )
                if answer_checkpoint.get("verdict") != "PASS":
                    run_report["blocked_finish_count"] += 1
                    run_report["last_finish_block_reason"] = str(
                        answer_checkpoint.get("revision_hint")
                        or "finish blocked by answer checkpoint"
                    ).strip()
                    guard_observation = self._build_answer_guard_observation(answer_checkpoint)

                    if self._is_no_repair_mode():
                        proposed_answer = str(args.get("answer") or "").strip()
                        current_step = len(run_report.get("tool_sequence") or [])
                        if proposed_answer:
                            run_report["best_candidate_answer"] = {
                                "answer_text": proposed_answer,
                                "sufficiency_status": str(
                                    (run_report.get("evidence_state") or {}).get("sufficiency_status") or ""
                                ),
                                "cached_at_step": current_step,
                                "source": "no_repair_ablation",
                            }
                        run_report["final_response_kind"] = (
                            "answer_forced" if proposed_answer else "insufficient_evidence"
                        )
                        run_report["final_output_text"] = proposed_answer or guard_observation
                        run_report["stop_reason"] = "no_repair_finish_block"
                        normalized = self._normalize_tool_observation(
                            name,
                            guard_observation,
                            question=question,
                            asset_id=run_report.get("_current_asset_id"),
                        )
                        run_report.pop("_current_asset_id", None)
                        self._record_action_trace(
                            run_report,
                            selected_tool=name,
                            stage_before=stage_before,
                            stage_after="stopped",
                            reason_source="no_repair_ablation",
                            reason_text=(
                                "The answer checkpoint blocked finish, and the no_repair ablation "
                                "finalizes immediately instead of reopening evidence collection."
                            ),
                            blocked=False,
                            critic_decision=answer_checkpoint.get("verdict", ""),
                            observation_summary=self._summarize_tool_content(name, guard_observation),
                        )
                        self._record_protocol_step(
                            run_report,
                            selected_tool=name,
                            tool_arguments=public_args,
                            stage_before=stage_before,
                            stage_after="stopped",
                            reason_source="no_repair_ablation",
                            reason_text=(
                                "The answer checkpoint blocked finish, and the no_repair ablation "
                                "finalizes immediately instead of reopening evidence collection."
                            ),
                            blocked=False,
                            raw_observation=guard_observation,
                            normalized_observation=normalized,
                            evidence_delta={"state_changed": False},
                            critic_verdict=run_report.get("latest_critic_verdict"),
                            answer_verdict=answer_checkpoint,
                            step_elapsed_ms=0,
                        )
                        raise StopException(run_report["final_output_text"])

                    current_revision_count = int(run_report.get("answer_revision_count") or 0)
                    reopens = min(
                        int(run_report.get("answer_revision_reopen_count") or 0),
                        MAX_ANSWER_REVISION_REOPENS,
                    )
                    if current_revision_count - reopens >= MAX_ANSWER_REVISIONS:
                        # Force-accept: bypass answer checkpoint, return answer as-is
                        run_report["final_response_kind"] = "answer_forced"
                        run_report["final_output_text"] = str(args.get("answer") or "").strip()
                        run_report["stop_reason"] = "answer_revision_depth_limit"
                        # Still record the step for traceability
                        self._record_action_trace(
                            run_report, selected_tool=name, stage_before=stage_before,
                            stage_after="stopped",
                            reason_source="answer_revision_depth_limit",
                            reason_text=f"Answer revision capped at {MAX_ANSWER_REVISIONS}; force-accepting current answer.",
                            blocked=False,
                            critic_decision="FORCE_ACCEPT",
                            observation_summary=f"Force-accepted answer after {current_revision_count} revision attempts.",
                        )
                        raise StopException(run_report["final_output_text"])

                    run_report["answer_revision_count"] = current_revision_count + 1
                    self._append_tool_msg(tool_call_id, name, guard_observation, msgs)
                    state_before = copy.deepcopy(run_report.get("evidence_state") or self._init_evidence_state(question))
                    self._recompute_state_fields(state_before, question=question, run_report=run_report)
                    run_report["evidence_state"] = state_before
                    run_report["latest_state_delta"] = self._build_state_delta(
                        copy.deepcopy(run_report.get("evidence_state") or {}), state_before,
                    )
                    run_report["latest_state_digest"] = self._build_evidence_state_digest(state_before)
                    normalized = self._normalize_tool_observation(
                        name,
                        guard_observation,
                        question=question,
                        asset_id=run_report.get("_current_asset_id"),
                    )
                    run_report.pop("_current_asset_id", None)
                    self._record_action_trace(
                        run_report, selected_tool=name, stage_before=stage_before,
                        stage_after=self._infer_protocol_stage(run_report),
                        reason_source="answer_critic",
                        reason_text="The latest answer checkpoint blocked this finish attempt and requested a revision or more evidence.",
                        blocked=True, critic_decision=answer_checkpoint.get("verdict", ""),
                        observation_summary=self._summarize_tool_content(name, guard_observation),
                    )
                    self._record_protocol_step(
                        run_report, selected_tool=name, tool_arguments=public_args,
                        stage_before=stage_before, stage_after=self._infer_protocol_stage(run_report),
                        reason_source="answer_critic",
                        reason_text="The latest answer checkpoint blocked this finish attempt and requested a revision or more evidence.",
                        blocked=True, raw_observation=guard_observation,
                        normalized_observation=normalized,
                        evidence_delta={"state_changed": False},
                        critic_verdict=run_report.get("latest_critic_verdict"),
                        answer_verdict=answer_checkpoint,
                        step_elapsed_ms=0,
                    )
                    self._append_state_digest_message(msgs, run_report)
                    if self._should_try_inline_answer_revision(answer_checkpoint, run_report):
                        self._attempt_inline_answer_revision(msgs, tool_history, question, run_report)
                    return
                reason_source = "answer_critic"
                reason_text = "The latest answer checkpoint approved the current final answer."

        # Argument normalization
        if name == "frame_confirm_tool" and "time_ranges" in args and "time_ranges_hhmmss" not in args:
            args["time_ranges_hhmmss"] = args.pop("time_ranges")
        if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
            if "topk" in args:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK
            if "top_k" in args:
                args["top_k"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # Guard 4: repeat call detection
        signature_payload = json.dumps(
            {"name": name, "args": {k: v for k, v in args.items() if k != "database"}},
            ensure_ascii=False, sort_keys=True,
        )
        tool_history[signature_payload] = tool_history.get(signature_payload, 0) + 1
        if tool_history[signature_payload] > REPEAT_GUARD_MAX:
            self._record_blocked_step(
                tool_call_id, name,
                self._build_repeat_guard_observation(name),
                msgs, run_report, public_args, stage_before,
                "guardrail",
                "The same tool call arguments were already used multiple times in this run.",
                question,
            )
            return

        # Working memory dedup check
        if name not in {"finish"}:
            memory_state = run_report.get("evidence_state") or self._init_evidence_state(question)
            memory_query_key = json.dumps(public_args, ensure_ascii=False, sort_keys=True)
            _cached_payload = self._get_memory_slot(
                memory_state, tool_name=name, query=memory_query_key
            )
            if _cached_payload:
                memory_hit_observation = {
                    "status": "memory_hit",
                    "source_modality": "cache",
                    "summary": f"[Working Memory] Returning cached result for {name} (dedup).",
                    "fallback_hint": "This result was served from working memory. Do NOT repeat this call.",
                    "cached_from_step": _cached_payload.get("created_at_step", 0),
                }
                self._append_tool_msg(
                    tool_call_id, name,
                    dumps_llm_payload(memory_hit_observation), msgs,
                )
                if not tool_sequence_recorded:
                    run_report["tool_sequence"].append(name)
                    tool_sequence_recorded = True
                normalized = self._normalize_tool_observation(
                    name,
                    json.dumps(memory_hit_observation, ensure_ascii=False),
                    question=question,
                    asset_id=run_report.get("_current_asset_id"),
                )
                run_report.pop("_current_asset_id", None)
                self._record_protocol_step(
                    run_report, selected_tool=name, tool_arguments=public_args,
                    stage_before=stage_before, stage_after=self._infer_protocol_stage(run_report),
                    reason_source="memory_hit",
                    reason_text="Working memory dedup: identical tool+args already executed, returning cached result.",
                    blocked=True, raw_observation=json.dumps(memory_hit_observation, ensure_ascii=False),
                    normalized_observation=normalized, evidence_delta={"state_changed": False},
                    critic_verdict=run_report.get("latest_critic_verdict"),
                    answer_verdict=run_report.get("latest_answer_verdict"),
                    step_elapsed_ms=0,
                )
                self._append_state_digest_message(msgs, run_report)
                return

        # Guard 5: missing required tool argument -> recoverable observation, never an uncaught
        # crash. The try-block below only catches StopException, so a tool call that omits a
        # required positional arg (e.g. frame_confirm_tool without time_ranges_hhmmss) would
        # otherwise raise TypeError and abort the entire case. In no_state_react there is no
        # readiness gate and no state-synthesized args, so the planner can issue such calls; a
        # fair ReAct must surface an error observation it can react to rather than crash. This is
        # mode-agnostic robustness — it never fires when every required arg is present, which is
        # the normal full-state path.
        if "__precomputed_result__" not in args:
            missing_required = self._missing_required_tool_args(name, args)
            if missing_required:
                missing_csv = ", ".join(missing_required)
                self._record_blocked_step(
                    tool_call_id, name,
                    {"status": "error", "source_modality": "unknown",
                     "summary": f"Tool '{name}' call is missing required argument(s): {missing_csv}.",
                     "fallback_hint": (
                         f"Re-issue {name} with {missing_csv}. For time_ranges_hhmmss, first ground a "
                         "candidate range with a grounding tool, then pass [[start, end]] in HH:MM:SS."
                     )},
                    msgs, run_report, public_args, stage_before,
                    "guardrail",
                    f"Tool call missing required argument(s): {missing_csv}.",
                    question,
                )
                return

        try:
            tool_started_at = time.perf_counter()
            self._debug_print(f"Calling function `{name}` with args: {args}")
            self._reset_asset_cache_tracking()
            if not tool_sequence_recorded:
                run_report["tool_sequence"].append(name)
                tool_sequence_recorded = True
            if "__precomputed_result__" in args:
                result = args.pop("__precomputed_result__")
            else:
                result = self.name_to_function_map[name](**args)
            self._capture_asset_cache_summary(run_report, name)
            self._append_tool_msg(tool_call_id, name, result, msgs)
            if name == "web_search_tool":
                self._record_web_search_feedback(result, run_report)
            if name in VISUAL_CONFIRM_TOOLS:
                run_report["high_cost_call_count"] += 1
            run_report["_message_snapshot"] = copy.deepcopy(msgs)
            if self._is_no_state_react_mode():
                run_report.pop("_current_asset_id", None)
                if name != "finish":
                    run_report.pop("last_finish_block_reason", None)
                state_before = copy.deepcopy(
                    run_report.get("evidence_state") or self._inert_evidence_state(question)
                )
                normalized_observation = None
                evidence_delta = {}
            else:
                state_before = copy.deepcopy(
                    run_report.get("evidence_state") or self._init_evidence_state(question)
                )
                normalized_observation = self._normalize_tool_observation(
                    name,
                    result,
                    question=question,
                    asset_id=run_report.get("_current_asset_id"),
                )
                try:
                    _, evidence_delta = self._ingest_normalized_observation(
                        run_report,
                        normalized_observation,
                        question=question,
                    )
                finally:
                    run_report.pop("_current_asset_id", None)
                if name != "finish":
                    run_report.pop("last_finish_block_reason", None)
                    memory_query_key = json.dumps(public_args, ensure_ascii=False, sort_keys=True)
                    memory_summary = str(normalized_observation.get("summary") or "")[:TRUNC_RESULT_SUMMARY]
                    memory_result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    self._store_memory_slot(
                        run_report["evidence_state"],
                        tool_name=name,
                        query=memory_query_key,
                        result_summary=memory_summary,
                        full_payload={
                            "result_text": memory_result_text[:TRUNC_MEMORY_RESULT_TEXT],
                            "created_at_step": len(run_report.get("tool_sequence") or []),
                        },
                        step_index=len(run_report.get("tool_sequence") or []),
                    )
            stage = (
                "after_grounding" if name in GROUNDING_TOOLS else "after_visual_verification"
            )
            if self._is_no_state_react_mode():
                critic_result = None
            else:
                with self._span("critic.check") as _h_critic:
                    if _h_critic is not None:
                        _h_critic.extra["stage"] = stage
                        _h_critic.extra["after_tool"] = name
                    critic_result = self._run_critic_checkpoint(
                        stage, name, result, msgs, question, run_report
                    )
                    if _h_critic is not None and isinstance(critic_result, dict):
                        _h_critic.extra["verdict"] = str(
                            critic_result.get("verdict")
                            or critic_result.get("sufficiency_status")
                            or ""
                        )
                        _h_critic.extra["gap_diagnosis"] = str(
                            (critic_result.get("gap_diagnosis") or {}).get("description")
                            if isinstance(critic_result.get("gap_diagnosis"), dict)
                            else critic_result.get("gap_diagnosis") or ""
                        )[:240]
                        observation_input = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                        gap_text = str(
                            (critic_result.get("gap_diagnosis") or {}).get("description")
                            if isinstance(critic_result.get("gap_diagnosis"), dict)
                            else critic_result.get("gap_diagnosis") or ""
                        )
                        verdict_text = str(
                            critic_result.get("verdict")
                            or critic_result.get("sufficiency_status")
                            or ""
                        )
                        output_text = verdict_text + (": " + gap_text if gap_text else "")
                        _h_critic.attach_io(
                            input_preview=observation_input,
                            output_preview=output_text,
                        )
            evidence_delta = run_report.get("latest_state_delta") or {}
            self._record_action_trace(
                run_report,
                selected_tool=name,
                stage_before=stage_before,
                stage_after=self._infer_protocol_stage(run_report),
                reason_source=reason_source,
                reason_text=reason_text,
                blocked=False,
                critic_decision=(run_report.get("latest_critic_verdict") or {}).get(
                    "verdict", ""
                ),
                observation_summary=self._summarize_tool_content(name, result),
            )
            self._record_protocol_step(
                run_report,
                selected_tool=name,
                tool_arguments=public_args,
                stage_before=stage_before,
                stage_after=self._infer_protocol_stage(run_report),
                reason_source=reason_source,
                reason_text=reason_text,
                blocked=False,
                raw_observation=result,
                normalized_observation=normalized_observation,
                evidence_delta=evidence_delta,
                critic_verdict=run_report.get("latest_critic_verdict"),
                answer_verdict=run_report.get("latest_answer_verdict")
                if name == "finish"
                else None,
                step_elapsed_ms=int((time.perf_counter() - tool_started_at) * 1000),
            )
            self._inject_strategy_switch(msgs, run_report)
            self._append_state_digest_message(msgs, run_report)
        except StopException as exc:
            self._debug_print(f"Finish task with message: '{exc!s}'")
            answer_verdict = run_report.get("latest_answer_verdict") or {}
            latest_verdict = run_report.get("latest_critic_verdict") or {}
            run_report["final_response_kind"] = "answer"
            run_report["final_output_text"] = str(exc)
            run_report["stop_reason"] = (
                run_report.get("stop_reason")
                or str((answer_verdict.get("revision_hint") or "")).strip()
                or str(((latest_verdict.get("gap_diagnosis") or {}).get("description") or "")).strip()
                or "finish tool called after critic approval."
            )
            finish_observation = {
                "status": "ok",
                "source_modality": "unknown",
                "summary": str(exc),
                "confidence": 0.95,
                "fallback_hint": "",
            }
            if self._is_no_state_react_mode():
                run_report.pop("_current_asset_id", None)
                normalized_observation = finish_observation
                evidence_delta = {}
            else:
                normalized_observation = self._normalize_tool_observation(
                    name,
                    finish_observation,
                    question=question,
                    asset_id=run_report.get("_current_asset_id"),
                )
                run_report.pop("_current_asset_id", None)
                state_before = copy.deepcopy(
                    run_report.get("evidence_state") or self._init_evidence_state(question)
                )
                final_state = copy.deepcopy(state_before)
                final_state["current_stage"] = "stopped"
                final_state["sufficiency_status"] = "answered"
                final_state["stop_ready"] = True
                final_state["budget_state"] = {
                    "high_cost_call_count": int(run_report.get("high_cost_call_count", 0)),
                    "reground_count": int(run_report.get("reground_count", 0)),
                    "blocked_finish_count": int(run_report.get("blocked_finish_count", 0)),
                    "answer_revision_count": int(run_report.get("answer_revision_count", 0)),
                }
                final_state["latest_observation"] = copy.deepcopy(normalized_observation)
                run_report["evidence_state"] = final_state
                evidence_delta = self._build_state_delta(state_before, final_state)
                final_state["latest_state_delta"] = copy.deepcopy(evidence_delta)
                run_report["latest_state_delta"] = evidence_delta
                run_report["latest_state_digest"] = self._build_evidence_state_digest(final_state)
            self._record_action_trace(
                run_report,
                selected_tool=name,
                stage_before=stage_before,
                stage_after="stopped",
                reason_source=reason_source,
                reason_text=reason_text,
                blocked=False,
                critic_decision=(
                    answer_verdict.get("verdict")
                    or latest_verdict.get("verdict")
                    or ""
                ),
                observation_summary=str(exc),
            )
            self._record_protocol_step(
                run_report,
                selected_tool=name,
                tool_arguments=public_args,
                stage_before=stage_before,
                stage_after="stopped",
                reason_source=reason_source,
                reason_text=reason_text,
                blocked=False,
                raw_observation=finish_observation,
                normalized_observation=normalized_observation,
                evidence_delta=evidence_delta,
                critic_verdict=run_report.get("latest_critic_verdict"),
                answer_verdict=run_report.get("latest_answer_verdict"),
                step_elapsed_ms=int((time.perf_counter() - tool_started_at) * 1000),
            )
            raise

    def _validate_message_history(self, msgs: list[dict]) -> list[dict]:
        """Fix message ordering issues before each LLM call.

        1. Remove orphaned tool_calls that have no matching tool response.
        2. Reorder interleaved messages so that all tool responses for a
           parallel tool_call batch appear consecutively (OpenAI API requires
           this; system messages injected between tool responses cause 400s).

        Returns a list of repair actions taken, for logging/observability.
        """
        repair_actions: list[dict] = []
        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            tool_call_ids = {
                tc.get("id") for tc in msg["tool_calls"] if tc.get("id")
            }
            # Determine the batch range: from i+1 to next assistant message
            batch_start = i + 1
            batch_end = len(msgs)
            for j in range(batch_start, len(msgs)):
                if msgs[j].get("role") == "assistant":
                    batch_end = j
                    break

            # Find matching tool responses
            responded_ids: set[str] = set()
            for j in range(batch_start, batch_end):
                if msgs[j].get("role") == "tool" and msgs[j].get("tool_call_id"):
                    responded_ids.add(msgs[j]["tool_call_id"])

            # Repair 1: remove orphaned tool_calls
            orphaned = tool_call_ids - responded_ids
            if orphaned:
                repair_actions.append({
                    "action": "remove_orphaned_tool_calls",
                    "assistant_index": i,
                    "removed_tool_call_ids": sorted(orphaned),
                })
                msg["tool_calls"] = [
                    tc for tc in msg["tool_calls"]
                    if tc.get("id") not in orphaned
                ]
                if not msg["tool_calls"]:
                    del msg["tool_calls"]

            # Repair 2: fix interleaved non-tool messages in parallel batches.
            # When _exec_tool runs for each parallel call it appends both a tool
            # response and an internal state digest, producing:
            #   [tool_A, digest_A, tool_B, digest_B]
            # The API requires all tool responses consecutive, and intermediate
            # digests are stale anyway.  Fix: keep all tool responses in order,
            # drop intermediate non-tool messages, keep only the last one.
            batch = msgs[batch_start:batch_end]
            tool_msgs = [m for m in batch if m.get("role") == "tool"]
            other_msgs = [m for m in batch if m.get("role") != "tool"]
            if len(tool_msgs) > 1 and other_msgs:
                # Keep only the last non-tool message (final state digest)
                reordered = tool_msgs + other_msgs[-1:]
                dropped_count = len(other_msgs) - 1
                if reordered != batch:
                    msgs[batch_start:batch_end] = reordered
                    repair_actions.append({
                        "action": "dedup_parallel_batch_digests",
                        "assistant_index": i,
                        "tool_count": len(tool_msgs),
                        "dropped_intermediate_digests": dropped_count,
                    })
        return repair_actions

    def _truncate_history_for_weak_llm(self, msgs: list[dict], prefix_len: int) -> int:
        """Multi-pass context budget enforcement for weak-LLM mode (qwen3-omni-flash 49K).

        Pass 1 (sliding-window): drop middle rounds, keeping only the latest
            assistant_with_tool_calls + its tool responses (or last user msg).
        Pass 2 (per-message clip): cap each tool/system message to a hard size.
        Pass 3 (prefix asset clip): if total still exceeds budget, truncate the
            single largest message in the prefix (typically asset description).
        Logs final char totals to stderr for diagnosis.
        Returns the number of dropped middle messages (0 if none).
        """
        if not getattr(config, "WEAK_LLM_MODE", False):
            return 0

        max_total_chars = 100_000  # ~25K tokens; leaves headroom for response under 49K limit
        max_single_tool_chars = 8000
        max_single_system_chars = 3000

        # Pass 1: sliding-window
        dropped = 0
        if len(msgs) > prefix_len + 1:
            anchor = None
            for i in range(len(msgs) - 1, prefix_len - 1, -1):
                msg = msgs[i]
                role = msg.get("role")
                if role == "assistant" and msg.get("tool_calls"):
                    anchor = i
                    break
                if role == "user":
                    anchor = i
                    break
            if anchor is not None and anchor > prefix_len:
                dropped = anchor - prefix_len
                del msgs[prefix_len:anchor]

        # Pass 2: clip oversized individual messages
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if role == "tool" and len(content) > max_single_tool_chars:
                m["content"] = content[:max_single_tool_chars] + "\n...[truncated for weak-LLM mode]"
            elif role == "system" and len(content) > max_single_system_chars:
                m["content"] = content[:max_single_system_chars] + "\n...[truncated]"

        def _total_chars(seq):
            return sum(len(m.get("content") or "") for m in seq if isinstance(m.get("content"), str))

        # Pass 3: if total still over budget, truncate single largest message in prefix
        total = _total_chars(msgs)
        if total > max_total_chars and prefix_len >= 1:
            prefix_msgs = msgs[:prefix_len]
            idx_largest = max(
                range(len(prefix_msgs)),
                key=lambda i: len(prefix_msgs[i].get("content") or "")
                if isinstance(prefix_msgs[i].get("content"), str)
                else 0,
            )
            m = prefix_msgs[idx_largest]
            c = m.get("content")
            if isinstance(c, str):
                overshoot = total - max_total_chars + 1024
                target = max(2000, len(c) - overshoot)
                if target < len(c):
                    m["content"] = c[:target] + "\n...[asset description truncated for weak-LLM mode]"

        # Optional diagnostic (off by default). Enable via
        # OMNIVQA_WEAK_LLM_DEBUG_PAYLOAD=1 to trace per-message sizes during
        # OOC root-cause analysis.
        import os, sys
        if os.environ.get("OMNIVQA_WEAK_LLM_DEBUG_PAYLOAD", "").strip() == "1":
            try:
                payload_bytes = len(json.dumps(msgs, ensure_ascii=False))
                per_msg = []
                for i, m in enumerate(msgs):
                    role = m.get("role")
                    content = m.get("content")
                    if isinstance(content, str):
                        c_len = len(content)
                    elif isinstance(content, list):
                        c_len = sum(
                            len(p.get("text", "")) if isinstance(p, dict) else 0
                            for p in content
                        )
                    else:
                        c_len = 0
                    tc = m.get("tool_calls") or []
                    tc_len = sum(
                        len((t.get("function", {}).get("arguments") or ""))
                        for t in tc
                    )
                    per_msg.append(f"[{i}]{role}c={c_len}tc={tc_len}")
                sys.stderr.write(
                    f"[weak_llm_truncate] dropped={dropped} total_chars={_total_chars(msgs)} "
                    f"payload_bytes={payload_bytes} n_msgs={len(msgs)} prefix_len={prefix_len} "
                    f"per_msg={','.join(per_msg)}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        return dropped

    def _compact_planner_history(self, msgs: list[dict], prefix_len: int) -> int:
        """Replace all but the newest tool observations with their summaries."""
        if getattr(config, "WEAK_LLM_MODE", False):
            # The weak-LLM path already runs a far more aggressive sliding
            # window; layering compaction on top would only fight it.
            return 0

        window = max(1, int(PLANNER_RAW_OBSERVATION_WINDOW))
        tool_indices = [
            index
            for index in range(prefix_len, len(msgs))
            if msgs[index].get("role") == "tool"
        ]
        if len(tool_indices) <= window:
            return 0

        compacted = 0
        for index in tool_indices[:-window]:
            message = msgs[index]
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            payload = safe_load_json_object(content)
            if payload is None or payload.get(COMPACTED_OBSERVATION_MARKER):
                continue
            summary = self._summarize_observation(
                message.get("name", "unknown"), payload
            )
            summary[COMPACTED_OBSERVATION_MARKER] = True
            rendered = json.dumps(summary, ensure_ascii=False)
            if len(rendered) >= len(content):
                # Nothing to reclaim on an already-small observation; leaving it
                # raw is both cheaper and more faithful.
                continue
            message["content"] = rendered
            compacted += 1
        return compacted

    def _execute_loop(self, question):
        msgs = copy.deepcopy(self.messages)
        tool_history: dict[str, int] = {}
        token_tracker.reset()
        provider_failure_ledger.reset()
        case_id = self._derive_case_id(question)
        if self._step_logger is not None:
            self._step_logger.case_start(case_id, question)

        run_report = None
        _final_status = "ok"
        _error_type: str | None = None
        _error_msg: str | None = None
        try:
            if self._is_no_state_react_mode():
                run_report = self._init_run_report(question)
            else:
                with self._span("setup.question_profile") as _h_setup:
                    run_report = self._init_run_report(question)
                    if _h_setup is not None:
                        try:
                            _h_setup.attach_io(
                                output_preview=json.dumps(
                                    run_report.get("_question_profile") or {},
                                    ensure_ascii=False,
                                )
                            )
                        except Exception:
                            pass
            msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)
            run_report["_message_snapshot"] = copy.deepcopy(msgs)
            _prime_transcripts = getattr(
                self, "_append_transcript_priming_message", None
            )
            if _prime_transcripts is not None:
                _prime_transcripts(msgs, run_report)
            self._append_state_digest_message(msgs, run_report)
            run_report["_message_snapshot"] = copy.deepcopy(msgs)
            initial_message_count = len(msgs)
            empty_response_retries = 0
            repair_retry_attempted = False  # Per-call flag, not per-instance

            for i in range(self.max_iterations):
                # Round number for anything that budgets in rounds rather than
                # steps: the step recorder stamps it onto every step_record.
                run_report["_iteration_index"] = i + 1
                if i == self.max_iterations - 1:
                    msgs.append(
                        {
                            "role": "user",
                            "content": "Please call the `finish` function to finish the task.",
                        }
                    )

                # Weak-LLM sliding-window truncation (no-op when WEAK_LLM_MODE is off).
                dropped = self._truncate_history_for_weak_llm(msgs, initial_message_count)
                if dropped:
                    run_report.setdefault("repair_log", []).append({
                        "action": "weak_llm_history_truncate",
                        "iteration": i,
                        "dropped_messages": dropped,
                    })

                compacted = self._compact_planner_history(msgs, initial_message_count)
                if compacted:
                    run_report.setdefault("planner_history_compaction", []).append({
                        "iteration": i,
                        "compacted_observations": compacted,
                        "raw_window": PLANNER_RAW_OBSERVATION_WINDOW,
                    })

                # Defense-in-depth: validate message history before every LLM call
                pre_call_repairs = self._validate_message_history(msgs)
                if pre_call_repairs:
                    run_report.setdefault("repair_log", []).extend(pre_call_repairs)
                run_report["_message_snapshot"] = copy.deepcopy(msgs)
                is_last_iteration = i == self.max_iterations - 1
                finish_available = self._compute_finish_available(
                    run_report,
                    is_last_iteration=is_last_iteration,
                    ablation_mode=self._current_ablation_mode(),
                )
                active_schemas = (
                    self.function_schemas
                    if finish_available
                    else [
                        schema
                        for schema in self.function_schemas
                        if schema.get("function", {}).get("name") != "finish"
                    ]
                )

                with self._span("planner.decide") as _h_planner:
                    try:
                        response = call_openai_model_with_tools(
                            msgs,
                            endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                            model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                            tools=active_schemas,
                            temperature=0.0,
                            api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                            base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                            caller="orchestrator",
                        )
                    except LLMCallFailure as failure:
                        self._record_llm_call_failure(
                            run_report, failure, layer="planner"
                        )
                        response = None
                    self._attach_planner_span_fields(_h_planner, response, run_report)
                if response is None:
                    if (
                        not repair_retry_attempted
                        and not self._has_deterministic_provider_failure(run_report)
                    ):
                        repair_retry_attempted = True
                        retry_repairs = self._validate_message_history(msgs)
                        if retry_repairs:
                            run_report.setdefault("repair_log", []).extend(retry_repairs)
                        with self._span("planner.decide", retry=True) as _h_retry:
                            try:
                                response = call_openai_model_with_tools(
                                    msgs,
                                    endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                                    model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                                    tools=active_schemas,
                                    temperature=0.0,
                                    api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                                    base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                                    caller="orchestrator",
                                )
                            except LLMCallFailure as failure:
                                self._record_llm_call_failure(
                                    run_report, failure, layer="planner_repair_retry"
                                )
                                response = None
                            self._attach_planner_span_fields(_h_retry, response, run_report)
                    if response is None:
                        run_report["stop_reason"] = "orchestrator_unavailable"
                        self._best_effort_finalize(msgs, question, run_report)
                        self.last_run_report = run_report
                        if len(msgs) > initial_message_count or run_report.get("tool_sequence"):
                            return msgs
                        return msgs or None
                    # Reset on successful recovery
                    repair_retry_attempted = False

                response.setdefault("role", "assistant")
                msgs.append(response)
                run_report["_message_snapshot"] = copy.deepcopy(msgs)
                yield response
                pending_tool_calls = list(response.get("tool_calls") or [])
                if not pending_tool_calls:
                    synthetic_finish = self._maybe_promote_text_revision_to_finish(
                        response,
                        msgs,
                        run_report,
                    )
                    if synthetic_finish is not None:
                        pending_tool_calls = [synthetic_finish]
                        try:
                            promoted_args = json.loads(
                                synthetic_finish.get("function", {}).get("arguments", "{}")
                            )
                            promoted_answer = str(promoted_args.get("answer") or "").strip()
                            if promoted_answer:
                                current_step = len(run_report.get("tool_sequence") or [])
                                existing = run_report.get("best_candidate_answer")
                                if not existing or current_step > (existing.get("cached_at_step") or 0):
                                    run_report["best_candidate_answer"] = {
                                        "answer_text": promoted_answer,
                                        "sufficiency_status": str(
                                            (run_report.get("evidence_state") or {}).get("sufficiency_status", "")
                                        ),
                                        "cached_at_step": current_step,
                                        "source": "text_revision_promotion",
                                    }
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            pass
                    else:
                        synthetic_compute = self._maybe_promote_pending_compute_to_tool_call(
                            response,
                            msgs,
                            run_report,
                        )
                        if synthetic_compute is not None:
                            pending_tool_calls = [synthetic_compute]
                if not pending_tool_calls:
                    empty_response_retries += 1
                    if empty_response_retries <= MAX_EMPTY_RETRIES:
                        nudge_parts = ["You must call a tool or use `finish` to provide a final answer."]
                        if not self._is_history_only_mode():
                            latest_gap = (run_report.get("latest_critic_verdict") or {}).get("gap_diagnosis") or {}
                            gap_desc = str(latest_gap.get("description") or "").strip()
                            unresolved = list((run_report.get("latest_state_digest") or {}).get("unresolved_questions") or [])
                            if gap_desc:
                                nudge_parts.append(f"Current evidence gap: {gap_desc}")
                            if unresolved:
                                nudge_parts.append(f"Unresolved: {', '.join(str(q) for q in unresolved[:3])}")
                        msgs.append(
                            {
                                "role": "user",
                                "content": " ".join(nudge_parts),
                            }
                        )
                        run_report["_message_snapshot"] = copy.deepcopy(msgs)
                        continue
                    cached = run_report.get("best_candidate_answer") or {}
                    cached_text = str(cached.get("answer_text") or "").strip()
                    run_report["stop_reason"] = (
                        "empty_response_fallback"
                        if cached_text
                        else "empty_response_no_cached_answer"
                    )
                    self._best_effort_finalize(msgs, question, run_report)
                    self.last_run_report = run_report
                    return msgs

                # Item ③: admit the planner's batch before executing any of it.
                # Runs after the synthetic-finish/compute promotions above, which
                # always produce a 1-wide batch and therefore pass through
                # untouched. Blocked-step messages recorded here are yielded so a
                # streaming consumer sees the same history the transport will.
                composed_at = len(msgs)
                pending_tool_calls = self._compose_tool_call_batch(
                    pending_tool_calls, msgs, run_report, question
                )
                if len(msgs) > composed_at:
                    run_report["_message_snapshot"] = copy.deepcopy(msgs)
                    for new_message in msgs[composed_at:]:
                        yield new_message

                try:
                    for tool_call in pending_tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        yield {
                            "role": "tool_call",
                            "name": tool_name,
                            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                        }
                        previous_len = len(msgs)
                        full_args_str = str(
                            tool_call.get("function", {}).get("arguments") or ""
                        )
                        args_preview = full_args_str[:240]
                        stop_exc = None
                        with self._span(f"tool:{tool_name}") as _h_tool:
                            if _h_tool is not None:
                                _h_tool.extra["arguments_preview"] = args_preview
                            try:
                                self._exec_tool(tool_call, msgs, tool_history, question, run_report)
                            except StopException as exc:
                                stop_exc = exc
                            empty_response_retries = 0
                            self._attach_tool_span_fields(
                                _h_tool,
                                tool_name,
                                msgs[previous_len:],
                                run_report,
                                full_args=full_args_str,
                            )
                        if stop_exc is not None:
                            raise stop_exc
                        for new_message in msgs[previous_len:]:
                            yield new_message
                        final_kind = str(run_report.get("final_response_kind") or "").strip()
                        if final_kind and final_kind != "incomplete":
                            if (
                                final_kind != "answer"
                                or not str(run_report.get("final_output_text") or "").strip()
                            ):
                                self._best_effort_finalize(msgs, question, run_report)
                            self.last_run_report = run_report
                            return msgs
                except StopException:
                    if run_report.get("final_response_kind", "incomplete") == "incomplete":
                        run_report["stop_reason"] = (
                            run_report.get("stop_reason") or "tool_call_aborted_best_effort"
                        )
                        self._best_effort_finalize(msgs, question, run_report)
                    self.last_run_report = run_report
                    return msgs

            if not self._is_no_state_react_mode():
                answer_phase_messages = self._run_answer_phase(
                    msgs,
                    tool_history,
                    question,
                    run_report,
                )
                for message in answer_phase_messages:
                    yield message

            if run_report.get("final_response_kind", "incomplete") == "incomplete":
                run_report["stop_reason"] = (
                    run_report.get("stop_reason") or "answer_phase_best_effort"
                )
                self._best_effort_finalize(msgs, question, run_report)
            self.last_run_report = run_report
            return msgs
        except BaseException as exc:
            _final_status = "error"
            _error_type = type(exc).__name__
            _error_msg = repr(exc)[:240]
            raise
        finally:
            if self._step_logger is not None:
                final_answer: str | None = None
                extra_payload: dict[str, Any] = {}
                if isinstance(run_report, dict):
                    final_answer = str(run_report.get("final_output_text") or "") or None
                    stop_reason = str(run_report.get("stop_reason") or "").strip()
                    if stop_reason:
                        extra_payload["stop_reason"] = stop_reason
                if _final_status == "error":
                    extra_payload["error_type"] = _error_type
                    extra_payload["error_msg"] = _error_msg
                try:
                    self._step_logger.case_end(
                        case_id,
                        status=_final_status,
                        answer=final_answer,
                        token_snapshot=token_tracker.snapshot(),
                        extra=extra_payload or None,
                    )
                except Exception:
                    pass
            self._pending_case_id = None

    def run(self, question) -> list[dict] | None:
        """Run the ReAct-style loop with OpenAI function calling."""
        iterator = self._execute_loop(question)
        try:
            while True:
                next(iterator)
        except StopIteration as stop:
            # Attach token usage snapshot to run report
            if hasattr(self, "last_run_report") and isinstance(self.last_run_report, dict):
                self.last_run_report["token_usage"] = token_tracker.snapshot()
                self.last_run_report["provider_failure_summary"] = (
                    provider_failure_ledger.snapshot()
                )
            return stop.value

    def stream_run(self, question):
        """Yield assistant and tool messages during execution."""
        return (yield from self._execute_loop(question))
