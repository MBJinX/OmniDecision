"""Synthetic tool-call construction and inline answer revision."""

import copy
import json
from typing import Any

import agents.config as config
from agents.shared.utils import (
    LLMCallFailure,
    call_openai_model_with_tools,
    extract_answer,
)


class SyntheticTools:
    """Provides synthetic finish promotion and inline answer revision helpers.

    Compute promotion is disabled under the diagnostic-only answer critic schema.
    """

    def _build_synthetic_finish_tool_call(
        self,
        answer: str,
        run_report: dict[str, Any],
    ) -> dict[str, Any]:
        synthetic_id = f"synthetic_finish_{len(run_report.get('answer_checkpoints') or []) + 1}"
        return {
            "id": synthetic_id,
            "type": "function",
            "function": {
                "name": "finish",
                "arguments": json.dumps({"answer": answer}, ensure_ascii=False),
            },
        }

    def _maybe_promote_pending_compute_to_tool_call(
        self,
        response: dict[str, Any],
        msgs: list[dict[str, Any]],
        run_report: dict[str, Any],
    ) -> dict[str, Any] | None:
        return None

    def _attach_tool_call_to_latest_assistant(
        self,
        tool_call: dict[str, Any],
        msgs: list[dict[str, Any]],
    ) -> bool:
        tool_call_id = str(tool_call.get("id") or "").strip()
        for message in reversed(msgs):
            if message.get("role") != "assistant":
                continue
            existing_tool_calls = list(message.get("tool_calls") or [])
            if tool_call_id and any(
                str(existing.get("id") or "").strip() == tool_call_id
                for existing in existing_tool_calls
                if isinstance(existing, dict)
            ):
                return True
            existing_tool_calls.append(tool_call)
            message["tool_calls"] = existing_tool_calls
            return True
        return False

    def _maybe_promote_text_revision_to_finish(
        self,
        response: dict[str, Any],
        msgs: list[dict[str, Any]],
        run_report: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(response, dict) or response.get("tool_calls"):
            return None

        latest_answer_verdict = run_report.get("latest_answer_verdict") or {}
        if latest_answer_verdict.get("verdict") != "BLOCK":
            return None

        revised_answer = extract_answer(response)
        if not revised_answer:
            return None

        synthetic_tool_call = self._build_synthetic_finish_tool_call(
            revised_answer,
            run_report,
        )
        if not self._attach_tool_call_to_latest_assistant(synthetic_tool_call, msgs):
            return None
        return synthetic_tool_call

    def _should_try_inline_answer_revision(
        self,
        checkpoint: dict[str, Any],
        run_report: dict[str, Any],
    ) -> bool:
        if checkpoint.get("verdict") != "BLOCK":
            return False
        if int(run_report.get("_inline_answer_revision_count") or 0) >= 1:
            return False
        return True

    def _attempt_inline_answer_revision(
        self,
        msgs: list[dict[str, Any]],
        tool_history: dict[str, int],
        question: str,
        run_report: dict[str, Any],
    ) -> bool:
        run_report["_inline_answer_revision_count"] = (
            int(run_report.get("_inline_answer_revision_count") or 0) + 1
        )
        try:
            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                caller="synthetic_tools",
            )
        except LLMCallFailure:
            # An inline revision the provider refuses is not retried here; the
            # caller treats a failed revision the same as a declined one.
            return False
        if response is None:
            return False

        response.setdefault("role", "assistant")
        msgs.append(response)
        run_report["_message_snapshot"] = copy.deepcopy(msgs)
        pending_tool_calls = list(response.get("tool_calls") or [])
        if not pending_tool_calls:
            synthetic_finish = self._maybe_promote_text_revision_to_finish(
                response,
                msgs,
                run_report,
            )
            if synthetic_finish is not None:
                pending_tool_calls = [synthetic_finish]

        for followup_tool_call in pending_tool_calls:
            self._exec_tool(
                followup_tool_call,
                msgs,
                tool_history,
                question,
                run_report,
            )
            if str(run_report.get("final_response_kind") or "").strip() != "incomplete":
                break
        return True
