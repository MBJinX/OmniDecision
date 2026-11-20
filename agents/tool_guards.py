"""Tool call guardrails and blocking logic."""

import json
import re
from typing import Any

# How many times a blocked answer may reopen the loop to acquire a missing input
# before the revision cap applies again.
#
# `MAX_ANSWER_REVISIONS` counts *rewrites*: after one blocked attempt the run
# force-accepts. That is the right budget for a presentation block, and the wrong
# one for a missing-input block, where the critic's hint can only be satisfied by
# a tool call. Charging an acquisition against the rewrite budget force-accepts
# the agent for doing what it was told, so reopens carry their own allowance of 1
# — worst case one extra acquisition round per case, not an unbounded loop.
#
# Lives here rather than in `agent.py` beside `MAX_ANSWER_REVISIONS` because the
# guard that consults it is here, and importing it from `agent.py` would need a
# deferred import inside the predicate: `agent.py` imports this module.
MAX_ANSWER_REVISION_REOPENS = 1

# Tools that can add evidence atoms to the state. A blocked answer whose critic
# hint asks for a missing input can only satisfy it by calling one of these.
# `finish` is added at the call site, since it is always permitted.
EVIDENCE_ACQUIRING_TOOLS = (
    "web_search_tool",
    "audio_scout_tool",
    "subtitle_grounding_tool",
    "clip_grounding_tool",
    "frame_confirm_tool",
)

# The answer critic's own vocabulary for "you are missing an input", taken from
# the phrasings its prompt directs it to produce ("use the sourced value already
# in evidence if one exists, otherwise acquire it before finishing"). This reads
# the critic's verdict.
_ACQUISITION_DEMAND = re.compile(
    r"\b(acquire|cite a|search|look ?up|retrieve|obtain"
    r"|unsupported|not supported by any evidence"
    r"|is not in the accumulated evidence|no evidence atom"
    r"|(is|are) not identified in|ground(ed)? (it|this|the)"
    r"|verify (via|with|by))",
    re.I,
)


def _hint_demands_acquisition(revision_hint: str) -> bool:
    """Whether a BLOCK's revision hint asks for evidence rather than a rewrite.

    A hint naming a missing or unsourced input cannot be satisfied by rewording
    the answer, so the guard has to let the agent go and fetch it. A hint about
    which parts to report, or how to present them, is satisfiable in place.
    """
    return bool(_ACQUISITION_DEMAND.search(str(revision_hint or "")))


class ToolGuards:
    """Provides repeat/finish/answer guard observations and auto-stop-insufficient stub."""

    # ------------------------------------------------------------------
    # Repeat / finish / answer / web-search guard observations
    # ------------------------------------------------------------------

    def _build_repeat_guard_observation(self, name: str) -> str:
        return json.dumps(
            {
                "status": "guardrail",
                "source_modality": "unknown",
                "summary": f"Repeated tool call blocked for `{name}` because the same arguments were already used multiple times in this run.",
                "fallback_hint": "Choose a different tool, change the arguments, or finish with the available evidence.",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_answer_guard_observation(self, verdict: dict[str, Any]) -> str:
        revision_hint = verdict.get(
            "revision_hint",
            "Revise the answer to be more specific and evidence-based.",
        )
        if _hint_demands_acquisition(revision_hint):
            directive = (
                "IMPORTANT: This is a missing-evidence block, not a wording problem. "
                "Call an evidence tool to acquire the input named above, then call "
                "finish again with the sourced value. If the acquisition attempt "
                "fails, finish with your best-supported value and say which source "
                "it came from — do NOT weaken the answer into 'I cannot determine'."
            )
        else:
            directive = (
                "IMPORTANT: Revise towards a MORE SPECIFIC, evidence-supported answer. "
                "Do NOT weaken the answer into 'I cannot determine' or 'insufficient evidence' "
                "if you have collected evidence that points to an answer."
            )
        return json.dumps(
            {
                "status": "guardrail",
                "source_modality": "unknown",
                "summary": (
                    "The proposed final answer did not pass the "
                    f"OmniVQA {self._agent_label()} answer checkpoint."
                ),
                "fallback_hint": f"{revision_hint} {directive}",
                "answer_verdict": verdict,
            },
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Answer-action guard (post-answer-checkpoint tool constraint)
    # ------------------------------------------------------------------

    def _pending_answer_action_guard(
        self,
        msgs: list[dict],
        run_report: dict[str, Any],
    ) -> dict[str, Any] | None:
        latest_answer_verdict = run_report.get("latest_answer_verdict") or {}
        verdict = str(latest_answer_verdict.get("verdict") or "").strip().upper()
        if verdict != "BLOCK":
            return None

        checkpoint_signature = self._latest_answer_checkpoint_signature(run_report)
        if checkpoint_signature:
            current_signature = self._build_evidence_signature(msgs)
            if current_signature != checkpoint_signature:
                return None

        hint = str(latest_answer_verdict.get("revision_hint") or "").strip()
        # A block that names a missing input has to be allowed to go and get it.
        # Restricting the agent to `finish` under unchanged evidence is right for
        # a presentation block — reword and try again — but for a missing-input
        # block it is a deadlock: evidence cannot change without a tool call, and
        # no tool call but `finish` is permitted, so the only exit is to
        # re-answer with the same gap.
        #
        # This predicate stays read-only. The budget it consults is written in one
        # place — the reopen accounting in `agent.py` when an acquisition tool
        # actually runs after a block — so merely attempting `finish` cannot spend
        # the allowance that the following acquisition needs.
        if _hint_demands_acquisition(hint) and self._answer_block_reopen_available(
            run_report
        ):
            allowed_tools = {"finish", *EVIDENCE_ACQUIRING_TOOLS}
            reason = (
                "The latest answer checkpoint blocked the answer for a missing or "
                "unsourced input. Acquire that input, then finish."
            )
        else:
            allowed_tools = {"finish"}
            reason = (
                "The latest answer checkpoint blocked the answer under unchanged evidence. "
                "Revise the answer and try finish again."
            )

        return {
            "verdict": verdict,
            "allowed_tools": sorted(allowed_tools),
            "reason": reason,
            "revision_hint": hint or "Revise the answer based on accumulated evidence.",
            "answer_verdict": latest_answer_verdict,
        }

    # ------------------------------------------------------------------
    # Reopen budget for missing-input blocks
    # ------------------------------------------------------------------

    def _answer_block_reopen_available(self, run_report: dict[str, Any]) -> bool:
        """Whether a blocked answer may still reopen the loop to acquire an input.

        Read-only: the counter is written in one place, when an acquisition tool
        actually runs after a block, so a `finish` attempt cannot spend the
        allowance the following acquisition needs.
        """
        spent = int(run_report.get("answer_revision_reopen_count") or 0)
        return spent < MAX_ANSWER_REVISION_REOPENS

    def _note_answer_block_reopen(
        self,
        run_report: dict[str, Any],
        tool_name: str,
    ) -> None:
        """Record that a blocked answer went back for evidence instead of rewording.

        Updates `answer_revision_reopen_count` when
        an evidence tool runs while the latest answer verdict is a BLOCK whose
        hint asked for acquisition — that is the reopen this budget governs.
        """
        if tool_name not in EVIDENCE_ACQUIRING_TOOLS:
            return
        latest = run_report.get("latest_answer_verdict") or {}
        if str(latest.get("verdict") or "").strip().upper() != "BLOCK":
            return
        if not _hint_demands_acquisition(latest.get("revision_hint")):
            return
        run_report["answer_revision_reopen_count"] = (
            int(run_report.get("answer_revision_reopen_count") or 0) + 1
        )

    def _build_answer_action_guard_observation(
        self,
        guard: dict[str, Any],
        attempted_tool: str,
    ) -> str:
        allowed_tools = list(guard.get("allowed_tools") or [])
        allowed_text = ", ".join(allowed_tools) if allowed_tools else "no tool call"
        return json.dumps(
            {
                "status": "guardrail",
                "source_modality": "unknown",
                "summary": (
                    f"The attempted tool `{attempted_tool}` does not match the latest "
                    "OmniVQA answer-check verdict."
                ),
                "fallback_hint": (
                    f"Allowed next action under unchanged evidence: {allowed_text}. "
                    + str(guard.get("revision_hint") or "").strip()
                ).strip(),
                "answer_action_guard": {
                    "verdict": guard.get("verdict"),
                    "allowed_tools": allowed_tools,
                    "reason": guard.get("reason"),
                },
                "answer_verdict": guard.get("answer_verdict") or {},
            },
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Auto-stop insufficient
    # ------------------------------------------------------------------

    def _should_auto_stop_insufficient(
        self,
        run_report: dict[str, Any],
    ) -> bool:
        """Compatibility hook."""
        return False
