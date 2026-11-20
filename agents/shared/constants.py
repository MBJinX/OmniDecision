"""Shared protocol constants for the OmniVQA runtime."""

from collections import Counter

# ---------------------------------------------------------------------------
# Tool classification sets
# ---------------------------------------------------------------------------
VIDEO_GROUNDING_TOOLS = {
    "audio_scout_tool",
    "subtitle_grounding_tool",
    "clip_grounding_tool",
}
EXTERNAL_LOOKUP_TOOLS = {"web_search_tool"}
COMPUTATION_TOOLS = {"code_executor_tool"}
GROUNDING_TOOLS = VIDEO_GROUNDING_TOOLS | EXTERNAL_LOOKUP_TOOLS
VISUAL_CONFIRM_TOOLS = {"frame_confirm_tool"}

# ---------------------------------------------------------------------------
# Critic / answer-critic verdict enums
# ---------------------------------------------------------------------------
# Evidence critic verdicts
CRITIC_VERDICTS = {"SUFFICIENT", "INSUFFICIENT", "CONFLICTING"}

# Answer critic verdicts
ANSWER_CHECK_VERDICTS = {"PASS", "BLOCK"}

# ---------------------------------------------------------------------------
# Engineering thresholds
# ---------------------------------------------------------------------------
ENTITY_NO_MATCH_STOP_THRESHOLD = 2
CONSECUTIVE_STALE_STEP_THRESHOLD = 3  # Consecutive stale steps before strategy switch injection
BATCH_MAX_WIDTH = 4

# ---------------------------------------------------------------------------
# Truncation limits (character caps for display/payload fields)
# ---------------------------------------------------------------------------
TRUNC_SOURCE_TEXT = 200              # entity_bridge: source_text
TRUNC_RESULT_SUMMARY = 300          # entity_bridge, memory: raw_result_summary, result_summary
TRUNC_PROPOSED_ANSWER_PREVIEW = 240 # critic: proposed_answer_preview
TRUNC_TOOL_CONTENT_SUMMARY = 400   # critic: tool_content fallback summary
TRUNC_EVIDENCE_BUNDLE_TEXT = 420    # web_evidence: evidence bundle text
TRUNC_GRAPH_SUMMARY = 500          # evidence_graph: graph_summary
TRUNC_KEY_FINDINGS_PREVIEW = 360
TRUNC_EVIDENCE_BUNDLE_TITLE = 180  # web_evidence: evidence bundle title
TRUNC_WEB_EVIDENCE_TO_LLM = 1500  # entity_bridge: web_evidence_text to LLM
TRUNC_MEMORY_RESULT_TEXT = 2000    # agent: memory slot result_text
TRUNC_GRAPH_NODE_LABEL = 80        # evidence_graph: graph node label
TRUNC_BEST_EFFORT_ANSWER_SPAN = 240  # agent: last-assistant span reused as the terminal answer

COMPACTED_OBSERVATION_MARKER = "_planner_history_compacted"

# ---------------------------------------------------------------------------
# Answer policy
# ---------------------------------------------------------------------------
# Terminal string for the one branch where nothing in the run committed to a
# value. It must not match _is_refusal_or_abort_answer: the previous static
# string opened with "I could not determine", which the refusal detector
# classifies as an abstention, so the last line of defense emitted exactly what
# the answer policy forbids. Reaching this branch is a defect signal — read
# stop_reason and best_effort_answer_source in attribution.
NO_VERIFIED_VALUE_ANSWER = "No verified value was obtained."

# ---------------------------------------------------------------------------
# Window sizes (sliding window lengths for history arrays)
# ---------------------------------------------------------------------------
WINDOW_ACCUMULATED_OBS = 6          # critic: accumulated/normalized observations
WINDOW_RECENT_FEEDBACK = 3          # critic: recent feedback/step window
WINDOW_EVIDENCE_SIGNATURE_OBS = 8   # agent: evidence signature observation window
WINDOW_EVIDENCE_ATOMS_CAP = 24      # state_machine: evidence atoms cap
WINDOW_CANDIDATE_RANGES_CAP = 6     # state_machine: candidate ranges cap
WINDOW_ATOMS_PER_OBSERVATION = 8    # state_machine: evidence atoms per observation cap
WINDOW_NORMALIZE_ITEMS = 3          # state_machine: candidate range/evidence items per normalization
TOTAL_ATOM_BUDGET = 24              # state_machine: multi-asset digest total atom budget

# ---------------------------------------------------------------------------
# Thresholds (behavioral decision boundaries)
# ---------------------------------------------------------------------------
REPEAT_GUARD_MAX = 2                    # agent: repeat call detection threshold
MIN_TOOL_CALLS_BEFORE_FINISH = 6       # agent: minimum tool calls before finish allowed
WEAK_GROUNDING_CONFIDENCE = 0.45       # state_machine: weak grounding confidence threshold
TERMINAL_FACT_MAX_SCORE = 0.75         # web_evidence: max_score threshold for terminal fact
TEMPORAL_ANCHOR_COUNT_THRESHOLD = 2    # state_machine: temporal anchor count threshold
EVIDENCE_ATOM_COUNT_THRESHOLD = 2      # state_machine: evidence atom count threshold
TOOL_COUNT_EVIDENCE_COLLECTED = 3      # state_machine: tool count for evidence_collected promotion

COUNTABLE_TOOL_NAMES = (
    "subtitle_grounding_tool",
    "clip_grounding_tool",
    "frame_confirm_tool",
    "audio_scout_tool",
    "web_search_tool",
    "code_executor_tool",
)


def build_tool_call_counts(tool_sequence: list | None) -> dict[str, int]:
    """Build a compact tool call count dict from a tool_sequence list."""
    seq = list(tool_sequence) if tool_sequence else []
    counts = Counter(name for name in seq if isinstance(name, str))
    return {
        canonical.replace("_tool", ""): counts.get(canonical, 0)
        for canonical in COUNTABLE_TOOL_NAMES
    }
