"""Shared type definitions for OmniVQA grounding, verification, and protocol state."""

from typing import Any, Literal, TypedDict


GroundingModality = Literal[
    "audio",
    "subtitle",
    "clip",
    "visual",
    "web",
    "computation",
    "unknown",
]
ObservationStatus = Literal["ok", "no_match", "unavailable", "guardrail", "error"]
ToolRole = Literal[
    "grounding_video",
    "grounding_external",
    "visual_verify",
    "computation",
    "finish",
    "unknown",
]
ProtocolStage = Literal[
    "start",
    "after_grounding",
    "after_visual_verification",
    "ready_to_finish",
    "answer_revision",
    "stopped",
]
ActionReasonSource = Literal[
    "policy",
    "critic",
    "answer_critic",
    "guardrail",
    "orchestrator",
    "unknown",
]
EstimatedCostTier = Literal["low", "medium", "high"]
SufficiencyStatus = Literal[
    "seeking_grounding",
    "evidence_collected",
    "needs_reground",
    "needs_computation",
    "ready_to_finish",
    "answered",
]
DecisionAction = Literal[
    "ground",
    "verify",
    "compute",
    "reground",
    "answer",
    "stop_insufficient",
    "revise_answer",
    "unknown",
]


class EstimatedCost(TypedDict, total=False):
    tier: EstimatedCostTier
    score: float
    high_cost_units: int
    rationale: str


class ActionTraceItem(TypedDict, total=False):
    step_index: int
    selected_tool: str
    tool_role: ToolRole
    stage_before: ProtocolStage
    stage_after: ProtocolStage
    reason_source: ActionReasonSource
    reason_text: str
    critic_decision: str
    blocked: bool
    observation_summary: str


class TimeAnchor(TypedDict, total=False):
    start_time: str
    end_time: str
    confidence: float
    source_tool: str
    source_modality: GroundingModality
    rationale: str


class EvidenceAtom(TypedDict, total=False):
    evidence_id: str
    source_tool: str
    source_modality: GroundingModality
    text: str
    confidence: float
    anchor: TimeAnchor
    evidence_kind: Literal[
        "candidate_range",
        "text",
        "visual_observation",
        "web_fact",
        "computation_result",
        "guardrail",
        "summary",
    ]
    metadata: dict
    asset_id: str


class ConflictMarker(TypedDict, total=False):
    conflict_id: str
    conflict_type: Literal[
        "temporal_mismatch",
        "entity_mismatch",
        "cross_modal_mismatch",
        "unsupported_grounding",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    evidence_ids: list[str]


class NormalizedToolObservation(TypedDict, total=False):
    status: ObservationStatus
    source_tool: str
    source_modality: GroundingModality
    tool_role: ToolRole
    summary: str
    confidence: float
    estimated_cost: EstimatedCost
    fallback_hint: str
    candidate_ranges: list[TimeAnchor]
    evidence_atoms: list[EvidenceAtom]
    source_uncertainty: list[str]
    raw_payload: dict


class EvidenceStateDigest(TypedDict, total=False):
    query_id: str
    video_id: str
    current_stage: ProtocolStage
    sufficiency_status: SufficiencyStatus
    unresolved_questions: list[str]
    uncertainty_summary: dict
    bridge_status: dict
    budget_state: dict
    candidate_range_preview: list[TimeAnchor]
    recent_evidence: list[dict]
    conflict_count: int
    evidence_atom_count: int
    stop_ready: bool
    evidence_needs: list[dict]
    actionable_need_indices: list[int]
    blocked_need_indices: list[int]
    decision_state: dict
    gap_diagnosis: dict
    entity_cards: list[dict]
    fact_bridge_records: list[dict]
    graph_summary: str   # LLM-built evidence graph summary (≤120 words)


class EntityCard(TypedDict, total=False):
    entity_id: str          # sha1(canonical_name.lower())[:10]
    canonical_name: str     # normalized primary name, e.g. "Mevo Plus"
    aliases: list[str]      # alternative names
    entity_type: str        # "product" | "person" | "song" | "event" | "unknown"
    source_tool: str        # originating tool
    source_text: str        # triggering raw text (truncated to 200 chars)
    lookup_status: str      # "pending" | "in_flight" | "resolved" | "failed"
    lookup_slot: str        # attribute slot to query, e.g. "battery_capacity_mah"
    resolved_fact: str      # conclusion after external lookup, e.g. "1200 mAh"
    resolved_at_step: int


class FactBridgeRecord(TypedDict, total=False):
    record_id: str          # sha1(entity_id + query_used)[:10]
    entity_id: str
    query_type: str         # "fact_completion" | "numeric_lookup" | "background_knowledge"
    query_used: str         # actual query sent to web_search_tool
    raw_result_summary: str # web observation summary (truncated to 300 chars)
    extracted_value: str    # extracted key value, e.g. "1200" or "1937"
    confidence: float
    source_urls: list[str]  # up to 3 source URLs
    status: str             # "ok" | "no_match" | "ambiguous" | "error"
    created_at_step: int


VALID_PREFERRED_SOURCES = frozenset(
    {"visual", "audio", "subtitle", "web", "computation"}
)


class EvidenceNeed(TypedDict, total=False):
    need: str
    preferred_source: str
    closure_level: str
    status: str
    filled_by: str
    need_role: str
    depends_on: list[int]
    query_focus: str | None


# Answer Readiness
# These must be defined BEFORE EvidenceState so EvidenceState can reference them.

class AnswerCandidate(TypedDict, total=False):
    """Structured answer produced by the agent before final emission."""
    candidate_id: str           # sha1(answer_text)[:10]
    answer_text: str            # raw answer string
    normalized_text: str        # lower-cased, punctuation-stripped form
    answer_kind: Literal[
        "short_text",           # single entity / yes-no
        "numeric",              # a number or range
        "date_or_duration",     # a date, year, or time span
        "list",                 # comma-separated items
        "sentence",             # full-sentence answer
        "unknown",
    ]
    supporting_evidence_ids: list[str]  # EvidenceAtom.evidence_id refs
    confidence: float
    produced_at_step: int


class MemorySlot(TypedDict, total=False):
    """Deduplication slot for a single tool invocation within one run."""
    slot_id: str            # sha1(tool_name + normalized_query)[:10]
    tool_name: str
    normalized_query: str   # lowercase, stripped query used as dedup key
    result_summary: str     # short text summary of the cached result
    full_payload: dict      # full raw payload for re-injection
    created_at_step: int
    hit_count: int          # how many times this slot was reused


class EvidenceGraphNode(TypedDict, total=False):
    """A single node in the local evidence graph."""
    node_id: str                    # same as EvidenceAtom.evidence_id
    label: str                      # short human-readable description
    source_tool: str
    confidence: float


class EvidenceGraphEdge(TypedDict, total=False):
    """A directed relationship between two evidence nodes or a node and the query."""
    from_id: str    # node_id or "query"
    to_id: str      # node_id or "answer"
    edge_type: Literal[
        "supports",             # evidence supports the answer
        "conflicts",            # evidence conflicts with another node
        "requires_computation", # node value needs further computation
        "requires_lookup",      # node needs an external fact lookup
        "chain",                # temporal / causal chain link
    ]
    weight: float               # 0.0–1.0 importance of this edge
    rationale: str


class EvidenceGraph(TypedDict, total=False):
    """Local evidence graph for one query run."""
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]
    graph_summary: str          # LLM-generated text summary (≤120 words)
    conflict_paths: list[list[str]]  # node_id paths that form conflicts
    support_paths: list[list[str]]   # node_id paths that support the answer
    built_at_step: int


AssetType = Literal["video", "audio", "image"]


class EvidenceState(TypedDict, total=False):
    query_id: str
    video_id: str
    current_stage: ProtocolStage
    candidate_ranges: list[TimeAnchor]
    evidence_atoms: list[EvidenceAtom]
    conflicts: list[ConflictMarker]
    unresolved_questions: list[str]
    uncertainty_summary: dict
    bridge_status: dict
    budget_state: dict
    sufficiency_status: SufficiencyStatus
    stop_ready: bool
    decision_state: dict
    latest_observation: NormalizedToolObservation
    latest_state_delta: dict
    entity_cards: list[EntityCard]
    fact_bridge_records: list[FactBridgeRecord]
    pending_entity_lookups: list[str]
    evidence_needs: list[EvidenceNeed]
    # Additional readiness fields
    memory_slots: list[MemorySlot]
    evidence_graph: EvidenceGraph
    answer_candidates: list[AnswerCandidate]


class DecisionRecord(TypedDict, total=False):
    step_index: int
    selected_tool: str
    tool_role: ToolRole
    stage_before: ProtocolStage
    stage_after: ProtocolStage
    reason_source: ActionReasonSource
    reason_text: str
    blocked: bool
    critic_decision: str
    evidence_state_digest: EvidenceStateDigest


class StepRecord(TypedDict, total=False):
    step_index: int
    action: DecisionRecord
    tool_arguments: dict
    raw_observation: dict | str
    normalized_observation: NormalizedToolObservation
    evidence_delta: dict
    critic_verdict: dict[str, Any] | None
    answer_verdict: dict[str, Any] | None
    state_after: EvidenceState
    step_cost: EstimatedCost
    step_elapsed_ms: int
    state_changed: bool
    round_index: int  # planner round that produced this step (progress guard budgets in rounds)
