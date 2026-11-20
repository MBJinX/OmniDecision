"""Question constraint profile inference."""

import os
import re
import sys
from typing import Any

import agents.config as config
from agents.shared.json_utils import safe_load_json_object
from agents.shared.types import VALID_PREFERRED_SOURCES
from agents.shared.utils import LLMCallFailure, call_openai_model_with_tools


DEFAULT_BOOLEAN_PROFILE: dict[str, Any] = {
    "_schema": "boolean",
    "requires_exact_temporal": False,
    "requires_exact_numeric": False,
    "likely_entity_bridge_required": False,
    "prefers_subtitle_grounding": False,
    "prefers_audio_grounding": False,
    "prefers_visual_grounding": False,
    "requires_external_lookup": False,
}

FALLBACK_EVIDENCE_NEEDS = [
    {
        "need": "ground relevant evidence for the question",
        "preferred_source": "visual",
        "closure_level": "blocking",
    },
    {
        "need": "verify or supplement evidence from available sources",
        "preferred_source": "subtitle",
        "closure_level": "supplementary",
    },
]

PREFERRED_SOURCE_ALIASES = {
    "external": "web",
    "web_search": "web",
    "text": "subtitle",
    "video": "visual",
    "math": "computation",
}

TEMPORAL_KEYWORDS = (
    "temporal",
    "date",
    "day",
    "time",
    "year",
    "month",
    "duration",
    "when",
    "timeline",
    "span",
)
NUMERIC_KEYWORDS = (
    "numeric",
    "number",
    "count",
    "total",
    "amount",
    "score",
    "rank",
    "ratio",
    "math",
    "compute",
)
ENTITY_BRIDGE_KEYWORDS = (
    "entity",
    "identity",
    "person",
    "product",
    "model",
    "brand",
    "song",
    "event",
    "who",
)
BOOLEAN_SOURCE_MAP = {
    "subtitle": "prefers_subtitle_grounding",
    "audio": "prefers_audio_grounding",
    "visual": "prefers_visual_grounding",
    "web": "requires_external_lookup",
}
VALID_CLOSURE_LEVELS = frozenset({"blocking", "supplementary"})
VALID_NEED_ROLES = frozenset({"anchor", "external_fact", "applicability", "compute"})


def _boolean_profile_defaults() -> dict[str, Any]:
    return dict(DEFAULT_BOOLEAN_PROFILE)


def _normalize_preferred_source(raw_source: Any) -> str:
    source = str(raw_source or "").strip().lower()
    if source in VALID_PREFERRED_SOURCES:
        return source
    return PREFERRED_SOURCE_ALIASES.get(source, "visual")


def _normalize_closure_level(raw_level: Any) -> str:
    level = str(raw_level or "").strip().lower()
    if level in VALID_CLOSURE_LEVELS:
        return level
    return "blocking"


def _normalize_need_role(raw: Any) -> str:
    role = str(raw or "").strip().lower()
    if role in VALID_NEED_ROLES:
        return role
    return "anchor"


def _normalize_depends_on(raw: Any, self_index: int, max_index: int) -> list[int]:
    """Normalize dependency indices while filtering invalid and self references."""
    if not isinstance(raw, list):
        return []

    normalized: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if 0 <= item < max_index and item != self_index:
            normalized.append(item)
    return normalized


def _normalize_query_focus(raw: Any) -> str | None:
    if raw is None:
        return None
    query_focus = str(raw).strip()[:100]
    return query_focus or None


def _clear_cycle_participants(needs: list[dict[str, Any]]) -> None:
    """Clear depends_on only for nodes participating in dependency cycles."""
    node_count = len(needs)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    self_loops: set[int] = set()

    for idx, need in enumerate(needs):
        for dep in list(need.get("depends_on") or []):
            if isinstance(dep, bool) or not isinstance(dep, int):
                continue
            if 0 <= dep < node_count:
                adjacency[dep].append(idx)
                if dep == idx:
                    self_loops.add(idx)

    index = 0
    indices = [-1] * node_count
    lowlink = [0] * node_count
    stack: list[int] = []
    on_stack = [False] * node_count
    cycle_nodes: set[int] = set()

    def strongconnect(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in adjacency[node]:
            if indices[neighbor] == -1:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif on_stack[neighbor]:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] != indices[node]:
            return

        component: list[int] = []
        while stack:
            member = stack.pop()
            on_stack[member] = False
            component.append(member)
            if member == node:
                break

        if len(component) > 1 or any(member in self_loops for member in component):
            cycle_nodes.update(component)

    for idx in range(node_count):
        if indices[idx] == -1:
            strongconnect(idx)

    for idx in cycle_nodes:
        needs[idx]["depends_on"] = []


def _validate_evidence_needs(raw_needs: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_needs, list):
        return None

    valid: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    for item in raw_needs:
        if not isinstance(item, dict):
            continue
        need_text = str(item.get("need") or "").strip()
        if not need_text:
            continue
        valid.append(
            {
                "need": need_text,
                "preferred_source": _normalize_preferred_source(
                    item.get("preferred_source")
                ),
                "closure_level": _normalize_closure_level(item.get("closure_level")),
            }
        )
        raw_items.append(item)

    if len(valid) < 2 or len(valid) > 5:
        return None

    for idx, entry in enumerate(valid):
        raw_item = raw_items[idx]
        entry["need_role"] = _normalize_need_role(raw_item.get("need_role"))
        entry["depends_on"] = _normalize_depends_on(
            raw_item.get("depends_on"), idx, len(valid)
        )
        entry["query_focus"] = _normalize_query_focus(raw_item.get("query_focus"))

    _clear_cycle_participants(valid)
    return valid


def _iter_evidence_needs(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(profile, dict):
        return []
    if profile.get("_schema") != "evidence_needs":
        return []
    needs = profile.get("evidence_needs")
    if not isinstance(needs, list):
        return []
    return [need for need in needs if isinstance(need, dict)]


def _need_text_matches(need: dict[str, Any], keywords: tuple[str, ...]) -> bool:
    haystack = str(need.get("need") or "").strip().lower()
    return any(keyword in haystack for keyword in keywords)


def _evidence_needs_prefer_source(
    profile: dict[str, Any] | None,
    source: str,
    unfilled_only: bool = True,
) -> bool:
    normalized_source = _normalize_preferred_source(source)
    needs = _iter_evidence_needs(profile)
    if needs:
        for need in needs:
            if need.get("preferred_source") != normalized_source:
                continue
            status = str(need.get("status") or "unfilled").strip().lower()
            if not unfilled_only or status != "filled":
                return True
        return False

    if not isinstance(profile, dict):
        return False
    field_name = BOOLEAN_SOURCE_MAP.get(normalized_source)
    if not field_name:
        return False
    return bool(profile.get(field_name))


def _evidence_needs_require_entity_bridge(profile: dict[str, Any] | None) -> bool:
    needs = _iter_evidence_needs(profile)
    if needs:
        return any(
            need.get("preferred_source") == "web"
            and _need_text_matches(need, ENTITY_BRIDGE_KEYWORDS)
            for need in needs
        )
    if not isinstance(profile, dict):
        return False
    return bool(profile.get("likely_entity_bridge_required"))


def _evidence_needs_require_temporal(profile: dict[str, Any] | None) -> bool:
    needs = _iter_evidence_needs(profile)
    if needs:
        return any(_need_text_matches(need, TEMPORAL_KEYWORDS) for need in needs)
    if not isinstance(profile, dict):
        return False
    return bool(profile.get("requires_exact_temporal"))


def _evidence_needs_require_numeric(profile: dict[str, Any] | None) -> bool:
    needs = _iter_evidence_needs(profile)
    if needs:
        return any(
            need.get("preferred_source") == "computation"
            or _need_text_matches(need, NUMERIC_KEYWORDS)
            for need in needs
        )
    if not isinstance(profile, dict):
        return False
    return bool(profile.get("requires_exact_numeric"))


def _build_evidence_needs_profile(raw_needs: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_needs = [
        {
            "need": need["need"],
            "preferred_source": need["preferred_source"],
            "closure_level": need["closure_level"],
            "status": "unfilled",
            "filled_by": None,
            "need_role": str(need.get("need_role") or "anchor"),
            "depends_on": list(need.get("depends_on") or []),
            "query_focus": need.get("query_focus"),
        }
        for need in raw_needs
    ]
    profile: dict[str, Any] = {
        "_schema": "evidence_needs",
        "evidence_needs": evidence_needs,
    }
    profile["requires_exact_temporal"] = _evidence_needs_require_temporal(profile)
    profile["requires_exact_numeric"] = _evidence_needs_require_numeric(profile)
    profile["likely_entity_bridge_required"] = _evidence_needs_require_entity_bridge(
        profile
    )
    profile["prefers_subtitle_grounding"] = _evidence_needs_prefer_source(
        profile, "subtitle", unfilled_only=False
    )
    profile["prefers_audio_grounding"] = _evidence_needs_prefer_source(
        profile, "audio", unfilled_only=False
    )
    profile["prefers_visual_grounding"] = _evidence_needs_prefer_source(
        profile, "visual", unfilled_only=False
    )
    profile["requires_external_lookup"] = _evidence_needs_prefer_source(
        profile, "web", unfilled_only=False
    )
    return profile


class QuestionProfile:
    """Provides _build_question_constraint_profile, _infer_question_profile_with_llm."""

    def _call_model_with_tools(self, *args, **kwargs):
        core_agent_module = sys.modules.get("agents.core.agent") or sys.modules.get(
            "agents.core_agent"
        )
        try:
            if core_agent_module is not None:
                patched_caller = getattr(
                    core_agent_module, "call_openai_model_with_tools", None
                )
                if callable(patched_caller):
                    return patched_caller(*args, caller="question_profile", **kwargs)
            return call_openai_model_with_tools(*args, caller="question_profile", **kwargs)
        except LLMCallFailure:
            # Profiling falls back to its defaults; re-issuing a request the
            # provider deterministically refused would only repeat the cost.
            return None

    def _build_question_constraint_profile(
        self, question: str, run_report: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a question constraint profile inferred by LLM."""
        if run_report is not None:
            cached = run_report.get("_question_profile")
            if isinstance(cached, dict) and cached:
                return cached

        profile = self._infer_question_profile_with_llm(question)

        if run_report is not None:
            run_report["_question_profile"] = profile
        return profile

    def _infer_question_profile_with_llm(self, question: str) -> dict[str, Any]:
        """Return a structured question profile."""
        if os.environ.get("OMNIVQA_DISABLE_QUESTION_PROFILE_LLM", "").strip() == "1":
            return _boolean_profile_defaults()
        if config.DISABLE_EVIDENCE_NEEDS:
            return self._infer_boolean_question_profile(question)
        return self._infer_evidence_needs_question_profile(question)

    def _infer_boolean_question_profile(self, question: str) -> dict[str, Any]:
        prompt = (
            "You are a question-type classifier for a video QA system. "
            "Given a question, output a JSON object with exactly these boolean keys:\n\n"
            "{\n"
            '  "requires_exact_temporal": true/false,\n'
            '  "requires_exact_numeric": true/false,\n'
            '  "likely_entity_bridge_required": true/false,\n'
            '  "prefers_subtitle_grounding": true/false,\n'
            '  "prefers_audio_grounding": true/false,\n'
            '  "prefers_visual_grounding": true/false,\n'
            '  "requires_external_lookup": true/false\n'
            "}\n\n"
            "Output valid JSON only, no explanation.\n\n"
            f"Question: {question}"
        )
        return self._call_boolean_profile_prompt(prompt)

    def _infer_evidence_needs_question_profile(self, question: str) -> dict[str, Any]:
        prompt = (
            "You are a question-type classifier for a video QA system.\n"
            "Given a question, identify the 2-5 distinct evidence needs required to answer it.\n\n"
            "Output a JSON object:\n"
            "{\n"
            '  "evidence_needs": [\n'
            '    {"need": "...", "preferred_source": "visual|audio|subtitle|web|computation", "closure_level": "blocking|supplementary", "need_role": "anchor|external_fact|applicability|compute", "depends_on": [], "query_focus": "short phrase describing what to look for"}\n'
            "  ]\n"
            "}\n\n"
            "Each need should be medium-abstract:\n"
            '  Good: "visual evidence of on-screen text showing a score or date"\n'
            '  Bad (too vague): "need visual"\n'
            '  Bad (too specific): "scoreboard at minute 23:15"\n\n'
            "preferred_source is a hint for which modality would best fill this need:\n"
            "- visual: on-screen text, graphical annotations, frame-level details\n"
            "- audio: non-speech sounds, music identity, environmental audio\n"
            "- subtitle: spoken words, dialogue, lyrics, narration\n"
            "- web: external facts, entity attributes, public knowledge\n"
            "- computation: arithmetic, date math, counting\n\n"
            "closure_level indicates how important the need is for answer correctness:\n"
            '- "blocking": without this evidence the answer would be wrong or impossible\n'
            '- "supplementary": improves completeness but the core answer is possible without it\n\n'
            "For each evidence need, also provide three additional fields:\n\n"
            '- need_role: one of "anchor", "external_fact", "applicability", "compute"\n'
            '  - anchor: a fact that must be grounded from the video itself (entity identity, date, number, label, product name)\n'
            '  - external_fact: a fact that must be looked up from external sources (birth date, spec, regulation value, chart ranking)\n'
            '  - applicability: applying an external rule or regulation to a video-grounded entity (which zone applies, which category matches)\n'
            '  - compute: a deterministic calculation using other filled needs as inputs (date difference, numeric gap, unit conversion)\n\n'
            "- depends_on: a list of need indices (0-based) that must reach \"filled\" status before this need can be productively pursued. Must form a DAG (no cycles). Use an empty list [] if no dependencies.\n"
            "  Example: an external_fact that requires knowing a specific entity from the video should depend on the anchor need that identifies that entity.\n\n"
            "- query_focus: a short phrase (under 15 words) describing what exactly this need is looking for. Downstream tools and web searches will use this as guidance.\n\n"
            "When the question text contains quoted terms, proper nouns, or specific named references, include them verbatim in the relevant need's description and query_focus. Do not paraphrase or generalize these references.\n\n"
            "Output valid JSON only, no explanation.\n\n"
            f"Question: {question}"
        )
        try:
            response = self._call_model_with_tools(
                [{"role": "user", "content": prompt}],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                max_tokens=512,
            )
            raw = self._extract_model_content(response)
            parsed = safe_load_json_object(raw)
            validated = _validate_evidence_needs(
                parsed.get("evidence_needs") if isinstance(parsed, dict) else None
            )
            if validated is None:
                validated = list(FALLBACK_EVIDENCE_NEEDS)
            return _build_evidence_needs_profile(validated)
        except Exception:
            return _build_evidence_needs_profile(list(FALLBACK_EVIDENCE_NEEDS))

    def _call_boolean_profile_prompt(self, prompt: str) -> dict[str, Any]:
        default_profile = _boolean_profile_defaults()
        try:
            response = self._call_model_with_tools(
                [{"role": "user", "content": prompt}],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                max_tokens=256,
            )
            raw = self._extract_model_content(response)
            parsed = safe_load_json_object(raw)
            if not isinstance(parsed, dict):
                return default_profile
            profile = _boolean_profile_defaults()
            for key in DEFAULT_BOOLEAN_PROFILE:
                if key == "_schema":
                    continue
                profile[key] = bool(parsed.get(key, profile[key]))
            return profile
        except Exception:
            return default_profile

    @staticmethod
    def _extract_model_content(response: dict[str, Any] | None) -> str:
        if response is None:
            return ""
        raw = str(response.get("content") or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        return raw
