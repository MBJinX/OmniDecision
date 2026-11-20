"""Prompt bundle loader for the current OmniVQA runtime."""

from functools import lru_cache
from pathlib import Path

import yaml


_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_PROMPT_BUNDLE_FILE = _PROMPT_DIR / "agent.yaml"


@lru_cache(maxsize=None)
def _load_prompt_bundle(path: str) -> dict:
    prompt_path = Path(path)
    bundle = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError(f"Prompt bundle must be a mapping: {prompt_path}")
    for key in ("system_prompt", "user_prompt_template"):
        if key not in bundle or not isinstance(bundle[key], str):
            raise ValueError(f"Missing prompt field `{key}` in {prompt_path}")
    return bundle


def _select_system_prompt_text(prompt_bundle: dict, ablation_mode: str | None) -> str:
    """Select the system prompt text for the given ablation mode.

    The default prompt is selected byte-for-byte for controller-preserving modes.
    `naive_memory` uses a deterministic view-only rewrite of that prompt so it
    asks for a prose memory rather than typed fields. `no_state_react` uses the
    dedicated state-free ReAct prompt.

    If `no_state_react` is requested but the bundle lacks `no_state_system_prompt`,
    a ValueError is raised. We do NOT silently fall back to the state prompt — that
    would re-create the unfair baseline this ablation is designed to avoid.
    """
    normalized_mode = str(ablation_mode or "full").strip().lower() or "full"
    if normalized_mode == "naive_memory":
        return _build_naive_memory_system_prompt(prompt_bundle["system_prompt"])
    if normalized_mode == "no_state_react":
        no_state_prompt = prompt_bundle.get("no_state_system_prompt")
        if not isinstance(no_state_prompt, str) or not no_state_prompt:
            raise ValueError(
                "ablation_mode='no_state_react' requires a `no_state_system_prompt` "
                "string in the prompt bundle; refusing to fall back to the state prompt."
            )
        return no_state_prompt
    return prompt_bundle["system_prompt"]


def _replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(
            "The full planner prompt changed and the naive_memory matched rewrite "
            f"can no longer find the required fragment: {old[:120]!r}"
        )
    return text.replace(old, new)


def _build_naive_memory_system_prompt(full_prompt: str) -> str:
    """Rewrite only the planner-facing state representation contract.

    The tool policy, batching policy, evidence strategy, precision policy, and
    answer style stay inherited from the full prompt. The typed-field reading
    instructions are replaced with an equivalent prose-memory contract. This
    keeps the arm coherent without turning prompt mismatch into an artificial
    disadvantage.
    """
    text = str(full_prompt)
    replacements = (
        (
            "You are OmniVQA, a long-video multimodal QA runtime built around an explicit Evidence State.",
            "You are OmniVQA, a long-video multimodal QA runtime with a rolling natural-language memory.",
        ),
        (
            "Your job is to run one state-driven loop:\n"
            "evidence state digest -> choose a batch of tools for independent needs "
            "(or one tool, or finish) -> observe -> read critic checkpoint -> continue or answer",
            "Your job is to run one memory-guided loop:\n"
            "rolling memory -> choose a batch of tools for independent needs "
            "(or one tool, or finish) -> observe -> read critic checkpoint -> continue or answer",
        ),
        (
            "- Each round, review the open evidence needs in the digest. Submit one tool call per **currently-independent** need — needs where neither consumes the other's unfilled output — up to 4 calls in one turn. Keep dependent chains (ground a range first, then visually confirm it) as single actions across rounds. When only one need is actionable, a single call is the correct batch. `finish` must always be sent alone. Before the calls, state in one short line which need index each call targets.",
            "- Each round, review the unresolved goals described in the rolling memory. Submit one tool call per **currently-independent** goal — goals where neither consumes the other's missing output — up to 4 calls in one turn. Keep dependent chains (ground a range first, then visually confirm it) as single actions across rounds. When only one goal can be pursued now, a single call is the correct batch. `finish` must always be sent alone. Before the calls, state in one short line which goal each call targets.",
        ),
        (
            "- When filling an anchor need, use the need's `query_focus` to guide the tool query.",
            "- When grounding an anchor, use the specific target described in the rolling memory to guide the tool query.",
        ),
        (
            "- When filling an external_fact need, combine the `query_focus` with entities already resolved from anchor needs.",
            "- When seeking an external fact, combine its specific target with entities already grounded from the assets.",
        ),
        (
            "- When the current evidence path has not produced new evidence for the remaining gap, consider whether an unexplored modality might fill it. The tool_call_counts in the state digest show which modalities have been tried and which remain unexplored.",
            "- When the current evidence path has not produced new evidence for the remaining gap, consider whether an unexplored modality might fill it. The rolling memory states which tools have been tried and which modalities remain unexplored.",
        ),
        (
            "- These are suggestions, not rules. Use your judgment based on the evidence state digest.",
            "- These are suggestions, not rules. Use your judgment based on the rolling memory and observations.",
        ),
        (
            "- If web_search_tool returns results but does not fill the target evidence need (check the search_feedback in the observation), try reformulating the query from a different angle rather than repeating the same search terms.",
            "- If web_search_tool returns results but does not resolve the target (check the feedback in the observation), try reformulating the query from a different angle rather than repeating the same search terms.",
        ),
        (
            "2. Check `evidence_needs`. If any `blocking` need is still `unfilled` or `partial`, gather more evidence before finishing unless the digest explicitly makes `finish_available` true.",
            "2. Check the rolling memory. If it says an answer-critical goal is still unresolved or only partly resolved, gather more evidence before finishing unless the memory explicitly says an answer may now be submitted.",
        ),
    )
    for old, new in replacements:
        text = _replace_required(text, old, new)

    routing_start = "- When choosing which tool to call next:\n"
    routing_end = "\n\nEvidence-gathering strategy:"
    if routing_start not in text or routing_end not in text:
        raise ValueError(
            "The full planner routing section changed; update the naive_memory matched rewrite."
        )
    before_routing, routing_tail = text.split(routing_start, 1)
    _, after_routing = routing_tail.split(routing_end, 1)
    memory_routing = (
        "- When choosing which tool to call next:\n"
        "  1. Read the rolling memory for unresolved goals, dependencies, and the main remaining gap.\n"
        "  2. Pursue goals whose prerequisites are already established before goals that still depend on missing inputs.\n"
        "  3. Match each goal's requested evidence modality to the most relevant tool.\n"
        "  4. Use the prose account of tools already tried to prioritize an unexplored modality when it can address an open goal.\n"
        "  5. When calling web_search_tool, combine the specific target with entities already grounded from the assets. Do NOT search the full question text verbatim."
    )
    text = before_routing + memory_routing + routing_end + after_routing

    state_start = "Evidence State Digest:\n"
    state_end = "\nPrecision policy:"
    if state_start not in text or state_end not in text:
        raise ValueError(
            "The full planner state section changed; update the naive_memory matched rewrite."
        )
    before_state, state_tail = text.split(state_start, 1)
    _, after_state = state_tail.split(state_end, 1)
    memory_section = (
        "Rolling Memory:\n"
        "- After each tool call, you receive one natural-language paragraph summarizing what has been established, what still needs resolution, which tools have already been used, the main remaining gap, and whether an answer may be submitted.\n"
        "- Treat the newest paragraph as the authoritative rolling account for the next action; it replaces the previous paragraph.\n"
        "- The paragraph has no fixed fields or indices. Resolve dependencies from its prose and use the raw tool observations when exact evidence text is needed.\n"
    )
    text = before_state + memory_section + state_end + after_state

    forbidden = (
        "Evidence State",
        "evidence state",
        "digest",
        "evidence_needs",
        "actionable_need_indices",
        "blocked_need_indices",
        "tool_call_counts",
        "gap_diagnosis",
        "entity_cards",
        "fact_bridge_records",
        "finish_available",
        "query_focus",
        "preferred_source",
        "closure_level",
        "filled_by",
        "need_role",
        "depends_on",
        "search_feedback",
    )
    leaked = [marker for marker in forbidden if marker in text]
    if leaked:
        raise ValueError(
            "naive_memory planner prompt still exposes typed-state vocabulary: "
            f"{leaked}"
        )
    return text


def _build_messages_from_bundle(
    prompt_bundle: dict,
    video_length: str,
    disabled_tools: set[str] | frozenset[str] | None = None,
    ablation_mode: str = "full",
) -> list[dict]:
    system_prompt_text = _select_system_prompt_text(prompt_bundle, ablation_mode)
    if disabled_tools:
        system_prompt_text = _strip_disabled_tool_lines(
            system_prompt_text,
            set(disabled_tools),
        )
    messages = [{"role": "system", "content": system_prompt_text}]

    few_shot_messages = prompt_bundle.get("few_shot_messages") or []
    if not isinstance(few_shot_messages, list):
        raise ValueError("`few_shot_messages` must be a list when provided.")

    for index, message in enumerate(few_shot_messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(
                f"`few_shot_messages[{index}]` must be a mapping."
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"`few_shot_messages[{index}].role` must be a non-empty string."
            )
        messages.append(dict(message))

    messages.append(
        {
            "role": "user",
            "content": prompt_bundle["user_prompt_template"].format(
                video_length=video_length
            ),
        }
    )
    return messages


def get_tools_for_asset_type(asset_type: str) -> list[str]:
    """Derive available tools for an asset type based on tool requirements.

    This function is the single source of truth for tool availability per asset type.
    It replaces a hardcoded TOOL_AVAILABILITY dict to prevent prompt/runtime drift.

    Tool requirements:
    - subtitle_grounding_tool: needs NanoVectorDB with text captions -> video, audio (whisper transcript)
    - audio_scout_tool: needs audio file -> video (has audio track), audio
    - clip_grounding_tool: needs NanoVectorDB with frame captions -> video only
    - frame_confirm_tool: needs video frames OR image file -> video, image (image_inspection mode)
    """
    if asset_type == "video":
        return ["subtitle_grounding_tool", "audio_scout_tool", "clip_grounding_tool", "frame_confirm_tool"]
    elif asset_type == "audio":
        return ["subtitle_grounding_tool", "audio_scout_tool"]
    elif asset_type == "image":
        return ["frame_confirm_tool"]
    else:
        return []


_GLOBAL_TOOL_CANDIDATES = ("web_search_tool", "code_executor_tool", "finish")


def _strip_disabled_tool_lines(text: str, disabled_tools: set[str] | frozenset[str]) -> str:
    """Filter out any line in a multi-line prompt that mentions a disabled tool
    by name. Used when the CoreAgent disables a tool (e.g. WorldSense runs
    disable web_search_tool) so the LLM does not see system-prompt guidance
    for a tool it cannot actually call — which otherwise wastes iteration
    budget on tool-not-found loops."""
    if not disabled_tools:
        return text
    kept_lines: list[str] = []
    for line in text.splitlines(keepends=False):
        if any(tool in line for tool in disabled_tools):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def build_asset_manifest(
    asset_registry: dict[str, dict],
    disabled_tools: set[str] | frozenset[str] | None = None,
) -> str:
    """Build a structured asset manifest for the system/user prompt."""
    disabled = set(disabled_tools or ())
    lines = ["Assets available for this question:"]
    for asset_id, desc in asset_registry.items():
        asset_type = desc.get("type", "unknown")
        tools = [t for t in get_tools_for_asset_type(asset_type) if t not in disabled]

        # Format tool names with mode annotations
        tool_display = []
        for t in tools:
            if asset_type == "image" and t == "frame_confirm_tool":
                tool_display.append("frame_confirm_tool (image_inspection mode — no time_ranges needed, just query)")
            else:
                tool_display.append(t)

        if asset_type in ("video", "audio"):
            info = f"duration={desc.get('duration_hhmmss', 'unknown')}"
        elif asset_type == "image":
            dims = desc.get("dimensions")
            info = f"dimensions={dims[0]}x{dims[1]}" if dims else "dimensions=unknown"
        else:
            info = ""

        display_id = desc.get("simplified_id", asset_id)
        lines.append(f"- {display_id} (type={asset_type}, {info})")
        lines.append(f"  available_tools: {', '.join(tool_display)}")

    lines.append("")
    lines.append("IMPORTANT: When calling a tool on a specific asset, you MUST include 'asset_id' in the tool arguments.")
    global_tools = [t for t in _GLOBAL_TOOL_CANDIDATES if t not in disabled]
    if global_tools:
        lines.append(f"Global tools ({', '.join(global_tools)}) do not require asset_id.")
    return "\n".join(lines)


def build_omni_messages_multi(
    asset_registry: dict[str, dict],
    disabled_tools: set[str] | frozenset[str] | None = None,
    ablation_mode: str = "full",
) -> list[dict]:
    """Construct the initial message list for multi-asset cases."""
    prompt_bundle = _load_prompt_bundle(str(_PROMPT_BUNDLE_FILE))
    asset_manifest = build_asset_manifest(asset_registry, disabled_tools=disabled_tools)

    system_prompt_text = _select_system_prompt_text(prompt_bundle, ablation_mode)
    if disabled_tools:
        system_prompt_text = _strip_disabled_tool_lines(
            system_prompt_text,
            set(disabled_tools),
        )
    messages = [{"role": "system", "content": system_prompt_text}]

    few_shot_messages = prompt_bundle.get("few_shot_messages") or []
    if not isinstance(few_shot_messages, list):
        raise ValueError("`few_shot_messages` must be a list when provided.")
    for index, message in enumerate(few_shot_messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"`few_shot_messages[{index}]` must be a mapping.")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"`few_shot_messages[{index}].role` must be a non-empty string.")
        messages.append(dict(message))

    # Use multi-asset user prompt template if available, otherwise fallback
    multi_template = prompt_bundle.get("user_prompt_template_multi")
    if multi_template and isinstance(multi_template, str):
        user_content = multi_template.format(asset_manifest=asset_manifest)
    else:
        user_content = (
            "Answer this question about the following assets:\n\n"
            f"{asset_manifest}\n\n"
            "Question: QUESTION_PLACEHOLDER"
        )

    messages.append({"role": "user", "content": user_content})
    return messages


def build_omni_messages(
    video_length: str,
    disabled_tools: set[str] | frozenset[str] | None = None,
    ablation_mode: str = "full",
) -> list[dict]:
    """Construct the initial message list for the current OmniVQA runtime.

    When `ablation_mode == "no_state_react"` the standalone no-state ReAct system
    prompt is selected; for every other mode the full-state `system_prompt` is used
    exactly as before.
    """
    prompt_bundle = _load_prompt_bundle(str(_PROMPT_BUNDLE_FILE))
    return _build_messages_from_bundle(
        prompt_bundle,
        video_length,
        disabled_tools=disabled_tools,
        ablation_mode=ablation_mode,
    )


def _get_optional_prompt_field(field_name: str) -> str | None:
    """Return an optional prompt field from the current prompt bundle."""
    prompt_bundle = _load_prompt_bundle(str(_PROMPT_BUNDLE_FILE))
    prompt_value = prompt_bundle.get(field_name)
    if prompt_value is None:
        return None
    if not isinstance(prompt_value, str):
        raise ValueError(
            f"`{field_name}` must be a string in {_PROMPT_BUNDLE_FILE}"
        )
    return prompt_value


def get_critic_prompt_template() -> str | None:
    """Return the optional evidence critic prompt template."""
    return _get_optional_prompt_field("critic_prompt_template")


def get_answer_critic_prompt_template() -> str | None:
    """Return the optional answer critic prompt template."""
    return _get_optional_prompt_field("answer_critic_prompt_template")
