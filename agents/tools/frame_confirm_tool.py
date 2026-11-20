"""Visual confirmation wrapper for OmniVQA."""

import json
import os
import re
from typing import Annotated as A

import numpy as np
from nano_vectordb import NanoVectorDB

import agents.config as config
from agents.asset_runtime import resolve_video_asset_runtime
from agents.build_database import (
    convert_hhmmss_to_seconds,
    frame_inspect_tool,
)
from agents.shared.func_call_schema import doc as D
from agents.shared.perception import perception_call
from agents.shared.utils import LLMCallFailure, call_openai_model_with_tools

from .common import build_failure_observation, build_visual_observation


# ---------------------------------------------------------------------------
# Merge-mode helpers
# ---------------------------------------------------------------------------

_SEGMENT_PATTERN = re.compile(
    r"\[Segment\s+([A-C])\]\s*(.*?)(?=\[Segment\s+[A-C]\]|\Z)",
    re.DOTALL,
)


def _split_time_ranges(time_ranges_hhmmss) -> tuple[list[list[str]], list[str]]:
    """Split planner-supplied ranges into usable ``[start, end]`` pairs and rejects.

    The planner owns this argument, so its shape is not guaranteed: a weaker
    orchestrator can emit ``[[]]``, ``[""]``, a bare string, or a one-element
    range. Unpacking those raises out of the tool and kills the whole case, so
    the shape is checked here and anything unusable is reported back as a
    rejected entry the caller turns into a failure observation.
    """
    rejected: list[str] = []
    if isinstance(time_ranges_hhmmss, (str, bytes)) or not isinstance(
        time_ranges_hhmmss, (list, tuple)
    ):
        return [], [repr(time_ranges_hhmmss)]

    pairs: list[list[str]] = []
    for entry in time_ranges_hhmmss:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, (list, tuple)):
            rejected.append(repr(entry))
            continue
        if len(entry) != 2:
            rejected.append(repr(entry))
            continue
        start, end = (str(entry[0] or "").strip(), str(entry[1] or "").strip())
        if not start or not end:
            rejected.append(repr(entry))
            continue
        pairs.append([start, end])
    return pairs, rejected


def _malformed_ranges_observation(rejected: list[str]) -> str:
    detail = f" Rejected: {', '.join(rejected)}." if rejected else ""
    return build_failure_observation(
        "visual",
        (
            "No usable candidate range was provided; each entry of "
            f"time_ranges_hhmmss must be [start, end] in HH:MM:SS.{detail}"
        ),
        (
            "Ground a candidate range with a grounding tool first, then re-issue "
            "frame_confirm_tool with [[start, end]] in HH:MM:SS. If this asset has no "
            "video track, use a tool that supports its modality instead."
        ),
        status="error",
    )


def _sample_frame_paths_for_range(database, start_hhmmss: str, end_hhmmss: str) -> list[str]:
    """Sample frame file paths for a single time range (no VLM call).

    Replicates the frame sampling algorithm from ``frame_inspect_tool`` in
    ``build_database.py`` but returns file paths instead of calling the VLM.
    """
    video_meta = database.get_additional_data()
    video_length_secs = convert_hhmmss_to_seconds(video_meta["video_length"])
    video_file_root = video_meta["video_file_root"]
    fps = video_meta.get("fps", config.VIDEO_FPS)

    start_secs = convert_hhmmss_to_seconds(start_hhmmss)
    end_secs = convert_hhmmss_to_seconds(end_hhmmss)

    # Boundary guards (same as frame_inspect_tool lines 54-56)
    if start_secs > video_length_secs:
        raise ValueError(
            f"Start time {start_hhmmss} ({start_secs}s) exceeds video length {video_length_secs}s"
        )
    end_secs = min(end_secs, video_length_secs)

    total_time = end_secs - start_secs
    assert total_time > 0, (
        f"Range [{start_hhmmss} - {end_hhmmss}] has non-positive duration ({total_time}s)"
    )

    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM

    # Uniformly sample on the range timeline
    offsets = np.linspace(0, total_time, num=max_timepoints, endpoint=False, dtype=float)
    timepoints = [start_secs + off for off in offsets]

    # Convert timestamps to frame indices
    max_frame_idx = int(video_length_secs * fps) - 1
    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]

    paths = [
        os.path.join(video_file_root, "frames", f"frame_n{fn:06d}.jpg")
        for fn in framepoints
    ]
    return _trim_frames_to_payload_budget(paths)


_VLM_RAW_BUDGET_BYTES = 4 * 1024 * 1024


def _trim_frames_to_payload_budget(
    frame_paths: list[str],
    max_raw_bytes: int = _VLM_RAW_BUDGET_BYTES,
) -> list[str]:
    """Uniformly subsample *frame_paths* so total file size stays within budget.

    BUG-2 fix: filter to existing files first so sizes/paths stay aligned.
    """
    if not frame_paths:
        return frame_paths
    # Only keep paths that actually exist on disk
    existing = [p for p in frame_paths if os.path.isfile(p)]
    if not existing:
        return frame_paths  # let downstream handle missing files
    sizes = [os.path.getsize(p) for p in existing]
    total = sum(sizes)
    if total <= max_raw_bytes:
        return existing
    avg = total / len(existing)
    target = max(int(max_raw_bytes / avg), 4)  # floor at 4 frames, not 2
    indices = np.linspace(0, len(existing) - 1, num=target, dtype=int)
    return [existing[i] for i in sorted(set(indices))]


def _build_merged_prompt(
    query: str,
    ranges_with_frames: list[tuple[str, str, int]],
) -> list[dict]:
    """Build VLM messages for a merged multi-segment call.

    ``ranges_with_frames`` is a list of ``(start_hhmmss, end_hhmmss, frame_count)``
    tuples for the successfully-sampled ranges.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a visual-only assistant. You can inspect the provided video frames, "
            "but you cannot hear audio, recover missing speech, or infer off-screen facts "
            "that are not visibly shown."
        ),
    }

    # Build segment description lines and marker instructions
    segment_lines = []
    marker_lines = []
    cumulative = 0
    for i, (start, end, count) in enumerate(ranges_with_frames):
        label = chr(ord("A") + i)
        first = cumulative + 1
        last = cumulative + count
        segment_lines.append(f"Segment {label} [{start} - {end}]: frames {first}-{last}")
        marker_lines.append(f"[Segment {label}] <your observations for segment {label}>")
        cumulative += count

    user_text = (
        "Carefully watch the video frames. Pay attention to the cause and sequence of "
        "events, the detail and movement of objects, visible on-screen text, and the "
        "action and pose of persons.\n\n"
        f"This inspection covers {len(ranges_with_frames)} segments:\n"
        + "\n".join(segment_lines)
        + "\n\n"
        "For each segment, note your observations using the segment marker:\n"
        + "\n".join(marker_lines)
        + "\n\n"
        "Important constraints:\n"
        "- Use only what is visibly present in the frames.\n"
        "- Do not claim to hear dialogue, narration, music, or other audio.\n"
        "- Do not infer exact spoken wording, dates, or timing unless they are visibly "
        "shown on screen.\n"
        "- If the question depends on speech/audio content or a fact that is not visibly "
        "shown in the provided frames, return: `Error: Cannot find corresponding result "
        "in the given time range.`.\n\n"
        "Based on your observations from all segments, answer the question concisely. "
        "If no relevant visible content is found, return: `Error: Cannot find "
        "corresponding result in the given time range.`\n\n"
        f"Question: {query}\n"
    )

    user_msg = {"role": "user", "content": user_text}
    return [system_msg, user_msg]


def _parse_segment_observations(vlm_text: str, range_count: int) -> list[str] | None:
    """Parse per-segment observations from VLM output.

    Returns a list of per-segment text strings, or ``None`` when:
    - Zero segment markers are found (triggers shared fallback), OR
    - Fewer markers than ``range_count`` are found (partial = parse failure = fallback).
    """
    matches = _SEGMENT_PATTERN.findall(vlm_text)
    if not matches:
        return None  # zero markers -> fallback

    label_to_text: dict[str, str] = {}
    for label, text in matches:
        label_to_text[label] = text.strip()

    labels = [chr(ord("A") + i) for i in range(range_count)]
    # ALL expected labels must be present; partial = parse failure
    if not all(lbl in label_to_text for lbl in labels):
        return None  # partial markers -> fallback

    return [label_to_text[lbl] for lbl in labels]


def _merged_frame_confirm(
    database,
    runtime,
    query: str,
    time_ranges_hhmmss: list[list[str]],
) -> str:
    """Orchestrate a merged multi-range VLM call with per-segment parsing."""

    ranges, rejected = _split_time_ranges(time_ranges_hhmmss)
    ranges = ranges[:3]
    if not ranges:
        return _malformed_ranges_observation(rejected)

    if not runtime.paths.frames_dir.is_dir():
        return build_failure_observation(
            "visual",
            f"Frame assets are missing under {runtime.paths.frames_dir}.",
            "Rebuild frame assets or continue with other evidence.",
            status="unavailable",
        )

    # Per-range frame sampling with fault tolerance
    all_frame_paths: list[str] = []
    ranges_with_frames: list[tuple[str, str, int]] = []
    failed_ranges: list[tuple[str, str, str]] = []

    for start, end in ranges:
        try:
            paths = _sample_frame_paths_for_range(database, start, end)
            all_frame_paths.extend(paths)
            ranges_with_frames.append((start, end, len(paths)))
        except Exception as exc:
            failed_ranges.append((start, end, str(exc)))

    # All ranges failed
    if not ranges_with_frames:
        return build_failure_observation(
            "visual",
            "Frame sampling failed for all requested ranges.",
            "Try different candidate ranges or continue with other evidence.",
            status="unavailable",
        )

    all_frame_paths = _trim_frames_to_payload_budget(all_frame_paths)
    # Update per-range frame counts after trimming
    # (conservative: keep original counts for prompt labels, VLM sees fewer frames)

    # Build prompt and call VLM
    messages = _build_merged_prompt(query, ranges_with_frames)
    if config.PERCEPTION_MODE == "unified":
        system_prompt = (
            messages[0]["content"]
            if messages and messages[0].get("role") == "system"
            else None
        )
        user_prompt = next(
            (msg.get("content", "") for msg in messages if msg.get("role") == "user"),
            "",
        )
        _pc_result = perception_call(
            "vision",
            prompt=user_prompt,
            image_paths=all_frame_paths,
            system_prompt=system_prompt,
            max_tokens=512 * len(ranges_with_frames),
            temperature=0,
            caller="frame_confirm",
        )
        if _pc_result["error"]:
            result = None
        else:
            result = {
                "content": _pc_result["content"],
                "tool_calls": _pc_result.get("tool_calls"),
                "role": "assistant",
            }
    else:
        try:
            result = call_openai_model_with_tools(
                messages=messages,
                endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
                model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
                api_key=config.AOAI_TOOL_VLM_API_KEY,
                base_url=config.AOAI_TOOL_VLM_BASE_URL,
                image_paths=all_frame_paths,
                max_tokens=512 * len(ranges_with_frames),
                temperature=0,
                caller="frame_confirm",
            )
        except LLMCallFailure:
            # Surface as an unavailable observation the planner can route
            # around; the refusal is already recorded in the shared ledger.
            result = None

    if result is None:
        return build_failure_observation(
            "visual",
            "Visual model backend unavailable for merged frame confirmation.",
            "Retry later or continue with other evidence.",
            status="unavailable",
        )

    vlm_text = result["content"]
    if vlm_text.startswith("Error:"):
        return build_failure_observation(
            "visual",
            vlm_text,
            "Try different candidate ranges from a grounding tool.",
            status="no_match",
        )

    # Parse segment observations
    parsed = _parse_segment_observations(vlm_text, len(ranges_with_frames))
    merge_parse_fallback = parsed is None

    if merge_parse_fallback:
        # Fallback: use entire VLM text for all successful ranges
        segment_texts = [vlm_text] * len(ranges_with_frames)
    else:
        segment_texts = parsed

    # Build range_observations
    range_observations: list[dict] = []
    aligned_observations: list[str] = []

    for i, (start, end, _count) in enumerate(ranges_with_frames):
        obs_text = segment_texts[i]
        aligned_observations.append(f"[{start} - {end}] {obs_text}")
        range_observations.append({
            "time_range": [start, end],
            "status": "ok",
            "observation": obs_text,
        })

    # Add failed ranges with error status
    for start, end, err_msg in failed_ranges:
        range_observations.append({
            "time_range": [start, end],
            "status": "error",
            "observation": "",
            "summary": f"Frame sampling failed: {err_msg}",
        })

    success_count = len(ranges_with_frames)
    total_count = max(len(ranges), 1)
    coverage_ratio = success_count / total_count
    all_ranges_verified = success_count == len(ranges)
    confidence = 0.35 + 0.47 * coverage_ratio

    if all_ranges_verified:
        summary = "Visual verification finished for the provided candidate ranges with time-range aligned observations."
    else:
        summary = (
            f"Visual verification finished with partial matches; "
            f"{success_count} of {len(ranges)} candidate ranges returned usable evidence."
        )

    serialized_ranges = [[s, e] for s, e, _ in ranges_with_frames]

    obs_json_str = build_visual_observation(
        aligned_observations,
        serialized_ranges,
        summary,
        confidence=confidence,
        range_observations=range_observations,
        verified_range_count=success_count,
        requested_range_count=len(ranges),
        coverage_ratio=coverage_ratio,
        all_ranges_verified=all_ranges_verified,
    )

    # Inject merge_parse_fallback flag if needed
    if merge_parse_fallback:
        payload = json.loads(obs_json_str)
        payload["merge_parse_fallback"] = True
        obs_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    return obs_json_str


def image_inspect_asset(image_path: str, query: str) -> str:
    """Send a raw image directly to VLM for visual inspection."""
    import agents.config as config
    from agents.build_database import call_openai_model_with_tools

    input_msgs = [
        {
            "role": "system",
            "content": (
                "You are a visual analyst. Describe what you see in the image "
                "to answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Analyze this image and answer: {query}",
        },
    ]
    if config.PERCEPTION_MODE == "unified":
        _pc_result = perception_call(
            "vision",
            prompt=input_msgs[1]["content"],
            image_paths=[image_path],
            system_prompt=input_msgs[0]["content"],
            max_tokens=512,
            temperature=0,
            caller="frame_confirm",
        )
        if _pc_result["error"]:
            result = None
        else:
            result = {
                "content": _pc_result["content"],
                "tool_calls": _pc_result.get("tool_calls"),
                "role": "assistant",
            }
    else:
        try:
            result = call_openai_model_with_tools(
                messages=input_msgs,
                endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
                model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
                api_key=config.AOAI_TOOL_VLM_API_KEY,
                base_url=config.AOAI_TOOL_VLM_BASE_URL,
                image_paths=[image_path],
                temperature=0,
                max_tokens=512,
                caller="frame_confirm",
            )
        except LLMCallFailure:
            result = None
    if result is None:
        return build_failure_observation(
            "visual",
            "Visual model backend unavailable for image inspection.",
            "Retry later or continue with other evidence.",
            status="unavailable",
        )
    observation = result["content"]
    return build_visual_observation(
        [f"[image] {observation}"],
        [["image", "image"]],
        "Image inspection completed.",
        confidence=0.70,
        range_observations=[{
            "time_range": ["image", "image"],
            "status": "ok",
            "observation": observation,
        }],
        verified_range_count=1,
        requested_range_count=1,
        coverage_ratio=1.0,
        all_ranges_verified=True,
    )


def frame_confirm_tool(
    database: A[NanoVectorDB, D("The video database for the current video.")],
    query: A[str, D("The visual question to answer for the provided candidate time ranges.")],
    time_ranges_hhmmss: A[list[list[str]], D("One or more candidate time ranges in HH:MM:SS format. Each item must be [start_time, end_time].")],
    merge_ranges: A[bool, D("When true, merge all ranges into a single VLM call with per-segment boundary labels. Useful for cross-range visual comparison.")] = False,
) -> str:
    """
    Visually confirm a grounded candidate range by sampling frames and asking the VLM for evidence.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"

    runtime = resolve_video_asset_runtime(database)

    serialized_ranges, rejected_ranges = _split_time_ranges(time_ranges_hhmmss)
    if not serialized_ranges:
        return _malformed_ranges_observation(rejected_ranges)

    if merge_ranges and len(serialized_ranges) > 1:
        return _merged_frame_confirm(database, runtime, query, serialized_ranges)

    aligned_observations: list[str] = []
    range_observations: list[dict] = []

    for start, end in serialized_ranges:
        if not runtime.paths.frames_dir.is_dir():
            range_observations.append(
                {
                    "time_range": [start, end],
                    "status": "unavailable",
                    "observation": "",
                    "summary": f"Frame assets are missing under {runtime.paths.frames_dir}.",
                }
            )
            continue
        try:
            observation = frame_inspect_tool(database, query, [(start, end)])
        except Exception as exc:
            range_observations.append(
                {
                    "time_range": [start, end],
                    "status": "unavailable",
                    "observation": "",
                    "summary": f"Visual verification backend failed: {exc}",
                }
            )
            continue

        if observation.startswith("Error: Visual model backend unavailable"):
            range_observations.append(
                {
                    "time_range": [start, end],
                    "status": "unavailable",
                    "observation": "",
                    "summary": observation,
                }
            )
            continue

        if observation.startswith("Error:"):
            range_observations.append(
                {
                    "time_range": [start, end],
                    "status": "no_match",
                    "observation": "",
                    "summary": observation,
                }
            )
            continue

        aligned_text = f"[{start} - {end}] {observation}"
        aligned_observations.append(aligned_text)
        range_observations.append(
            {
                "time_range": [start, end],
                "status": "ok",
                "observation": observation,
            }
        )

    if not aligned_observations:
        unavailable_ranges = [
            item for item in range_observations if item.get("status") == "unavailable"
        ]
        fallback_hint = "Try a different candidate range from a grounding tool."
        if unavailable_ranges:
            fallback_hint = (
                "Retry later, rebuild frame assets if needed, or continue with other evidence "
                "until visual verification becomes available."
            )
        return build_failure_observation(
            "visual",
            (
                "Visual verification was unavailable for all requested candidate ranges."
                if unavailable_ranges
                else "Error: Cannot find corresponding result in the given time range."
            ),
            fallback_hint,
            status="unavailable" if unavailable_ranges else "no_match",
        )

    total_ranges = max(len(serialized_ranges), 1)
    success_count = len(aligned_observations)
    coverage_ratio = success_count / total_ranges
    all_ranges_verified = success_count == len(serialized_ranges)
    confidence = 0.35 + 0.47 * coverage_ratio
    if all_ranges_verified:
        summary = "Visual verification finished for the provided candidate ranges with time-range aligned observations."
    else:
        summary = (
            "Visual verification finished with partial matches; "
            f"{success_count} of {len(serialized_ranges)} candidate ranges returned usable evidence."
        )

    return build_visual_observation(
        aligned_observations,
        serialized_ranges,
        summary,
        confidence=confidence,
        range_observations=range_observations,
        verified_range_count=success_count,
        requested_range_count=len(serialized_ranges),
        coverage_ratio=coverage_ratio,
        all_ranges_verified=all_ranges_verified,
    )
