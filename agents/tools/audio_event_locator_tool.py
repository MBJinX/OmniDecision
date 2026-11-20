"""Independent AudioLLM fallback locator for OmniVQA."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Annotated as A

from nano_vectordb import NanoVectorDB

from agents import config
from agents.asset_runtime import (
    load_grounding_cache,
    resolve_video_asset_runtime,
    save_grounding_cache,
)
from agents.shared.func_call_schema import doc as D
from agents.shared.utils import (
    LLMCallFailure,
    call_openai_model_with_tools,
    normalize_openai_base_url,
)

from .common import (
    build_failure_observation,
    build_grounding_observation,
    dumps_observation,
    make_candidate_range,
)
from .transcript_utils import ensure_audio_track


_IMPLEMENTATION_VERSION = "audio_event_locator_audiollm_fallback_v1"
_DEFAULT_FALLBACK_HINT = (
    "Use `frame_confirm_tool` on the best returned range. "
    "If the fallback still cannot localize the event, switch back to "
    "`audio_scout_tool`, `subtitle_grounding_tool`, or `clip_grounding_tool`."
)


def _cache_build_params() -> dict:
    return {
        "implementation": _IMPLEMENTATION_VERSION,
        "model_name": str(config.AOAI_TOOL_VLM_MODEL_NAME or ""),
        "base_url": normalize_openai_base_url(config.AOAI_TOOL_VLM_BASE_URL),
    }


def _load_cached_observation(runtime, *, query: str, top_k: int) -> str | None:
    cached = load_grounding_cache(
        runtime,
        tool_name="audio_event_locator_tool",
        query=query,
        top_k=top_k,
        build_params=_cache_build_params(),
        upstream_asset_keys=["raw_audio", "raw_video"],
    )
    if cached is None:
        return None
    return dumps_observation(cached)


def _save_observation(runtime, *, query: str, top_k: int, observation_text: str) -> str:
    try:
        payload = json.loads(observation_text)
    except Exception:
        return observation_text
    save_grounding_cache(
        runtime,
        tool_name="audio_event_locator_tool",
        query=query,
        top_k=top_k,
        observation=payload,
        build_params=_cache_build_params(),
        upstream_asset_keys=["raw_audio", "raw_video"],
    )
    return observation_text


def _resolve_audio_input_path(runtime) -> Path:
    raw_audio_path = Path(runtime.paths.raw_audio_path)
    if raw_audio_path.is_file() and raw_audio_path.suffix.lower() == ".wav":
        return raw_audio_path

    normalized_wav_path = raw_audio_path.with_suffix(".wav")
    if normalized_wav_path.is_file():
        return normalized_wav_path

    raw_video_path = Path(runtime.paths.raw_video_path)
    if raw_video_path.is_file():
        return ensure_audio_track(raw_video_path, normalized_wav_path)

    if raw_audio_path.is_file():
        return raw_audio_path

    raise FileNotFoundError(
        f"No usable audio input found. raw_audio_path={raw_audio_path} raw_video_path={raw_video_path}"
    )


def _read_audio_base64(audio_path: Path) -> str:
    return base64.b64encode(audio_path.read_bytes()).decode("utf-8")


def _audio_request_format(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower().lstrip(".")
    return suffix or "wav"


def _build_locator_prompt(query: str, *, top_k: int) -> str:
    limit = max(int(top_k), 1)
    return (
        "Localize the user-described audio event directly from the full audio track.\n"
        "Return strict JSON only.\n\n"
        "Rules:\n"
        "- Use only evidence from the provided audio.\n"
        f"- Return at most {limit} candidates ordered from best to worst.\n"
        "- Use seconds as floating-point numbers.\n"
        "- For point events, you may use the same second for start_secs and end_secs.\n"
        "- If the event is not present, set not_found=true and return an empty candidates list.\n"
        "- Do not include markdown, prose outside JSON, or extra keys.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "not_found": boolean,\n'
        '  "candidates": [\n'
        "    {\n"
        '      "start_secs": number,\n'
        '      "end_secs": number,\n'
        '      "confidence": number,\n'
        '      "rationale": string\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"User query: {query}"
    )


def _coerce_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return str(content or "").strip()


def _call_audio_locator_model(audio_path: Path, query: str, *, top_k: int) -> dict:
    if not config.AOAI_TOOL_VLM_API_KEY:
        raise RuntimeError("AOAI_TOOL_VLM_API_KEY is not configured.")
    if not config.AOAI_TOOL_VLM_MODEL_NAME:
        raise RuntimeError("AOAI_TOOL_VLM_MODEL_NAME is not configured.")

    endpoint = normalize_openai_base_url(config.AOAI_TOOL_VLM_BASE_URL)
    payload = {
        "endpoints": config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        "model_name": config.AOAI_TOOL_VLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise audio event localization model. "
                    "Your only job is to return strict JSON that pinpoints the timestamps "
                    "for the requested sound, speech, lyric, or music event."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _build_locator_prompt(query, top_k=top_k),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _read_audio_base64(audio_path),
                            "format": _audio_request_format(audio_path),
                        },
                    },
                ],
            },
        ],
        "return_json": True,
        "api_key": config.AOAI_TOOL_VLM_API_KEY,
        "base_url": endpoint,
        "temperature": 0,
        "max_tokens": 900,
    }

    try:
        response = call_openai_model_with_tools(
            payload["messages"],
            endpoints=payload["endpoints"],
            model_name=payload["model_name"],
            return_json=payload["return_json"],
            api_key=payload["api_key"],
            base_url=payload["base_url"],
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
            caller="audio_event_locator",
        )
    except LLMCallFailure as failure:
        raise RuntimeError(
            f"Audio locator backend refused the request ({failure.category})."
        ) from failure
    if response is None:
        raise RuntimeError("Audio locator backend returned no response.")
    content_text = _coerce_message_content(response.get("content"))
    if not content_text:
        raise RuntimeError("Audio locator backend returned empty content.")

    parsed = json.loads(content_text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Audio locator backend did not return a JSON object.")
    return parsed


def _clip_confidence(value: float) -> float:
    return min(max(float(value), 0.05), 0.99)


def _fallback_confidence(rank: int) -> float:
    return _clip_confidence(0.78 - 0.08 * rank)


def _normalize_candidates(locator_payload: dict, *, top_k: int) -> tuple[list[dict], list[dict]]:
    if bool(locator_payload.get("not_found")):
        return [], []

    raw_candidates = locator_payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raise RuntimeError("Audio locator JSON payload has a non-list `candidates` field.")

    candidate_ranges: list[dict] = []
    evidence: list[dict] = []
    for rank, candidate in enumerate(raw_candidates):
        if len(candidate_ranges) >= top_k:
            break
        if not isinstance(candidate, dict):
            continue

        try:
            start_secs = float(candidate.get("start_secs"))
            end_secs = float(candidate.get("end_secs", start_secs))
        except Exception:
            continue

        end_secs = max(end_secs, start_secs)
        rationale = str(candidate.get("rationale") or "").strip()
        if not rationale:
            rationale = "The full-audio fallback model marked this span as a likely match."

        raw_confidence = candidate.get("confidence")
        try:
            confidence = _clip_confidence(
                _fallback_confidence(rank) if raw_confidence is None else float(raw_confidence)
            )
        except Exception:
            confidence = _fallback_confidence(rank)

        candidate_ranges.append(
            make_candidate_range(
                start_secs,
                end_secs,
                confidence,
                rationale,
            )
        )
        evidence.append(
            {
                "text": rationale,
                "score": round(confidence, 4),
                "metadata": {
                    "event_type": "audio_event",
                    "implementation": _IMPLEMENTATION_VERSION,
                    "locator_type": "end_to_end_audiollm_fallback",
                    "model_name": str(config.AOAI_TOOL_VLM_MODEL_NAME or ""),
                    "start_secs": round(start_secs, 3),
                    "end_secs": round(end_secs, 3),
                },
            }
        )

    return candidate_ranges, evidence


def _build_success_summary(candidate_count: int, audio_path: Path) -> str:
    return (
        "Audio event locator used the full audio track with the configured AudioLLM fallback path "
        f"and returned {candidate_count} query-conditioned temporal anchor(s). "
        f"Input audio: {audio_path.name}."
    )


def audio_event_locator_tool(
    database: A[NanoVectorDB, D("The video database for the current video.")],
    query: A[
        str,
        D(
            "Describe the spoken phrase, lyric clue, background music change, or sound event that should be localized as a temporal anchor."
        ),
    ],
    top_k: A[
        int,
        D("Maximum number of query-conditioned audio anchor ranges to return. Use the default value unless needed."),
    ] = 5,
) -> str:
    """
    Fallback-only end-to-end audio event localization that sends the whole audio
    track to the configured audio-capable model and returns temporal anchors.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"

    top_k = max(int(top_k), 1)
    runtime = resolve_video_asset_runtime(database)

    cached_observation = _load_cached_observation(runtime, query=query, top_k=top_k)
    if cached_observation is not None:
        return cached_observation

    try:
        audio_path = _resolve_audio_input_path(runtime)
    except Exception as exc:
        return build_failure_observation(
            "audio",
            f"Audio event locator could not resolve a full-track audio input: {exc}",
            _DEFAULT_FALLBACK_HINT,
            status="unavailable",
        )

    try:
        locator_payload = _call_audio_locator_model(audio_path, query, top_k=top_k)
        candidate_ranges, evidence = _normalize_candidates(locator_payload, top_k=top_k)
    except Exception as exc:
        return build_failure_observation(
            "audio",
            f"Audio event locator failed while using the end-to-end AudioLLM fallback path: {exc}",
            _DEFAULT_FALLBACK_HINT,
            status="unavailable",
        )

    if not candidate_ranges:
        return _save_observation(
            runtime,
            query=query,
            top_k=top_k,
            observation_text=build_failure_observation(
                "audio",
                "Audio event locator used the full audio track but could not find a stable match for the query.",
                _DEFAULT_FALLBACK_HINT,
                status="no_match",
            ),
        )

    observation_text = build_grounding_observation(
        "audio",
        candidate_ranges,
        _build_success_summary(len(candidate_ranges), audio_path),
        evidence=evidence,
        confidence=max((item["confidence"] for item in candidate_ranges), default=0.0),
        fallback_hint=(
            "Use `frame_confirm_tool` on the best candidate range. "
            "This result came from the end-to-end AudioLLM fallback path over the full audio track."
        ),
    )
    return _save_observation(
        runtime,
        query=query,
        top_k=top_k,
        observation_text=observation_text,
    )
