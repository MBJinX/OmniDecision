"""Dual-path audio grounding tool for OmniVQA."""

import json
import logging
from pathlib import Path
from typing import Annotated as A

import numpy as np
from nano_vectordb import NanoVectorDB

from agents import config
from agents.asset_runtime import (
    load_asr_segments,
    load_dialogue_segments,
    load_grounding_cache,
    load_subtitle_segments,
    resolve_video_asset_runtime,
    save_grounding_cache,
)
from agents.shared.func_call_schema import doc as D
from agents.shared.perception import perception_call

from .common import (
    cosine_retrieval_score_to_confidence,
    build_failure_observation,
    build_grounding_observation,
    dumps_observation,
    expand_time_range,
    make_candidate_range,
    score_texts,
)
from .transcript_utils import ensure_audio_track


_NON_SPEECH_ROUTE_TRIGGER_CONFIDENCE = 0.62
logger = logging.getLogger(__name__)


def _ranges_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    max_gap_secs: float = 1.0,
) -> bool:
    return min(end_a, end_b) >= max(start_a, start_b) - max_gap_secs


def _make_route_hit(
    *,
    start_secs: float,
    end_secs: float,
    confidence: float,
    rationale: str,
    evidence_text: str,
    metadata: dict,
) -> dict:
    return {
        "start_secs": float(start_secs),
        "end_secs": float(end_secs),
        "confidence": float(confidence),
        "rationale": rationale,
        "evidence_text": evidence_text,
        "metadata": metadata,
    }


def _max_hit_confidence(result: dict) -> float:
    hits = result.get("hits") or []
    return max((float(hit.get("confidence") or 0.0) for hit in hits), default=0.0)


def _should_expand_proxy_with_asr(subtitle_segments: list[dict]) -> bool:
    return not subtitle_segments


def _should_run_non_speech_retrieval(
    speech_result: dict,
    proxy_result: dict,
) -> bool:
    if proxy_result.get("status") == "ok":
        return True
    if speech_result.get("status") != "ok":
        return True
    return _max_hit_confidence(speech_result) < _NON_SPEECH_ROUTE_TRIGGER_CONFIDENCE


def _run_speech_route(query: str, runtime, top_k: int) -> dict:
    try:
        segments, transcript_source, transcript_asset_key = load_dialogue_segments(
            runtime,
            allow_asr_fallback=True,
        )
    except Exception as exc:
        return {
            "hits": [],
            "status": "unavailable",
            "summary": f"Speech-oriented transcript retrieval failed: {exc}",
        }

    if not segments:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "Speech-oriented transcript retrieval found no subtitle or ASR segments.",
        }

    asr_backend = "whisper_asr"
    if transcript_source != "subtitle":
        if any(str(segment.get("source") or "").startswith("vlm_asr") for segment in segments):
            asr_backend = "vlm_asr"

    texts = [segment["spoken_text"] or segment["clean_text"] or "" for segment in segments]
    non_empty_mask = [bool(t) for t in texts]
    texts = [t for t, keep in zip(texts, non_empty_mask) if keep]
    segments = [s for s, keep in zip(segments, non_empty_mask) if keep]
    if not texts:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "All loaded speech segments have empty text content.",
        }
    scores = score_texts(query, texts)
    if scores.size == 0:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "Speech-oriented transcript retrieval could not score any segments.",
        }

    ranked_indices = scores.argsort()[::-1][: max(top_k * 2, 4)]
    video_length_secs = max(float(segment["end_secs"]) for segment in segments)
    hits: list[dict] = []
    for index in ranked_indices.tolist():
        confidence = cosine_retrieval_score_to_confidence(float(scores[index]))
        if confidence < 0.33 and hits:
            break
        segment = segments[index]
        start_secs, end_secs = expand_time_range(
            float(segment["start_secs"]),
            float(segment["end_secs"]),
            max_duration_secs=video_length_secs,
        )
        rationale = (
            "Subtitle semantic retrieval indicates that this spoken segment is highly related to the query."
            if transcript_source == "subtitle"
            else (
                "VLM ASR retrieval indicates that this spoken segment is highly related to the query."
                if asr_backend == "vlm_asr"
                else "audio transcription retrieval indicates that this spoken segment is highly related to the query."
            )
        )
        hits.append(
            _make_route_hit(
                start_secs=start_secs,
                end_secs=end_secs,
                confidence=confidence,
                rationale=rationale,
                evidence_text=segment["clean_text"],
                metadata={
                    "route": "speech",
                    "implementation": transcript_source if transcript_source == "subtitle" else asr_backend,
                    "start_secs": round(float(segment["start_secs"]), 3),
                    "end_secs": round(float(segment["end_secs"]), 3),
                },
            )
        )

    return {
        "hits": hits,
        "status": "ok" if hits else "no_match",
        "summary": (
            "Speech-oriented audio grounding used subtitle retrieval."
            if transcript_source == "subtitle"
            else (
                "Speech-oriented audio grounding fell back to VLM ASR retrieval."
                if asr_backend == "vlm_asr"
                else "Speech-oriented audio grounding fell back to audio transcription retrieval."
            )
        ),
    }


def _run_proxy_non_speech_route(
    query: str,
    runtime,
    subtitle_segments: list[dict],
    top_k: int,
) -> dict:
    asr_segments: list[dict] = []
    asr_error: str | None = None
    asr_asset_key: str | None = None
    if _should_expand_proxy_with_asr(subtitle_segments):
        try:
            asr_segments, asr_asset_key = load_asr_segments(runtime)
        except Exception as exc:
            asr_error = str(exc)

    segments: list[dict] = []
    for segment in subtitle_segments:
        item = dict(segment)
        item["source"] = "subtitle"
        segments.append(item)
    for segment in asr_segments:
        segments.append(dict(segment))

    if not segments:
        return {
            "hits": [],
            "status": "unavailable",
            "summary": (
                "Audio proxy retrieval could not load subtitle or ASR evidence."
                if not asr_error
                else f"Audio proxy retrieval could not load ASR evidence: {asr_error}"
            ),
        }

    texts = [segment["spoken_text"] or segment["clean_text"] or "" for segment in segments]
    non_empty_mask = [bool(t) for t in texts]
    texts = [t for t, keep in zip(texts, non_empty_mask) if keep]
    segments = [s for s, keep in zip(segments, non_empty_mask) if keep]
    if not texts:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "All loaded proxy segments have empty text content.",
        }
    upstream_asset_keys: list[str] = []
    if subtitle_segments:
        upstream_asset_keys.append("raw_subtitle")
    if asr_asset_key:
        upstream_asset_keys.append(asr_asset_key)
    text_scores = score_texts(query, texts)
    video_length_secs = max(float(segment["end_secs"]) for segment in segments)

    scored_segments = []
    for index, segment in enumerate(segments):
        text_score = (
            cosine_retrieval_score_to_confidence(float(text_scores[index]))
            if text_scores.size
            else 0.0
        )
        if segment.get("source") in {"asr", "vlm_asr", "vlm_asr_summary"}:
            final_score = text_score * 0.85
        else:
            final_score = text_score * 0.7
        if final_score <= 0.35:
            continue
        scored_segments.append((final_score, text_score, segment))

    if not scored_segments:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "Audio proxy retrieval did not find a strong cue-tag or transcript-aligned non-speech candidate.",
        }

    scored_segments.sort(key=lambda item: item[0], reverse=True)
    hits: list[dict] = []
    for final_score, text_score, segment in scored_segments[: max(top_k * 2, 4)]:
        start_secs, end_secs = expand_time_range(
            float(segment["start_secs"]),
            float(segment["end_secs"]),
            max_duration_secs=video_length_secs,
        )
        rationale = (
            "Matched transcript text as a weak proxy for the sound-related query."
        )
        if segment.get("cue_tags"):
            rationale += " Subtitle cue tags in the same segment provide auxiliary evidence."
        hits.append(
            _make_route_hit(
                start_secs=start_secs,
                end_secs=end_secs,
                confidence=final_score,
                rationale=rationale,
                evidence_text=segment["clean_text"],
                metadata={
                    "route": "non_speech_proxy",
                    "implementation": segment.get("source", "subtitle_cue_proxy"),
                    "cue_tags": segment.get("cue_tags", []),
                    "start_secs": round(float(segment["start_secs"]), 3),
                    "end_secs": round(float(segment["end_secs"]), 3),
                },
            )
        )

    return {
        "hits": hits,
        "status": "ok",
        "summary": "Audio proxy retrieval used transcript retrieval as weak support and only exposed subtitle cue tags as auxiliary evidence.",
    }


def _collect_supporting_cues(
    subtitle_segments: list[dict],
    start_secs: float,
    end_secs: float,
) -> list[str]:
    cue_tags: list[str] = []
    for segment in subtitle_segments:
        if not _ranges_overlap(
            start_secs,
            end_secs,
            float(segment["start_secs"]),
            float(segment["end_secs"]),
        ):
            continue
        for cue_tag in segment.get("cue_tags", []):
            if cue_tag not in cue_tags:
                cue_tags.append(cue_tag)
    return cue_tags


def _resolve_audio_for_vlm(runtime) -> Path:
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
        f"No usable audio for VLM non-speech grounding. "
        f"raw_audio_path={raw_audio_path} raw_video_path={raw_video_path}"
    )


def _vlm_search_non_speech_audio(
    query: str,
    runtime,
    *,
    top_k: int = 5,
) -> list[dict]:
    audio_path = _resolve_audio_for_vlm(runtime)
    prompt = (
        "Localize the user-described audio event directly from the full audio track.\n"
        "Return strict JSON only.\n\n"
        "Rules:\n"
        "- Use only evidence from the provided audio.\n"
        f"- Return at most {top_k} candidates ordered from best to worst.\n"
        "- Use seconds as floating-point numbers.\n"
        "- For point events, you may use the same second for start_secs and end_secs.\n"
        '- If the event is not present, return {"not_found": true, "candidates": []}.\n'
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
    result = perception_call(
        "audio_nonspeech",
        prompt=prompt,
        audio_path=str(audio_path),
        system_prompt=(
            "You are a precise audio event localization model. "
            "Return strict JSON that pinpoints the requested sound event."
        ),
        return_json=True,
        max_tokens=900,
        temperature=0,
        caller="audio_scout_vlm",
    )
    if result["error"]:
        logger.warning("VLM non-speech audio grounding failed: %s", result["error"])
        return []

    try:
        parsed = json.loads(result["content"])
    except json.JSONDecodeError as exc:
        logger.warning("VLM non-speech audio parse failed: %s", exc)
        return []

    if bool(parsed.get("not_found")):
        return []

    raw_candidates = parsed.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []

    top_confidence = 0.5
    normalized_confidences: list[float] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            normalized_confidences.append(float(candidate.get("confidence", 0.5)))
        except (TypeError, ValueError):
            continue
    if normalized_confidences:
        top_confidence = max(normalized_confidences)

    candidates: list[dict] = []
    for candidate in raw_candidates[:top_k]:
        if not isinstance(candidate, dict):
            continue
        try:
            start_secs = float(candidate.get("start_secs", 0.0))
            end_secs = float(candidate.get("end_secs", start_secs))
            confidence = float(candidate.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        score = confidence / max(top_confidence, 1e-6)
        candidates.append(
            {
                "start_secs": start_secs,
                "end_secs": max(end_secs, start_secs + 0.1),
                "score": float(np.clip(score, 0.05, 0.99)),
            }
        )
    return candidates


def _run_non_speech_route(
    query: str,
    runtime,
    subtitle_segments: list[dict],
    top_k: int,
) -> dict:
    try:
        candidates = _vlm_search_non_speech_audio(
            query,
            runtime,
            top_k=max(top_k * 2, 4),
        )
    except Exception as exc:
        return {
            "hits": [],
            "status": "unavailable",
            "summary": f"VLM non-speech audio grounding failed: {exc}",
        }

    if not candidates:
        return {
            "hits": [],
            "status": "no_match",
            "summary": "VLM non-speech audio grounding did not find the requested sound event.",
        }

    hits: list[dict] = []
    for candidate in candidates[: max(top_k * 2, 4)]:
        cue_tags = _collect_supporting_cues(
            subtitle_segments,
            float(candidate["start_secs"]),
            float(candidate["end_secs"]),
        )
        confidence = float(candidate["score"])
        rationale = "VLM-based audio event localization identified this non-speech audio window."
        if cue_tags:
            rationale += " Overlapping subtitle cue tags are attached as auxiliary evidence only."
        hits.append(
            _make_route_hit(
                start_secs=float(candidate["start_secs"]),
                end_secs=float(candidate["end_secs"]),
                confidence=confidence,
                rationale=rationale,
                evidence_text=", ".join(cue_tags) if cue_tags else query,
                metadata={
                    "route": "non_speech",
                    "implementation": "vlm_perception",
                    "raw_score": round(float(candidate["score"]), 4),
                    "cue_tags": cue_tags,
                },
            )
        )

    return {
        "hits": hits,
        "status": "ok",
        "summary": "Non-speech audio grounding used VLM-based audio event localization.",
    }


def _merge_hits(hits: list[dict], top_k: int) -> tuple[list[dict], list[dict]]:
    if not hits:
        return [], []

    sorted_hits = sorted(hits, key=lambda item: (item["start_secs"], item["end_secs"]))
    merged_clusters: list[dict] = []
    for hit in sorted_hits:
        if not merged_clusters:
            merged_clusters.append(
                {
                    "start_secs": hit["start_secs"],
                    "end_secs": hit["end_secs"],
                    "hits": [hit],
                }
            )
            continue

        previous = merged_clusters[-1]
        if hit["start_secs"] <= previous["end_secs"] + 1.0:
            previous["start_secs"] = min(previous["start_secs"], hit["start_secs"])
            previous["end_secs"] = max(previous["end_secs"], hit["end_secs"])
            previous["hits"].append(hit)
        else:
            merged_clusters.append(
                {
                    "start_secs": hit["start_secs"],
                    "end_secs": hit["end_secs"],
                    "hits": [hit],
                }
            )

    ranked_clusters: list[dict] = []
    for cluster in merged_clusters:
        cluster_hits = sorted(
            cluster["hits"],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        implementations = {
            item["metadata"].get("implementation", "unknown") for item in cluster_hits
        }
        confidence = min(
            cluster_hits[0]["confidence"] + 0.04 * max(len(implementations) - 1, 0),
            0.99,
        )
        if len(implementations) > 1:
            rationale = (
                "Multiple audio routes agree on this time range, including "
                + ", ".join(sorted(implementations))
                + "."
            )
        else:
            rationale = cluster_hits[0]["rationale"]
        ranked_clusters.append(
            {
                "start_secs": cluster["start_secs"],
                "end_secs": cluster["end_secs"],
                "confidence": confidence,
                "rationale": rationale,
                "hits": cluster_hits,
            }
        )

    ranked_clusters.sort(key=lambda item: item["confidence"], reverse=True)
    candidate_ranges: list[dict] = []
    evidence: list[dict] = []
    for cluster in ranked_clusters[:top_k]:
        candidate_ranges.append(
            make_candidate_range(
                cluster["start_secs"],
                cluster["end_secs"],
                cluster["confidence"],
                cluster["rationale"],
            )
        )
        for hit in cluster["hits"][:3]:
            evidence.append(
                {
                    "text": hit["evidence_text"],
                    "score": round(hit["confidence"], 4),
                    "metadata": hit["metadata"],
                }
            )
    return candidate_ranges, evidence


def audio_scout_tool(
    database: A[NanoVectorDB, D("The video database for the current video.")],
    query: A[
        str,
        D(
            "Describe the spoken clue, lyric, music cue, environmental sound, or non-speech audio event that should be located."
        ),
    ],
    top_k: A[
        int,
        D("Maximum number of audio-grounded candidate ranges to return. Use the default value unless needed."),
    ] = 5,
    fallback_mode: A[
        bool,
        D("Set true only when you explicitly want the full-audio fallback locator instead of the default speech/proxy retrieval path."),
    ] = False,
) -> str:
    """
    Ground audio-related questions with a dual-path strategy:
    speech queries use subtitle/transcript retrieval, while non-speech queries use
    audio-model localization with transcript proxy support.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"

    if fallback_mode:
        from .audio_event_locator_tool import audio_event_locator_tool as locator_impl

        return locator_impl(
            database=database,
            query=query,
            top_k=top_k,
        )

    runtime = resolve_video_asset_runtime(database)
    subtitle_segments, _ = load_subtitle_segments(runtime)
    tool_name = "audio_scout_tool"
    upstream_asset_keys = ["raw_subtitle", "raw_audio", "raw_video", "subtitle_segments"]
    try:
        _, _, transcript_asset_key = load_dialogue_segments(
            runtime,
            allow_asr_fallback=True,
        )
        upstream_asset_keys.append(transcript_asset_key)
    except Exception:
        pass
    if _should_expand_proxy_with_asr(subtitle_segments):
        try:
            _, asr_asset_key = load_asr_segments(runtime)
            upstream_asset_keys.append(asr_asset_key)
        except Exception:
            pass
    cache_build_params = {
        "tool_version": "audio_scout_v3",
        "execution_strategy": "speech_then_proxy_then_evidence_driven_non_speech",
        "retrieval_backend": "api_embedding|audio_perception|transcript_proxy",
        "subtitle_proxy_enabled": True,
        "perception_mode": config.PERCEPTION_MODE,
        "perception_model": config.PERCEPTION_MODEL if config.PERCEPTION_MODE == "unified" else "",
    }
    cached_observation = load_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=cache_build_params,
        upstream_asset_keys=sorted(set(upstream_asset_keys)),
    )
    if cached_observation is not None:
        return dumps_observation(cached_observation)

    route_results: list[dict] = []
    executed_paths = ["speech", "proxy_non_speech"]
    speech_result = _run_speech_route(query, runtime, top_k)
    proxy_result = _run_proxy_non_speech_route(query, runtime, subtitle_segments, top_k)
    route_results.extend([speech_result, proxy_result])
    if _should_run_non_speech_retrieval(speech_result, proxy_result):
        route_results.append(
            _run_non_speech_route(query, runtime, subtitle_segments, top_k)
        )
        executed_paths.append("non_speech")

    hits = [hit for result in route_results for hit in result.get("hits", [])]
    candidate_ranges, evidence = _merge_hits(hits, top_k)
    successful_paths = [
        result["summary"] for result in route_results if result.get("status") == "ok"
    ]
    failed_paths = [
        result["summary"] for result in route_results if result.get("status") != "ok"
    ]

    if not candidate_ranges:
        fallback_hint = (
            "Fallback to `subtitle_grounding_tool` for transcript evidence, "
            "or use `clip_grounding_tool` for coarse localization."
        )
        observation_text = build_failure_observation(
            "audio",
            " | ".join(failed_paths) if failed_paths else "Audio scout could not ground the query.",
            fallback_hint,
            status="no_match",
        )
        save_grounding_cache(
            runtime,
            tool_name=tool_name,
            query=query,
            top_k=top_k,
            observation=json.loads(observation_text),
            build_params=cache_build_params,
            upstream_asset_keys=sorted(set(upstream_asset_keys)),
        )
        return observation_text

    summary_parts = [
        "Audio scout used transcript retrieval first and expanded to other audio paths only when the retrieved evidence stayed weak."
    ]
    summary_parts.append("Executed paths: " + ", ".join(executed_paths) + ".")
    summary_parts.extend(successful_paths)
    if failed_paths:
        summary_parts.append("Fallback notes: " + " | ".join(failed_paths))

    fallback_hint = (
        "Use `frame_confirm_tool` on the best candidate range. If the evidence is still weak, "
        "fallback to `subtitle_grounding_tool` for spoken clues or `clip_grounding_tool` for coarse localization."
    )
    observation_text = build_grounding_observation(
        "audio",
        candidate_ranges,
        " ".join(summary_parts),
        evidence=evidence,
        confidence=max((item["confidence"] for item in candidate_ranges), default=0.0),
        fallback_hint=fallback_hint,
    )
    save_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        observation=json.loads(observation_text),
        build_params=cache_build_params,
        upstream_asset_keys=sorted(set(upstream_asset_keys)),
    )
    return observation_text
