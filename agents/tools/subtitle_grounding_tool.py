"""Subtitle-grounding tool for OmniVQA."""

import json
from typing import Annotated as A

from nano_vectordb import NanoVectorDB

from agents.asset_runtime import (
    load_dialogue_segments,
    load_grounding_cache,
    resolve_video_asset_runtime,
    save_grounding_cache,
)
from agents.shared.func_call_schema import doc as D

from .common import (
    cosine_retrieval_score_to_confidence,
    build_failure_observation,
    build_grounding_observation,
    dumps_observation,
    expand_time_range,
    make_candidate_range,
    score_texts,
)


def subtitle_grounding_tool(
    database: A[NanoVectorDB, D("The video database for the current video.")],
    query: A[str, D("Describe the spoken content, subtitle semantics, or dialogue clue to locate.")],
    top_k: A[int, D("Maximum number of subtitle-based candidate ranges to return. Use the default value unless needed.")] = 5,
) -> str:
    """
    Ground the question using subtitle or transcript segments before visual confirmation.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"

    runtime = resolve_video_asset_runtime(database)
    subtitle_path = runtime.paths.raw_subtitle_path
    tool_name = "subtitle_grounding_tool"
    try:
        segments, transcript_source, transcript_asset_key = load_dialogue_segments(
            runtime,
            allow_asr_fallback=True,
        )
    except Exception as exc:
        return build_failure_observation(
            "subtitle",
            f"Subtitle grounding could not load subtitle or ASR transcript evidence: {exc}",
            "Fallback to `clip_grounding_tool`, or fix the local subtitle/ASR assets inside OmniVQA before using subtitle grounding.",
        )

    if not segments:
        return build_failure_observation(
            "subtitle",
            f"No subtitle file was found at {subtitle_path}, and audio transcription fallback produced no usable transcript segments.",
            "Fallback to `clip_grounding_tool`, or rebuild the local subtitle/ASR assets before relying on subtitle grounding.",
        )

    cache_build_params = {
        "tool_version": "subtitle_grounding_v1",
        "segment_scope": "dialogue_segments",
        "retrieval_backend": "api_embedding",
        "transcript_source": transcript_source,
    }
    cached_observation = load_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=cache_build_params,
        upstream_asset_keys=[transcript_asset_key],
    )
    if cached_observation is not None:
        return dumps_observation(cached_observation)

    texts = [segment["spoken_text"] or segment["clean_text"] or "" for segment in segments]
    # Filter segments where both spoken_text and clean_text are empty/None.
    non_empty_mask = [bool(t) for t in texts]
    texts = [t for t, keep in zip(texts, non_empty_mask) if keep]
    segments = [s for s, keep in zip(segments, non_empty_mask) if keep]
    if not texts:
        return build_failure_observation(
            "subtitle",
            "All loaded subtitle/ASR segments have empty text content.",
            "Fallback to `clip_grounding_tool`.",
        )
    scores = score_texts(query, texts)
    if scores.size == 0:
        observation_text = build_failure_observation(
            "subtitle",
            "Subtitle retrieval could not score any subtitle segments.",
            "Fallback to `clip_grounding_tool`.",
            status="no_match",
        )
        save_grounding_cache(
            runtime,
            tool_name=tool_name,
            query=query,
            top_k=top_k,
            observation=json.loads(observation_text),
            build_params=cache_build_params,
            upstream_asset_keys=[transcript_asset_key],
        )
        return observation_text

    ranked_indices = scores.argsort()[::-1][:top_k]
    video_length_secs = max(
        float(segment["end_secs"]) for segment in segments
    )
    candidate_ranges = []
    evidence = []
    for index in ranked_indices.tolist():
        segment = segments[index]
        confidence = cosine_retrieval_score_to_confidence(float(scores[index]))
        start_secs, end_secs = expand_time_range(
            float(segment["start_secs"]),
            float(segment["end_secs"]),
            max_duration_secs=video_length_secs,
        )
        candidate_ranges.append(
            make_candidate_range(
                start_secs,
                end_secs,
                confidence,
                (
                    "Subtitle semantic retrieval indicates that this line is highly related to the query."
                    if transcript_source == "subtitle"
                    else "audio transcription transcript retrieval indicates that this spoken segment is highly related to the query."
                ),
            )
        )
        evidence.append(
            {
                "text": segment["clean_text"],
                "score": round(confidence, 4),
                "metadata": {
                    "start_secs": round(float(segment["start_secs"]), 3),
                    "end_secs": round(float(segment["end_secs"]), 3),
                    "implementation": transcript_source,
                },
            }
        )

    observation_text = build_grounding_observation(
        "subtitle",
        candidate_ranges,
        (
            "Subtitle retrieval found dialogue-aligned candidate ranges for visual confirmation."
            if transcript_source == "subtitle"
            else "Subtitle grounding fell back to audio transcription transcript because no usable .srt subtitle file was available."
        ),
        evidence=evidence,
        confidence=max((item["confidence"] for item in candidate_ranges), default=0.0),
        fallback_hint="Use `frame_confirm_tool` to verify the top subtitle candidate visually. If the subtitle evidence is weak, fallback to `clip_grounding_tool`.",
    )
    save_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        observation=json.loads(observation_text),
        build_params=cache_build_params,
        upstream_asset_keys=[transcript_asset_key],
    )
    return observation_text
