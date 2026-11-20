"""Unified clip grounding wrapper for OmniVQA."""

import json
from typing import Annotated as A

from nano_vectordb import NanoVectorDB

from agents.asset_runtime import load_grounding_cache, resolve_video_asset_runtime, save_grounding_cache
from agents.build_database import _get_single_query_embedding
from agents.shared.func_call_schema import doc as D

from .common import (
    build_grounding_observation,
    cosine_retrieval_score_to_confidence,
    dumps_observation,
    make_candidate_range,
)


def clip_grounding_tool(
    database: A[NanoVectorDB, D("The video database that stores clip captions and metadata.")],
    query: A[str, D("Describe the event, state, or moment that should be grounded in the video.")],
    top_k: A[int, D("Maximum number of candidate time ranges to return. Use the default value unless needed.")] = 5,
) -> str:
    """
    Retrieve the most relevant clip-level candidate ranges from the caption database.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"

    runtime = resolve_video_asset_runtime(database)
    tool_name = "clip_grounding_tool"
    cache_build_params = {
        "tool_version": "clip_grounding_v1",
        "retrieval_backend": "nano_vectordb_caption_query",
        "query_embedding_backend": "aoai_embedding_large",
    }
    cached_observation = load_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=cache_build_params,
        upstream_asset_keys=["database", "raw_video"],
    )
    if cached_observation is not None:
        return dumps_observation(cached_observation)

    query_embedding = _get_single_query_embedding(query)
    results = database.query(query_embedding, top_k=top_k)
    if not results:
        observation_text = build_grounding_observation(
            "clip",
            [],
            "No caption-aligned clip candidates were found for the query.",
            confidence=0.0,
            fallback_hint="Try `subtitle_grounding_tool` or rephrase the event for `clip_grounding_tool`.",
            status="no_match",
        )
        save_grounding_cache(
            runtime,
            tool_name=tool_name,
            query=query,
            top_k=top_k,
            observation=json.loads(observation_text),
            build_params=cache_build_params,
            upstream_asset_keys=["database", "raw_video"],
        )
        return observation_text

    candidate_ranges = []
    evidence = []
    total = max(len(results), 1)
    for rank, item in enumerate(results):
        raw_retrieval_score = item.get("__metrics__")
        if raw_retrieval_score is None:
            confidence = max(0.15, 1.0 - (rank / (total + 1)))
            rationale = "Caption embedding ranking suggests this clip is relevant."
        else:
            raw_retrieval_score = float(raw_retrieval_score)
            confidence = cosine_retrieval_score_to_confidence(raw_retrieval_score)
            rationale = (
                f"Caption embedding cosine similarity {raw_retrieval_score:.4f} suggests this clip is relevant."
            )
        candidate_ranges.append(
            make_candidate_range(
                item["time_start_secs"],
                item["time_end_secs"],
                confidence,
                rationale,
                raw_retrieval_score=raw_retrieval_score,
            )
        )
        evidence.append(
            {
                "text": item["caption"],
                "score": round(confidence, 4),
                "metadata": {
                    "start_secs": item["time_start_secs"],
                    "end_secs": item["time_end_secs"],
                    "raw_retrieval_score": (
                        round(raw_retrieval_score, 4)
                        if raw_retrieval_score is not None
                        else None
                    ),
                },
            }
        )

    observation_text = build_grounding_observation(
        "clip",
        candidate_ranges,
        "Caption retrieval produced candidate clip ranges for visual verification.",
        evidence=evidence,
        confidence=max((item["confidence"] for item in candidate_ranges), default=0.0),
        fallback_hint="Use `frame_confirm_tool` on the best candidate range, or try `subtitle_grounding_tool` if the question depends on speech content.",
    )
    save_grounding_cache(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        observation=json.loads(observation_text),
        build_params=cache_build_params,
        upstream_asset_keys=["database", "raw_video"],
    )
    return observation_text
