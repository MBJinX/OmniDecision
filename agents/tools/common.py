"""Shared helpers for OmniVQA grounding and verification tools."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from agents.build_database import convert_seconds_to_hhmmss


def emit_runtime_progress(message: str) -> None:
    setting = str(os.environ.get("OMNIVQA_PROGRESS_LOGS", "auto")).strip().lower()
    if setting in {"0", "false", "off", "no"}:
        return
    if setting == "auto" and not sys.stdout.isatty():
        return
    print(f"[progress] {message}", flush=True)


def dumps_observation(payload: dict) -> str:
    """Serialize a tool observation to a stable JSON string."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_failure_observation(
    source_modality: str,
    summary: str,
    fallback_hint: str,
    status: str = "unavailable",
) -> str:
    return dumps_observation(
        {
            "status": status,
            "source_modality": source_modality,
            "summary": summary,
            "fallback_hint": fallback_hint,
        }
    )


def build_grounding_observation(
    source_modality: str,
    candidate_ranges: list[dict],
    summary: str,
    evidence: list[dict] | None = None,
    confidence: float | None = None,
    fallback_hint: str | None = None,
    status: str = "ok",
) -> str:
    payload = {
        "status": status,
        "source_modality": source_modality,
        "candidate_ranges": candidate_ranges,
        "summary": summary,
        "confidence": round(
            confidence
            if confidence is not None
            else max((item.get("confidence", 0.0) for item in candidate_ranges), default=0.0),
            4,
        ),
        "evidence": evidence or [],
        "fallback_hint": fallback_hint
        or "Use `frame_confirm_tool` to visually verify the top candidate range.",
    }
    return dumps_observation(payload)


def build_visual_observation(
    observations: list[str],
    time_ranges: list[list[str]],
    summary: str,
    confidence: float | None = None,
    status: str = "ok",
    range_observations: list[dict] | None = None,
    verified_range_count: int | None = None,
    requested_range_count: int | None = None,
    coverage_ratio: float | None = None,
    all_ranges_verified: bool | None = None,
) -> str:
    resolved_confidence = 0.0 if confidence is None else confidence
    payload = {
        "status": status,
        "source_modality": "visual",
        "time_ranges": time_ranges,
        "observations": observations,
        "summary": summary,
        "confidence": round(resolved_confidence, 4),
        "fallback_hint": "Answer only if the visual evidence is sufficient; otherwise re-ground.",
    }
    if range_observations:
        payload["range_observations"] = range_observations
    if verified_range_count is not None:
        payload["verified_range_count"] = int(verified_range_count)
    if requested_range_count is not None:
        payload["requested_range_count"] = int(requested_range_count)
    if coverage_ratio is not None:
        payload["coverage_ratio"] = round(float(coverage_ratio), 4)
    if all_ranges_verified is not None:
        payload["all_ranges_verified"] = bool(all_ranges_verified)
    return dumps_observation(payload)


def build_web_observation(
    evidence: list[dict],
    summary: str,
    confidence: float = 0.72,
    status: str = "ok",
    fallback_hint: str | None = None,
    search_feedback: dict | None = None,
) -> str:
    payload = {
        "status": status,
        "source_modality": "web",
        "candidate_ranges": [],
        "summary": summary,
        "confidence": round(confidence, 4),
        "evidence": evidence,
        "fallback_hint": fallback_hint
        or "Use web evidence only to fill exact gaps that are not explicitly available in the video, and keep it consistent with the grounded video evidence.",
    }
    if isinstance(search_feedback, dict) and search_feedback:
        payload["search_feedback"] = search_feedback
    return dumps_observation(payload)


def build_computation_observation(
    summary: str,
    *,
    stdout: str = "",
    stderr: str = "",
    result: str = "",
    status: str = "ok",
    confidence: float = 1.0,
    fallback_hint: str | None = None,
    code_preview: str | None = None,
    error_type: str | None = None,
    blocked_import: str | None = None,
    missing_symbol: str | None = None,
    stderr_preview: str | None = None,
    retry_hint: str | None = None,
    exception_type: str | None = None,
) -> str:
    payload = {
        "status": status,
        "source_modality": "computation",
        "summary": summary,
        "confidence": round(float(confidence), 4),
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
        "fallback_hint": fallback_hint
        or "If the computation output is usable, incorporate it into the final answer faithfully.",
    }
    if code_preview:
        payload["code_preview"] = code_preview
    if error_type:
        payload["error_type"] = error_type
    if blocked_import:
        payload["blocked_import"] = blocked_import
    if missing_symbol:
        payload["missing_symbol"] = missing_symbol
    if stderr_preview:
        payload["stderr_preview"] = stderr_preview
    if retry_hint:
        payload["retry_hint"] = retry_hint
    if exception_type:
        payload["exception_type"] = exception_type
    return dumps_observation(payload)


def seconds_to_hhmmss(seconds: float) -> str:
    return convert_seconds_to_hhmmss(max(0, int(round(seconds))))


def make_candidate_range(
    start_secs: float,
    end_secs: float,
    confidence: float,
    rationale: str,
    raw_retrieval_score: float | None = None,
) -> dict:
    payload = {
        "start_time": seconds_to_hhmmss(start_secs),
        "end_time": seconds_to_hhmmss(max(end_secs, start_secs + 1)),
        "confidence": round(float(confidence), 4),
        "rationale": rationale,
    }
    if raw_retrieval_score is not None:
        payload["raw_retrieval_score"] = round(float(raw_retrieval_score), 4)
    return payload


def expand_time_range(
    start_secs: float,
    end_secs: float,
    *,
    left_pad: float = 1.5,
    right_pad: float = 1.5,
    max_duration_secs: float | None = None,
) -> tuple[float, float]:
    start = max(0.0, start_secs - left_pad)
    end = max(start + 1.0, end_secs + right_pad)
    if max_duration_secs is not None:
        end = min(end, max_duration_secs)
    return start, end


def score_texts(query: str, texts: list[str]) -> np.ndarray:
    """Rank transcript segments by cosine similarity in the configured embedding space."""
    from agents import config
    from agents.shared.utils import AzureOpenAIEmbeddingService

    items = [query, *texts]
    vectors = []
    for offset in range(0, len(items), config.AOAI_EMBEDDING_BATCH_SIZE):
        rows = AzureOpenAIEmbeddingService.get_embeddings(
            (), config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            items[offset:offset + config.AOAI_EMBEDDING_BATCH_SIZE],
            config.EMBEDDING_API_KEY, config.EMBEDDING_BASE_URL, config.AOAI_EMBEDDING_LARGE_DIM,
        )
        if not rows:
            raise RuntimeError("Transcript embedding request failed.")
        vectors.extend(row["embedding"] for row in rows)
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix[1:] @ matrix[0]


def cosine_retrieval_score_to_confidence(score: float) -> float:
    normalized = (float(score) + 1.0) / 2.0
    return float(np.clip(0.05 + 0.94 * normalized, 0.05, 0.99))
