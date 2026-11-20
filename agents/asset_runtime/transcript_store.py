"""Transcript asset store for OmniVQA."""

from __future__ import annotations

import json
from pathlib import Path

from agents.tools.transcript_utils import parse_subtitle_segments, transcribe_audio_segments

from .resolver import refresh_runtime_source_fingerprints
from .types import VideoAssetRuntime


SUBTITLE_SEGMENTS_VERSION = "subtitle_segments_v2"
ASR_SEGMENTS_VERSION = "audio_perception_v1"


def _read_segments(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_segments(path: Path, segments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def _subtitle_upstreams(runtime: VideoAssetRuntime) -> dict[str, dict]:
    return {"raw_subtitle": dict(runtime.source_fingerprints.get("raw_subtitle") or {})}


def _asr_upstreams(runtime: VideoAssetRuntime) -> dict[str, dict]:
    return {
        "raw_audio": dict(runtime.source_fingerprints.get("raw_audio") or {}),
        "raw_video": dict(runtime.source_fingerprints.get("raw_video") or {}),
    }


def _load_subtitle_segments(runtime: VideoAssetRuntime) -> list[dict]:
    subtitle_path = runtime.paths.raw_subtitle_path
    if not subtitle_path.is_file():
        return []

    asset_key = "subtitle_segments"
    cache_path = runtime.paths.subtitle_cache_path
    build_params = {"parser_version": SUBTITLE_SEGMENTS_VERSION}
    upstreams = _subtitle_upstreams(runtime)
    should_rebuild, _ = runtime.manifest.should_rebuild(
        asset_key,
        expected_path=cache_path,
        upstream_fingerprints=upstreams,
        build_params=build_params,
    )
    if not should_rebuild and cache_path.is_file():
        runtime.manifest.record_asset_hit(asset_key)
        runtime.record_cache_event("hit", {"asset_key": asset_key, "path": str(cache_path)})
        return _read_segments(cache_path)

    segments = [
        dict(segment, source="subtitle")
        for segment in parse_subtitle_segments(subtitle_path)
    ]
    _write_segments(cache_path, segments)
    runtime.manifest.record_asset_build(
        asset_key,
        path=cache_path,
        format="json",
        upstream_keys=["raw_subtitle"],
        upstream_fingerprints=upstreams,
        build_params=build_params,
        record_count=len(segments),
    )
    runtime.record_cache_event("build", {"asset_key": asset_key, "path": str(cache_path)})
    return segments


def _load_asr_segments(runtime: VideoAssetRuntime) -> list[dict]:
    asset_key = "whisper_segments"
    cache_path = runtime.paths.whisper_cache_path
    build_params = {
        "asr_model_version": ASR_SEGMENTS_VERSION,
        "chunk_length_s": 30,
        "batch_size": 8,
    }
    upstreams = _asr_upstreams(runtime)
    should_rebuild, _ = runtime.manifest.should_rebuild(
        asset_key,
        expected_path=cache_path,
        upstream_fingerprints=upstreams,
        build_params=build_params,
    )
    if not should_rebuild and cache_path.is_file():
        runtime.manifest.record_asset_hit(asset_key)
        runtime.record_cache_event("hit", {"asset_key": asset_key, "path": str(cache_path)})
        return _read_segments(cache_path)

    segments = transcribe_audio_segments(
        runtime.paths.raw_video_path,
        runtime.paths.raw_audio_path,
        cache_path,
        force_rebuild=True,
    )
    refresh_runtime_source_fingerprints(runtime)
    upstreams = _asr_upstreams(runtime)
    runtime.manifest.record_asset_build(
        asset_key,
        path=cache_path,
        format="json",
        upstream_keys=["raw_audio", "raw_video"],
        upstream_fingerprints=upstreams,
        build_params=build_params,
        record_count=len(segments),
    )
    runtime.record_cache_event("build", {"asset_key": asset_key, "path": str(cache_path)})
    return segments


def load_subtitle_segments(runtime: VideoAssetRuntime) -> tuple[list[dict], str]:
    return _load_subtitle_segments(runtime), "subtitle_segments"


def load_asr_segments(runtime: VideoAssetRuntime) -> tuple[list[dict], str]:
    return _load_asr_segments(runtime), "whisper_segments"


def load_dialogue_segments(
    runtime: VideoAssetRuntime,
    *,
    allow_asr_fallback: bool = True,
) -> tuple[list[dict], str, str]:
    """
    Load normalized dialogue segments.

    Returns `(segments, source, asset_key)`.
    """
    subtitle_segments = _load_subtitle_segments(runtime)
    if subtitle_segments or not allow_asr_fallback:
        return subtitle_segments, "subtitle", "subtitle_segments"

    asr_segments = _load_asr_segments(runtime)
    return asr_segments, "asr", "whisper_segments"
