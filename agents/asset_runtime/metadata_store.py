"""Metadata cache store for OmniVQA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VIDEO_METADATA_INDEX_VERSION = "video_metadata_index_v1"
_VIDEO_METADATA_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_hhmmss_to_seconds(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    text = text.split(".", 1)[0]
    parts = text.split(":")
    if len(parts) == 2:
        parts = ["00"] + parts
    if len(parts) != 3:
        return 0.0
    try:
        hours, minutes, seconds = [int(part) for part in parts]
    except Exception:
        return 0.0
    return float(hours * 3600 + minutes * 60 + seconds)


def _format_hhmmss(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours = total_seconds // 3600
    remain = total_seconds % 3600
    minutes = remain // 60
    secs = remain % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _count_frames(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return sum(1 for path in frames_dir.glob("frame_n*.jpg") if path.is_file())


def _frame_index_from_secs(
    secs: float,
    *,
    fps: float,
    frame_count: int,
    inclusive_end: bool = False,
) -> int:
    if frame_count <= 0 or fps <= 0:
        return 0
    raw_index = int(round(float(secs) * float(fps)))
    if inclusive_end:
        raw_index -= 1
    return max(0, min(raw_index, frame_count - 1))


def _extract_clip_ranges(payload: dict[str, Any], fps: float, frame_count: int) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("data") or []):
        if not isinstance(item, dict):
            continue
        start_secs = _safe_float(item.get("time_start_secs"))
        end_secs = max(start_secs, _safe_float(item.get("time_end_secs"), start_secs))
        duration_secs = max(0.0, end_secs - start_secs)
        clips.append(
            {
                "clip_index": int(index),
                "start_secs": round(start_secs, 3),
                "end_secs": round(end_secs, 3),
                "duration_secs": round(duration_secs, 3),
                "start_time": _format_hhmmss(start_secs),
                "end_time": _format_hhmmss(end_secs),
                "start_frame_idx": _frame_index_from_secs(
                    start_secs,
                    fps=fps,
                    frame_count=frame_count,
                ),
                "end_frame_idx": _frame_index_from_secs(
                    end_secs,
                    fps=fps,
                    frame_count=frame_count,
                    inclusive_end=True,
                ),
            }
        )
    return clips


def _build_video_metadata_index(runtime) -> dict[str, Any]:
    try:
        payload = json.loads(runtime.paths.database_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    additional_data = payload.get("additional_data") if isinstance(payload, dict) else None
    if not isinstance(additional_data, dict):
        additional_data = {}

    frame_count = _count_frames(runtime.paths.frames_dir)
    fps = _safe_float(additional_data.get("fps"), default=0.0)
    if fps <= 0:
        fps = 1.0

    clip_ranges = _extract_clip_ranges(payload if isinstance(payload, dict) else {}, fps, frame_count)
    video_length_secs = _parse_hhmmss_to_seconds(additional_data.get("video_length"))
    if video_length_secs <= 0.0 and clip_ranges:
        video_length_secs = max(float(item["end_secs"]) for item in clip_ranges)

    subject_registry = additional_data.get("subject_registry")
    subject_registry_size = len(subject_registry) if isinstance(subject_registry, dict) else 0

    return {
        "video_id": runtime.video_id,
        "schema_version": VIDEO_METADATA_INDEX_VERSION,
        "video_root": str(runtime.paths.video_root),
        "database_path": str(runtime.paths.database_path),
        "frames_dir": str(runtime.paths.frames_dir),
        "frames_dir_exists": runtime.paths.frames_dir.is_dir(),
        "frame_count": int(frame_count),
        "fps": round(float(fps), 4),
        "video_length": additional_data.get("video_length") or _format_hhmmss(video_length_secs),
        "video_length_secs": round(float(video_length_secs), 3),
        "clip_count": len(clip_ranges),
        "subject_registry_size": int(subject_registry_size),
        "clip_ranges": clip_ranges,
    }


def get_video_metadata_index(runtime) -> dict[str, Any]:
    asset_key = "video_metadata_index"
    cache_path = runtime.paths.video_metadata_index_path
    memory_key = str(cache_path)
    cached = _VIDEO_METADATA_INDEX_CACHE.get(memory_key)
    if cached is not None:
        runtime.record_cache_event(
            "hit",
            {"asset_key": asset_key, "path": str(cache_path), "tier": "memory"},
        )
        return dict(cached)

    build_params = {"parser_version": VIDEO_METADATA_INDEX_VERSION}
    upstreams = {
        "database": dict(runtime.source_fingerprints.get("database") or {}),
        "frames_dir": dict(runtime.source_fingerprints.get("frames_dir") or {}),
    }
    should_rebuild, _ = runtime.manifest.should_rebuild(
        asset_key,
        expected_path=cache_path,
        upstream_fingerprints=upstreams,
        build_params=build_params,
    )
    if not should_rebuild and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        _VIDEO_METADATA_INDEX_CACHE[memory_key] = dict(cached)
        runtime.manifest.record_asset_hit(asset_key)
        runtime.record_cache_event(
            "hit",
            {"asset_key": asset_key, "path": str(cache_path), "tier": "disk"},
        )
        return dict(cached)

    metadata_index = _build_video_metadata_index(runtime)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(metadata_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runtime.manifest.record_asset_build(
        asset_key,
        path=cache_path,
        format="json",
        upstream_keys=["database", "frames_dir"],
        upstream_fingerprints=upstreams,
        build_params=build_params,
        record_count=len(metadata_index.get("clip_ranges") or []),
    )
    _VIDEO_METADATA_INDEX_CACHE[memory_key] = dict(metadata_index)
    runtime.record_cache_event("build", {"asset_key": asset_key, "path": str(cache_path)})
    return dict(metadata_index)
