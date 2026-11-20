"""Runtime resolver for OmniVQA per-video assets."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .manifest import AssetManifest, fingerprint_path
from .types import VideoAssetPaths, VideoAssetRuntime


_RUNTIME_CACHE: OrderedDict[str, VideoAssetRuntime] = OrderedDict()
_RUNTIME_CACHE_MAXSIZE = max(
    1,
    int(os.environ.get("OMNIVQA_RUNTIME_CACHE_MAXSIZE", "32")),
)
_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac")
_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass")


def _cache_layer_name(payload: dict[str, Any]) -> str:
    if str(payload.get("cache_layer") or "").strip().lower() == "grounding":
        return "grounding_cache"
    return "asset_cache"


def _resolve_database_path(database: Any, video_root: Path) -> Path:
    storage_file = getattr(database, "storage_file", None)
    if storage_file:
        return Path(str(storage_file)).expanduser().resolve()
    return (video_root / "database.json").resolve()


def _resolve_optional_path(value: Any, *, video_root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (video_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _candidate_raw_roots(video_root: Path, additional_data: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ("raw_asset_root", "raw_root"):
        resolved = _resolve_optional_path(additional_data.get(key), video_root=video_root)
        if resolved and resolved not in roots:
            roots.append(resolved)
    for root in (video_root.parent / "raw", video_root):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _resolve_asset_path(
    *,
    additional_data: dict[str, Any],
    explicit_key: str,
    root_candidates: list[Path],
    video_id: str,
    extensions: tuple[str, ...],
    default_extension: str,
    video_root: Path,
) -> Path:
    explicit_path = _resolve_optional_path(additional_data.get(explicit_key), video_root=video_root)
    if explicit_path is not None:
        return explicit_path

    for root in root_candidates:
        for extension in extensions:
            candidate = root / f"{video_id}{extension}"
            if candidate.is_file():
                return candidate

    return root_candidates[0] / f"{video_id}{default_extension}"


def build_video_asset_paths(database: Any) -> VideoAssetPaths:
    additional_data = database.get_additional_data()
    video_root = Path(str(additional_data["video_file_root"])).expanduser().resolve()
    video_id = video_root.name
    raw_roots = _candidate_raw_roots(video_root, additional_data)
    audio_cache_dir = video_root / "audio_cache"
    visual_cache_dir = video_root / "visual_cache"
    grounding_cache_dir = video_root / "grounding_cache"
    captions_dir = video_root / "captions"
    audio_bge_dir = audio_cache_dir / "bge"
    audio_languagebind_dir = audio_cache_dir / "languagebind"
    database_path = _resolve_database_path(database, video_root)
    raw_video_path = _resolve_asset_path(
        additional_data=additional_data,
        explicit_key="raw_video_path",
        root_candidates=raw_roots,
        video_id=video_id,
        extensions=_VIDEO_EXTENSIONS,
        default_extension=".mp4",
        video_root=video_root,
    )
    raw_audio_path = _resolve_asset_path(
        additional_data=additional_data,
        explicit_key="raw_audio_path",
        root_candidates=raw_roots,
        video_id=video_id,
        extensions=_AUDIO_EXTENSIONS,
        default_extension=".wav",
        video_root=video_root,
    )
    raw_subtitle_path = _resolve_asset_path(
        additional_data=additional_data,
        explicit_key="raw_subtitle_path",
        root_candidates=raw_roots,
        video_id=video_id,
        extensions=_SUBTITLE_EXTENSIONS,
        default_extension=".srt",
        video_root=video_root,
    )
    return VideoAssetPaths(
        video_id=video_id,
        video_root=video_root,
        raw_video_path=raw_video_path,
        raw_audio_path=raw_audio_path,
        raw_subtitle_path=raw_subtitle_path,
        frames_dir=video_root / "frames",
        database_path=database_path,
        audio_cache_dir=audio_cache_dir,
        audio_bge_dir=audio_bge_dir,
        audio_languagebind_dir=audio_languagebind_dir,
        captions_dir=captions_dir,
        visual_cache_dir=visual_cache_dir,
        grounding_cache_dir=grounding_cache_dir,
        manifest_path=video_root / "asset_manifest.json",
        whisper_cache_path=audio_cache_dir / "whisper_segments.json",
        subtitle_cache_path=audio_cache_dir / "subtitle_segments.json",
        frame_caption_index_path=visual_cache_dir / "frame_caption_index.json",
        video_metadata_index_path=visual_cache_dir / "video_metadata_index.json",
    )


def compute_source_fingerprints(paths: VideoAssetPaths) -> dict[str, dict[str, Any]]:
    return {
        "raw_video": fingerprint_path(paths.raw_video_path),
        "raw_audio": fingerprint_path(paths.raw_audio_path),
        "raw_subtitle": fingerprint_path(paths.raw_subtitle_path),
        "database": fingerprint_path(paths.database_path),
        "frames_dir": fingerprint_path(paths.frames_dir),
    }


def refresh_runtime_source_fingerprints(runtime: VideoAssetRuntime) -> dict[str, dict[str, Any]]:
    runtime.source_fingerprints = compute_source_fingerprints(runtime.paths)
    runtime.manifest.sync_source_fingerprints(runtime.source_fingerprints)
    return runtime.source_fingerprints


def _remember_runtime(cache_key: str, runtime: VideoAssetRuntime) -> None:
    _RUNTIME_CACHE[cache_key] = runtime
    while len(_RUNTIME_CACHE) > _RUNTIME_CACHE_MAXSIZE:
        _RUNTIME_CACHE.popitem(last=False)


def resolve_video_asset_runtime(database: Any) -> VideoAssetRuntime:
    paths = build_video_asset_paths(database)
    cache_key = str(paths.database_path)
    runtime = _RUNTIME_CACHE.pop(cache_key, None)
    if runtime is None:
        for path in (
            paths.audio_cache_dir,
            paths.audio_bge_dir,
            paths.audio_languagebind_dir,
            paths.captions_dir,
            paths.visual_cache_dir,
            paths.grounding_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        manifest = AssetManifest(
            paths.manifest_path,
            video_id=paths.video_id,
            video_root=paths.video_root,
        )
        runtime = VideoAssetRuntime(
            video_id=paths.video_id,
            paths=paths,
            manifest=manifest,
        )
    else:
        runtime.paths = paths
        runtime.manifest.reload()

    _remember_runtime(cache_key, runtime)
    refresh_runtime_source_fingerprints(runtime)
    return runtime


def clear_runtime_cache_entry(database: Any) -> None:
    paths = build_video_asset_paths(database)
    _RUNTIME_CACHE.pop(str(paths.database_path), None)


def reset_runtime_cache_summary(database: Any) -> None:
    runtime = resolve_video_asset_runtime(database)
    runtime.clear_cache_events()
    runtime.manifest.consume_events()


def consume_runtime_cache_summary(database: Any) -> dict[str, Any]:
    runtime = resolve_video_asset_runtime(database)
    event_buckets = runtime.snapshot_cache_events()
    manifest_events = runtime.manifest.consume_events()
    runtime.clear_cache_events()
    manifest_invalidate_count = sum(
        1 for item in manifest_events if str(item.get("event") or "") == "invalidate"
    )
    payload = {
        "video_id": runtime.video_id,
        "events": event_buckets,
        "manifest_events": manifest_events,
    }
    layer_counts = {
        "asset_cache": {"hit_count": 0, "build_count": 0, "invalidate_count": 0},
        "grounding_cache": {"hit_count": 0, "build_count": 0, "invalidate_count": 0},
    }
    for event_name in ("hit", "build", "invalidate"):
        for item in event_buckets.get(event_name) or []:
            layer_counts[_cache_layer_name(item)][f"{event_name}_count"] += 1
    layer_counts["asset_cache"]["invalidate_count"] += manifest_invalidate_count
    payload.update(layer_counts)
    counts = {
        "hit_count": len(event_buckets.get("hit") or []),
        "build_count": len(event_buckets.get("build") or []),
        "invalidate_count": len(event_buckets.get("invalidate") or []) + manifest_invalidate_count,
    }
    payload.update(counts)
    payload["has_activity"] = bool(
        counts["hit_count"] or counts["build_count"] or counts["invalidate_count"] or manifest_events
    )
    return payload
