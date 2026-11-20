"""Inference cache store for OmniVQA."""

from __future__ import annotations

import json
from pathlib import Path

from .types import VideoAssetRuntime


FRAME_CAPTION_INDEX_VERSION = "frame_caption_index_v1"
_FRAME_CAPTION_INDEX_CACHE: dict[str, list[dict]] = {}


def _extract_frame_captions(database_path: Path) -> list[dict]:
    try:
        payload = json.loads(database_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    captions: list[dict] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        caption = str(item.get("caption") or "").strip()
        if not caption:
            continue
        try:
            start_secs = float(item.get("time_start_secs") or 0.0)
            end_secs = float(item.get("time_end_secs") or start_secs)
        except Exception:
            continue
        captions.append(
            {
                "start_secs": start_secs,
                "end_secs": max(end_secs, start_secs),
                "caption": caption,
            }
        )
    return captions


def get_frame_caption_index(runtime: VideoAssetRuntime) -> list[dict]:
    asset_key = "frame_caption_index"
    cache_path = runtime.paths.frame_caption_index_path
    memory_key = str(cache_path)
    cached = _FRAME_CAPTION_INDEX_CACHE.get(memory_key)
    if cached is not None:
        runtime.record_cache_event("hit", {"asset_key": asset_key, "path": str(cache_path), "tier": "memory"})
        return cached

    build_params = {"parser_version": FRAME_CAPTION_INDEX_VERSION}
    upstreams = {"database": dict(runtime.source_fingerprints.get("database") or {})}
    should_rebuild, _ = runtime.manifest.should_rebuild(
        asset_key,
        expected_path=cache_path,
        upstream_fingerprints=upstreams,
        build_params=build_params,
    )
    if not should_rebuild and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        _FRAME_CAPTION_INDEX_CACHE[memory_key] = cached
        runtime.manifest.record_asset_hit(asset_key)
        runtime.record_cache_event("hit", {"asset_key": asset_key, "path": str(cache_path), "tier": "disk"})
        return cached

    captions = _extract_frame_captions(runtime.paths.database_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime.manifest.record_asset_build(
        asset_key,
        path=cache_path,
        format="json",
        upstream_keys=["database"],
        upstream_fingerprints=upstreams,
        build_params=build_params,
        record_count=len(captions),
    )
    _FRAME_CAPTION_INDEX_CACHE[memory_key] = captions
    runtime.record_cache_event("build", {"asset_key": asset_key, "path": str(cache_path)})
    return captions
