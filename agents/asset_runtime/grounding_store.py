"""Query-level grounding cache store for OmniVQA."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .manifest import utc_now_iso
from .resolver import refresh_runtime_source_fingerprints
from .types import VideoAssetRuntime


GROUNDING_CACHE_SCHEMA_VERSION = "grounding_cache_v1"
_QUERY_SIGNATURE_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "around",
    "at",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "is",
    "of",
    "on",
    "or",
    "section",
    "specific",
    "talk",
    "talks",
    "that",
    "the",
    "their",
    "this",
    "to",
    "tips",
    "where",
    "with",
}


def normalize_grounding_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


def _normalize_signature_token(token: str) -> str:
    value = str(token or "").strip().lower()
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("es"):
        value = value[:-2]
    elif len(value) > 3 and value.endswith("s"):
        value = value[:-1]
    return value


def _query_signature_tokens(query: str) -> set[str]:
    normalized = normalize_grounding_query(query)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    signature: set[str] = set()
    for token in tokens:
        if token in _QUERY_SIGNATURE_STOPWORDS:
            continue
        reduced = _normalize_signature_token(token)
        if len(reduced) < 2 or reduced in _QUERY_SIGNATURE_STOPWORDS:
            continue
        signature.add(reduced)
    return signature


def _signature_overlap_ratio(current: set[str], cached: set[str]) -> tuple[float, int]:
    if not current or not cached:
        return 0.0, 0
    overlap_count = len(current & cached)
    ratio = overlap_count / max(min(len(current), len(cached)), 1)
    return ratio, overlap_count


def _stable_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _safe_stem(value: str, *, limit: int = 48) -> str:
    stem = "".join(ch if ch.isalnum() else "_" for ch in str(value or "")).strip("_").lower()
    if not stem:
        stem = "query"
    return stem[:limit].strip("_") or "query"


def _resolve_upstreams(
    runtime: VideoAssetRuntime,
    upstream_asset_keys: list[str] | None,
) -> dict[str, dict[str, Any]]:
    refresh_runtime_source_fingerprints(runtime)
    upstreams: dict[str, dict[str, Any]] = {}
    for key in upstream_asset_keys or []:
        if key in runtime.source_fingerprints:
            upstreams[str(key)] = dict(runtime.source_fingerprints[key] or {})
            continue
        record = runtime.manifest.get_asset_record(str(key))
        if record is not None:
            upstreams[str(key)] = dict(record.fingerprint or {})
    return upstreams


def _cache_payload(
    *,
    tool_name: str,
    query: str,
    top_k: int,
    build_params: dict[str, Any] | None,
    upstream_fingerprints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool_name": str(tool_name),
        "normalized_query": normalize_grounding_query(query),
        "top_k": int(top_k),
        "build_params": dict(build_params or {}),
        "upstreams": {
            str(key): dict(value or {})
            for key, value in sorted(upstream_fingerprints.items())
        },
    }


def build_grounding_cache_key(
    *,
    tool_name: str,
    query: str,
    top_k: int,
    build_params: dict[str, Any] | None,
    upstream_fingerprints: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    payload = _cache_payload(
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=build_params,
        upstream_fingerprints=upstream_fingerprints,
    )
    return _stable_hash(payload), payload


def _cache_path(
    runtime: VideoAssetRuntime,
    *,
    tool_name: str,
    query: str,
    top_k: int,
    build_params: dict[str, Any] | None,
    upstream_fingerprints: dict[str, dict[str, Any]],
) -> tuple[str, Path, dict[str, Any]]:
    cache_key, payload = build_grounding_cache_key(
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=build_params,
        upstream_fingerprints=upstream_fingerprints,
    )
    normalized_query = str(payload["normalized_query"])
    subdir = runtime.paths.grounding_cache_dir / _safe_stem(tool_name)
    path = subdir / f"{_safe_stem(normalized_query)}.{cache_key[:16]}.json"
    return cache_key, path, payload


def _load_cache_entry(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_cache_entry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _prepare_cached_observation(
    path: Path,
    cache_entry: dict[str, Any],
    *,
    tool_name: str,
    cache_key: str,
    normalized_query: str,
    match_type: str,
) -> dict[str, Any] | None:
    observation = cache_entry.get("observation")
    if not isinstance(observation, dict):
        return None
    cache_entry["last_used_at"] = utc_now_iso()
    cache_entry["hit_count"] = int(cache_entry.get("hit_count") or 0) + 1
    try:
        _write_cache_entry(path, cache_entry)
    except Exception:
        pass
    payload_with_cache = dict(observation)
    payload_with_cache["cache_info"] = {
        "hit": True,
        "scope": "grounding_result",
        "tool_name": tool_name,
        "cache_key": cache_key[:16],
        "match_type": match_type,
        "normalized_query": normalized_query,
    }
    return payload_with_cache


def _find_lexical_near_cache(
    runtime: VideoAssetRuntime,
    *,
    tool_name: str,
    tool_dir: Path,
    top_k: int,
    build_params: dict[str, Any],
    upstream_fingerprints: dict[str, dict[str, Any]],
    query: str,
) -> tuple[Path, dict[str, Any], str] | None:
    if not tool_dir.is_dir():
        return None
    current_signature = _query_signature_tokens(query)
    if len(current_signature) < 4:
        return None
    best_match: tuple[float, int, Path, dict[str, Any]] | None = None
    for candidate_path in tool_dir.glob("*.json"):
        candidate = _load_cache_entry(candidate_path)
        if candidate is None:
            continue
        if str(candidate.get("schema_version") or "") != GROUNDING_CACHE_SCHEMA_VERSION:
            continue
        if str(candidate.get("tool_name") or "") != str(tool_name):
            continue
        if int(candidate.get("top_k") or 0) != int(top_k):
            continue
        if dict(candidate.get("build_params") or {}) != dict(build_params or {}):
            continue
        candidate_upstreams = {
            str(key): dict(value or {})
            for key, value in (candidate.get("upstream_fingerprints") or {}).items()
        }
        if candidate_upstreams != dict(upstream_fingerprints or {}):
            continue
        cached_query = str(candidate.get("raw_query") or candidate.get("normalized_query") or "")
        cached_signature = _query_signature_tokens(cached_query)
        ratio, overlap_count = _signature_overlap_ratio(current_signature, cached_signature)
        if overlap_count < 6 or ratio < 0.88:
            continue
        score = (ratio, overlap_count)
        if best_match is None or score > (best_match[0], best_match[1]):
            best_match = (ratio, overlap_count, candidate_path, candidate)
    if best_match is None:
        return None
    _, _, candidate_path, candidate = best_match
    cache_key = str(candidate.get("cache_key") or "")
    return candidate_path, candidate, cache_key


def _record_stale_cache_invalidations(
    runtime: VideoAssetRuntime,
    *,
    tool_name: str,
    cache_path: Path,
    payload: dict[str, Any],
) -> None:
    tool_dir = cache_path.parent
    if not tool_dir.is_dir():
        return
    query_stem = _safe_stem(str(payload.get("normalized_query") or ""))
    for stale_path in tool_dir.glob(f"{query_stem}.*.json"):
        if stale_path.resolve() == cache_path.resolve():
            continue
        stale_entry = _load_cache_entry(stale_path)
        if stale_entry is None:
            continue
        if str(stale_entry.get("tool_name") or "") != str(tool_name):
            continue
        if str(stale_entry.get("normalized_query") or "") != str(payload.get("normalized_query") or ""):
            continue
        if int(stale_entry.get("top_k") or 0) != int(payload.get("top_k") or 0):
            continue
        stale_build_params = dict(stale_entry.get("build_params") or {})
        current_build_params = dict(payload.get("build_params") or {})
        stale_upstreams = {
            str(key): dict(value or {})
            for key, value in (stale_entry.get("upstream_fingerprints") or {}).items()
        }
        current_upstreams = {
            str(key): dict(value or {})
            for key, value in (payload.get("upstreams") or {}).items()
        }
        reason = None
        if str(stale_entry.get("schema_version") or "") != GROUNDING_CACHE_SCHEMA_VERSION:
            reason = "schema_version_changed"
        elif stale_build_params != current_build_params:
            reason = "build_params_changed"
        elif stale_upstreams != current_upstreams:
            reason = "upstream_changed"
        if reason is None:
            continue
        runtime.record_cache_event(
            "invalidate",
            {
                "asset_key": f"{tool_name}_grounding_cache",
                "path": str(stale_path),
                "cache_layer": "grounding",
                "tool_name": tool_name,
                "normalized_query": payload.get("normalized_query"),
                "reason": reason,
            },
        )
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue


def load_grounding_cache(
    runtime: VideoAssetRuntime,
    *,
    tool_name: str,
    query: str,
    top_k: int,
    build_params: dict[str, Any] | None = None,
    upstream_asset_keys: list[str] | None = None,
) -> dict[str, Any] | None:
    upstream_fingerprints = _resolve_upstreams(runtime, upstream_asset_keys)
    cache_key, path, payload = _cache_path(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=build_params,
        upstream_fingerprints=upstream_fingerprints,
    )
    if not path.is_file():
        _record_stale_cache_invalidations(
            runtime,
            tool_name=tool_name,
            cache_path=path,
            payload=payload,
        )
        lexical_match = _find_lexical_near_cache(
            runtime,
            tool_name=tool_name,
            tool_dir=path.parent,
            top_k=top_k,
            build_params=dict(build_params or {}),
            upstream_fingerprints=upstream_fingerprints,
            query=query,
        )
        if lexical_match is None:
            return None
        matched_path, cache_entry, matched_key = lexical_match
        resolved = _prepare_cached_observation(
            matched_path,
            cache_entry,
            tool_name=tool_name,
            cache_key=matched_key,
            normalized_query=payload["normalized_query"],
            match_type="lexical_near",
        )
        if resolved is None:
            return None
        runtime.record_cache_event(
            "hit",
            {
                "asset_key": f"{tool_name}_grounding_cache",
                "path": str(matched_path),
                "cache_layer": "grounding",
                "tool_name": tool_name,
                "cache_key": matched_key[:16],
                "normalized_query": payload["normalized_query"],
                "tier": "disk",
                "match_type": "lexical_near",
            },
        )
        return resolved
    cache_entry = _load_cache_entry(path)
    if cache_entry is None:
        return None
    if str(cache_entry.get("schema_version") or "") != GROUNDING_CACHE_SCHEMA_VERSION:
        runtime.record_cache_event(
            "invalidate",
            {
                "asset_key": f"{tool_name}_grounding_cache",
                "path": str(path),
                "cache_layer": "grounding",
                "tool_name": tool_name,
                "normalized_query": payload["normalized_query"],
                "reason": "schema_version_changed",
            },
        )
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    payload_with_cache = _prepare_cached_observation(
        path,
        cache_entry,
        tool_name=tool_name,
        cache_key=cache_key,
        normalized_query=payload["normalized_query"],
        match_type="exact",
    )
    if payload_with_cache is None:
        return None
    runtime.record_cache_event(
        "hit",
        {
            "asset_key": f"{tool_name}_grounding_cache",
            "path": str(path),
            "cache_layer": "grounding",
            "tool_name": tool_name,
            "cache_key": cache_key[:16],
            "normalized_query": payload["normalized_query"],
            "tier": "disk",
            "match_type": "exact",
        },
    )
    return payload_with_cache


def save_grounding_cache(
    runtime: VideoAssetRuntime,
    *,
    tool_name: str,
    query: str,
    top_k: int,
    observation: dict[str, Any] | None,
    build_params: dict[str, Any] | None = None,
    upstream_asset_keys: list[str] | None = None,
) -> None:
    if not isinstance(observation, dict):
        return
    status = str(observation.get("status") or "")
    if status not in {"ok", "no_match"}:
        return
    upstream_fingerprints = _resolve_upstreams(runtime, upstream_asset_keys)
    cache_key, path, payload = _cache_path(
        runtime,
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        build_params=build_params,
        upstream_fingerprints=upstream_fingerprints,
    )
    entry = {
        "schema_version": GROUNDING_CACHE_SCHEMA_VERSION,
        "tool_name": str(tool_name),
        "video_id": runtime.video_id,
        "cache_key": cache_key,
        "raw_query": str(query or ""),
        "normalized_query": payload["normalized_query"],
        "top_k": int(top_k),
        "build_params": dict(build_params or {}),
        "upstream_fingerprints": upstream_fingerprints,
        "created_at": utc_now_iso(),
        "last_used_at": utc_now_iso(),
        "hit_count": 0,
        "observation": dict(observation),
    }
    _record_stale_cache_invalidations(
        runtime,
        tool_name=tool_name,
        cache_path=path,
        payload=payload,
    )
    _write_cache_entry(path, entry)
    runtime.record_cache_event(
        "build",
        {
            "asset_key": f"{tool_name}_grounding_cache",
            "path": str(path),
            "cache_layer": "grounding",
            "tool_name": tool_name,
            "cache_key": cache_key[:16],
            "normalized_query": payload["normalized_query"],
            "status": status,
        },
    )
