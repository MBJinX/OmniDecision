"""Shared asset runtime data structures for OmniVQA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetFingerprint:
    size: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None
    exists: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exists": bool(self.exists),
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }
        if self.sha256:
            payload["sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AssetFingerprint":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            size=payload.get("size"),
            mtime_ns=payload.get("mtime_ns"),
            sha256=payload.get("sha256"),
            exists=bool(payload.get("exists", False)),
        )


@dataclass
class AssetRecord:
    key: str
    path: str
    format: str
    upstream_keys: list[str]
    fingerprint: dict[str, Any]
    build_params: dict[str, Any]
    upstream_fingerprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    record_count: int | None = None
    created_at: str | None = None
    last_used_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "format": self.format,
            "upstream_keys": list(self.upstream_keys),
            "fingerprint": dict(self.fingerprint),
            "build_params": dict(self.build_params),
            "upstream_fingerprints": dict(self.upstream_fingerprints),
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }
        if self.record_count is not None:
            payload["record_count"] = int(self.record_count)
        return payload

    @classmethod
    def from_dict(cls, key: str, payload: dict[str, Any] | None) -> "AssetRecord | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            key=key,
            path=str(payload.get("path") or ""),
            format=str(payload.get("format") or "unknown"),
            upstream_keys=[str(item) for item in payload.get("upstream_keys") or []],
            fingerprint=dict(payload.get("fingerprint") or {}),
            build_params=dict(payload.get("build_params") or {}),
            upstream_fingerprints={
                str(name): dict(value or {})
                for name, value in (payload.get("upstream_fingerprints") or {}).items()
            },
            record_count=payload.get("record_count"),
            created_at=payload.get("created_at"),
            last_used_at=payload.get("last_used_at"),
        )


@dataclass(frozen=True)
class VideoAssetPaths:
    video_id: str
    video_root: Path
    raw_video_path: Path
    raw_audio_path: Path
    raw_subtitle_path: Path
    frames_dir: Path
    database_path: Path
    audio_cache_dir: Path
    audio_bge_dir: Path
    audio_languagebind_dir: Path
    captions_dir: Path
    visual_cache_dir: Path
    grounding_cache_dir: Path
    manifest_path: Path
    whisper_cache_path: Path
    subtitle_cache_path: Path
    frame_caption_index_path: Path
    video_metadata_index_path: Path


@dataclass
class VideoAssetRuntime:
    video_id: str
    paths: VideoAssetPaths
    manifest: Any
    source_fingerprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_summary: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record_cache_event(self, event: str, payload: dict[str, Any]) -> None:
        bucket = self.cache_summary.setdefault(event, [])
        bucket.append(dict(payload))

    def clear_cache_events(self) -> None:
        self.cache_summary = {}

    def snapshot_cache_events(self) -> dict[str, list[dict[str, Any]]]:
        return {
            str(event): [dict(item) for item in items]
            for event, items in self.cache_summary.items()
        }
