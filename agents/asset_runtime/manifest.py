"""Manifest helpers for the OmniVQA asset/cache runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import AssetFingerprint, AssetRecord


SCHEMA_VERSION = "v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fingerprint_path(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return AssetFingerprint().to_dict()
    resolved = Path(path).expanduser()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return AssetFingerprint().to_dict()
    return AssetFingerprint(
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        exists=True,
    ).to_dict()


class AssetManifest:
    """Read/write the per-video asset manifest and track cache events."""

    def __init__(self, manifest_path: Path, *, video_id: str, video_root: Path) -> None:
        self.manifest_path = manifest_path
        self.video_id = video_id
        self.video_root = video_root
        self.data: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        if self.manifest_path.is_file():
            try:
                self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        if not isinstance(self.data, dict):
            self.data = {}
        self.data.setdefault("schema_version", SCHEMA_VERSION)
        self.data.setdefault("video_id", self.video_id)
        self.data.setdefault("video_root", str(self.video_root))
        self.data.setdefault("source_fingerprints", {})
        self.data.setdefault("assets", {})

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def sync_source_fingerprints(self, source_fingerprints: dict[str, dict[str, Any]]) -> None:
        self.data["source_fingerprints"] = {
            str(key): dict(value or {})
            for key, value in source_fingerprints.items()
        }
        self.save()

    def get_asset_record(self, asset_key: str) -> AssetRecord | None:
        assets = self.data.get("assets") or {}
        return AssetRecord.from_dict(asset_key, assets.get(asset_key))

    def _record_event(self, event: str, asset_key: str, **payload: Any) -> None:
        item = {"event": event, "asset_key": asset_key, **payload}
        self.events.append(item)

    def consume_events(self) -> list[dict[str, Any]]:
        events = [dict(item) for item in self.events]
        self.events = []
        return events

    def _make_relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.video_root.resolve()))
        except Exception:
            return str(path)

    def should_rebuild(
        self,
        asset_key: str,
        *,
        expected_path: Path,
        upstream_fingerprints: dict[str, dict[str, Any]],
        build_params: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        build_params = dict(build_params or {})
        record = self.get_asset_record(asset_key)
        if record is None:
            return True, "missing_manifest_record"

        if not expected_path.is_file():
            self._record_event("invalidate", asset_key, reason="asset_file_missing")
            return True, "asset_file_missing"

        expected_relpath = self._make_relative_path(expected_path)
        if record.path != expected_relpath:
            self._record_event("invalidate", asset_key, reason="path_changed")
            return True, "path_changed"

        if dict(record.build_params or {}) != build_params:
            self._record_event("invalidate", asset_key, reason="build_params_changed")
            return True, "build_params_changed"

        saved_upstreams = dict(record.upstream_fingerprints or {})
        if saved_upstreams != dict(upstream_fingerprints or {}):
            self._record_event("invalidate", asset_key, reason="upstream_changed")
            return True, "upstream_changed"

        return False, None

    def record_asset_build(
        self,
        asset_key: str,
        *,
        path: Path,
        format: str,
        upstream_keys: list[str],
        upstream_fingerprints: dict[str, dict[str, Any]],
        build_params: dict[str, Any] | None = None,
        record_count: int | None = None,
    ) -> None:
        now = utc_now_iso()
        record = AssetRecord(
            key=asset_key,
            path=self._make_relative_path(path),
            format=format,
            upstream_keys=list(upstream_keys),
            fingerprint=fingerprint_path(path),
            build_params=dict(build_params or {}),
            upstream_fingerprints={
                str(key): dict(value or {})
                for key, value in upstream_fingerprints.items()
            },
            record_count=record_count,
            created_at=now,
            last_used_at=now,
        )
        self.data.setdefault("assets", {})[asset_key] = record.to_dict()
        self._record_event("build", asset_key, path=record.path)
        self.save()

    def record_asset_hit(self, asset_key: str) -> None:
        assets = self.data.setdefault("assets", {})
        payload = assets.get(asset_key)
        if not isinstance(payload, dict):
            return
        payload["last_used_at"] = utc_now_iso()
        self._record_event("hit", asset_key, path=str(payload.get("path") or ""))
        self.save()
