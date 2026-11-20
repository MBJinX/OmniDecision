from __future__ import annotations

import atexit
import json
import re
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Iterator, Literal

try:
    from agents.logging.trace_render import (
        _strip_ansi,
        _truncate,
        render_record,
        resolve_ansi_enabled,
    )
except ModuleNotFoundError:
    from .trace_render import (
        _strip_ansi,
        _truncate,
        render_record,
        resolve_ansi_enabled,
    )


_BASE64_DATA_URI_RE = re.compile(r"data:[^;]{0,40};base64,[A-Za-z0-9+/=]{64,}")
_BASE64_STANDALONE_RE = re.compile(r"[A-Za-z0-9+/]{128,}={0,2}")

_JSONL_FIELD_CAP = 65536


def _sanitize_for_jsonl(value: Any) -> Any:
    """Sanitize a value for JSONL storage.

    Applies two safety transforms generically (no tool- or benchmark-specific logic):
    1. Replaces base64/binary blobs with a placeholder.
    2. Applies a 64KB hard cap per string field.

    Recursively processes dicts and lists. All other types pass through unchanged.
    """
    if isinstance(value, str):
        # Strip ANSI escape codes first
        text = _strip_ansi(value)
        # Apply 64KB hard cap
        if len(text) > _JSONL_FIELD_CAP:
            total = len(text)
            text = text[:_JSONL_FIELD_CAP] + f" ...[truncated: {total} chars total]"
        # Replace data URI base64 blobs
        def _replace_data_uri(m: re.Match) -> str:
            return f"<base64 {len(m.group(0))} bytes omitted>"
        text = _BASE64_DATA_URI_RE.sub(_replace_data_uri, text)
        # Replace standalone base64 runs
        def _replace_standalone(m: re.Match) -> str:
            return f"<base64 {len(m.group(0))} bytes omitted>"
        text = _BASE64_STANDALONE_RE.sub(_replace_standalone, text)
        return text
    if isinstance(value, dict):
        return {k: _sanitize_for_jsonl(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_jsonl(item) for item in value]
    return value


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


_current_handle: ContextVar["StepHandle | None"] = ContextVar("omnivqa_current_step_handle", default=None)


def current_step_handle() -> "StepHandle | None":
    return _current_handle.get()


@dataclass
class _CaseState:
    case_id: str
    step: int = 0
    question: str = ""
    depth: int = 0
    pending_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepHandle:
    role: str
    step: int
    case_id: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    latency_ms: int | None = None
    status: str | None = None
    input_preview: str = ""
    output_preview: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def attach_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        caller: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        response_text: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Attach one sub-call usage sample and accumulate top-level totals."""
        self.model = self.model or model
        self.tokens_in = (self.tokens_in or 0) + int(prompt_tokens)
        self.tokens_out = (self.tokens_out or 0) + int(completion_tokens)
        self.reasoning_tokens = (self.reasoning_tokens or 0) + int(reasoning_tokens)
        self.cached_tokens = (self.cached_tokens or 0) + int(cached_tokens)
        sub_call: dict[str, Any] = {
            "caller": caller,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }
        if messages is not None:
            sub_call["messages"] = messages
        if response_text is not None:
            sub_call["response_text"] = response_text
        if tool_calls:
            sub_call["tool_calls"] = tool_calls
        if finish_reason is not None:
            sub_call["finish_reason"] = finish_reason
        self.extra.setdefault("sub_calls", []).append(sub_call)

    def attach_io(
        self,
        *,
        input_preview: str | None = None,
        output_preview: str | None = None,
    ) -> None:
        if input_preview is not None:
            self.input_preview = input_preview
        if output_preview is not None:
            self.output_preview = output_preview

    def to_record(self) -> dict[str, Any]:
        status = self.status or "ok"
        return {
            "step": self.step,
            "ts": _iso_now(),
            "case_id": self.case_id,
            "role": self.role,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_ms": self.latency_ms,
            "status": status,
            "input_preview": _sanitize_for_jsonl(str(self.input_preview).strip()),
            "output_preview": _sanitize_for_jsonl(str(self.output_preview).strip()),
            "extra": _sanitize_for_jsonl(dict(self.extra)),
        }


class StepLogger:
    def __init__(
        self,
        run_dir: Path,
        *,
        ansi: Literal["auto", "always", "never"] = "auto",
        quiet: bool = False,
        path: Path | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._quiet = quiet
        self._ansi_enabled = resolve_ansi_enabled(ansi)
        self._jsonl_path = Path(path) if path else self._run_dir / "step_trace.jsonl"
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_fp = open(self._jsonl_path, "a", encoding="utf-8", buffering=1)
        txt_path = self._jsonl_path.with_suffix(".txt")
        self._txt_fp: IO[str] | None = open(txt_path, "a", encoding="utf-8", buffering=1)
        self._txt_sink_failed: bool = False
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._per_thread: dict[int, _CaseState] = {}
        self._closed = False

        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
        atexit.register(self.close)

    def case_start(self, case_id: str, question: str) -> None:
        with self._state_lock:
            self._per_thread[threading.get_ident()] = _CaseState(
                case_id=case_id,
                step=0,
                question=question,
            )
        record = self._build_case_record(
            case_id,
            event="start",
            status="ok",
            question=question,
            step_count=0,
        )
        self._emit_record(record)

    def case_end(
        self,
        case_id: str,
        *,
        status: str,
        answer: str | None,
        token_snapshot: dict | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._state_lock:
            state = self._per_thread.pop(threading.get_ident(), None)
        step_count = state.step if state else 0
        record = self._build_case_record(
            case_id,
            event="end",
            status=status,
            answer=answer,
            token_snapshot=token_snapshot,
            step_count=step_count,
            extra=extra,
        )
        self._emit_record(record)

    @contextmanager
    def span(self, role: str, **extra: Any) -> Iterator[StepHandle]:
        with self._state_lock:
            state = self._per_thread.get(threading.get_ident())
            if state is None:
                raise RuntimeError("span() called outside case_start/case_end")
            state.step += 1
            step_no = state.step
            case_id = state.case_id
            state.depth += 1

        handle = StepHandle(role=role, step=step_no, case_id=case_id, extra=dict(extra))
        token = _current_handle.set(handle)
        started = time.perf_counter()
        try:
            yield handle
            handle.status = handle.status or "ok"
        except Exception as exc:
            handle.status = "error"
            handle.extra["error_type"] = type(exc).__name__
            handle.extra["error_msg"] = _truncate(repr(exc), 240)
            raise
        finally:
            handle.latency_ms = int((time.perf_counter() - started) * 1000)
            _current_handle.reset(token)
            record = handle.to_record()
            records_to_emit: list[dict[str, Any]] = []
            with self._state_lock:
                state = self._per_thread.get(threading.get_ident())
                if state is None or state.depth <= 0:
                    records_to_emit = [record]
                elif state.depth == 1:
                    records_to_emit = [record]
                    if state.pending_records:
                        records_to_emit.extend(
                            sorted(
                                state.pending_records,
                                key=lambda item: int(item.get("step") or 0),
                            )
                        )
                        state.pending_records.clear()
                    state.depth = 0
                else:
                    state.pending_records.append(record)
                    state.depth -= 1
            for item in records_to_emit:
                self._emit_record(item)

    def close(self) -> None:
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            self._jsonl_fp.flush()
            self._jsonl_fp.close()
            if self._txt_fp is not None:
                self._txt_fp.flush()
                self._txt_fp.close()
                self._txt_fp = None

    def _build_case_record(
        self,
        case_id: str,
        *,
        event: str,
        status: str,
        question: str | None = None,
        answer: str | None = None,
        token_snapshot: dict | None = None,
        step_count: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_preview = answer or ""
        extra_payload: dict[str, Any] = {"event": event, "step_count": step_count}
        if event == "end":
            extra_payload["final_status"] = status
        if token_snapshot is not None:
            extra_payload["token_snapshot"] = token_snapshot
        if extra:
            extra_payload.update(extra)
        return {
            "step": 0,
            "ts": _iso_now(),
            "case_id": case_id,
            "role": "case",
            "event": event,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "latency_ms": None,
            "status": status,
            "input_preview": _sanitize_for_jsonl(str(question or "").strip()),
            "output_preview": _sanitize_for_jsonl(str(output_preview).strip()),
            "extra": extra_payload,
        }

    def _emit_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            if self._closed:
                return
            self._jsonl_fp.write(line + "\n")
            self._jsonl_fp.flush()
            if self._txt_fp is not None:
                try:
                    self._txt_fp.write(render_record(record, full=True, color="never"))
                    self._txt_fp.flush()
                except Exception as _exc:
                    if not self._txt_sink_failed:
                        self._txt_sink_failed = True
                        import sys as _sys

                        print(f"[step_logger] txt sink error (suppressed): {_exc}", file=_sys.stderr)
            if not self._quiet:
                sys.stdout.write(self._format_human_block(record))
                sys.stdout.flush()

    def _format_human_block(self, record: dict[str, Any]) -> str:
        return render_record(
            record,
            full=False,
            color="always" if self._ansi_enabled else "never",
        )
