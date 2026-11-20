from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OMNIVQA_ROOT = Path(__file__).resolve().parents[1]
if str(OMNIVQA_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIVQA_ROOT))

from gui.jsonl_tailer import tail_jsonl
from gui.model_profile import build_env_overrides
from gui.schemas import RunCreateRequest


TERMINAL_STATUSES = {"finished", "failed", "cancelled"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _classify_stdout_line(line: str) -> str:
    text = line.strip()
    if re.match(r"\[run_single\]|preprocess", text, re.IGNORECASE):
        return "pp"
    if re.match(r"[Aa]nswer:", text):
        return "ans"
    if re.search(r"Traceback|ERROR|Exception", text):
        return "err"
    return "log"


@dataclass
class RunState:
    run_id: str
    status: str = "created"
    created_at: str = field(default_factory=utc_now_iso)
    run_dir: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    process: asyncio.subprocess.Process | None = None
    tail_task: asyncio.Task[Any] | None = None
    pump_task: asyncio.Task[Any] | None = None
    finalize_task: asyncio.Task[Any] | None = None
    queue_task: asyncio.Task[Any] | None = None
    stop_event: asyncio.Event | None = None


class RunController:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, RunState] = {}

    async def create_run(self, request: RunCreateRequest) -> dict[str, str]:
        run_id = uuid.uuid4().hex
        run_dir = self.runs_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{run_id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_state = RunState(run_id=run_id, run_dir=run_dir)
        self._runs[run_id] = run_state
        await self._start_run(run_state, request)
        return {"run_id": run_id, "status": "created", "run_dir": str(run_dir)}

    async def cancel_run(self, run_id: str) -> dict[str, str]:
        run_state = self.get_run(run_id)
        if run_state.status in TERMINAL_STATUSES:
            return {"run_id": run_id, "status": run_state.status}

        if run_state.stop_event is not None:
            run_state.stop_event.set()

        process = run_state.process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

        await self._cancel_tasks(run_state)
        run_state.status = "cancelled"
        self._append_event(run_state, "run_failed", {"error": "Run cancelled.", "traceback": ""})
        return {"run_id": run_id, "status": run_state.status}

    def get_events_since(self, run_id: str, last_event_id: int = 0) -> list[dict[str, Any]]:
        run_state = self.get_run(run_id)
        return [
            dict(event)
            for event in run_state.events
            if int(event.get("event_id") or 0) > last_event_id
        ]

    def get_run_status(self, run_id: str) -> str:
        return self.get_run(run_id).status

    def get_run(self, run_id: str) -> RunState:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Run `{run_id}` not found.") from exc

    def _create_state_for_test(self, run_id: str, run_dir: Path) -> RunState:
        run_state = RunState(run_id=run_id, run_dir=run_dir)
        self._runs[run_id] = run_state
        return run_state

    async def _start_run(self, run_state: RunState, request: RunCreateRequest) -> None:
        if run_state.run_dir is None:
            raise ValueError("run_dir is required")
        jsonl_path = run_state.run_dir / "step_trace.jsonl"
        cmd = self._build_command(request, run_state.run_dir)
        env = {**os.environ, **build_env_overrides(request.planner_model, request.perception_model)}
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(OMNIVQA_ROOT),
        )
        stop_event = asyncio.Event()
        step_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        run_state.process = process
        run_state.stop_event = stop_event
        run_state.status = "running"
        run_state.queue_task = asyncio.create_task(self._consume_steps(run_state, step_queue))
        run_state.tail_task = asyncio.create_task(tail_jsonl(jsonl_path, step_queue, stop_event))
        run_state.pump_task = asyncio.create_task(self._pump_stdout(run_state))
        run_state.finalize_task = asyncio.create_task(
            self._wait_and_finalize(run_state, jsonl_path, step_queue)
        )

    def _build_command(self, request: RunCreateRequest, run_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            str(OMNIVQA_ROOT / "scripts" / "run_single.py"),
            "--quiet",
            "--run-dir",
            str(run_dir),
            "--question",
            request.question,
            "--max-iterations",
            str(request.max_iterations),
        ]
        for asset in request.assets:
            if asset.type == "video":
                cmd.extend(["--video", asset.path])
            elif asset.type == "image":
                cmd.extend(["--image", asset.path])
        return cmd

    async def _consume_steps(
        self,
        run_state: RunState,
        queue: asyncio.Queue[tuple[str, dict[str, Any]]],
    ) -> None:
        while True:
            event_type, record = await queue.get()
            if event_type == "step":
                self._append_event(run_state, "step", {"record": record})

    async def _pump_stdout(self, run_state: RunState) -> None:
        process = run_state.process
        if process is None or process.stdout is None:
            return
        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                return
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            self._append_event(
                run_state,
                "log_line",
                {"text": line, "kind": _classify_stdout_line(line)},
            )

    async def _wait_and_finalize(
        self,
        run_state: RunState,
        jsonl_path: Path,
        step_queue: asyncio.Queue[tuple[str, dict[str, Any]]],
    ) -> None:
        process = run_state.process
        if process is None:
            run_state.status = "failed"
            self._append_event(run_state, "run_failed", {"error": "Process was not started.", "traceback": ""})
            return

        try:
            returncode = await process.wait()
            if run_state.stop_event is not None:
                run_state.stop_event.set()
            if run_state.tail_task is not None:
                await run_state.tail_task
            await self._drain_step_queue(run_state, step_queue)

            if run_state.pump_task is not None and not run_state.pump_task.done():
                run_state.pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_state.pump_task

            if returncode == 0:
                payload = self._read_result(run_state)
                run_state.status = "finished"
                self._append_event(
                    run_state,
                    "run_finished",
                    {
                        "final_answer": payload.get("final_answer"),
                        "result_kind": payload.get("result_kind"),
                        "token_snapshot": (payload.get("run_report_summary") or {}).get("token_usage"),
                        "run_dir": str(run_state.run_dir),
                    },
                )
            else:
                payload = self._read_result(run_state)
                run_state.status = "failed"
                self._append_event(
                    run_state,
                    "run_failed",
                    {
                        "error": payload.get("error") or f"Subprocess exited with return code {returncode}.",
                        "traceback": self._read_failure_trace(jsonl_path),
                        "run_dir": str(run_state.run_dir),
                        "result_json_path": payload.get("result_json_path"),
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            run_state.status = "failed"
            self._append_event(
                run_state,
                "run_failed",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
        finally:
            if run_state.queue_task is not None and not run_state.queue_task.done():
                run_state.queue_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_state.queue_task

    async def _drain_step_queue(
        self,
        run_state: RunState,
        queue: asyncio.Queue[tuple[str, dict[str, Any]]],
    ) -> None:
        while not queue.empty():
            event_type, record = queue.get_nowait()
            if event_type == "step":
                self._append_event(run_state, "step", {"record": record})
            queue.task_done()

    async def _cancel_tasks(self, run_state: RunState) -> None:
        tasks = [
            run_state.tail_task,
            run_state.pump_task,
            run_state.finalize_task,
            run_state.queue_task,
        ]
        current = asyncio.current_task()
        for task in tasks:
            if task is not None and task is not current and not task.done():
                task.cancel()
        for task in tasks:
            if task is not None and task is not current:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def _append_event(self, run_state: RunState, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": run_state.next_event_id,
            "event_type": event_type,
            **payload,
        }
        run_state.next_event_id += 1
        run_state.events.append(event)
        return event

    def _read_result(self, run_state: RunState) -> dict[str, Any]:
        if run_state.run_dir is None:
            return {}
        result_path = run_state.run_dir / "result.json"
        if not result_path.is_file():
            return {}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _read_failure_trace(self, jsonl_path: Path) -> str:
        if not jsonl_path.is_file():
            return ""
        try:
            return jsonl_path.read_text(encoding="utf-8")[-8000:]
        except OSError:
            return ""
