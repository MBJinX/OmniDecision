from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


OMNIVQA_ROOT = Path(__file__).resolve().parents[1]
if str(OMNIVQA_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIVQA_ROOT))

from gui.run_controller import RunController
from gui.schemas import RunCreateRequest, StepTraceLoadRequest


GUI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OMNIVQA_ROOT / "outputs"
templates = Jinja2Templates(directory=str(GUI_ROOT / "templates"))
controller = RunController(runs_dir=PROJECT_ROOT / "gui")
STATIC_VERSION = str(
    int(
        max(
            (GUI_ROOT / "static" / "app.js").stat().st_mtime,
            (GUI_ROOT / "static" / "style.css").stat().st_mtime,
        )
    )
)
DEFAULT_FORM_VALUES = {
    "question": "",
    "video_db_path": "",
    "caption_path": "",
    "max_iterations": 15,
}
MAX_LOAD_SIZE = 50 * 1024 * 1024

app = FastAPI(title="OmniVQA GUI", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(GUI_ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_form": DEFAULT_FORM_VALUES,
            "static_version": STATIC_VERSION,
        },
    )


@app.get("/api/meta")
async def get_meta() -> JSONResponse:
    return JSONResponse({"version": "0.2.0"})


@app.post("/api/runs")
async def create_run(payload: RunCreateRequest) -> JSONResponse:
    try:
        result = await controller.create_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    try:
        controller.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        initial_cursor = int(request.headers.get("last-event-id", 0) or 0)
    except ValueError:
        initial_cursor = 0

    async def event_stream():
        cursor = initial_cursor
        while True:
            try:
                events = controller.get_events_since(run_id, cursor)
                status = controller.get_run_status(run_id)
            except KeyError:
                break

            if events:
                for event in events:
                    event_id = int(event.get("event_id") or 0)
                    cursor = max(cursor, event_id)
                    event_type = str(event.get("event_type") or "message")
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"
                    if event_type in {"run_finished", "run_failed"}:
                        return
                continue

            if status in {"finished", "failed", "cancelled"}:
                break

            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> JSONResponse:
    try:
        result = await controller.cancel_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/load")
async def load_step_trace(payload: StepTraceLoadRequest) -> JSONResponse:
    path = _resolve_load_path(payload.path)
    max_size = _get_max_load_size()
    if path.stat().st_size > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Open the trace locally.",
        )
    return JSONResponse({"records": _read_jsonl_records(path)})


def _resolve_load_path(path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    path = raw_path.resolve(strict=False)
    if path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl step trace files are supported.")
    if not _is_allowed_load_path(path):
        raise HTTPException(status_code=400, detail=f"Path is outside allowed roots: {path}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Step trace not found: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
    return path


def _is_allowed_load_path(path: Path) -> bool:
    allowed_roots = [PROJECT_ROOT]
    configured_root = os.environ.get("OMNIVQA_RUNS_ROOT")
    if configured_root:
        allowed_roots.append(Path(configured_root).expanduser().resolve(strict=False))
    return any(_is_relative_to(path, root) for root in allowed_roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _get_max_load_size() -> int:
    value = os.environ.get("OMNIVQA_GUI_MAX_LOAD_MB")
    if value is None or not value.strip():
        return MAX_LOAD_SIZE
    try:
        megabytes = float(value)
    except ValueError:
        return MAX_LOAD_SIZE
    return max(0, int(megabytes * 1024 * 1024))


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OmniVQA local GUI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
