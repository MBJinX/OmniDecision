from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


POLL_INTERVAL_SECS = 0.1
READ_CHUNK_SIZE = 65536


async def tail_jsonl(
    path: Path,
    queue: asyncio.Queue[tuple[str, dict[str, Any]]],
    stop_event: asyncio.Event,
) -> None:
    """Incrementally tail a JSONL file and enqueue parsed step records."""
    while not path.exists():
        if stop_event.is_set():
            return
        await asyncio.sleep(POLL_INTERVAL_SECS)

    buffer = ""
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(READ_CHUNK_SIZE)
            if chunk:
                buffer += chunk
                buffer = await _drain_complete_lines(buffer, queue)
                continue

            if stop_event.is_set():
                final_chunk = fh.read()
                if final_chunk:
                    buffer += final_chunk
                buffer = await _drain_complete_lines(buffer, queue)
                stripped = buffer.strip()
                if stripped:
                    await _enqueue_line(stripped, queue)
                return

            await asyncio.sleep(POLL_INTERVAL_SECS)


async def _drain_complete_lines(
    buffer: str,
    queue: asyncio.Queue[tuple[str, dict[str, Any]]],
) -> str:
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        stripped = line.strip()
        if stripped:
            await _enqueue_line(stripped, queue)
    return buffer


async def _enqueue_line(
    line: str,
    queue: asyncio.Queue[tuple[str, dict[str, Any]]],
) -> None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return
    if isinstance(record, dict):
        await queue.put(("step", record))
