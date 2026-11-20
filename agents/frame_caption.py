import copy
import functools
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

import agents.config as config
from agents.shared.perception import perception_call
from agents.shared.utils import (
    ContentBlockedError,
    LLMCallFailure,
    call_openai_model_with_tools,
)

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

messages = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": "",
    },
]


CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output template:
{
  "clip_start_time": CLIP_START_TIME,
  "clip_end_time": CLIP_END_TIME,
  "subject_registry": {
    <subject_i>: {
      "name": <fill with short identity if name is unknown>,
      "appearance": <list of appearance descriptions>,
      "identity": <list of identity descriptions>,
      "first_seen": <timestamp>
    },
    ...
  },
  "clip_description": <smooth and detailed natural narration of the video clip>
}
"""


MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects obviously refer to the same person, merge them
   (keep earliest `first_seen` time and union all fields).

Input (list of JSON objects):
REGISTRIES_PLACEHOLDER

Return *only* the merged `subject_registry` JSON object.
"""

SYSTEM_PROMPT = "You are a helpful assistant."

# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #
def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def gather_clip_frames(
    video_frame_folder, clip_secs: int, subtitle_file_path: str = None
) -> Dict[str, Dict]:
    # Fix possible typo in the earlier list-comprehension and gather frames again
    frame_files = sorted(
        [f for f in os.listdir(video_frame_folder) if f.startswith("frame") and f.endswith(".jpg")],
        key=lambda x: float(x.split("_n")[-1].rstrip(".jpg")),
    )
    if not frame_files:
        return {}

    # Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # Map timestamps → file names for quick lookup
    frame_ts = [float(f.split("_n")[-1].rstrip(".jpg")) / config.VIDEO_FPS for f in frame_files]
    ts_to_file = dict(zip(frame_ts, frame_files))
    last_ts = int(max(frame_ts))

    result = []

    # Iterate over fixed-length clips
    clip_start = 0
    while clip_start <= last_ts:
        clip_end = min(clip_start + clip_secs - 1, last_ts)

        # Collect frames that fall inside the current clip
        clip_files = [
            os.path.join(video_frame_folder, ts_to_file[t])
            for t in frame_ts
            if clip_start <= t <= clip_end
        ]

        # Aggregate transcript text overlapping the clip interval
        transcript_parts: List[str] = []
        for key, text in subtitle_map.items():
            s, e = map(int, key.split("_"))
            if s <= clip_end and e >= clip_start:  # overlap check
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip() or "No transcript."

        result.append((
                f"{clip_start}_{clip_end}", 
                {"files": clip_files, "transcript": transcript}
        ))

        clip_start += clip_secs
    return result


# --------------------------------------------------------------------------- #
#                   Subtitle (.srt) parsing helper function                    #
# --------------------------------------------------------------------------- #
def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' to seconds (float)."""
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an .srt file and return a mapping
    '{startSec_endSec}': 'subtitle text'.
    """
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}
    with open(srt_path, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]

    idx = 0
    n = len(lines)
    while idx < n:
        # Skip sequential index if present
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break

        # Time-range line
        if "-->" not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split("-->")]
        start_sec = int(_timestamp_to_seconds(start_ts))
        end_sec = int(_timestamp_to_seconds(end_ts))
        idx += 1

        # Collect subtitle text (may span multiple lines)
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = " ".join(subtitle_lines)
        key = f"{start_sec}_{end_sec}"
        if key in result:  # append if duplicate key
            result[key] += " " + subtitle
        else:
            result[key] = subtitle
        # Skip blank line separating entries
        idx += 1
    return result


def _is_valid_caption_payload(payload) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("clip_description"), str)
        and isinstance(payload.get("subject_registry"), dict)
    )


# --------------------------------------------------------------------------- #
#                        LLM wrappers (single clip)                           #
# --------------------------------------------------------------------------- #
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript = info["files"], info["transcript"]

    clip_start_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = CAPTION_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript).replace(
        "CLIP_START_TIME", clip_start_time).replace(
        "CLIP_END_TIME", clip_end_time)

    if os.path.exists(os.path.join(caption_ckpt_folder, f"{timestamp}.json")):
        # If the caption already exists, skip processing
        with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "r") as f:
            cached = json.load(f)
        if _is_valid_caption_payload(cached):
            return timestamp, cached

    max_retries = config.LLM_MAX_RETRIES
    infinite = max_retries < 0
    attempt = 0
    try:
      while infinite or attempt <= max_retries:
        attempt += 1
        if config.PERCEPTION_MODE == "unified":
            _pc_result = perception_call(
                "vision",
                prompt=send_messages[1]["content"],
                image_paths=files,
                system_prompt=send_messages[0]["content"],
                return_json=True,
                caller="frame_caption",
            )
            resp = _pc_result["content"] if not _pc_result["error"] else None
        else:
            try:
                resp = call_openai_model_with_tools(
                    send_messages,
                    endpoints=config.AOAI_CAPTION_VLM_ENDPOINT_LIST,
                    model_name=config.AOAI_CAPTION_VLM_MODEL_NAME,
                    return_json=True,
                    image_paths=files,
                    api_key=config.AOAI_CAPTION_VLM_API_KEY,
                    base_url=config.AOAI_CAPTION_VLM_BASE_URL,
                    caller="frame_caption",
                )["content"]
            except LLMCallFailure:
                resp = None
        if resp is None:
            # Non-retryable: 6MB body limit means frames are too large, will never succeed
            _err = (_pc_result.get("error") or "") if config.PERCEPTION_MODE == "unified" else ""
            if "max bytes" in _err.lower() or "6291456" in _err:
                print(f"[frame_caption] clip {timestamp} skipped: request body exceeds 6MB limit", flush=True)
                return timestamp, {}
            if attempt >= 15:
                print(f"[frame_caption] clip {timestamp} skipped after {attempt} failed attempts", flush=True)
                return timestamp, {}
            backoff = min(2 ** attempt, 120)
            print(f"[frame_caption] clip {timestamp} attempt {attempt} got None, retrying in {backoff}s...", flush=True)
            time.sleep(backoff)
            continue
        try:
            assert isinstance(resp, str), f"Response must be a JSON string instead of {type(resp)}:{resp}."
            parsed = json.loads(resp)
            if not _is_valid_caption_payload(parsed):
                if attempt >= 15:
                    print(f"[frame_caption] clip {timestamp} skipped after {attempt} invalid payload attempts", flush=True)
                    return timestamp, {}
                backoff = min(2 ** attempt, 120)
                print(f"[frame_caption] clip {timestamp} attempt {attempt} invalid payload, retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
                continue
            parsed["clip_description"] += f"\n\nTranscript during this video clip: {transcript}." # add transcript to description
            resp = json.dumps(parsed)
            with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "w") as f:
                f.write(resp)
            return timestamp, parsed
        except json.JSONDecodeError:
            continue
    except ContentBlockedError:
        print(f"[frame_caption] clip {timestamp} skipped: content blocked by safety filter", flush=True)
        return timestamp, {}
    return timestamp, {}  # give up (only reachable when max_retries is finite)


# --------------------------------------------------------------------------- #
#                  LLM wrapper – merge subject registries                     #
# --------------------------------------------------------------------------- #
MERGE_BATCH_SIZE = 15  # max registries per LLM merge call


def _llm_merge_registries(registries: List[dict]) -> dict:
    """Single LLM call to merge a list of partial registries. Returns {} on failure."""
    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = MERGE_PROMPT.replace(
        "REGISTRIES_PLACEHOLDER", json.dumps(registries)
    )

    max_retries = config.LLM_MAX_RETRIES
    infinite = max_retries < 0
    attempt = 0
    while infinite or attempt <= max_retries:
        attempt += 1
        if config.PERCEPTION_MODE == "unified":
            _pc_result = perception_call(
                "vision",
                prompt=send_messages[1]["content"],
                system_prompt=send_messages[0]["content"],
                return_json=True,
                caller="frame_caption_merge",
                model=config.PLANNER_MODEL,
            )
            resp = _pc_result["content"] if not _pc_result["error"] else None
        else:
            try:
                resp = call_openai_model_with_tools(
                    send_messages,
                    endpoints=config.AOAI_CAPTION_VLM_ENDPOINT_LIST,
                    model_name=config.AOAI_CAPTION_VLM_MODEL_NAME,
                    return_json=True,
                    api_key=config.AOAI_CAPTION_VLM_API_KEY,
                    base_url=config.AOAI_CAPTION_VLM_BASE_URL,
                    caller="frame_caption",
                )["content"]
            except LLMCallFailure:
                resp = None
        if resp is None:
            backoff = min(2 ** attempt, 120)
            print(f"[frame_caption] merge registry attempt {attempt} got None, retrying in {backoff}s...", flush=True)
            time.sleep(backoff)
            continue
        try:
            parsed = json.loads(resp)
            if not isinstance(parsed, dict):
                backoff = min(2 ** attempt, 120)
                print(f"[frame_caption] merge registry attempt {attempt} invalid response, retrying in {backoff}s...", flush=True)
                time.sleep(backoff)
                continue
            return parsed
        except json.JSONDecodeError:
            backoff = min(2 ** attempt, 120)
            print(f"[frame_caption] merge registry attempt {attempt} JSON decode error, retrying in {backoff}s...", flush=True)
            time.sleep(backoff)
            continue
    return {}


def merge_subject_registries(registries: List[dict]) -> dict:
    """Merge all partial subject registries, batching to avoid output truncation on long videos."""
    registries = [r for r in registries if r]
    if not registries:
        return {}
    if len(registries) <= MERGE_BATCH_SIZE:
        return _llm_merge_registries(registries)

    batches = [registries[i:i + MERGE_BATCH_SIZE] for i in range(0, len(registries), MERGE_BATCH_SIZE)]
    print(f"[frame_caption] batched merge: {len(registries)} registries -> {len(batches)} batches of <={MERGE_BATCH_SIZE}", flush=True)

    partial_results = []
    for idx, batch in enumerate(batches):
        print(f"[frame_caption] merging batch {idx+1}/{len(batches)} ({len(batch)} registries)", flush=True)
        merged = _llm_merge_registries(batch)
        if merged:
            partial_results.append(merged)

    if not partial_results:
        return {}
    if len(partial_results) == 1:
        return partial_results[0]

    print(f"[frame_caption] final merge: {len(partial_results)} partial results", flush=True)
    return _llm_merge_registries(partial_results)


def _resolve_caption_worker_count(clip_count: int) -> int:
    clip_count = max(1, int(clip_count))

    configured_workers = int(getattr(config, "FRAME_CAPTION_WORKERS", 0) or 0)
    if configured_workers > 0:
        return min(clip_count, configured_workers)

    worker_count = min(clip_count, max(1, os.cpu_count() or 1))
    if config.PERCEPTION_MODE == "unified":
        worker_count = min(
            worker_count,
            max(1, int(getattr(config, "UNIFIED_FRAME_CAPTION_AUTO_MAX_WORKERS", 4) or 4)),
        )
    return max(1, worker_count)


# --------------------------------------------------------------------------- #
#                     Process one video (parallel caption)                    #
# --------------------------------------------------------------------------- #
def process_video(
    frame_folder: str,
    output_caption_folder: str,
    subtitle_file_path: str = None,
):
    caption_ckpt_folder = os.path.join(output_caption_folder, "ckpt")
    os.makedirs(caption_ckpt_folder, exist_ok=True)

    clips = gather_clip_frames(frame_folder, config.CLIP_SECS, subtitle_file_path)

    caption_clip = functools.partial(
        _caption_clip,
        caption_ckpt_folder=caption_ckpt_folder,
    )
    worker_count = _resolve_caption_worker_count(len(clips))
    print(
        f"[frame_caption] clip_count={len(clips)} worker_count={worker_count} mode={config.PERCEPTION_MODE}",
        flush=True,
    )
    # ---------------- Parallel captioning --------------- #
    if worker_count == 1:
        results = list(
            tqdm(
                map(caption_clip, clips),
                total=len(clips),
                desc=f"Captioning {frame_folder}",
            )
        )
    else:
        with mp.Pool(worker_count) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(caption_clip, clips, chunksize=1),
                    total=len(clips),
                    desc=f"Captioning {frame_folder}",
                )
            )

    # ---------------- Save per-clip JSON ---------------- #
    partial_registries = []
    frame_captions = {}
    results = sorted(results, key=lambda x: float(x[0].split("_")[0]))
    for ts, parsed in results:
        if _is_valid_caption_payload(parsed):
            frame_captions[ts] = {
                "caption": parsed["clip_description"],
            }
            partial_registries.append(parsed["subject_registry"])

    if not frame_captions:
        raise RuntimeError("No usable clip captions. Check the vision model and media format.")

    # ---------------- Merge subject registries ---------- #
    merged_registry = merge_subject_registries(partial_registries)
    frame_captions["subject_registry"] = merged_registry

    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)


