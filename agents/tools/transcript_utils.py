"""Subtitle parsing and ASR helpers for OmniVQA."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from agents import config
from agents.shared.perception import perception_call

from .common import emit_runtime_progress


logger = logging.getLogger(__name__)


_ASR_SEGMENT_CACHE: dict[str, list[dict]] = {}
_SUBTITLE_TIME_RANGE_PATTERN = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)"
)
_ASS_OVERRIDE_TAG_PATTERN = re.compile(r"\{[^}]*\}")
_VLM_ASR_CHUNK_SECS = 300
_VLM_ASR_MAX_TOKENS = 16384


def _split_subtitle_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.split(r"\n\s*\n", normalized)


def _parse_srt_like_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    for block in _split_subtitle_blocks(text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        pointer = 0
        if lines[0].isdigit():
            pointer = 1
        if pointer >= len(lines) or "-->" not in lines[pointer]:
            if pointer + 1 >= len(lines) or "-->" not in lines[pointer + 1]:
                continue
            pointer += 1

        match = _SUBTITLE_TIME_RANGE_PATTERN.search(lines[pointer])
        if match is None:
            continue

        raw_text = " ".join(lines[pointer + 1 :]).strip()
        clean_text = clean_subtitle_text(raw_text)
        cue_tags = [item.strip().lower() for item in re.findall(r"\[(.*?)\]", clean_text)]
        spoken_text = re.sub(r"\[[^\]]+\]", "", clean_text).strip()
        segments.append(
            {
                "start_secs": srt_timestamp_to_seconds(match.group("start")),
                "end_secs": srt_timestamp_to_seconds(match.group("end")),
                "raw_text": raw_text,
                "clean_text": clean_text,
                "spoken_text": spoken_text,
                "cue_tags": cue_tags,
            }
        )
    return segments


def _ass_timestamp_to_seconds(timestamp: str) -> float:
    normalized = timestamp.strip().replace(",", ".")
    hours, minutes, rest = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _clean_ass_text(text: str) -> str:
    text = text.replace(r"\N", " ").replace(r"\n", " ")
    text = _ASS_OVERRIDE_TAG_PATTERN.sub("", text)
    return clean_subtitle_text(text)


def _parse_ass_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    in_events_section = False
    format_fields: list[str] = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_events_section = line.lower() == "[events]"
            continue
        if not in_events_section:
            continue
        if line.lower().startswith("format:"):
            format_fields = [item.strip().lower() for item in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:") or not format_fields:
            continue

        payload = line.split(":", 1)[1]
        values = [item.strip() for item in payload.split(",", len(format_fields) - 1)]
        if len(values) != len(format_fields):
            continue
        record = dict(zip(format_fields, values))
        if "start" not in record or "end" not in record or "text" not in record:
            continue

        clean_text = _clean_ass_text(record["text"])
        cue_tags = [item.strip().lower() for item in re.findall(r"\[(.*?)\]", clean_text)]
        spoken_text = re.sub(r"\[[^\]]+\]", "", clean_text).strip()
        segments.append(
            {
                "start_secs": _ass_timestamp_to_seconds(record["start"]),
                "end_secs": _ass_timestamp_to_seconds(record["end"]),
                "raw_text": record["text"],
                "clean_text": clean_text,
                "spoken_text": spoken_text,
                "cue_tags": cue_tags,
            }
        )

    return segments


def parse_subtitle_segments(path_like: str | os.PathLike[str]) -> list[dict]:
    """Parse subtitle files in SRT, VTT, or ASS format into normalized segments."""
    path = Path(path_like)
    if not path.is_file():
        return []

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".ass":
        return _parse_ass_segments(text)
    return _parse_srt_like_segments(text)


def parse_srt_segments(srt_path: str | os.PathLike[str]) -> list[dict]:
    """Backward-compatible wrapper kept for existing call sites and tests."""
    return parse_subtitle_segments(srt_path)


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_transcribed_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_long_asr_segment(segment: dict) -> list[dict]:
    """Split a long ASR segment into shorter sentence-like pieces."""
    text = segment.get("clean_text", "").strip()
    start = float(segment.get("start_secs", 0.0))
    end = float(segment.get("end_secs", start))
    duration = max(end - start, 0.1)
    source = str(segment.get("source") or "asr")

    sentence_candidates = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text)
        if item.strip()
    ]
    if len(sentence_candidates) <= 1 or duration <= 15:
        return [segment]

    total_chars = max(sum(len(item) for item in sentence_candidates), 1)
    cursor = start
    refined_segments: list[dict] = []
    for idx, sentence in enumerate(sentence_candidates):
        ratio = len(sentence) / total_chars
        if idx == len(sentence_candidates) - 1:
            sentence_end = end
        else:
            sentence_end = min(end, cursor + max(duration * ratio, 0.8))
        refined_segments.append(
            {
                "start_secs": cursor,
                "end_secs": max(sentence_end, cursor + 0.2),
                "raw_text": sentence,
                "clean_text": sentence,
                "spoken_text": sentence,
                "cue_tags": [],
                "source": source,
            }
        )
        cursor = sentence_end
    return refined_segments


def srt_timestamp_to_seconds(timestamp: str) -> float:
    normalized = timestamp.strip().replace(",", ".")
    hours, minutes, rest = normalized.split(":")
    seconds_float = float(rest)
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + seconds_float
    )


def ensure_audio_track(raw_video_path: str | os.PathLike[str], raw_audio_path: str | os.PathLike[str]) -> Path:
    """Extract a mono 16k wav track for ASR when it does not already exist."""
    video_path = Path(raw_video_path)
    audio_path = Path(raw_audio_path)
    if audio_path.is_file():
        return audio_path
    if not video_path.is_file():
        raise FileNotFoundError(f"Raw video file not found: {video_path}")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "error",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


def _get_audio_duration_secs(audio_path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(audio_path)).duration)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _split_audio_into_chunks(
    audio_path: Path,
    chunk_secs: float = _VLM_ASR_CHUNK_SECS,
) -> list[tuple[Path, float]]:
    duration_secs = _get_audio_duration_secs(audio_path)
    if duration_secs <= 0.0 or duration_secs <= chunk_secs:
        return [(audio_path, 0.0)]

    chunk_root = audio_path.parent / ".asr_chunks" / audio_path.stem
    chunk_root.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[Path, float]] = []
    offset_secs = 0.0
    chunk_index = 0
    while offset_secs < duration_secs:
        current_duration = min(chunk_secs, duration_secs - offset_secs)
        chunk_path = chunk_root / f"chunk_{chunk_index:03d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(offset_secs),
                "-t",
                str(current_duration),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-loglevel",
                "error",
                str(chunk_path),
            ],
            check=True,
            timeout=120,
        )
        chunks.append((chunk_path, offset_secs))
        offset_secs += current_duration
        chunk_index += 1

    return chunks


def _vlm_transcribe_single_chunk(audio_path: Path, offset_secs: float = 0.0) -> list[dict]:
    prompt = (
        "Transcribe all spoken content in this audio clip with timestamps.\n"
        "Return strict JSON only.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "segments": [\n'
        "    {\n"
        '      "start_secs": number,\n'
        '      "end_secs": number,\n'
        '      "text": string\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Include dialogue, narration, and any spoken words.\n"
        "- Use seconds relative to the start of this audio clip.\n"
        '- If there is no speech, return {"segments": []}.\n'
        "- Do not describe music or sound effects.\n"
    )
    result = perception_call(
        "audio_asr",
        prompt=prompt,
        audio_path=str(audio_path),
        system_prompt="You are a precise audio transcription model. Return JSON only.",
        return_json=True,
        max_tokens=_VLM_ASR_MAX_TOKENS,
        temperature=0,
        caller="transcript_vlm",
    )
    if result["error"]:
        logger.warning("VLM ASR chunk transcription failed: %s", result["error"])
        return []

    try:
        parsed = json.loads(result["content"])
    except json.JSONDecodeError as exc:
        logger.warning("VLM ASR chunk returned non-JSON content: %s", exc)
        return []

    raw_segments = parsed.get("segments", [])
    if not isinstance(raw_segments, list):
        return []

    segments: list[dict] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        try:
            start_secs = float(raw_segment.get("start_secs", 0.0))
            end_secs = float(raw_segment.get("end_secs", start_secs))
        except (TypeError, ValueError):
            continue
        text = str(raw_segment.get("text") or "").strip()
        clean_text = clean_transcribed_text(text)
        if not clean_text:
            continue
        absolute_start = max(offset_secs + start_secs, 0.0)
        absolute_end = max(offset_secs + end_secs, absolute_start + 0.1)
        segments.append(
            {
                "start_secs": absolute_start,
                "end_secs": absolute_end,
                "raw_text": text,
                "clean_text": clean_text,
                "spoken_text": clean_text,
                "cue_tags": [],
                "source": "vlm_asr",
            }
        )
    return segments


def _vlm_transcribe_fallback(audio_path: Path) -> list[dict]:
    prompt = (
        "Provide a complete transcript of all spoken content in this audio.\n"
        "Return plain text only.\n"
        "If there is no speech, return an empty string."
    )
    result = perception_call(
        "audio_asr",
        prompt=prompt,
        audio_path=str(audio_path),
        system_prompt="You are an audio transcription assistant.",
        max_tokens=_VLM_ASR_MAX_TOKENS,
        temperature=0,
        caller="transcript_vlm_fallback",
    )
    if result["error"]:
        logger.warning("VLM ASR fallback failed: %s", result["error"])
        return []

    text = str(result["content"] or "").strip()
    clean_text = clean_transcribed_text(text)
    if not clean_text:
        return []

    duration_secs = _get_audio_duration_secs(audio_path)
    if duration_secs <= 0.0:
        return []

    return [
        {
            "start_secs": 0.0,
            "end_secs": duration_secs,
            "raw_text": text,
            "clean_text": clean_text,
            "spoken_text": clean_text,
            "cue_tags": [],
            "source": "vlm_asr_summary",
        }
    ]


def _vlm_transcribe_audio_segments(audio_path: Path) -> list[dict]:
    chunks = _split_audio_into_chunks(audio_path)
    if len(chunks) > 1:
        logger.info(
            "VLM ASR chunking enabled for %s: %.1fs split into %d chunks.",
            audio_path.name,
            _get_audio_duration_secs(audio_path),
            len(chunks),
        )

    all_segments: list[dict] = []
    for chunk_path, offset_secs in chunks:
        all_segments.extend(_vlm_transcribe_single_chunk(chunk_path, offset_secs))

    if all_segments:
        all_segments.sort(key=lambda segment: (segment["start_secs"], segment["end_secs"]))
        return all_segments
    return _vlm_transcribe_fallback(audio_path)


def _asr_cache_identity() -> dict[str, str]:
    if config.PERCEPTION_MODE == "unified":
        return {
            "mode": "unified",
            "model": str(config.PERCEPTION_MODEL or ""),
        }
    return {
        "mode": "specialist",
        "model": "whisper-large-v3-turbo",
    }


def _asr_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _is_matching_asr_file_cache(cache_path: Path) -> bool:
    meta_path = _asr_cache_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return False
    try:
        cached_identity = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return cached_identity == _asr_cache_identity()


def _write_asr_file_cache(cache_path: Path, segments: list[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = _asr_cache_meta_path(cache_path)
    meta_path.write_text(
        json.dumps(_asr_cache_identity(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_audio_segments(
    raw_video_path: str | os.PathLike[str],
    raw_audio_path: str | os.PathLike[str],
    cache_json_path: str | os.PathLike[str],
    *,
    force_rebuild: bool = False,
) -> list[dict]:
    """Generate or load ASR segments with timestamps."""
    cache_path = Path(cache_json_path)
    cache_identity = _asr_cache_identity()
    cache_suffix = (
        f"|{cache_identity['mode']}|{cache_identity['model']}"
    )
    cache_key = str(cache_path.resolve()) + cache_suffix
    cached = None if force_rebuild else _ASR_SEGMENT_CACHE.get(cache_key)
    if cached is not None:
        emit_runtime_progress("ASR: Reusing in-memory transcript cache.")
        return cached

    if not force_rebuild and _is_matching_asr_file_cache(cache_path):
        emit_runtime_progress(f"ASR: Loading transcript from cache {cache_path.name}.")
        segments = json.loads(cache_path.read_text(encoding="utf-8"))
        _ASR_SEGMENT_CACHE[cache_key] = segments
        return segments

    if force_rebuild:
        _ASR_SEGMENT_CACHE.pop(cache_key, None)

    audio_path = ensure_audio_track(raw_video_path, raw_audio_path)
    segments = _vlm_transcribe_audio_segments(audio_path)

    refined_segments: list[dict] = []
    for segment in segments:
        refined_segments.extend(split_long_asr_segment(segment))
    segments = refined_segments

    _write_asr_file_cache(cache_path, segments)
    _ASR_SEGMENT_CACHE[cache_key] = segments
    return segments
