"""Unified perception dispatch for OmniVQA."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Literal

from agents import config
from agents.shared.utils import (
    ContentBlockedError,
    LLMCallFailure,
    call_openai_model_with_tools,
    normalize_openai_base_url,
)

logger = logging.getLogger(__name__)

Modality = Literal["vision", "audio_asr", "audio_nonspeech"]

_AUDIO_SIZE_WARN_BYTES = 50 * 1024 * 1024


def _read_audio_base64(audio_path: Path) -> str:
    size_bytes = audio_path.stat().st_size
    if size_bytes > _AUDIO_SIZE_WARN_BYTES:
        logger.warning(
            "Audio file %s is %.1fMB; inline base64 payload will be large.",
            audio_path.name,
            size_bytes / (1024 * 1024),
        )
    return base64.b64encode(audio_path.read_bytes()).decode("utf-8")


def _audio_format_suffix(audio_path: Path) -> str:
    return audio_path.suffix.lower().lstrip(".") or "wav"


def _build_audio_content_inline(audio_path: Path, text_prompt: str) -> list[dict]:
    return [
        {"type": "text", "text": text_prompt},
        {
            "type": "input_audio",
            "input_audio": {
                "data": _read_audio_base64(audio_path),
                "format": _audio_format_suffix(audio_path),
            },
        },
    ]


def _dispatch_vision(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    image_paths: list[str],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    return_json: bool,
    caller: str,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = call_openai_model_with_tools(
        messages=messages,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        image_paths=image_paths,
        max_tokens=max_tokens,
        temperature=temperature,
        return_json=return_json,
        caller=caller,
    )
    if result is None:
        return {"content": "", "error": "VLM backend returned no response.", "tool_calls": None}
    return {
        "content": str(result.get("content") or ""),
        "error": None,
        "tool_calls": result.get("tool_calls"),
    }


def _dispatch_audio(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    audio_path: Path,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    return_json: bool,
    caller: str,
) -> dict[str, Any]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _build_audio_content_inline(audio_path, prompt)})

    result = call_openai_model_with_tools(
        messages=messages,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        return_json=return_json,
        caller=caller,
    )
    if result is None:
        return {"content": "", "error": "Audio VLM backend returned no response."}
    return {"content": str(result.get("content") or ""), "error": None}


def perception_call(
    modality: Modality,
    prompt: str,
    *,
    model: str | None = None,
    image_paths: list[str] | None = None,
    audio_path: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    return_json: bool = False,
    caller: str = "perception",
) -> dict[str, Any]:
    selected_model = model or config.PERCEPTION_MODEL
    api_key = config.AOAI_TOOL_VLM_API_KEY
    if not api_key:
        return {"content": "", "error": "AOAI_TOOL_VLM_API_KEY is not configured."}

    try:
        normalized_base_url = normalize_openai_base_url(config.AOAI_TOOL_VLM_BASE_URL)
        if modality == "vision":
            return _dispatch_vision(
                prompt=prompt,
                model=selected_model,
                api_key=api_key,
                base_url=normalized_base_url,
                image_paths=image_paths or [],
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                return_json=return_json,
                caller=caller,
            )
        if modality in ("audio_asr", "audio_nonspeech"):
            if not audio_path:
                return {
                    "content": "",
                    "error": f"audio_path is required for modality={modality}",
                }
            return _dispatch_audio(
                prompt=prompt,
                model=selected_model,
                api_key=api_key,
                base_url=normalized_base_url,
                audio_path=Path(audio_path),
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                return_json=return_json,
                caller=caller,
            )
        return {"content": "", "error": f"Unknown modality: {modality}"}
    except ContentBlockedError:
        raise
    except LLMCallFailure as failure:
        # Keep the failure classification on the returned envelope so a caller
        # can tell a policy rejection from a transient outage without parsing
        # the message text. The perception boundary still returns rather than
        # raises, because tools convert this into a failure observation.
        logger.error(
            "perception_call refused (modality=%s, model=%s, caller=%s, category=%s, status=%s)",
            modality,
            selected_model,
            caller,
            failure.category,
            failure.status_code,
        )
        return {
            "content": "",
            "error": f"[{caller}] {failure.category}: {failure}",
            "failure": failure.as_record(),
        }
    except Exception as exc:
        logger.error(
            "perception_call failed (modality=%s, model=%s, caller=%s): %s",
            modality,
            selected_model,
            caller,
            exc,
            exc_info=True,
        )
        return {"content": "", "error": f"[{caller}] {type(exc).__name__}: {exc}"}
