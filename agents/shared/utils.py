import base64
import copy
import json
import random
import re
import sys
import threading
import time
from mimetypes import guess_type
from pathlib import Path


import cv2
import requests


OMNIVQA_ROOT = Path(__file__).resolve().parents[2]
if str(OMNIVQA_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIVQA_ROOT))


# ---------------------------------------------------------------------------
# Token usage tracker — accumulates per-caller and total token counts
# ---------------------------------------------------------------------------

def _modality_split(details: object) -> dict[str, int]:
    """Normalize a provider's per-modality token detail into {MODALITY: count}.

    Two shapes reach this:
      * Gemini native — ``[{"modality": "VIDEO", "tokenCount": 10836}, ...]``
      * OpenAI-compatible — ``{"text_tokens": 40, "image_tokens": 0, ...}``

    Providers that report neither yield ``{}``, which reads as "not broken out"
    rather than "all text" — the distinction matters because a caller with no
    split must not be silently attributed to one modality.
    """
    out: dict[str, int] = {}
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue
            modality = str(entry.get("modality") or "UNKNOWN").upper()
            count = int(entry.get("tokenCount") or entry.get("token_count") or 0)
            if count:
                out[modality] = out.get(modality, 0) + count
    elif isinstance(details, dict):
        for key, value in details.items():
            if not key.endswith("_tokens"):
                continue
            modality = key[: -len("_tokens")].upper()
            if modality in ("CACHED", "REASONING", "ACCEPTED_PREDICTION", "REJECTED_PREDICTION"):
                continue
            try:
                count = int(value or 0)
            except (TypeError, ValueError):
                continue
            if count:
                out[modality] = out.get(modality, 0) + count
    return out


def _empty_bucket() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "call_count": 0,
        "prompt_by_modality": {},
        "completion_by_modality": {},
    }


def _merge_into(bucket: dict, counts: dict) -> None:
    for key, value in counts.items():
        if key in ("prompt_by_modality", "completion_by_modality"):
            target = bucket.setdefault(key, {})
            for modality, count in value.items():
                target[modality] = target.get(modality, 0) + count
        else:
            bucket[key] = bucket.get(key, 0) + value


class TokenTracker:
    """Thread-safe token usage accumulator.

    Each call to ``record()`` adds token counts under both a caller label and a
    model name.  ``snapshot()`` returns per-caller and per-model breakdowns plus
    grand totals.  ``reset()`` clears all counters (call between agent runs).

    The model dimension is not cosmetic. This runtime routes planning to one
    model and perception to another, and their tokens are neither the same unit
    nor the same price — a video token billed by the perception backend and a
    text token billed by the planner cannot be meaningfully added. The same
    applies within one model across modalities, which is why each bucket also
    carries the provider's own modality split rather than an estimate.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._callers: dict[str, dict] = {}
        self._models: dict[str, dict] = {}

    # -- mutators ----------------------------------------------------------

    def record(self, caller: str, usage: dict | None, model: str = "") -> None:
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0)
        total = int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0)
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details")
        reasoning = int(
            (completion_details or {}).get("reasoning_tokens")
            or usage.get("thoughtsTokenCount")
            or 0
        )
        cached = int(
            (prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else 0)
            or usage.get("cachedContentTokenCount")
            or 0
        )
        if total == 0:
            total = prompt + completion
        counts = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "reasoning_tokens": reasoning,
            "cached_tokens": cached,
            "call_count": 1,
            "prompt_by_modality": _modality_split(
                usage.get("promptTokensDetails") or prompt_details
            ),
            "completion_by_modality": _modality_split(
                usage.get("candidatesTokensDetails") or completion_details
            ),
        }
        with self._lock:
            _merge_into(self._callers.setdefault(caller, _empty_bucket()), counts)
            _merge_into(self._models.setdefault(model or "unknown", _empty_bucket()), counts)

    def reset(self) -> None:
        with self._lock:
            self._callers.clear()
            self._models.clear()

    # -- readers -----------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            callers = {k: copy.deepcopy(v) for k, v in self._callers.items()}
            models = {k: copy.deepcopy(v) for k, v in self._models.items()}
        grand = _empty_bucket()
        for bucket in callers.values():
            _merge_into(grand, bucket)
        return {"by_caller": callers, "by_model": models, "total": grand}


# Module-level singleton — importable from anywhere
token_tracker = TokenTracker()


def _emit_token_usage(
    caller: str,
    model: str,
    usage_dict: dict | None,
    *,
    messages: list[dict] | None = None,
    response_text: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
) -> None:
    """Record token counts in the current run and its active trace span."""
    token_tracker.record(caller, usage_dict, model)

    # Soft import: agents.logging may not be importable in legacy test paths
    # or partial installs. Step attachment is best-effort.
    try:
        from agents.logging.step_logger import current_step_handle
    except ImportError:
        return
    handle = current_step_handle()
    if handle is None or not usage_dict:
        return
    prompt_tokens = int(
        usage_dict.get("prompt_tokens")
        or usage_dict.get("promptTokenCount")
        or 0
    )
    completion_tokens = int(
        usage_dict.get("completion_tokens")
        or usage_dict.get("candidatesTokenCount")
        or 0
    )
    reasoning_tokens = int(
        (usage_dict.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    cached_tokens = int(
        (usage_dict.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    handle.attach_usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        caller=caller,
        messages=messages,
        response_text=response_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def normalize_openai_base_url(base_url: str | None) -> str:
    """Normalize an OpenAI-compatible base URL so callers can append API paths safely."""
    if not base_url:
        return "https://api.openai.com/v1"
    return base_url.rstrip("/")


class ContentBlockedError(Exception):
    """Raised when VLM blocks content due to safety filters (e.g. PROHIBITED_CONTENT)."""
    pass


class LLMCallFailure(Exception):
    """A model call that failed in a way the caller must be able to reason about.

    Returning ``None`` for every failure erases the distinction between "try
    again" and "this request will never succeed". Upper layers then re-issue
    the same deterministic failure independently — planner history repair,
    critic, web enrichment, and the finalizer each paying for it once. Carrying
    the HTTP status and a retryability category lets each layer stop at the
    right boundary and lets the run record say why.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        category: str = "unknown",
        retryable: bool = False,
        caller: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.retryable = retryable
        self.caller = caller
        self.model = model

    def as_record(self) -> dict:
        """Return the persistable failure summary for run reports and records."""
        return {
            "category": self.category,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "caller": self.caller,
            "model": self.model,
            "message": str(self)[:500],
        }


_HTTP_STATUS_RE = re.compile(r"status[\s:]+(\d{3})|http code[\s:]+(\d{3})")


class ProviderFailureLedger:
    """Per-thread record of deterministic provider failures within one run.

    Layers that call the model sit at different depths — planner, critic,
    question profiling, web enrichment, finalizer — and most of them cannot see
    the run report. Recording the failure once where it is classified, in
    thread-local storage, lets every layer ask whether a request of this shape
    has already been refused without threading the run report through the whole
    call graph. Thread-local scoping keeps parallel case workers separate.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _records(self) -> list[dict]:
        records = getattr(self._local, "records", None)
        if records is None:
            records = []
            self._local.records = records
        return records

    def record(self, failure: "LLMCallFailure") -> None:
        self._records().append(failure.as_record())

    def reset(self) -> None:
        self._local.records = []

    def has_deterministic_failure(self) -> bool:
        """Return whether a non-retryable failure was already seen this run."""
        return any(not record["retryable"] for record in self._records())

    def snapshot(self) -> dict:
        """Return the persistable failure history and per-category latch."""
        records = [dict(record) for record in self._records()]
        latch: dict[str, dict] = {}
        for record in records:
            if record["retryable"]:
                continue
            entry = latch.setdefault(
                record["category"],
                {
                    "status_code": record["status_code"],
                    "first_caller": record["caller"],
                    "count": 0,
                },
            )
            entry["count"] += 1
        return {"failures": records, "latch": latch}


provider_failure_ledger = ProviderFailureLedger()


def _extract_http_status(err_lower: str) -> int | None:
    """Return the HTTP status carried by a provider error string, when present."""
    match = _HTTP_STATUS_RE.search(err_lower)
    if match is None:
        return None
    for group in match.groups():
        if group:
            return int(group)
    return None


_CONTENT_POLICY_MARKERS = (
    "content",
    "safety",
    "policy",
    "moderation",
    "data_inspection_failed",
    "blocked",
    "prohibited",
)


def _classify_failure(err_lower: str, status_code: int | None) -> str:
    """Name the deterministic failure class so callers can latch on it."""
    if "prohibited_content" in err_lower:
        return "content_blocked"
    if status_code in (401, 403):
        return "provider_auth"
    if status_code == 400:
        # Some providers report a request refused for its content as a
        # plain 400 with a provider-specific marker, not a distinct status.
        if any(marker in err_lower for marker in _CONTENT_POLICY_MARKERS):
            return "content_policy"
        return "invalid_request"
    if any(
        marker in err_lower
        for marker in ("invalidparameter", "invalid parameter", "invalid_argument")
    ):
        return "invalid_request"
    if status_code is not None and 400 <= status_code < 500:
        return "client_error"
    return "unknown"


def _should_retry_error(err_lower: str) -> bool:
    non_retryable_markers = (
        "status 400",
        "status 401",
        "status 403",
        "invalidparameter",
        "invalid parameter",
        "batch size is invalid",
        '"status":"invalid_argument"',
        '"status": "invalid_argument"',
        "prohibited_content",
    )
    transient_network_markers = (
        "ssl",
        "ssleoferror",
        "unexpected_eof_while_reading",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    retryable_markers = (
        "rate limit",
        "timed out",
        "too many requests",
        "forbidden for url",
        "status 404",
        "status 429",
        "status 503",
        "status 504",
        "http code: 404",
        "http code: 429",
        "http code: 503",
        "http code: 504",
        '"code":404',
        '"code": 404',
        '"code":429',
        '"code": 429',
        '"status":"resource_exhausted"',
        '"status": "resource_exhausted"',
        "insufficient_quota",
        '"status":"not_found"',
        '"status": "not_found"',
        "resource exhausted",
        "try again later",
        "no candidates",
        "empty candidates",
        "finish_reason",
        "blocked_reason",
    )

    if any(marker in err_lower for marker in non_retryable_markers):
        return False
    if any(marker in err_lower for marker in retryable_markers):
        return True
    if any(marker in err_lower for marker in transient_network_markers):
        return True
    if "internal" in err_lower:
        return True
    return False


def retry_with_exponential_backoff(
    func=None,
    max_retries: int | None = None,
    min_delay: float = 2,
    max_delay: float = 5,
):
    """Retry a function with random delay between min_delay and max_delay.

    max_retries: number of retries. -1 means infinite. None reads from
    config.LLM_MAX_RETRIES.

    Exhausted requests raise LLMCallFailure with the provider status.
    """
    if max_retries is None:
        from agents import config
        max_retries = config.LLM_MAX_RETRIES

    infinite = max_retries < 0

    def decorator(fn):
        def wrapper(*args, **kwargs):
            num_retries = 0

            while True:
                try:
                    return fn(*args, **kwargs)
                except LLMCallFailure:
                    raise
                except Exception as e:
                    err = str(e)
                    err_lower = err.lower()
                    should_retry = _should_retry_error(err_lower)
                    if should_retry:
                        num_retries += 1

                        if not infinite and num_retries > max_retries:
                            failure = LLMCallFailure(
                                f"{kwargs.get('caller', 'provider')} failed after retries: {err}",
                                status_code=_extract_http_status(err_lower),
                                category="provider_unavailable", retryable=True,
                            )
                            provider_failure_ledger.record(failure)
                            raise failure from e

                        delay = min_delay + random.random() * (max_delay - min_delay)
                        print(
                            f"Retrying ({num_retries}{'/' + str(max_retries) if not infinite else '/∞'}) "
                            f"in {delay:.1f}s for {err}...",
                            flush=True,
                        )
                        time.sleep(delay)
                    else:
                        if "prohibited_content" in err_lower:
                            raise ContentBlockedError(err)
                        status_code = _extract_http_status(err_lower)
                        failure = LLMCallFailure(
                            err,
                            status_code=status_code,
                            category=_classify_failure(err_lower, status_code),
                            retryable=False,
                            caller=str(kwargs.get("caller") or ""),
                            model=str(kwargs.get("model_name") or ""),
                        )
                        provider_failure_ledger.record(failure)
                        raise failure from e

        return wrapper

    # Support both @retry_with_exponential_backoff and @retry_with_exponential_backoff(...)
    if func is not None:
        return decorator(func)
    return decorator

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def extract_answer(message: dict | None) -> str | None:
    """
    Extract the plain-text answer from an assistant message that may include
    tool calls.

    The function first checks the normal `content` field (for responses that
    are not using tools). If the assistant responded via a tool call, it
    attempts to parse the JSON string stored in
    `message["tool_calls"][i]["function"]["arguments"]` and returns the value
    associated with the key `"answer"`.

    Parameters
    ----------
    message : dict
        The assistant message returned by `call_openai_model_with_tools`.

    Returns
    -------
    str | None
        The extracted answer, or ``None`` if no answer could be found.
    """
    if not isinstance(message, dict):
        return None

    if message.get("role") != "assistant":
        return None

    # Tool-based response
    tool_calls = message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function_payload = call.get("function") or {}
            args_json = function_payload.get("arguments")
            if not isinstance(args_json, str):
                continue
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue
            answer = args.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

    # Direct text response
    if (content := message.get("content")):
        return content.strip()
    
    return None


def _post_model_request(base_url, route, api_key, payload):
    if not api_key:
        raise ValueError("Missing model provider API key.")
    response = requests.post(
        f"{normalize_openai_base_url(base_url)}/{route}",
        json=payload, headers={"Authorization": f"Bearer {api_key}"},
        timeout=(10, 180), allow_redirects=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"{route} returned status {response.status_code}")
    return response.json()


@retry_with_exponential_backoff
def call_openai_model_with_tools(
    messages, endpoints, model_name, api_key=None, base_url=None, tools=None,
    image_paths=None, max_tokens=4096, temperature=0.0, tool_choice="auto",
    return_json=False, caller="unknown",
):
    payload = {"model": model_name, "messages": copy.deepcopy(messages)}
    if tools:
        payload.update(tools=tools, tool_choice=tool_choice)
    if return_json:
        payload["response_format"] = {"type": "json_object"}
    if image_paths:
        parts = [{"type": "image_url", "image_url": {"url": local_image_to_data_url(path)}}
                 for path in image_paths]
        last = payload["messages"][-1]
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            last["content"] = [{"type": "text", "text": last["content"]}, *parts]
        else:
            payload["messages"].append({"role": "user", "content": parts})
    reply = _post_model_request(base_url, "chat/completions", api_key, payload)
    choice = reply["choices"][0]
    message = choice["message"]
    _emit_token_usage(caller, model_name, reply.get("usage"), messages=messages,
                      response_text=message.get("content") or "", tool_calls=message.get("tool_calls"),
                      finish_reason=choice.get("finish_reason") or "")
    return message


class AzureOpenAIEmbeddingService:
    @staticmethod
    @retry_with_exponential_backoff
    def get_embeddings(endpoints, model_name, input_text, api_key=None, base_url=None, dimensions=None):
        payload = {"model": model_name, "input": input_text}
        if dimensions:
            payload["dimensions"] = dimensions
        reply = _post_model_request(base_url, "embeddings", api_key, payload)
        rows = sorted(reply["data"], key=lambda row: row["index"])
        if len(rows) != len(input_text) or any(len(row["embedding"]) != dimensions for row in rows):
            raise ValueError("Embedding response count or dimensions differ from the configured index.")
        return rows
