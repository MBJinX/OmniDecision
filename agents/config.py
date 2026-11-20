"""Configuration for the public runtime. Shell values override the local .env."""
import os
from pathlib import Path
from urllib.parse import urlsplit

def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False

    for idx, char in enumerate(value):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()

    return value.strip()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    value = _strip_inline_comment(raw_value.strip())
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values

ROOT = Path(__file__).resolve().parents[1]
_FILE = _load_env_file(ROOT / ".env")


def setting(name, default=""):
    value = os.environ.get(name, _FILE.get(name))
    return (value or "").strip() or default


OPENAI_API_KEY = setting("OPENAI_API_KEY")
OPENAI_BASE_URL = setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
PLANNER_MODEL = setting("PLANNER_MODEL")
PERCEPTION_MODEL = setting("PERCEPTION_MODEL", PLANNER_MODEL)
EMBEDDING_API_KEY = setting("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = setting("EMBEDDING_BASE_URL", OPENAI_BASE_URL)
AOAI_EMBEDDING_LARGE_MODEL_NAME = setting("OMNIVQA_EMBEDDING_MODEL")
AOAI_EMBEDDING_LARGE_DIM = int(setting("AOAI_EMBEDDING_LARGE_DIM", "1536"))
AOAI_EMBEDDING_BATCH_SIZE = 10
AOAI_ORCHESTRATOR_API_KEY = OPENAI_API_KEY
AOAI_ORCHESTRATOR_BASE_URL = OPENAI_BASE_URL
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = PLANNER_MODEL
AOAI_TOOL_VLM_API_KEY = setting("PERCEPTION_API_KEY", OPENAI_API_KEY)
AOAI_TOOL_VLM_BASE_URL = setting("PERCEPTION_BASE_URL", OPENAI_BASE_URL)
AOAI_TOOL_VLM_MODEL_NAME = PERCEPTION_MODEL
AOAI_CAPTION_VLM_MODEL_NAME = PERCEPTION_MODEL
AOAI_CAPTION_VLM_API_KEY = AOAI_TOOL_VLM_API_KEY
AOAI_CAPTION_VLM_BASE_URL = AOAI_TOOL_VLM_BASE_URL
AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = AOAI_TOOL_VLM_ENDPOINT_LIST = ()
AOAI_CAPTION_VLM_ENDPOINT_LIST = AOAI_EMBEDDING_RESOURCE_LIST = ()
PERCEPTION_MODE = "unified"
TAVILY_API_KEY = setting("OMNIVQA_TAVILY_API_KEY")
JINA_API_KEY = setting("OMNIVQA_JINA_API_KEY")
JINA_READER_BASE_URL = "https://r.jina.ai"
WEB_SEARCH_TIMEOUT_CONNECT = 6
WEB_SEARCH_TIMEOUT_READ = 45
LLM_MAX_RETRIES = int(setting("LLM_MAX_RETRIES", "2"))
VIDEO_DATABASE_FOLDER = str(ROOT / "outputs")
VIDEO_FPS = 2
CLIP_SECS = 10
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0
AOAI_TOOL_VLM_MAX_FRAME_NUM = 50
TRANSCRIPT_PRIMING_ENABLED = DISABLE_EVIDENCE_NEEDS = WEAK_LLM_MODE = False
TRANSCRIPT_PRIMING_MAX_CHARS = 12000
TRANSCRIPT_PRIMING_MAX_SEGMENTS = 400
WEAK_LLM_WEB_SEARCH_MAX_CHARS = 1500
FRAME_CAPTION_WORKERS = 1


def validate(*, video):
    required = {"OPENAI_API_KEY": OPENAI_API_KEY, "PLANNER_MODEL": PLANNER_MODEL}
    urls = [OPENAI_BASE_URL, AOAI_TOOL_VLM_BASE_URL]
    if video:
        required["OMNIVQA_EMBEDDING_MODEL"] = AOAI_EMBEDDING_LARGE_MODEL_NAME
        urls.append(EMBEDDING_BASE_URL)
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Set " + ", ".join(missing) + " in .env or the environment.")
    for url in urls:
        parts = urlsplit(url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or not parts.path.rstrip("/").endswith("/v1")):
            raise ValueError("Each provider base URL must be an HTTP(S) API root ending in /v1.")
    if AOAI_EMBEDDING_LARGE_DIM < 1 or LLM_MAX_RETRIES < 0:
        raise ValueError("Use a positive embedding dimension and a non-negative retry count.")
