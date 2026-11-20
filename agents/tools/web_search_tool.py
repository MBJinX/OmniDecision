"""Web search tool for OmniVQA.

Single provider pipeline:

- Search:    Tavily (https://tavily.com) — high-quality URL discovery.
- Page read: Jina Reader (https://r.jina.ai) — markdown extraction of full page body.

Pipeline: tavily_search → LLM-rerank candidates → jina_read top-N → return evidence.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Annotated as A, TypedDict

import requests


class PipelineStats(TypedDict):
    """Pipeline-boundary statistics captured during _run_pipeline execution."""
    pre_filter_count: int       # results from search provider before relevance filter
    post_rerank_count: int      # results after LLM rerank, before relevance filter
    scrape_attempt_count: int   # number of URLs where page read was attempted
    scrape_success_count: int   # number of URLs where page_read_success=True


class PipelineResult(TypedDict):
    """Return type for _run_pipeline: evidence list + pipeline-boundary stats."""
    evidence: list[dict]
    pipeline_stats: PipelineStats


import agents.config as config
from agents.shared.func_call_schema import doc as D
from agents.shared.json_utils import safe_load_json_object
from agents.shared.utils import call_openai_model_with_tools

from .common import build_failure_observation, build_web_observation

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT = (config.WEB_SEARCH_TIMEOUT_CONNECT, config.WEB_SEARCH_TIMEOUT_READ)
# Deep page reads should fail fast and fall back to snippets instead of stalling
# the whole runtime chain on one slow page.
_SCRAPE_READ_TIMEOUT_SEC = min(float(config.WEB_SEARCH_TIMEOUT_READ), 12.0)
_SCRAPE_TIMEOUT = (config.WEB_SEARCH_TIMEOUT_CONNECT, _SCRAPE_READ_TIMEOUT_SEC)
_LLM_RELEVANCE_THRESHOLD = 4.0


# ---------------------------------------------------------------------------
# Search layer — Tavily
# ---------------------------------------------------------------------------

def _tavily_search(query: str, top_k: int) -> list[dict]:
    """Search via Tavily, return candidate URL list (no page body reading)."""
    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError as exc:
        raise RuntimeError("tavily-python is not installed. Run: pip install tavily-python") from exc

    api_key = str(config.TAVILY_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("Tavily API key not configured. Set OMNIVQA_TAVILY_API_KEY.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max(1, int(top_k)), search_depth="basic")
    results: list[dict] = []
    for item in (response or {}).get("results") or []:
        results.append({
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
            "snippet": str(item.get("content") or ""),
            "text": "",  # filled by scrape layer
            "score": float(item.get("score") or 0.0),
            "source": "tavily",
        })
    return results


# ---------------------------------------------------------------------------
# Scrape layer — Jina Reader
#
# Jina Reader returns a clean markdown rendering of any public web page when
# the page URL is appended to ``https://r.jina.ai/``. The free anonymous tier
# rate-limits aggressively; setting ``JINA_API_KEY`` raises the limits.
# ---------------------------------------------------------------------------

def _jina_read(url: str, max_chars: int) -> dict:
    """Fetch *url* via Jina Reader. Returns dict with text + scrape_error_type."""
    if not url:
        return {"text": "", "scrape_error_type": "connection_error", "status_code": None}

    base = (config.JINA_READER_BASE_URL or "https://r.jina.ai").rstrip("/")
    target = f"{base}/{url}"

    headers = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
    }
    api_key = str(config.JINA_API_KEY or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(target, headers=headers, timeout=_SCRAPE_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        logger.info("jina read url=%s error=timeout: %s", url, exc)
        return {"text": "", "scrape_error_type": "timeout", "status_code": None}
    except requests.exceptions.ConnectionError as exc:
        logger.info("jina read url=%s error=connection_error: %s", url, exc)
        return {"text": "", "scrape_error_type": "connection_error", "status_code": None}

    if resp.status_code in (408, 504):
        return {"text": "", "scrape_error_type": "timeout", "status_code": resp.status_code}
    if 400 <= resp.status_code < 500:
        return {"text": "", "scrape_error_type": "http_4xx", "status_code": resp.status_code}
    if resp.status_code >= 500:
        return {"text": "", "scrape_error_type": "http_5xx", "status_code": resp.status_code}

    text = (resp.text or "").strip()
    if not text:
        return {"text": "", "scrape_error_type": "empty_content", "status_code": resp.status_code}

    return {"text": text[:max_chars], "scrape_error_type": None, "status_code": resp.status_code}


# ---------------------------------------------------------------------------
# Relevance filtering
# ---------------------------------------------------------------------------

def _annotate_provider_results(results: list[dict]) -> list[dict]:
    ranked: list[dict] = []
    for provider_rank, item in enumerate(results):
        enriched = dict(item)
        enriched["provider_rank"] = provider_rank
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            -int(item.get("provider_rank") or 0),
        ),
        reverse=True,
    )
    return ranked


def _retain_relevant_results(results: list[dict]) -> list[dict]:
    if not results:
        return []

    has_llm_scores = any("llm_relevance" in item for item in results)
    if not has_llm_scores:
        return list(results)
    return [
        item
        for item in results
        if float(item.get("llm_relevance") or 0.0) >= _LLM_RELEVANCE_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# LLM rerank
# ---------------------------------------------------------------------------

def _llm_rerank_results(query: str, results: list[dict], top_k: int = 8) -> list[dict]:
    """Semantically rerank search results using a lightweight LLM call."""
    if not results or len(results) <= 1:
        return results

    candidates = results[:top_k]
    summaries: list[str] = []
    for i, item in enumerate(candidates):
        title = str(item.get("title") or "").strip()[:120]
        snippet = str(item.get("snippet") or item.get("text") or "").strip()[:250]
        url = str(item.get("url") or "").strip()[:120]
        summaries.append(f"[{i}] {title} | {url}\n    {snippet}")

    prompt = (
        "You are a search result ranker for a factual QA system.\n\n"
        f'Search query: "{query}"\n\n'
        "Candidate results:\n" + "\n".join(summaries) + "\n\n"
        "Rate each result for relevance (0-10), source authority, and page family.\n"
        "Output a JSON array only:\n"
        '[{"index": 0, "relevance": 8, "is_authoritative": true, "page_family": "detail"}, ...]\n\n'
        "Rules:\n"
        "- authoritative = official government, manufacturer, encyclopedia, academic sources\n"
        "- NOT authoritative = forum posts, Q&A sites, reseller pages, user-generated wikis\n"
        "- higher relevance = directly provides the specific fact requested in the query\n"
        "- 0 relevance = completely off-topic or about a different entity with the same name\n"
        "- page_family: classify the page type\n"
        "  - 'detail' = specific article, product page, or single-entity page with concrete facts\n"
        "  - 'authority' = official government, manufacturer spec sheet, academic publication\n"
        "  - 'index' = search results page, category listing, tag listing, site-internal search\n"
        "  - 'aggregator' = Q&A forum, social media, review aggregator, comparison table\n"
        "- Output valid JSON only, no explanation outside the JSON."
    )

    try:
        response = call_openai_model_with_tools(
            [{"role": "user", "content": prompt}],
            endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
            model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
            temperature=0.0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            max_tokens=300,
            caller="web_search",
        )
        if response is None:
            return results

        raw = str(response.get("content") or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())

        parsed = safe_load_json_object(raw)
        if not isinstance(parsed, list):
            return results

        llm_scores: dict[int, tuple[float, bool, str]] = {}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            rel = entry.get("relevance", 0)
            auth = entry.get("is_authoritative", False)
            pf = str(entry.get("page_family") or "unknown").strip().lower()
            if pf not in {"detail", "authority", "index", "aggregator"}:
                pf = "unknown"
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                llm_scores[idx] = (float(rel), bool(auth), pf)

        if not llm_scores:
            return results

        reranked: list[dict] = []
        for i, item in enumerate(candidates):
            llm_rel, llm_auth, llm_pf = llm_scores.get(i, (5.0, False, "unknown"))
            enriched = dict(item)
            enriched["llm_relevance"] = llm_rel
            enriched["llm_is_authoritative"] = llm_auth
            enriched["page_family"] = llm_pf
            provider_score = min(max(float(item.get("score") or 0.0), 0.0), 1.0)
            enriched["blended_score"] = (llm_rel / 10.0) * 0.9 + provider_score * 0.1
            reranked.append(enriched)

        # Drop items the LLM scored 0 (completely off-topic).
        reranked = [r for r in reranked if r.get("llm_relevance", 5.0) > 0]
        reranked.sort(key=lambda x: x.get("blended_score", 0.0), reverse=True)
        return reranked if reranked else results

    except Exception:
        logger.debug("LLM rerank failed, falling back to provider order", exc_info=True)
        return results


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    query: str,
    top_k: int,
    read_top_n: int,
    max_chars: int,
) -> PipelineResult:
    """Execute the Tavily + Jina pipeline.

    1. Tavily search returns ``top_k`` candidate URLs.
    2. LLM rerank scores candidates for relevance, authority, page family.
    3. Below-threshold candidates are dropped.
    4. The top ``read_top_n`` survivors are read in full via Jina Reader.
    5. Pages whose body is empty fall back to the Tavily snippet.
    """
    search_results = _annotate_provider_results(_tavily_search(query, top_k))
    search_results = _llm_rerank_results(query, search_results, top_k=8)
    pre_filter_count = len(search_results)
    post_rerank_count = len(search_results)
    search_results = _retain_relevant_results(search_results)

    evidence: list[dict] = []
    scrape_attempt_count = 0
    scrape_success_count = 0
    for item in search_results[:read_top_n]:
        url = item.get("url") or ""
        if not url:
            continue
        scrape_attempt_count += 1
        scrape_result = _jina_read(url, max_chars)
        text_from_scrape = scrape_result["text"]
        scrape_error_type = scrape_result.get("scrape_error_type")

        if text_from_scrape.strip():
            text = text_from_scrape
            page_read_success = True
            scrape_success_count += 1
        else:
            snippet = item.get("snippet") or ""
            if snippet.strip():
                logger.info(
                    "jina read failed for url=%s error_type=%s, "
                    "falling back to tavily snippet (%d chars)",
                    url, scrape_error_type, len(snippet),
                )
                text = snippet
            else:
                logger.warning(
                    "jina read failed for url=%s error_type=%s "
                    "and no tavily snippet available, skipping",
                    url, scrape_error_type,
                )
                text = ""
            page_read_success = False

        if not text.strip():
            continue

        meta: dict = {
            "title": item["title"],
            "url": url,
            "provider": "tavily+jina",
            "query_used": query,
            "page_read_success": page_read_success,
            "scrape_error_type": scrape_error_type,
            "provider_rank": int(item.get("provider_rank") or 0),
            "provider_score": round(float(item.get("score") or 0.0), 4),
        }
        if "llm_relevance" in item:
            meta["llm_relevance"] = item["llm_relevance"]
            meta["llm_is_authoritative"] = item.get("llm_is_authoritative", False)
            meta["page_family"] = item.get("page_family", "unknown")
        evidence.append({
            "text": text[:max_chars],
            "score": float(item.get("blended_score") or float(item.get("score") or 0.0)),
            "metadata": meta,
        })

    return PipelineResult(
        evidence=evidence[:top_k],
        pipeline_stats=PipelineStats(
            pre_filter_count=pre_filter_count,
            post_rerank_count=post_rerank_count,
            scrape_attempt_count=scrape_attempt_count,
            scrape_success_count=scrape_success_count,
        ),
    )


# ---------------------------------------------------------------------------
# Search feedback aggregation
# ---------------------------------------------------------------------------

_SEARCH_FEEDBACK_KEYS = frozenset({
    "result_count", "query_used", "pre_filter_count",
    "scrape_attempt_count", "scrape_success_count",
    "snippet_fallback_count", "avg_relevance",
    "has_authoritative_source", "dominant_page_family",
})


def _build_search_feedback(
    evidence: list[dict],
    query: str,
    pipeline_stats: PipelineStats | None = None,
) -> dict:
    """Build aggregated diagnostic summary from pipeline results."""
    stats = pipeline_stats or PipelineStats(
        pre_filter_count=0, post_rerank_count=0,
        scrape_attempt_count=0, scrape_success_count=0,
    )

    if not evidence:
        return {
            "result_count": 0,
            "query_used": query,
            "pre_filter_count": stats["pre_filter_count"],
            "scrape_attempt_count": stats["scrape_attempt_count"],
            "scrape_success_count": stats["scrape_success_count"],
            "snippet_fallback_count": 0,
            "avg_relevance": 0.0,
            "has_authoritative_source": False,
            "dominant_page_family": "none",
        }

    scrape_ok = sum(1 for e in evidence if e.get("metadata", {}).get("page_read_success"))
    snippet_only = len(evidence) - scrape_ok

    relevance_scores = [
        float(e.get("metadata", {}).get("llm_relevance", 0.0))
        for e in evidence
        if "llm_relevance" in (e.get("metadata") or {})
    ]
    avg_relevance = (
        round(sum(relevance_scores) / len(relevance_scores), 1)
        if relevance_scores else 0.0
    )

    page_families = [
        str(e.get("metadata", {}).get("page_family", "unknown"))
        for e in evidence
    ]
    family_counts = Counter(page_families)
    dominant = family_counts.most_common(1)[0][0] if family_counts else "unknown"

    has_auth = any(
        e.get("metadata", {}).get("llm_is_authoritative") for e in evidence
    )

    return {
        "result_count": len(evidence),
        "query_used": query,
        "pre_filter_count": stats["pre_filter_count"],
        "scrape_attempt_count": stats["scrape_attempt_count"],
        "scrape_success_count": stats["scrape_success_count"],
        "snippet_fallback_count": snippet_only,
        "avg_relevance": avg_relevance,
        "has_authoritative_source": has_auth,
        "dominant_page_family": dominant,
    }


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------

def web_search_tool(
    query: A[str, D(
        "Search query for external facts not available in the video. "
        "Rules: "
        "(1) Keep the query factual and compact. "
        "(2) Prefer the format: 'entity + attribute/metric + qualifier', "
        "e.g. 'Mevo Plus battery capacity mAh specs' or 'Charles Trenet La Mer first recording year'. "
        "(3) Do NOT include irrelevant entities from the video (e.g. accessories being reviewed). "
        "(4) Do NOT describe what happens in the video. Search only for the missing external fact. "
        "(5) If you need exact data such as years, specs, or distances, name the target entity and target attribute directly."
    )],
    top_k: A[int, D("Maximum number of search results to return. Keep small.")] = 5,
    read_top_n: A[int, D("Number of top URLs to deep-read via Jina Reader.")] = 3,
) -> str:
    """
    Search the public web and return page-level evidence to supplement missing external facts.
    Only use when video content alone is insufficient to answer the question.
    """
    top_k = max(1, min(int(top_k), 10))
    read_top_n = max(1, min(int(read_top_n), top_k, 5))
    if getattr(config, "WEAK_LLM_MODE", False):
        # Weak-LLM context budget: cap fan-out aggressively.
        top_k = min(top_k, 3)
        read_top_n = min(read_top_n, 2)
        max_chars = config.WEAK_LLM_WEB_SEARCH_MAX_CHARS
    else:
        max_chars = 5000

    try:
        pipeline_result = _run_pipeline(query, top_k, read_top_n, max_chars)
        evidence = pipeline_result["evidence"]
        pipeline_stats = pipeline_result["pipeline_stats"]
    except RuntimeError as exc:
        return build_failure_observation(
            "web", str(exc),
            "Set OMNIVQA_TAVILY_API_KEY (search) and optionally JINA_API_KEY (page read).",
            status="unavailable",
        )
    except requests.exceptions.ConnectionError as exc:
        return build_failure_observation(
            "web", f"Cannot connect to search service: {exc}",
            "Verify outbound HTTPS connectivity to api.tavily.com and r.jina.ai.",
            status="unavailable",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search_tool error: %s", exc, exc_info=True)
        return build_failure_observation(
            "web", f"Unexpected search error: {exc}",
            "Try a more specific query, or verify the search service is returning content properly.",
            status="unavailable",
        )

    if not evidence:
        feedback = _build_search_feedback([], query, pipeline_stats)
        return build_web_observation(
            evidence=[],
            summary=(
                f"Search for '{query}' returned no usable evidence. "
                f"Provider returned {feedback['pre_filter_count']} raw results; "
                f"after relevance filtering: 0 remained. "
                f"Scrape attempts: {feedback['scrape_attempt_count']}, "
                f"successes: {feedback['scrape_success_count']}."
            ),
            status="no_match",
            search_feedback=feedback,
            fallback_hint=(
                "Try reformulating the query from a different angle -- "
                "use a different entity name, different attribute phrasing, "
                "or target a specific authoritative source."
            ),
        )

    summary = f"Searched '{query}' via tavily+jina, obtained {len(evidence)} external evidence items."

    all_snippet_fallback = all(
        not e.get("metadata", {}).get("page_read_success") for e in evidence
    )
    if all_snippet_fallback:
        summary += " (Note: all page reads failed; evidence is search snippets only.)"

    feedback = _build_search_feedback(evidence, query, pipeline_stats)

    return build_web_observation(
        evidence=evidence,
        summary=summary,
        search_feedback=feedback,
    )
