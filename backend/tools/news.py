"""
tools/serper.py
---------------
Web news search tool using the Google Serper API.

Now async — uses httpx for async HTTP calls with retry support.
Returns up to `num_results` recent news articles as a list of dicts
with: title, snippet, url, date (when available).
"""

from __future__ import annotations

import json
import structlog
import httpx
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config.setting import settings

log = structlog.get_logger(__name__)

_SERPER_URL = "https://google.serper.dev/news"

_http_retry = retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


# ── Arg schema ────────────────────────────────────────────────────────────────

class SerperInput(BaseModel):
    """Input for web news search."""
    query: str = Field(description="News search query, e.g. 'AI regulation 2025' or 'Tesla earnings'.")
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of news articles to return (1–10).",
    )


# ── Tool (async) ──────────────────────────────────────────────────────────────

@tool("search_news", args_schema=SerperInput)
async def search_news(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web for recent news articles on a given topic using the Serper API.

    Returns a list of dicts, each with: title, snippet, url, date.
    Use this when you need current news, recent events, or breaking stories on any topic.
    """
    log.info("tool.search_news", query=query, num=num_results)

    @_http_retry
    async def _search() -> list[dict]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _SERPER_URL,
                headers={
                    "X-API-KEY":    settings.serper_api_key,
                    "Content-Type": "application/json",
                },
                content=json.dumps({
                    "q":           query,
                    "num":         num_results,
                    "autocorrect": False,
                    "tbs":         "qdr:d",   # past 24 hours
                }),
            )
            resp.raise_for_status()
            raw = resp.json().get("news", [])
            # Normalise to consistent shape
            return [
                {
                    "title":   a.get("title", ""),
                    "snippet": a.get("snippet", ""),
                    "url":     a.get("link", ""),
                    "date":    a.get("date", ""),
                    "source":  a.get("source", ""),
                }
                for a in raw
            ]

    try:
        articles = await _search()
        log.info("tool.search_news.done", query=query, count=len(articles))
        return articles
    except httpx.HTTPError as exc:
        log.error("tool.search_news.error", query=query, error=str(exc))
        return [{"error": str(exc)}]
    except Exception as exc:
        log.error("tool.search_news.error", query=query, error=str(exc))
        return [{"error": str(exc)}]