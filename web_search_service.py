"""
web_search_service.py — Live tool & pricing lookups.

Primary  : Tavily Search API  (set TAVILY_API_KEY in .env)
Fallback : DuckDuckGo Search  (no key required, uses duckduckgo-search pkg)

Each function returns structured dicts with:
    tool_name    – name of the tool / result
    latest_price – pricing string extracted from the snippet
    source       – URL of the result
"""

import os
import re
import logging
import textwrap
from typing import TypedDict

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_TIMEOUT    = 15.0


# ── Return type ───────────────────────────────────────────────────────────────

class ToolResult(TypedDict):
    tool_name:    str
    latest_price: str
    source:       str


# ── Helpers ───────────────────────────────────────────────────────────────────

_PRICE_RE = re.compile(
    r"""
    (?:free|gratis                          # free tier keyword
    |\$\s?\d[\d,]*(?:\.\d+)?               # $10 / $10.00 / $1,200
    |€\s?\d[\d,]*(?:\.\d+)?               # €10
    |\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP) # 10 USD
    )
    (?:[^\n.]{0,40}(?:\/mo|\/month|\/yr|\/year|per\s+(?:month|year)))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_price(text: str) -> str:
    """Pull the first price-like phrase from text, or return 'See website'."""
    match = _PRICE_RE.search(text)
    if match:
        return textwrap.shorten(match.group().strip(), width=60, placeholder="…")
    if re.search(r"\bfree\b", text, re.IGNORECASE):
        return "Free (see website for details)"
    return "See website"


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Raw Tavily search. Returns list of {title, url, content} or []."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = httpx.post(
            _TAVILY_URL,
            json={
                "api_key":      api_key,
                "query":        query,
                "search_depth": "basic",
                "max_results":  max_results,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except httpx.HTTPStatusError as e:
        logger.warning("Tavily HTTP %s: %s", e.response.status_code, e.response.text[:200])
        return []
    except httpx.RequestError as e:
        logger.warning("Tavily request error: %s", e)
        return []


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo fallback. Returns list of {title, url, content} or []."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "content": r.get("body", ""),
                })
        return results
    except ImportError:
        logger.warning("duckduckgo-search not installed; install it for DDG fallback.")
        return []
    except Exception as e:
        logger.warning("DuckDuckGo search error: %s", e)
        return []


def _search(query: str, max_results: int = 5) -> list[dict]:
    """Try Tavily first; fall back to DuckDuckGo."""
    results = _tavily_search(query, max_results)
    if results:
        logger.info("search via Tavily: %s", query)
        return results
    logger.info("search via DuckDuckGo (Tavily unavailable): %s", query)
    return _ddg_search(query, max_results)


# ── Public API ────────────────────────────────────────────────────────────────

def search_tools(query: str, max_results: int = 5) -> list[ToolResult]:
    """
    Search for automation tools matching *query*.

    Returns a list of ToolResult dicts:
        tool_name    – result title
        latest_price – first price phrase found in the snippet
        source       – URL

    Example
    -------
    >>> results = search_tools("auto-triage support emails with Slack alerts")
    >>> for r in results:
    ...     print(r["tool_name"], r["latest_price"])
    """
    full_query = f"best automation tools for {query} 2025"
    raw = _search(full_query, max_results)
    return [
        ToolResult(
            tool_name=r.get("title", "Unknown"),
            latest_price=_extract_price(r.get("content", "")),
            source=r.get("url", ""),
        )
        for r in raw
    ]


def search_pricing(tool_name: str) -> ToolResult:
    """
    Look up the latest pricing for a specific *tool_name*.

    Returns a single ToolResult dict:
        tool_name    – the queried tool name
        latest_price – first price phrase found in the top result
        source       – URL of the best result

    Example
    -------
    >>> result = search_pricing("Zapier")
    >>> print(result["latest_price"])
    'Free / $19.99/month (Starter)'
    """
    query = f"{tool_name} pricing plans 2025"
    raw   = _search(query, max_results=3)

    if not raw:
        return ToolResult(tool_name=tool_name, latest_price="Unavailable", source="")

    best = raw[0]
    # Prefer a result whose URL looks like an official pricing page
    for r in raw:
        if "pricing" in r.get("url", "").lower():
            best = r
            break

    return ToolResult(
        tool_name=tool_name,
        latest_price=_extract_price(best.get("content", "")),
        source=best.get("url", ""),
    )
