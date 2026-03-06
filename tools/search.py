"""TrustGraph search bridge — drop-in replacement for the hackathon tools/search.py.

Wraps trustandverify.search backends with a synchronous interface
so they can be called directly from Jac walkers. The hackathon's
web_search(query, max_results) signature is preserved exactly.

New additions (not in hackathon):
    multi_search()  — fans out to all configured backends concurrently
    brave_search()  — Brave Search API
    bing_search()   — Bing Web Search API
"""

from __future__ import annotations

import asyncio
import os


def _run(coro):
    """Run an async coroutine synchronously — safe to call from Jac walkers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)
    # Already inside an event loop (e.g. Jupyter) — use nest_asyncio
    import nest_asyncio
    nest_asyncio.apply()
    return loop.run_until_complete(coro)


def _to_dicts(results: list) -> list[dict]:
    return [
        {"title": r.title, "url": r.url, "content": r.content, "score": r.score}
        for r in results
    ]


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via Tavily. Drop-in replacement for hackathon tools/search.py.

    Returns list of dicts with: title, url, content, score.
    Returns empty list (never raises) on any error.
    """
    from trustandverify.search.tavily import TavilySearch
    return _to_dicts(_run(TavilySearch().search(query, max_results)))


def multi_search(query: str, max_results: int = 5) -> list[dict]:
    """Fan out to all configured search backends concurrently.

    Uses whichever of Tavily / Brave / Bing have API keys set.
    Falls back to web_search() if only Tavily is available.
    """
    from trustandverify.search.tavily import TavilySearch
    from trustandverify.search.brave import BraveSearch
    from trustandverify.search.bing import BingSearch
    from trustandverify.search.multi import MultiSearch

    backends = [b for b in [TavilySearch(), BraveSearch(), BingSearch()] if b.is_available()]
    if len(backends) <= 1:
        return web_search(query, max_results)
    return _to_dicts(_run(MultiSearch(backends).search(query, max_results)))


def brave_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via Brave Search API. Requires BRAVE_API_KEY."""
    from trustandverify.search.brave import BraveSearch
    return _to_dicts(_run(BraveSearch().search(query, max_results)))


def bing_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via Bing Web Search API. Requires BING_API_KEY."""
    from trustandverify.search.bing import BingSearch
    return _to_dicts(_run(BingSearch().search(query, max_results)))
