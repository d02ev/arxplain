"""Web search for learning resources, with caching and hard caps.

Only two providers, by design:
  tavily   TAVILY_API_KEY     https://api.tavily.com/search (primary)
  ddgs     no key             DuckDuckGo via the `ddgs` package when
                              installed, else its HTML endpoint (fallback)
  off      no network at all

auto = Tavily when TAVILY_API_KEY is set, otherwise DDGS.
Search calls are cached on disk (same query = 0 new calls) and callers
cap how many distinct searches a run may make, so this cannot run away.
Install the DDGS library with: uv sync --extra search
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Resource


class SearchError(RuntimeError):
    pass


def _request(url: str, *, data: bytes | None = None, headers: dict | None = None,
             timeout: int = 20) -> str:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.read().decode("utf-8", "ignore")
    except Exception as exc:  # noqa: BLE001
        raise SearchError(f"search request failed: {exc}") from exc


class BaseSearch(ABC):
    name = "base"
    calls = 0
    cache_hits = 0

    @abstractmethod
    def search(self, query: str, *, max_results: int = 3) -> list[Resource]:
        ...


class NoSearch(BaseSearch):
    name = "off"

    def search(self, query: str, *, max_results: int = 3) -> list[Resource]:
        return []


class TavilySearch(BaseSearch):
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not self.api_key:
            raise SearchError("TAVILY_API_KEY is not set")

    def search(self, query: str, *, max_results: int = 3) -> list[Resource]:
        self.calls += 1
        body = json.dumps({"api_key": self.api_key, "query": query,
                           "max_results": max_results,
                           "search_depth": "basic"}).encode()
        raw = _request("https://api.tavily.com/search", data=body,
                       headers={"Content-Type": "application/json"})
        data = json.loads(raw)
        return [Resource(title=r.get("title", ""), url=r.get("url", ""),
                         source="tavily", snippet=(r.get("content", "") or "")[:220])
                for r in data.get("results", []) if r.get("url")]


def _ddgs_client():
    """Return the DDGS class from `ddgs` (or legacy duckduckgo_search)."""
    for mod in ("ddgs", "duckduckgo_search"):
        try:
            module = __import__(mod)
            return module.DDGS
        except Exception:  # noqa: BLE001
            continue
    return None


class DDGSSearch(BaseSearch):
    """DuckDuckGo search — the free fallback. No API key, no payment.

    Uses the `ddgs` package when installed (uv sync --extra search);
    otherwise scrapes DuckDuckGo's HTML endpoint directly.
    """

    name = "ddgs"

    def _via_library(self, query: str, max_results: int) -> list[Resource] | None:
        cls = _ddgs_client()
        if cls is None:
            return None
        try:
            with cls() as client:
                try:
                    results = client.text(query, max_results=max_results)
                except TypeError:
                    results = client.text(query)
                out = []
                for r in list(results or [])[:max_results]:
                    url = r.get("href") or r.get("url", "")
                    if url:
                        out.append(Resource(
                            title=r.get("title", ""), url=url, source="ddgs",
                            snippet=(r.get("body", "") or "")[:220]))
                return out
        except Exception:  # noqa: BLE001
            return None

    def search(self, query: str, *, max_results: int = 3) -> list[Resource]:
        self.calls += 1
        via_lib = self._via_library(query, max_results)
        if via_lib is not None:
            return via_lib
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        try:
            raw = _request(url, headers={"User-Agent": "Mozilla/5.0 arxplain/0.1"})
        except SearchError:
            return []
        out: list[Resource] = []
        for m in re.finditer(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', raw, re.S):
            href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
            if "uddg=" in href:
                href = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
            out.append(Resource(title=_html.unescape(title).strip(), url=href,
                                source="ddgs"))
            if len(out) >= max_results:
                break
        return out


# Backwards-compatible alias (provider name is now canonically "ddgs").
DuckDuckGoSearch = DDGSSearch


def _cache_path() -> Path:
    base = os.getenv("ARXPLAIN_CACHE_DIR") or str(
        Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "arxplain")
    return Path(base) / "search-cache.json"


class CachedSearch(BaseSearch):
    """Disk-cached, call-capped wrapper around any provider."""

    def __init__(self, inner: BaseSearch, *, max_calls: int = 8,
                 cache_path: Path | None = None, use_cache: bool = True):
        self.inner = inner
        self.name = inner.name
        self.max_calls = max_calls
        self.cache_path = cache_path if use_cache else None
        self._mem: dict[str, list[Resource]] = {}

    @property
    def calls(self) -> int:  # type: ignore[override]
        return self.inner.calls

    @calls.setter
    def calls(self, v: int) -> None:
        self.inner.calls = v

    def _key(self, query: str, max_results: int) -> str:
        return hashlib.sha256(f"{self.name}|{query}|{max_results}".encode()).hexdigest()

    def _disk(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        except Exception:  # noqa: BLE001
            return {}

    def search(self, query: str, *, max_results: int = 3) -> list[Resource]:
        key = self._key(query, max_results)
        if key in self._mem:
            self.cache_hits += 1
            return self._mem[key]
        disk = self._disk()
        if key in disk:
            self.cache_hits += 1
            res = [Resource(**r) for r in disk[key]]
            self._mem[key] = res
            return res
        if self.inner.calls >= self.max_calls:
            return []  # cap reached: no more network, links are best-effort
        try:
            res = self.inner.search(query, max_results=max_results)
        except Exception:  # noqa: BLE001
            return []
        self._mem[key] = res
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                disk[key] = [r.to_dict() for r in res]
                self.cache_path.write_text(json.dumps(disk))
            except Exception:  # noqa: BLE001
                pass
        return res


_PROVIDERS = {"tavily": TavilySearch, "ddgs": DDGSSearch,
              "duckduckgo": DDGSSearch, "ddg": DDGSSearch,
              "off": NoSearch, "none": NoSearch}


def get_search(provider: str | None = None, *, max_calls: int = 8,
               use_cache: bool = True) -> BaseSearch:
    """auto: Tavily when TAVILY_API_KEY is set, otherwise DDGS (free).

    Only tavily / ddgs / off exist. Any other name (e.g. a removed paid
    provider) raises SearchError instead of silently changing provider.
    Explicit tavily without its key also raises.
    """
    name = (provider or os.getenv("ARXPLAIN_SEARCH", "auto")).lower()
    if name == "auto":
        name = "tavily" if os.getenv("TAVILY_API_KEY") else "ddgs"
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise SearchError(
            f"unknown search provider '{name}'; use tavily, ddgs, or off")
    inner = cls()  # may raise SearchError for a missing explicit key
    if isinstance(inner, NoSearch):
        return inner
    return CachedSearch(inner, max_calls=max_calls, use_cache=use_cache,
                        cache_path=_cache_path())
