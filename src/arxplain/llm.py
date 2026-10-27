"""Provider-agnostic LLM layer with cost controls.

Providers (select with --llm or ARXPLAIN_LLM, default ``mock`` = offline):
  mock       heuristic stand-in, zero cost, no network
  meta       Meta Model API (Muse Spark) — key from MODEL_API_KEY
  openai     OpenAI — key from OPENAI_API_KEY
  anthropic  Anthropic — key from ANTHROPIC_API_KEY
  ollama     local Ollama — no key

Cost controls apply to every real provider (see BudgetedCachedLLM):
  * hard cap on paid calls per run  (ARXPLAIN_MAX_LLM_CALLS, default 2)
  * on-disk response cache          (repeat paper+profile costs 0 calls)
  * usage accounting                (calls / cache hits / est. tokens)

Keys are read from the environment (or a local .env) only. Never put a
key in source, a profile JSON, or a committed file.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    pass


def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). For budgeting, not billing."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class LLMUsage:
    calls: int = 0            # paid/network calls actually made
    cache_hits: int = 0
    blocked: int = 0          # calls refused by the budget cap
    prompt_chars: int = 0
    completion_chars: int = 0
    api_input_tokens: int = 0   # real counts from the API, when it reports them
    api_output_tokens: int = 0  # (includes reasoning tokens on reasoning models)

    @property
    def est_input_tokens(self) -> int:
        return self.api_input_tokens or self.prompt_chars // 4

    @property
    def est_output_tokens(self) -> int:
        return self.api_output_tokens or self.completion_chars // 4

    @property
    def est_total_tokens(self) -> int:
        return self.est_input_tokens + self.est_output_tokens

    def record_api_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.api_input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.api_output_tokens += int(usage.get("completion_tokens", 0) or 0)

    def to_dict(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "blocked": self.blocked,
                "est_input_tokens": self.est_input_tokens,
                "est_output_tokens": self.est_output_tokens,
                "est_total_tokens": self.est_total_tokens}


class BaseLLM(ABC):
    name = "base"
    model = ""

    def __init__(self) -> None:
        self.usage = LLMUsage()

    @abstractmethod
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        ...

    def chat(self, messages: list[dict[str, str]], **kw) -> str:  # pragma: no cover
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        return self.complete(prompt, system=system, **kw)


class MockLLM(BaseLLM):
    """Deterministic offline stand-in; callers use heuristic fallbacks."""

    name = "mock"

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        self.usage.calls += 1
        low = prompt.lower()
        if "summarize" in low or "summary" in low:
            return ("This paper's contribution is explained step by step in the "
                    "sections below, adapted to your current level on each dimension.")
        return ""


def _post(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"LLM request failed: {exc}") from exc


def _extract_content(data: dict) -> str:
    """Chat content may be str, a parts list, or None (reasoning spent the
    whole token budget — finish_reason='length' — on reasoning models)."""
    try:
        choice = data["choices"][0]
        content = (choice.get("message") or {}).get("content")
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Unexpected LLM response shape: {list(data)[:5]}") from exc
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                          for p in content)
    if not content or not str(content).strip():
        raise LLMError(
            "LLM returned empty content "
            f"(finish_reason={choice.get('finish_reason')}); on reasoning "
            "models this usually means max_tokens was too small.")
    return str(content).strip()


def _chat_complete(llm: BaseLLM, url: str, headers: dict, prompt: str,
                   system: str, max_tokens: int, extra: dict | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    payload = {"model": llm.model, "messages": msgs,
               "max_tokens": max_tokens, "temperature": 0.3, **(extra or {})}
    data = _post(url, payload, headers)
    out = _extract_content(data)
    llm.usage.calls += 1
    llm.usage.prompt_chars += len(system) + len(prompt)
    llm.usage.completion_chars += len(out)
    llm.usage.record_api_usage(data.get("usage"))
    return out


class MetaLLM(BaseLLM):
    """Meta Model API — OpenAI-compatible chat completions.

    Base URL: https://api.meta.ai/v1  (override: META_BASE_URL)
    Model:    muse-spark-1.3          (override: ARXPLAIN_MODEL / META_MODEL)
    Key:      MODEL_API_KEY (also META_API_KEY / ARXPLAIN_META_API_KEY)
    """

    name = "meta"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None,
                 reasoning_effort: str | None = None):
        super().__init__()
        self.model = (model or os.getenv("ARXPLAIN_MODEL")
                      or os.getenv("META_MODEL", "muse-spark-1.3"))
        self.base_url = (base_url or os.getenv("META_BASE_URL")
                         or "https://api.meta.ai/v1").rstrip("/")
        self.api_key = (api_key or os.getenv("MODEL_API_KEY")
                        or os.getenv("META_API_KEY")
                        or os.getenv("ARXPLAIN_META_API_KEY", ""))
        # Muse Spark is a reasoning model: hidden reasoning tokens are
        # billed. 'low' is ample for rewriting explanations from supplied
        # evidence; 'minimal' is ~5x cheaper again on short outputs.
        self.reasoning_effort = (reasoning_effort
                                 or os.getenv("ARXPLAIN_REASONING_EFFORT", "low"))
        if not self.api_key:
            raise LLMError("MODEL_API_KEY is not set (Meta Model API key)")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        # Reasoning tokens come out of max_tokens; too small a budget
        # returns empty content, so floor it.
        return _chat_complete(
            self, f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"}, prompt, system,
            max(max_tokens, 768), extra={"reasoning_effort": self.reasoning_effort})


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__()
        self.model = model or os.getenv("ARXPLAIN_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        return _chat_complete(self, "https://api.openai.com/v1/chat/completions",
                              {"Authorization": f"Bearer {self.api_key}"},
                              prompt, system, max_tokens)


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__()
        self.model = model or os.getenv("ARXPLAIN_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        payload: dict = {"model": self.model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        data = _post("https://api.anthropic.com/v1/messages", payload,
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        out = "".join(b.get("text", "") for b in data.get("content", [])).strip()
        self.usage.calls += 1
        self.usage.prompt_chars += len(system) + len(prompt)
        self.usage.completion_chars += len(out)
        return out


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        super().__init__()
        self.model = model or os.getenv("ARXPLAIN_MODEL", "llama3.1")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        data = _post(f"{self.host}/api/generate",
                     {"model": self.model, "prompt": full, "stream": False}, {})
        out = str(data.get("response", "")).strip()
        self.usage.calls += 1
        self.usage.prompt_chars += len(full)
        self.usage.completion_chars += len(out)
        return out


def default_cache_path() -> Path:
    base = os.getenv("ARXPLAIN_CACHE_DIR") or str(
        Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "arxplain")
    return Path(base) / "llm-cache.json"


class BudgetedCachedLLM(BaseLLM):
    """Wraps a real provider: hard call cap + persistent response cache.

    A cache hit costs nothing and does not count against the cap. When the
    cap is reached, complete() raises LLMError and callers fall back to
    their (free) heuristic text — a run can never silently overspend.
    """

    def __init__(self, inner: BaseLLM, *, max_calls: int = 2,
                 cache_path: Path | None = None, use_cache: bool = True):
        super().__init__()
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.max_calls = max(0, max_calls)
        self.cache_path = cache_path if use_cache else None
        self._mem: dict[str, str] = {}
        # Share one usage object so pipeline/CLI read the inner counters too.
        self.usage = inner.usage

    def _key(self, prompt: str, system: str, max_tokens: int) -> str:
        raw = f"{self.name}|{self.model}|{max_tokens}|{system}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _disk_get(self, key: str) -> str | None:
        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text()).get(key)
        except Exception:  # noqa: BLE001
            return None

    def _disk_put(self, key: str, value: str) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text())
            data[key] = value
            self.cache_path.write_text(json.dumps(data))
        except Exception:  # noqa: BLE001
            pass  # cache is best-effort; never fail a run over it

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1200) -> str:
        key = self._key(prompt, system, max_tokens)
        hit = self._mem.get(key) or self._disk_get(key)
        if hit is not None:
            self.usage.cache_hits += 1
            self._mem[key] = hit
            return hit
        if self.usage.calls >= self.max_calls:
            self.usage.blocked += 1
            raise LLMError(
                f"LLM budget exhausted ({self.max_calls} paid call(s) this run); "
                "using free heuristic explanations. Raise --max-llm-calls to spend more.")
        out = self.inner.complete(prompt, system=system, max_tokens=max_tokens)
        self._mem[key] = out
        self._disk_put(key, out)
        return out


_PROVIDERS = {"mock": MockLLM, "meta": MetaLLM, "meta-ai": MetaLLM,
              "muse": MetaLLM, "openai": OpenAILLM,
              "anthropic": AnthropicLLM, "ollama": OllamaLLM}


def get_llm(provider: str | None = None, *, max_calls: int | None = None,
            use_cache: bool = True, cache_path: Path | None = None,
            reasoning_effort: str | None = None) -> BaseLLM:
    """Resolve a provider, wrapped in budget+cache controls.

    An explicitly requested real provider raises LLMError if its key is
    missing (no silent downgrade). The implicit default falls back to mock.
    """
    explicit = provider is not None
    name = (provider or os.getenv("ARXPLAIN_LLM", "mock")).lower()
    cls = _PROVIDERS.get(name, MockLLM)
    if cls is MockLLM:
        return MockLLM()
    try:
        inner = (cls(reasoning_effort=reasoning_effort)
                 if cls is MetaLLM and reasoning_effort else cls())
    except LLMError:
        if explicit:
            raise
        return MockLLM()
    cap = max_calls if max_calls is not None else int(
        os.getenv("ARXPLAIN_MAX_LLM_CALLS", "2"))
    return BudgetedCachedLLM(inner, max_calls=cap, use_cache=use_cache,
                             cache_path=cache_path or default_cache_path())
