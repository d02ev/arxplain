"""Cost controls: the whole point is NOT calling the LLM per concept."""
import json
import re

import pytest

from arxplain.agents import ExplainerAgent, GapAnalysisAgent
from arxplain.cli import DEMO_TEXT
from arxplain.extractor import extract_concepts
from arxplain.ingestion import paper_from_text
from arxplain.llm import (BaseLLM, BudgetedCachedLLM, LLMError, MetaLLM,
                          MockLLM, get_llm)
from arxplain.models import Dimension, Level, UserProfile
from arxplain.pipeline import ArXplainPipeline


class CountingLLM(BaseLLM):
    name = "counting"
    model = "counting-1"

    def complete(self, prompt, *, system="", max_tokens=1200):
        self.usage.calls += 1
        self.usage.prompt_chars += len(prompt) + len(system)
        n = len(re.findall(r"^\d+\. ", prompt, re.M))
        if n:  # batched explanations request -> JSON array
            out = json.dumps([{"n": i, "body": f"LLM explanation {i}."}
                              for i in range(1, n + 1)])
        else:
            out = "LLM summary sentence."
        self.usage.completion_chars += len(out)
        return out


def demo_paper():
    p = paper_from_text(DEMO_TEXT, source="demo")
    extract_concepts(p)
    return p


def profile(level):
    p = UserProfile()
    for d in Dimension:
        p.set_level(d, level)
    return p


def test_meta_provider_shape(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout=60):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr("arxplain.llm._post", fake_post)
    llm = MetaLLM(api_key="dummy-key")
    assert llm.complete("Explain attention", system="Be brief") == "hello"
    assert seen["url"] == "https://api.meta.ai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer dummy-key"
    assert seen["payload"]["model"] == "muse-spark-1.3"
    assert seen["payload"]["reasoning_effort"] == "low"  # billed reasoning kept low
    assert seen["payload"]["max_tokens"] >= 768  # floored: reasoning eats max_tokens
    assert llm.usage.calls == 1


def test_empty_reasoning_response_is_an_error_not_a_crash():
    from arxplain.llm import _extract_content
    with pytest.raises(LLMError):
        _extract_content({"choices": [{"finish_reason": "length",
                                        "message": {"content": None}}]})
    parts = {"choices": [{"finish_reason": "stop", "message": {
        "content": [{"type": "text", "text": "hi"}]}}]}
    assert _extract_content(parts) == "hi"


def test_meta_missing_key_is_loud_when_explicit(monkeypatch):
    for var in ("MODEL_API_KEY", "META_API_KEY", "ARXPLAIN_META_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError):
        get_llm("meta")


def test_budget_cap_and_cache(tmp_path):
    cache = tmp_path / "cache.json"
    llm = BudgetedCachedLLM(CountingLLM(), max_calls=1, cache_path=cache)
    assert "summary" in llm.complete("Summarize x").lower()
    assert llm.complete("Summarize x")  # cache hit: free, no new call
    assert llm.usage.calls == 1 and llm.usage.cache_hits >= 1
    with pytest.raises(LLMError):
        llm.complete("A different prompt")
    assert llm.usage.blocked == 1
    # A fresh wrapper over the same disk cache also hits for free.
    llm2 = BudgetedCachedLLM(CountingLLM(), max_calls=0, cache_path=cache)
    assert llm2.complete("Summarize x")
    assert llm2.usage.calls == 0 and llm2.usage.cache_hits == 1


def test_explainer_makes_one_batched_call_not_one_per_concept():
    paper, prof = demo_paper(), profile(Level.NOVICE)
    gaps = GapAnalysisAgent().run(paper, prof)
    llm = CountingLLM()
    exps = ExplainerAgent(llm).run(paper, gaps, prof)
    assert llm.usage.calls == 1  # was: one call per concept (18)
    assert any(e.body.startswith("LLM explanation") for e in exps)


def test_known_concepts_cost_zero_calls():
    paper, prof = demo_paper(), profile(Level.EXPERT)
    gaps = GapAnalysisAgent().run(paper, prof)
    llm = CountingLLM()
    ExplainerAgent(llm).run(paper, gaps, prof)
    assert llm.usage.calls == 0


def test_pipeline_economy_vs_quality_call_counts():
    econ = ArXplainPipeline(CountingLLM(), llm_mode="economy").run(
        demo_paper(), profile(Level.NOVICE))
    assert econ.llm_usage["calls"] == 1 and econ.llm_provider == "counting"
    qual = ArXplainPipeline(CountingLLM(), llm_mode="quality").run(
        demo_paper(), profile(Level.NOVICE))
    assert qual.llm_usage["calls"] == 2
    off = ArXplainPipeline(CountingLLM(), llm_mode="off").run(
        demo_paper(), profile(Level.NOVICE))
    assert off.llm_usage["calls"] == 0 and off.llm_provider == "mock"
    assert isinstance(get_llm(), MockLLM)  # default stays free/offline
