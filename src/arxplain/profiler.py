"""Profiling step: find what a paper demands of its reader, before explaining.

Flow (cost-capped at every step):
  1. heuristic concept extraction            (free, offline)
  2. one web search for how the paper/topic is explained elsewhere
                                               (1 search, cached)
  3. ONE batched LLM call classifying topics into
     "Hard Requirement" vs "Good To Know (Optional)"
     and naming prerequisites the heuristics missed
                                               (1 LLM call; 0 with mock/off)
  4. per-topic resource search, hard topics first
                                               (capped, cached; results are
                                               reused later as the explanation's
                                               links, so nothing is searched twice)
The user then rates each topic 1-10 and that becomes profile.json.
"""
from __future__ import annotations

import json
import re

from .extractor import extract_concepts
from .llm import BaseLLM, MockLLM
from .models import AgentStep, Dimension, Paper, Topic
from .search import BaseSearch, NoSearch

_SYSTEM = ("You classify what a research paper requires of its reader. "
           "Reply with ONLY a JSON array, no prose around it.")


def _parse_topic_items(raw: str) -> list[dict]:
    """Parse the JSON array; salvage complete objects if it was truncated."""
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
        except Exception:  # noqa: BLE001
            pass
    items = []
    for obj in re.findall(r"\{[^{}]*\}", raw):
        try:
            items.append(json.loads(obj))
        except Exception:  # noqa: BLE001
            continue
    return items


def find_context(text: str, name: str, *, window: int = 300) -> str:
    """The passage of THIS paper where a topic appears (for the prompt)."""
    low = text.lower()
    phrase = name.lower().split(":")[0].strip()
    idx = low.find(phrase) if len(phrase) >= 4 else -1
    if idx < 0:
        tokens = [w for w in re.findall(r"[a-z]{4,}", phrase)]
        for w in tokens:
            idx = low.find(w)
            if idx >= 0:
                break
    if idx < 0:
        return ""
    start = max(0, idx - 70)
    snippet = re.sub(r"\s+", " ", text[start:idx + window]).strip()
    return ("…" + snippet) if start else snippet


def heuristic_topics(paper: Paper) -> list[Topic]:
    topics: list[Topic] = []
    for c in paper.concepts:
        hard = c.required_level >= 2 or c.dimension in (
            Dimension.MATHEMATICS, Dimension.THEORY)
        topics.append(Topic(
            name=c.name, dimension=c.dimension,
            requirement="hard" if hard else "optional",
            why=c.description,
            search_query=f"{c.name} explained tutorial",
            paper_context=(c.evidence or "")[:320]))
    return _ordered(topics)


def _ordered(topics: list[Topic]) -> list[Topic]:
    seen: dict[str, Topic] = {}
    for t in topics:
        seen.setdefault(t.name.lower(), t)
    vals = list(seen.values())
    return sorted(vals, key=lambda t: (0 if t.is_hard else 1, t.name.lower()))


class ProfilerAgent:
    name = "profiler"

    def __init__(self, llm: BaseLLM | None = None, search: BaseSearch | None = None,
                 *, max_topics: int = 14, max_optional: int = 8,
                 max_searches: int = 6, resources_per_topic: int = 2):
        self.llm = llm or MockLLM()
        self.search = search or NoSearch()
        self.max_topics = max_topics
        self.max_optional = max_optional
        self.max_searches = max_searches
        self.resources_per_topic = resources_per_topic

    # -- LLM classification (single batched call) -------------------------
    def _llm_topics(self, paper: Paper, base: list[Topic],
                    web_titles: list[str]) -> list[Topic] | None:
        listing = "\n".join(
            f"- {t.name} [{t.dimension.label}] (heuristic guess: {t.requirement})"
            for t in base[:20])
        web = "\n".join(f"- {t}" for t in web_titles[:6]) or "(none)"
        # Slim schema on purpose: only the classification is asked of the
        # LLM. Longer outputs (reasons, queries) get truncated by reasoning
        # models and waste tokens; both are derived locally instead.
        prompt = (
            f"Paper: {paper.title}\nAbstract: {paper.abstract[:600]}\n\n"
            f"Topics found in the paper:\n{listing}\n\n"
            f"How the web explains this area (titles):\n{web}\n\n"
            "Classify each topic: \"hard\" = cannot follow this paper without "
            "it (Hard Requirement), \"optional\" = helpful background "
            "(Good To Know). You may add up to 3 missing prerequisites. "
            "Return ONLY a compact JSON array like "
            '[{"n":"Topic name","d":"theory","r":"hard"}] where d is one of '
            "mathematics|experimentation|keywords|frameworks|language|tools|"
            f"domain|theory and r is hard|optional. At most {self.max_topics}.")
        try:
            raw = self.llm.complete(prompt, system=_SYSTEM, max_tokens=2400)
        except Exception:  # noqa: BLE001
            return None
        items = _parse_topic_items(raw or "")
        if not items:
            return None
        by_name = {t.name.lower(): t for t in base}
        topics: list[Topic] = []
        for it in items:
            name = str(it.get("n") or it.get("name", "")).strip()
            if not name:
                continue
            try:
                dim = Dimension(str(it.get("d") or it.get("dimension", "theory")).lower())
            except Exception:
                dim = Dimension.THEORY
            src = by_name.get(name.lower())
            topics.append(Topic(
                name=name, dimension=dim,
                requirement="optional" if (it.get("r") or it.get("requirement")) == "optional" else "hard",
                why=(src.why if src else f"Prerequisite the paper assumes: {name}."),
                search_query=f"{name} explained tutorial"))
        return _ordered(topics) or None

    # -- paper context + self-check questions ------------------------------
    def _add_contexts_and_checks(self, paper: Paper,
                                 topics: list[Topic]) -> bool:
        """Attach each topic's paper passage and a self-check question.

        Returns True if the LLM wrote the questions. One batched call;
        on mock/off/budget-exhaustion a template question is used so
        the prompt is never bare.
        """
        for t in topics:
            if not t.paper_context:
                t.paper_context = find_context(paper.text, t.name)
            t.self_check = (
                f"Do you understand {t.name} as this paper uses it — "
                "could you explain its role in the passage above?")
        if isinstance(self.llm, MockLLM) or not topics:
            return False
        listing = "\n".join(
            f"- {t.name}: {(t.paper_context or t.why)[:220]}" for t in topics)
        prompt = (
            "For each topic below, write ONE specific self-check question "
            "starting with \"Do you understand\", based on how THIS paper "
            "uses the topic in the quoted passage. Mention the concrete "
            "notation, mechanism, dataset or decision from the passage "
            "(e.g. \"Do you understand why the attention scores are "
            "divided by sqrt(d_k) before the softmax?\"). Keep each "
            "question under 30 words. Return ONLY a compact JSON array "
            'like [{"n":"Topic name","q":"Do you understand ...?"}].\n'
            + listing)
        try:
            raw = self.llm.complete(prompt, system=_SYSTEM, max_tokens=1800)
        except Exception:  # noqa: BLE001 - templates above remain
            return False
        by_name: dict[str, str] = {}
        for it in _parse_topic_items(raw or ""):
            n = str(it.get("n") or it.get("name", "")).strip().lower()
            q = str(it.get("q") or it.get("question", "")).strip()
            if n and q:
                by_name[n] = q
        hit = 0
        for t in topics:
            q = by_name.get(t.name.lower())
            if q is None:  # fuzzy: topic name contained in the returned name
                q = next((v for k, v in by_name.items()
                          if k in t.name.lower() or t.name.lower() in k), None)
            if q:
                t.self_check = q if q.endswith("?") else q + "?"
                hit += 1
        return hit > 0

    # -- main entry ---------------------------------------------------------
    def run(self, paper: Paper) -> tuple[list[Topic], list[AgentStep]]:
        trace: list[AgentStep] = []
        if not paper.concepts:
            extract_concepts(paper)
        base = heuristic_topics(paper)
        trace.append(AgentStep("profiler", "extract paper topics (free)",
                               f"{len(base)} candidate topics from the paper text"))
        web_titles: list[str] = []
        searches_used = 0
        if not isinstance(self.search, NoSearch) and self.max_searches > 0:
            res = self.search.search(
                f"{paper.title} prerequisites explained", max_results=3)
            searches_used += 1
            web_titles = [r.title for r in res if r.title]
            trace.append(AgentStep("profiler", "web search: paper overview",
                                   f"{len(web_titles)} result(s) via {self.search.name}"))
        topics: list[Topic] | None = None
        if not isinstance(self.llm, MockLLM):
            topics = self._llm_topics(paper, base, web_titles)
            trace.append(AgentStep(
                "profiler", "LLM classifies Hard Requirement vs Good To Know",
                f"{len(topics)} topics" if topics else
                "LLM output unusable — kept heuristic classification"))
        if topics is None:
            topics = base
        else:
            # Never silently drop a topic the paper actually mentions:
            # anything the LLM omitted stays, as Good To Know.
            known = {t.name.lower() for t in topics}
            topics = topics + [
                Topic(name=t.name, dimension=t.dimension, requirement="optional",
                      why=t.why, search_query=t.search_query,
                      paper_context=t.paper_context)
                for t in base if t.name.lower() not in known]
        hard = [t for t in topics if t.is_hard]
        optional = [t for t in topics if not t.is_hard][:self.max_optional]
        # Hard topics are never dropped by the cap; only Good-To-Know is.
        topics = _ordered(hard + optional)
        llm_checks = self._add_contexts_and_checks(paper, topics)
        trace.append(AgentStep(
            "profiler", "write self-check questions from the paper",
            "LLM questions grounded in each topic's passage"
            if llm_checks else "template questions (LLM off/unavailable)"))
        # Resources now, hard first, so explaining later needs no new searches.
        if not isinstance(self.search, NoSearch):
            for t in topics:
                if searches_used >= self.max_searches:
                    break
                t.resources = self.search.search(
                    t.search_query or f"{t.name} explained",
                    max_results=self.resources_per_topic)
                searches_used += 1
            linked = sum(1 for t in topics if t.resources)
            trace.append(AgentStep("profiler", "find learning resources",
                                   f"resources for {linked}/{len(topics)} topics "
                                   f"({searches_used} search call(s), cached)"))
        trace.append(AgentStep("profiler", "ready for your 1-10 ratings",
                               f"{len(hard)} hard requirement(s), "
                               f"{len(optional)} good-to-know"))
        return topics, trace
