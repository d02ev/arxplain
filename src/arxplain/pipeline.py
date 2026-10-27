"""Orchestrator: wires the agents into one multi-step run, with a trace.

LLM spend by mode (real providers only; mock is always free):
  off      0 calls — heuristics only
  economy  1 call  — one batched explanation call; summary is heuristic (default)
  quality  2 calls — batched concept explanations + LLM rewrite of the paper sections
The budget wrapper additionally hard-caps paid calls and serves repeats
from cache, so a re-run of the same paper+profile costs 0 new calls.
"""
from __future__ import annotations

from .agents import (ExplainerAgent, GapAnalysisAgent, PaperExplainerAgent,
                     PlannerAgent, QuizAgent)
from .extractor import extract_concepts
from .llm import BaseLLM, MockLLM, get_llm
from .models import AgentStep, ExplainResult, Paper, UserProfile

MODES = ("off", "economy", "quality")


class ArXplainPipeline:
    def __init__(self, llm: BaseLLM | None = None, *, llm_mode: str = "economy",
                 max_llm_concepts: int = 8, search=None,
                 max_resource_searches: int = 5):
        if llm_mode not in MODES:
            raise ValueError(f"llm_mode must be one of {MODES}")
        self.llm_mode = llm_mode
        self.llm = MockLLM() if llm_mode == "off" else (llm or get_llm())
        self.search = search  # None = only links already stored in the profile
        self.max_resource_searches = max_resource_searches
        self.gap_agent = GapAnalysisAgent()
        self.planner = PlannerAgent()
        self.explainer = ExplainerAgent(self.llm, max_llm_concepts=max_llm_concepts)
        # The paper itself is explained extractively for free; only quality
        # mode spends its second call letting the LLM rewrite the sections.
        self.paper_explainer = PaperExplainerAgent(
            self.llm if llm_mode == "quality" else MockLLM())
        self.quiz = QuizAgent()

    def run(self, paper: Paper, profile: UserProfile) -> ExplainResult:
        trace: list[AgentStep] = []
        trace.append(AgentStep("ingestion", "load paper",
                               f"'{paper.title}' from {paper.source} "
                               f"({len(paper.sections)} sections)"))
        if not paper.complete:
            trace.append(AgentStep(
                "ingestion", "WARNING: full text unavailable",
                "only the abstract could be retrieved, so only that can be "
                "explained — try a PDF file for the complete paper"))
        if not paper.concepts:
            extract_concepts(paper)
        trace.append(AgentStep("extractor", "extract concepts",
                               f"{len(paper.concepts)} concepts across "
                               f"{len({c.dimension for c in paper.concepts})} dimensions"))
        gaps = self.gap_agent.run(paper, profile)
        large = sum(1 for g in gaps if g.gap >= 2)
        trace.append(AgentStep("gap-analysis", "compare paper vs. your profile",
                               f"{large} large gap(s), "
                               f"{sum(1 for g in gaps if g.gap == 1)} small, "
                               f"{sum(1 for g in gaps if g.gap <= 0)} already known"))
        plan = self.planner.run(paper, gaps)
        trace.append(AgentStep("planner", "order a learning path",
                               f"{len(plan.steps)} steps, ~{plan.estimated_minutes} min"))
        explanations = self.explainer.run(paper, gaps, profile)
        n_linked = self._attach_resources(explanations, gaps, profile)
        if n_linked:
            trace.append(AgentStep("resources", "link help where confidence is low",
                                   f"{n_linked} explanation(s) carry article/blog links"))
        paper_explanation = self.paper_explainer.run(paper, gaps, explanations, profile)
        trace.append(AgentStep("paper-explainer", "explain the paper section by section",
                               f"{len(paper_explanation)} section(s), adapted to your ratings"))
        if paper.figures:
            trace.append(AgentStep("figures", "place the paper's figures in their sections",
                                   f"{len(paper.figures)} figure(s) with captions"))
        u = self.llm.usage
        trace.append(AgentStep(
            "explainer", "write adaptive explanations",
            f"mode={self.llm_mode}, provider={self.llm.name}: "
            f"{u.calls} paid call(s), {u.cache_hits} cache hit(s) so far — "
            "known concepts cost 0"))
        summary = self._summary(paper, profile, gaps)
        trace.append(AgentStep("explainer", "summarize", "heuristic summary (free)"))
        usage = self.llm.usage.to_dict()
        trace.append(AgentStep("budget", "LLM usage",
                               f"{usage['calls']} paid call(s), "
                               f"{usage['cache_hits']} cache hit(s), "
                               f"~{usage['est_total_tokens']} est. tokens"))
        return ExplainResult(paper=paper, profile=profile, gaps=gaps, plan=plan,
                             explanations=explanations,
                             paper_explanation=paper_explanation, summary=summary,
                             trace=trace, llm_usage=usage,
                             llm_provider=self.llm.name)

    def _attach_resources(self, explanations, gaps, profile: UserProfile) -> int:
        """Links go where the user lacks confidence: low 1-10 rating, or a
        real gap. Links stored in profile.json are reused for free; only if
        a search provider was explicitly enabled do we fetch new ones
        (hard-capped, hard topics first via gap order)."""
        gap_by_name = {g.concept.name: g for g in gaps}
        linked, searches = 0, 0
        for e in sorted(explanations, key=lambda x: -gap_by_name[x.concept_name].gap
                        if x.concept_name in gap_by_name else 0):
            g = gap_by_name.get(e.concept_name)
            topic = profile.topic_for(e.concept_name)
            if topic is not None:
                e.familiarity = topic.familiarity
            weak = (topic is not None and topic.low_confidence) or \
                (g is not None and g.gap >= 1)
            if not weak:
                continue
            if topic is not None and topic.resources:
                e.resources = list(topic.resources)
            elif self.search is not None and searches < self.max_resource_searches:
                e.resources = self.search.search(
                    f"{e.concept_name} explained tutorial", max_results=2)
                searches += 1
            linked += bool(e.resources)
        return linked

    def _summary(self, paper: Paper, profile: UserProfile, gaps) -> str:
        large = [g.concept.name for g in gaps if g.gap >= 2]
        focus = (f" Your biggest lifts are: {', '.join(large[:3])} — those get "
                 "from-zero treatment below." if large else
                 " You already have the background this paper assumes, "
                 "so explanations stay brief.")
        return (f"{paper.title} — {paper.abstract[:500]}{focus}")


def explain_paper(paper: Paper, profile: UserProfile | None = None,
                  llm: BaseLLM | None = None, *,
                  llm_mode: str = "economy") -> ExplainResult:
    return ArXplainPipeline(llm, llm_mode=llm_mode).run(
        paper, profile or UserProfile())
