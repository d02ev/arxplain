"""Multi-step agents.

Pipeline: Ingest -> Extract -> GapAnalysis -> Planner -> Explainer -> Quiz.
Each agent is a small class with run(); the orchestrator records an
AgentStep trace so the CLI output can show exactly what happened.
Explanations adapt depth to (required_level - user_level) per concept.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from .extractor import extract_concepts
from .llm import BaseLLM, MockLLM
from .models import (AgentStep, Concept, Dimension, Explanation, Gap, Level,
                     LearningPlan, Paper, PlanStep, QuizQuestion, QuizResult,
                     UserProfile, text_id)

DEPTH_FOR_GAP = {0: "skip-known", 1: "concise"}
ANALOGIES: dict[Dimension, str] = {
    Dimension.MATHEMATICS: "Think of it as a recipe: inputs go in, a rule transforms them, and the output tells you what to do next.",
    Dimension.EXPERIMENTATION: "Think of it as a taste-test: same recipe, different tweaks, then score which plate people prefer.",
    Dimension.KEYWORDS: "Think of it as learning the local slang before visiting a new city.",
    Dimension.RESULTS: "Think of it as the scoreboard after the match: what actually happened, not what was hoped.",
    Dimension.ASSUMPTIONS: "Think of it as the fine print: things taken for granted before the argument starts.",
    Dimension.LIMITATIONS: "Think of it as the edge of the map: where this stops working and why.",
    Dimension.DOMAIN: "Think of it as background knowledge locals take for granted.",
    Dimension.THEORY: "Think of it as the 'why it works' story behind the trick.",
}


def depth_for_gap(gap: int) -> str:
    if gap <= 0:
        return "skip-known"
    if gap == 1:
        return "guided"
    return "foundation"


class GapAnalysisAgent:
    name = "gap-analysis"

    def run(self, paper: Paper, profile: UserProfile) -> list[Gap]:
        gaps = []
        for c in paper.concepts:
            # A per-topic 1-10 rating wins over the coarse dimension level.
            user = profile.level_for_concept(c)
            gaps.append(Gap(concept=c, user_level=user,
                            gap=c.required_level - user))
        return sorted(gaps, key=lambda g: (-g.gap, g.concept.name))


class PlannerAgent:
    name = "planner"

    def run(self, paper: Paper, gaps: list[Gap]) -> LearningPlan:
        steps: list[PlanStep] = []
        ordered = sorted(gaps, key=lambda g: (-g.gap, g.concept.required_level))
        for i, g in enumerate(ordered, 1):
            depth = depth_for_gap(g.gap)
            why = ("Already in your comfort zone — quick recap only."
                   if depth == "skip-known" else
                   f"Needs level {g.concept.required_level} "
                   f"({Level.from_any(g.concept.required_level).label}); "
                   f"you are at {Level.from_any(g.user_level).label} here.")
            steps.append(PlanStep(
                order=i, concept_name=g.concept.name, dimension=g.concept.dimension,
                title=f"{g.concept.name}", why=why, target_depth=depth,
                prerequisites=g.concept.prerequisites))
        biggest = ordered[0] if ordered else None
        overview = (f"Start with '{biggest.concept.name}' — your largest gap — "
                    "then work down. Familiar items get a one-line recap."
                    if biggest and biggest.gap > 0 else
                    "You already cover most of what this paper needs; "
                    "the plan is mostly quick recaps.")
        minutes = sum({"foundation": 8, "guided": 4, "concise": 2,
                       "skip-known": 1}[s.target_depth] for s in steps)
        return LearningPlan(paper_id=paper.id, steps=steps, overview=overview,
                            estimated_minutes=max(5, minutes))


def _fallback_body(concept: Concept, depth: str, profile: UserProfile) -> tuple[str, str, str]:
    label = Level.from_any(profile.level(concept.dimension)).label
    ev = f' In the paper: "{concept.evidence}"' if concept.evidence else ""
    if depth == "skip-known":
        body = (f"You already have this covered ({label} in "
                f"{concept.dimension.label}). {concept.description}{ev}")
        return body, "", ""
    if depth == "guided":
        body = (f"{concept.description} At your level ({label}), focus on how it is "
                f"used here rather than re-learning it from scratch.{ev} "
                "Read the surrounding paragraph, then ask: what would change "
                "if this piece were removed?")
        return body, "", "In your own words: what role does this play in the paper?"
    body = (f"Start from zero on this one — no prior knowledge assumed. "
            f"{concept.description}{ev} "
            "Take it in three passes: (1) what it is in plain words, "
            "(2) why the paper needs it, (3) a tiny concrete instance. "
            "Only then re-read the paper's paragraph.")
    return body, ANALOGIES.get(concept.dimension, ""), \
        "Can you give a one-sentence example of this idea outside this paper?"


class ExplainerAgent:
    """Cost rule: at most ONE paid LLM call per run, however many concepts.

    Concepts the user already knows (gap <= 0) are never sent to the LLM.
    All remaining concepts go in a single batched prompt and the reply is
    parsed back per concept; anything unparseable falls back to the free
    heuristic text. No LLM (mock) also costs nothing.
    """

    name = "explainer"
    system = ("You are ArXplain, a patient tutor explaining research-paper "
              "concepts. Reply with ONLY a JSON array, no prose around it.")

    def __init__(self, llm: BaseLLM | None = None, *, max_llm_concepts: int = 8):
        self.llm = llm or MockLLM()
        self.max_llm_concepts = max_llm_concepts

    def explain_one(self, concept: Concept, gap: Gap,
                    profile: UserProfile) -> Explanation:
        """Heuristic-only single explanation (never calls the LLM)."""
        depth = depth_for_gap(gap.gap)
        body, analogy, check = _fallback_body(concept, depth, profile)
        return Explanation(concept_name=concept.name, dimension=concept.dimension,
                           depth=depth, title=concept.name, body=body,
                           analogy=analogy,
                           example=f'From the paper: "{concept.evidence}"'
                           if concept.evidence and depth != "skip-known" else "",
                           check_question=check)

    def _batched_bodies(self, targets: list[tuple[Concept, Gap]]) -> dict[str, str]:
        lines = []
        for i, (c, g) in enumerate(targets, 1):
            depth = depth_for_gap(g.gap)
            words = 110 if depth == "foundation" else 70
            lines.append(
                f"{i}. \"{c.name}\" [{c.dimension.label}] learner="
                f"{Level.from_any(g.user_level).label}, required="
                f"{Level.from_any(c.required_level).label}, depth={depth}, "
                f"max {words} words. Paper evidence: {(c.evidence or c.description)[:220]}")
        prompt = (
            "Explain each numbered concept for this learner: plain language "
            "first, then how the paper uses it. Return ONLY a JSON array like "
            '[{"n":1,"body":"..."}] with one object per concept.\n' + "\n".join(lines))
        max_tokens = min(3200, 500 + 200 * len(targets))
        raw = self.llm.complete(prompt, system=self.system, max_tokens=max_tokens)
        return _parse_batched(raw, targets)

    def run(self, paper: Paper, gaps: list[Gap],
            profile: UserProfile) -> list[Explanation]:
        by_name = {g.concept.name: g for g in gaps}
        out = {c.name: self.explain_one(c, by_name[c.name], profile)
               for c in paper.concepts}
        if isinstance(self.llm, MockLLM):
            return [out[c.name] for c in paper.concepts]
        targets = [(c, by_name[c.name]) for c in paper.concepts
                   if by_name[c.name].gap >= 1]
        # Largest gaps first if we must cap how many concepts the LLM writes.
        targets.sort(key=lambda t: -t[1].gap)
        targets = targets[:self.max_llm_concepts]
        if not targets:
            return [out[c.name] for c in paper.concepts]
        try:
            bodies = self._batched_bodies(targets)
        except Exception:  # noqa: BLE001 - budget/network: heuristics are free
            bodies = {}
        for c, _g in targets:
            if bodies.get(c.name):
                out[c.name].body = bodies[c.name]
        return [out[c.name] for c in paper.concepts]


def _parse_batched(raw: str, targets: list[tuple[Concept, Gap]]) -> dict[str, str]:
    import json
    import re as _re
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    m = _re.search(r"\[.*\]", text, _re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            bodies: dict[str, str] = {}
            for item in items:
                n = int(item.get("n", 0))
                body = str(item.get("body", "")).strip()
                if 1 <= n <= len(targets) and body:
                    bodies[targets[n - 1][0].name] = body
            if bodies:
                return bodies
        except Exception:  # noqa: BLE001
            pass
    # Fallback: numbered blocks "1. ..." in plain text.
    bodies = {}
    parts = _re.split(r"\n\s*(\d+)[.)]\s*", "\n" + text)
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i])
        except ValueError:
            continue
        if 1 <= n <= len(targets) and parts[i + 1].strip():
            bodies[targets[n - 1][0].name] = parts[i + 1].strip()[:1200]
    return bodies


def _sentences(text: str, *, limit: int = 4, max_chars: int = 720) -> list[str]:
    import re as _re
    parts = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
             if len(s.strip()) > 25]
    out, total = [], 0
    for s in parts:
        if total + len(s) > max_chars and out:
            break
        out.append(s)
        total += len(s)
        if len(out) >= limit:
            break
    return out


def concepts_in_section(section_text: str, concepts: list[Concept]) -> list[Concept]:
    low = section_text.lower()
    found: list[Concept] = []
    for c in concepts:
        tokens = [t for t in re.findall(r"[a-z]{4,}", c.name.lower())
                  if t not in ("with", "from", "that", "this")]
        ev = (c.evidence or "")[:60].lower()
        if (ev and ev in low) or (tokens and sum(t in low for t in tokens)
                                  >= max(1, len(tokens) // 2)):
            found.append(c)
    return found


class PaperExplainerAgent:
    """Explains the ACTUAL paper, section by section, at the user's level.

    Offline/extractive by default (free): it works from the paper's own
    sentences, then scaffolds the hard topics the user is weak on using
    the concept explanations, and attaches their resource links. With a
    real LLM (quality mode) one batched call rewrites every section.
    """

    name = "paper-explainer"
    system = ("You explain a research paper section by section to one learner. "
              "Reply with ONLY a JSON array, no prose around it.")

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or MockLLM()

    def _heuristic_body(self, section, used: list[Concept],
                        by_name: dict[str, Explanation],
                        gap_by_name: dict[str, Gap],
                        profile: UserProfile) -> tuple[str, list[str]]:
        sents = _sentences(section.text)
        what = " ".join(sents)
        paras = [f"This part of the paper is the **{section.title}** section. "
                 f"In the authors' own terms: {what}"]
        weak = [c for c in used if gap_by_name[c.name].gap >= 1]
        known = [c for c in used if gap_by_name[c.name].gap < 1]
        if weak:
            bits = []
            for c in weak[:3]:
                topic = profile.topic_for(c.name)
                rating = (f" (you rated this {topic.familiarity}/10)"
                          if topic is not None and topic.familiarity is not None else "")
                exp = by_name[c.name].body if c.name in by_name else c.description
                if len(exp) > 260:  # cut at a sentence end, never mid-word
                    exp = exp[:260].rsplit(".", 1)[0].strip() + "."
                bits.append(f"**{c.name}**{rating}: {exp}")
            paras.append("Background you need to follow this section — "
                         "explained at your level: " + " ".join(bits))
        if known:
            paras.append("You already know "
                         + ", ".join(c.name for c in known[:3])
                         + ", so the authors' argument above can be read at "
                           "full speed; the novelty is how they combine these pieces.")
        keys = sents[:3]
        return "\n\n".join(paras), keys

    def run(self, paper: Paper, gaps: list[Gap], explanations: list[Explanation],
            profile: UserProfile):
        """One explanation per paper section — EVERY section, in order.

        Nothing is dropped: short sections and References are explained
        too (briefly), because the reader asked for the whole paper.
        """
        from .models import SectionExplanation, Topic
        by_name = {e.concept_name: e for e in explanations}
        gap_by_name = {g.concept.name: g for g in gaps}
        out: list[SectionExplanation] = []
        # Title-page/licence text ("Preamble") is not a paper section.
        explained_sections = [s for s in paper.sections
                              if s.title.strip().lower() != "preamble"]
        for s in explained_sections:
            used = concepts_in_section(s.text, paper.concepts)
            if len(s.text.strip()) >= 40:
                body, keys = self._heuristic_body(s, used, by_name,
                                                  gap_by_name, profile)
            else:
                body = (f"This section (\"{s.title}\") is very short in the "
                        f"paper. In full, it says: {s.text.strip() or '(no text)'}")
                keys = [s.text.strip()] if s.text.strip() else []
            # Hard Requirements FOR THIS SECTION: profile topics marked
            # hard that this section uses, else concepts needing level >= 2.
            reqs: list[Topic] = []
            weak_res, seen = [], set()
            for c in used:
                topic = profile.topic_for(c.name)
                is_hard = (topic.is_hard if topic is not None
                           else c.required_level >= 2)
                if not is_hard:
                    continue
                req = topic if topic is not None else Topic(
                    name=c.name, dimension=c.dimension, requirement="hard",
                    why=c.description, search_query=f"{c.name} explained tutorial")
                reqs.append(req)
                exp = by_name.get(c.name)
                weak = gap_by_name[c.name].gap >= 1 if c.name in gap_by_name else False
                if weak and exp is not None:
                    for r in exp.resources:
                        if r.url not in seen:
                            seen.add(r.url)
                            weak_res.append(r)
            figs = [f for f in paper.figures
                    if f.section_title.lower() == s.title.lower()]
            refs = (list(paper.references)
                    if s.title.strip().lower() == "references" else [])
            if refs:
                body = (f"This is the paper's bibliography: the {len(refs)} "
                        "works the authors cite and build on. Each entry "
                        "below links to the actual cited paper (publisher/"
                        "DOI or arXiv where available, otherwise a Scholar "
                        "lookup of its title). Following the entries marked "
                        "as background above is the fastest way to fill "
                        "the gaps this paper assumes.")
                keys = [r.title or r.text[:80] for r in refs[:3]]
            out.append(SectionExplanation(
                section_title=s.title, body=body, key_points=keys,
                concepts_used=[c.name for c in used], resources=weak_res[:3],
                hard_requirements=reqs, figures=figs, references=refs))
        attached = {id(f) for se in out for f in se.figures}
        for f in paper.figures:  # never silently lose a figure
            if id(f) in attached:
                continue
            for se, s in zip(out, explained_sections):
                if f.label and f.label.lower() in s.text.lower():
                    se.figures.append(f)
                    break
        if not isinstance(self.llm, MockLLM) and out:
            try:
                bodies = self._llm_bodies(paper, explained_sections, profile)
            except Exception:  # noqa: BLE001
                bodies = {}
            for se, s in zip(out, explained_sections):
                if bodies.get(s.title):
                    se.body = bodies[s.title]
        return out

    def _llm_bodies(self, paper: Paper, sections,
                    profile: UserProfile) -> dict[str, str]:
        lines = []
        for i, s in enumerate(sections, 1):
            lines.append(f"{i}. Section \"{s.title}\": {s.text[:650]}")
        prompt = (
            f"Paper: {paper.title}. Learner profile (dimension levels): "
            + ", ".join(f"{d.label}={Level.from_any(profile.level(d)).label}"
                        for d in Dimension)
            + ". For each numbered section, explain WHAT THE PAPER SAYS and "
              "WHY IT MATTERS, in plain language adapted to this learner: "
              "slow down and define terms where the paper uses ideas from a "
              "dimension they are weak in; be brief where they are strong. "
              "120-170 words each. Return ONLY a JSON array like "
              '[{"n":1,"body":"..."}].\n' + "\n".join(lines))
        max_tokens = min(3600, 600 + 260 * len(sections))
        raw = self.llm.complete(prompt, system=self.system, max_tokens=max_tokens)
        return _parse_batched(raw, [(SimpleNamespace(name=s.title), None)
                                    for s in sections])


_QUIZ_TEMPLATES: dict[Dimension, tuple[str, list[str], int]] = {
    Dimension.MATHEMATICS: ("What best describes '{c}' as used in this paper?",
                            ["A core calculation the results depend on",
                             "A decorative symbol with no role",
                             "A dataset name", "A programming library"], 0),
    Dimension.EXPERIMENTATION: ("Why does the paper use '{c}'?",
                                ["To measure whether the method actually works",
                                 "To make the paper longer",
                                 "To replace the method section",
                                 "It is not used in the paper"], 0),
    Dimension.RESULTS: ("What does '{c}' tell you about this paper?",
                           ["What the experiments actually found",
                            "A definition of a new term",
                            "A software tool used", "A dataset name"], 0),
    Dimension.ASSUMPTIONS: ("Why does '{c}' matter when reading this paper?",
                            ["It is something the method takes for granted",
                             "It is the main result", "It is a metric",
                             "It is a baseline"], 0),
    Dimension.LIMITATIONS: ("What does '{c}' warn the reader about?",
                            ["Where or why the approach may break",
                             "How to install the code", "A new theorem",
                             "A dataset split"], 0),
    Dimension.KEYWORDS: ("What does '{c}' mean in this paper's context?",
                         ["A key term the paper defines or relies on",
                          "An unrelated buzzword", "A typo",
                          "A reference to another paper's title"], 0),
    Dimension.THEORY: ("What best captures '{c}'?",
                       ["A foundational idea the method builds on",
                        "A software tool", "A dataset split",
                        "A formatting choice"], 0),
    Dimension.DOMAIN: ("Why does '{c}' matter for understanding this paper?",
                       ["It is background the authors assume the reader has",
                        "It is irrelevant background", "It is a tool name",
                        "It is a metric"], 0),
}


class QuizAgent:
    """Adaptive quiz: diagnostic (before) and check (after) questions.

    Difficulty follows the user's level on that dimension; answering
    updates the suggested level used by the next explanation pass.
    """

    name = "quiz"

    def generate(self, paper: Paper, profile: UserProfile,
                 *, count: int = 6) -> list[QuizQuestion]:
        qs: list[QuizQuestion] = []
        ranked = sorted(paper.concepts,
                        key=lambda c: -(c.required_level - int(profile.level(c.dimension))))
        for c in ranked[:count]:
            tmpl, options, ans = _QUIZ_TEMPLATES[c.dimension]
            qs.append(QuizQuestion(
                id=text_id(f"q:{c.name}"), question=tmpl.format(c=c.name),
                dimension=c.dimension, concept_name=c.name, kind="mcq",
                options=list(options), answer=options[ans],
                explanation=f"'{c.name}': {c.description}",
                difficulty=max(0, min(4, c.required_level))))
        return qs

    def evaluate(self, question: QuizQuestion, user_answer: str,
                 profile: UserProfile) -> QuizResult:
        correct = user_answer.strip().lower() == question.answer.strip().lower()
        cur = int(profile.level(question.dimension))
        suggested = min(4, cur + 1) if correct else max(0, cur - 1)
        feedback = ("Correct. " + question.explanation if correct else
                    f"Not quite — the expected answer is \"{question.answer}\". "
                    + question.explanation)
        return QuizResult(question_id=question.id, correct=correct,
                          feedback=feedback,
                          suggested_level=suggested if suggested != cur else None)

    def diagnostic(self, profile: UserProfile) -> list[QuizQuestion]:
        """Short per-dimension probe used before reading, to set levels."""
        probes = {
            Dimension.MATHEMATICS: ("If a model multiplies a query vector by key "
                                    "vectors and applies softmax, what is it computing?",
                                    ["Attention weights", "A loss value",
                                     "A learning rate", "A dataset split"]),
            Dimension.EXPERIMENTATION: ("What is an ablation study?",
                                        ["Removing one component to see its effect",
                                         "Training for more epochs",
                                         "A new dataset", "A plotting style"]),
            Dimension.RESULTS: ("A paper says it 'outperforms the baseline by 3 BLEU'. What happened?",
                                ["Its score was 3 points higher on that metric",
                                 "It trained 3 times faster",
                                 "It used 3 more datasets", "It has 3 authors"]),
            Dimension.ASSUMPTIONS: ("What is an assumption in a research paper?",
                                    ["Something taken for granted that the method relies on",
                                     "The main result", "A type of chart",
                                     "A citation style"]),
            Dimension.LIMITATIONS: ("Why do papers discuss complexity (e.g. O(n^2))?",
                                    ["It shows how cost grows and where the method stops scaling",
                                     "It is a citation format", "It names a dataset",
                                     "It describes the code style"]),
            Dimension.THEORY: ("What does a Transformer block mainly consist of?",
                               ["Attention + feed-forward layers",
                                "Only convolutions", "Only recurrent cells",
                                "A database query"]),
        }
        out = []
        for dim, (q, opts) in probes.items():
            out.append(QuizQuestion(
                id=text_id(f"diag:{dim.value}"), question=q, dimension=dim,
                concept_name=f"Diagnostic: {dim.label}", kind="mcq",
                options=list(opts), answer=opts[0],
                explanation=f"Probes your {dim.label} level.",
                difficulty=int(profile.level(dim))))
        return out

    def apply_diagnostic(self, profile: UserProfile,
                         results: list[QuizResult]) -> UserProfile:
        """Move each probed dimension one step toward the evidence."""
        by_q: dict[str, QuizResult] = {r.question_id: r for r in results}
        for q in self.diagnostic(profile):
            r = by_q.get(q.id)
            if r is not None and r.suggested_level is not None:
                profile.set_level(q.dimension, r.suggested_level)
        return profile
