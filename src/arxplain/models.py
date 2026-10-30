"""Core domain models for ArXplain."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Dimension(str, Enum):
    """What a paper demands of its reader — its ideas, not its tooling.

    Frameworks / programming languages / tools are deliberately NOT
    dimensions: most papers present a strategy or algorithm that is
    independent of any implementation. Such mentions are demoted to a
    single experimental-setup note (see extractor). Old profiles that
    still carry the removed keys are ignored on load.
    """

    THEORY = "theory"                # core idea: strategy / algorithm / approach
    MATHEMATICS = "mathematics"
    EXPERIMENTATION = "experimentation"  # design: datasets, baselines, metrics
    RESULTS = "results"              # what the experiments actually showed
    ASSUMPTIONS = "assumptions"      # assumptions & considerations made
    LIMITATIONS = "limitations"      # constraints, costs, failure cases
    KEYWORDS = "keywords"            # terminology / vocabulary of the field
    DOMAIN = "domain"                # subject-matter knowledge

    @property
    def label(self) -> str:
        return {
            "theory": "Core Ideas & Theory",
            "mathematics": "Mathematics",
            "experimentation": "Experimentation",
            "results": "Results & Interpretation",
            "assumptions": "Assumptions & Considerations",
            "limitations": "Constraints & Limitations",
            "keywords": "Keywords & Terminology",
            "domain": "Domain Knowledge",
        }[self.value]


class Level(int, Enum):
    NOVICE = 0
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

    @property
    def label(self) -> str:
        return ["Novice", "Beginner", "Intermediate", "Advanced", "Expert"][int(self)]

    @classmethod
    def from_any(cls, v: Any) -> "Level":
        if isinstance(v, Level):
            return v
        if isinstance(v, str):
            m = {x.label.lower(): x for x in cls}
            m.update({x.name.lower(): x for x in cls})
            if v.lower() in m:
                return m[v.lower()]
            v = int(v)
        return cls(max(0, min(4, int(v))))


def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:12]


@dataclass
class Section:
    title: str
    text: str
    level: int = 1  # heading depth
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Concept:
    name: str
    dimension: Dimension
    description: str = ""
    required_level: int = 2  # Level value needed to follow the paper
    evidence: str = ""       # snippet from the paper where it appears
    prerequisites: list[str] = field(default_factory=list)
    section: str = ""

    @property
    def id(self) -> str:
        return text_id(f"{self.dimension.value}:{self.name.lower()}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimension"] = self.dimension.value
        d["id"] = self.id
        return d


@dataclass
class Figure:
    """An image from the paper, kept with the section it belongs to."""
    src: str  # absolute URL (or local path) of the image
    caption: str = ""
    alt: str = ""
    section_title: str = ""
    label: str = ""  # e.g. "Figure 1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Reference:
    """One cited work from the paper's bibliography, with a real link."""
    text: str  # full citation as printed in the paper
    title: str = ""
    authors: str = ""
    year: str = ""
    label: str = ""  # e.g. "[13]"
    url: str = ""    # doi.org / arxiv.org / publisher / scholar search
    url_kind: str = ""  # doi | arxiv | publisher | scholar

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Paper:
    title: str
    text: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    source: str = "text"  # arxiv | pdf | url | text
    source_id: str = ""
    sections: list[Section] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    complete: bool = True  # False = only the abstract could be retrieved
    id: str = field(default_factory=lambda: _uid("paper_"))

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "title": self.title, "authors": self.authors,
            "abstract": self.abstract, "source": self.source,
            "source_id": self.source_id, "complete": self.complete,
            "figures": [f.to_dict() for f in self.figures],
            "references": [r.to_dict() for r in self.references],
            "sections": [s.to_dict() for s in self.sections],
            "concepts": [c.to_dict() for c in self.concepts],
        }
        if include_text:
            d["text"] = self.text
        return d

    # -- Canonical PaperDocument JSON -------------------------------------
    # The contract between ingestion adapters (arXiv HTML, PDF, text,
    # MarkItDown, future parsers) and every downstream agent. Agents
    # consume this document; they never re-parse a source file.
    def to_document(self) -> dict[str, Any]:
        doc = self.to_dict(include_text=True)
        doc["schema_version"] = PAPER_SCHEMA_VERSION
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "Paper":
        if int(doc.get("schema_version", 1)) > PAPER_SCHEMA_VERSION:
            raise ValueError(
                f"paper document schema {doc.get('schema_version')} is newer "
                f"than supported {PAPER_SCHEMA_VERSION}")
        concepts = []
        for c in doc.get("concepts") or []:
            try:
                dim = Dimension(str(c.get("dimension", "theory")).lower())
            except Exception:
                dim = Dimension.EXPERIMENTATION
            concepts.append(Concept(
                name=c.get("name", ""), dimension=dim,
                description=c.get("description", ""),
                required_level=int(c.get("required_level", 2)),
                evidence=c.get("evidence", ""),
                prerequisites=list(c.get("prerequisites") or []),
                section=c.get("section", "")))
        paper = cls(
            title=doc.get("title", "Untitled paper"),
            text=doc.get("text", ""),
            authors=list(doc.get("authors") or []),
            abstract=doc.get("abstract", ""),
            source=doc.get("source", "text"),
            source_id=doc.get("source_id", ""),
            sections=[Section(title=s.get("title", ""), text=s.get("text", ""),
                              level=int(s.get("level", 1)),
                              index=int(s.get("index", i)))
                      for i, s in enumerate(doc.get("sections") or [])],
            concepts=concepts,
            figures=[Figure(**{k: f.get(k, "") for k in
                               ("src", "caption", "alt", "section_title", "label")})
                     for f in doc.get("figures") or []],
            references=[Reference(**{k: r.get(k, "") for k in
                                     ("text", "title", "authors", "year",
                                      "label", "url", "url_kind")})
                        for r in doc.get("references") or []],
            complete=bool(doc.get("complete", True)))
        if doc.get("id"):
            paper.id = doc["id"]
        if not paper.text and paper.sections:
            paper.text = "\n\n".join(f"{s.title}\n{s.text}"
                                     for s in paper.sections)
        return paper


PAPER_SCHEMA_VERSION = 1


def familiarity_to_level(familiarity: int) -> int:
    """Map a user's 1-10 self-rating onto the 0-4 Level scale."""
    f = max(1, min(10, int(familiarity)))
    return (f - 1) * 4 // 9  # 1-3->0..1, 4-6->1..2, 7-8->3, 9-10->4 approx


@dataclass
class Resource:
    """A web article/blog/doc that explains a topic the user is weak on."""
    title: str
    url: str
    source: str = ""  # tavily | ddgs
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Topic:
    """One thing a paper assumes/discusses, as found by the profiler."""
    name: str
    dimension: Dimension = Dimension.THEORY
    requirement: str = "hard"  # hard | optional  (Good To Know)
    why: str = ""
    search_query: str = ""
    paper_context: str = ""  # passage from THIS paper where the topic appears
    self_check: str = ""     # "Do you understand ...?" drawn from that passage
    familiarity: int | None = None  # user's 1-10 self-rating, None = unrated
    resources: list[Resource] = field(default_factory=list)

    @property
    def is_hard(self) -> bool:
        return self.requirement == "hard"

    @property
    def low_confidence(self) -> bool:
        return self.familiarity is not None and self.familiarity <= 4

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension.value,
                "requirement": self.requirement, "why": self.why,
                "search_query": self.search_query,
                "paper_context": self.paper_context,
                "self_check": self.self_check,
                "familiarity": self.familiarity,
                "resources": [r.to_dict() for r in self.resources]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Topic":
        try:
            dim = Dimension(str(d.get("dimension", "theory")).lower())
        except Exception:
            # Profiles written before frameworks/language/tools were
            # removed: such topics were implementation details, i.e.
            # experimental setup, not core ideas.
            dim = Dimension.EXPERIMENTATION
        return cls(name=d.get("name", ""), dimension=dim,
                   requirement="optional" if d.get("requirement") == "optional" else "hard",
                   why=d.get("why", ""), search_query=d.get("search_query", ""),
                   paper_context=d.get("paper_context", ""),
                   self_check=d.get("self_check", ""),
                   familiarity=d.get("familiarity"),
                   resources=[Resource(**{k: r.get(k, "") for k in
                                          ("title", "url", "source", "snippet")})
                              for r in (d.get("resources") or []) if r.get("url")])


@dataclass
class UserProfile:
    """What the user already understands, per dimension and per paper topic."""

    name: str = "learner"
    levels: dict[Dimension, int] = field(default_factory=dict)
    interests: list[str] = field(default_factory=list)
    goal: str = ""  # e.g. "implement this paper", "exam prep", "survey"
    topics: list[Topic] = field(default_factory=list)
    paper_title: str = ""
    paper_id: str = ""

    def __post_init__(self) -> None:
        for dim in Dimension:
            self.levels.setdefault(dim, int(Level.BEGINNER))

    def level(self, dim: Dimension) -> Level:
        return Level.from_any(self.levels.get(dim, 1))

    def set_level(self, dim: Dimension, level: Any) -> None:
        self.levels[dim] = int(Level.from_any(level))

    def topic_for(self, concept_name: str) -> Topic | None:
        key = concept_name.lower()
        for t in self.topics:
            if t.name.lower() == key or t.name.lower() in key or key in t.name.lower():
                return t
        return None

    def level_for_concept(self, concept: "Concept") -> int:
        """Topic familiarity (1-10) overrides the coarse dimension level."""
        t = self.topic_for(concept.name)
        if t is not None and t.familiarity is not None:
            return familiarity_to_level(t.familiarity)
        return int(self.level(concept.dimension))

    def sync_levels_from_topics(self) -> None:
        """Derive dimension levels from rated topics (mean, per dimension)."""
        by_dim: dict[Dimension, list[int]] = {}
        for t in self.topics:
            if t.familiarity is not None:
                by_dim.setdefault(t.dimension, []).append(
                    familiarity_to_level(t.familiarity))
        for dim, vals in by_dim.items():
            self.levels[dim] = round(sum(vals) / len(vals))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UserProfile":
        if not data:
            return cls()
        levels: dict[Dimension, int] = {}
        for k, v in (data.get("levels") or {}).items():
            try:
                levels[Dimension(str(k).lower())] = int(Level.from_any(v))
            except Exception:
                continue
        paper = data.get("paper") or {}
        p = cls(name=data.get("name", "learner"),
                interests=list(data.get("interests") or []),
                goal=data.get("goal", ""),
                topics=[Topic.from_dict(t) for t in (data.get("topics") or [])],
                paper_title=data.get("paper_title", "") or paper.get("title", ""),
                paper_id=data.get("paper_id", "") or paper.get("source_id", ""))
        p.levels.update(levels)
        if p.topics and not levels:
            p.sync_levels_from_topics()
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paper": {"title": self.paper_title, "source_id": self.paper_id},
            "levels": {d.value: int(v) for d, v in self.levels.items()},
            "level_labels": {d.value: Level.from_any(v).label for d, v in self.levels.items()},
            "topics": [t.to_dict() for t in self.topics],
            "interests": self.interests, "goal": self.goal,
        }


@dataclass
class Gap:
    concept: Concept
    user_level: int
    gap: int  # required - user; >0 means explanation must scaffold

    @property
    def severity(self) -> str:
        return "none" if self.gap <= 0 else "small" if self.gap == 1 else "large"

    def to_dict(self) -> dict[str, Any]:
        return {"concept": self.concept.to_dict(), "user_level": self.user_level,
                "required_level": self.concept.required_level, "gap": self.gap,
                "severity": self.severity}


@dataclass
class PlanStep:
    order: int
    concept_name: str
    dimension: Dimension
    title: str
    why: str
    target_depth: str  # foundation | guided | concise | skip-known
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimension"] = self.dimension.value
        return d


@dataclass
class LearningPlan:
    paper_id: str
    steps: list[PlanStep] = field(default_factory=list)
    overview: str = ""
    estimated_minutes: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "overview": self.overview,
                "estimated_minutes": self.estimated_minutes,
                "steps": [s.to_dict() for s in self.steps]}


@dataclass
class Explanation:
    concept_name: str
    dimension: Dimension
    depth: str
    title: str
    body: str
    analogy: str = ""
    example: str = ""
    check_question: str = ""
    familiarity: int | None = None  # user's 1-10 rating, when profiled
    resources: list[Resource] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimension"] = self.dimension.value
        d["resources"] = [r.to_dict() for r in self.resources]
        return d


@dataclass
class QuizQuestion:
    id: str
    question: str
    dimension: Dimension
    concept_name: str
    kind: str = "mcq"  # mcq | short
    options: list[str] = field(default_factory=list)
    answer: str = ""
    explanation: str = ""
    difficulty: int = 1

    def public_dict(self) -> dict[str, Any]:
        """Question without the answer key (safe to show the learner)."""
        return {"id": self.id, "question": self.question,
                "dimension": self.dimension.value, "concept_name": self.concept_name,
                "kind": self.kind, "options": self.options, "difficulty": self.difficulty}

    def to_dict(self) -> dict[str, Any]:
        d = self.public_dict()
        d.update({"answer": self.answer, "explanation": self.explanation})
        return d


@dataclass
class QuizResult:
    question_id: str
    correct: bool
    feedback: str
    suggested_level: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStep:
    agent: str
    action: str
    detail: str = ""
    status: str = "done"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectionExplanation:
    """One section of the actual paper, explained at the user's level."""
    section_title: str
    body: str  # what the paper says here, explained plainly and adapted
    key_points: list[str] = field(default_factory=list)
    concepts_used: list[str] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    # Hard Requirements for THIS section, with the user's rating on each.
    hard_requirements: list[Topic] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resources"] = [r.to_dict() for r in self.resources]
        d["hard_requirements"] = [t.to_dict() for t in self.hard_requirements]
        d["figures"] = [f.to_dict() for f in self.figures]
        d["references"] = [r.to_dict() for r in self.references]
        return d


@dataclass
class ExplainResult:
    paper: Paper
    profile: UserProfile
    gaps: list[Gap] = field(default_factory=list)
    plan: LearningPlan | None = None
    explanations: list[Explanation] = field(default_factory=list)
    paper_explanation: list[SectionExplanation] = field(default_factory=list)
    summary: str = ""
    trace: list[AgentStep] = field(default_factory=list)
    llm_usage: dict[str, Any] = field(default_factory=dict)
    llm_provider: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper": self.paper.to_dict(),
            "profile": self.profile.to_dict(),
            "summary": self.summary,
            "gaps": [g.to_dict() for g in self.gaps],
            "plan": self.plan.to_dict() if self.plan else None,
            "explanations": [e.to_dict() for e in self.explanations],
            "paper_explanation": [s.to_dict() for s in self.paper_explanation],
            "trace": [t.to_dict() for t in self.trace],
            "llm_provider": self.llm_provider,
            "llm_usage": self.llm_usage,
        }
