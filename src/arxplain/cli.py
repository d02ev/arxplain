"""ArXplain command line.

  arxplain explain 1706.03762
  arxplain explain paper.pdf --profile me.json -o out.html
  arxplain explain --demo --level beginner
  arxplain profile-paper 1706.03762 --save profile.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .agents import QuizAgent
from .ingestion import (fetch_arxiv, fetch_url_text, normalize_arxiv_id,
                        paper_from_pdf_bytes, paper_from_text)
from .llm import LLMError, MockLLM, get_llm
from .models import Dimension, Level, Paper, Topic, UserProfile
from .profiler import ProfilerAgent
from .search import SearchError, get_search
from .pipeline import ArXplainPipeline
from .render import render_html

DEMO_TEXT = """Attention Is All You Need

Abstract
We propose the Transformer, a network architecture based solely on attention
mechanisms. Experiments on machine translation show superior quality while
being more parallelizable. Our model achieves 28.4 BLEU on the WMT 2014
English-to-German translation task.

1. Introduction
Recurrent and convolutional networks have dominated sequence transduction.
We propose the Transformer based solely on attention mechanisms.

2. Model Architecture
The Transformer uses scaled dot-product attention and softmax over
embeddings. Attention Is computed with matrix multiplication, layer
normalization and dropout for regularization. The implementation is in
Python using PyTorch and CUDA, trained with Docker containers and
Weights & Biases tracking.

3. Experiments
We evaluate on WMT 2014 with accuracy and BLEU metrics against strong
baselines. An ablation study removes multi-head attention. Training uses
a learning rate schedule, batch size 25,000 tokens, for 100,000 epochs
steps on NVIDIA GPUs. Hyperparameters were tuned carefully.
"""


def load_paper(raw: str) -> Paper:
    if raw == "--demo":
        return paper_from_text(DEMO_TEXT, source="demo", source_id="attention")
    p = Path(raw)
    if p.exists():
        if p.suffix.lower() == ".pdf":
            return paper_from_pdf_bytes(p.read_bytes(), source_id=str(p))
        return paper_from_text(p.read_text(encoding="utf-8", errors="ignore"),
                               source="file", source_id=str(p))
    if re.match(r"^https?://", raw):
        if "arxiv.org" in raw:
            return fetch_arxiv(normalize_arxiv_id(raw))
        return fetch_url_text(raw)
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", raw.strip()) or "arxiv" in raw.lower():
        return fetch_arxiv(raw)
    return paper_from_text(raw, source="text")


def load_profile(args) -> UserProfile:
    profile = UserProfile()
    path = getattr(args, "profile", None)
    if path and Path(path).exists():
        profile = UserProfile.from_dict(json.loads(Path(path).read_text()))
    if getattr(args, "level", None):
        lvl = Level.from_any(args.level)
        for d in Dimension:
            profile.set_level(d, lvl)
    for dim in Dimension:
        v = getattr(args, dim.value, None)
        if v is not None:
            profile.set_level(dim, v)
    if getattr(args, "name", None):
        profile.name = args.name
    if getattr(args, "goal", None):
        profile.goal = args.goal
    return profile


def _add_profile_args(sp) -> None:
    sp.add_argument("--profile", help="JSON profile file (from `arxplain profile-paper`)")
    sp.add_argument("--name", default=None)
    sp.add_argument("--goal", default=None)
    sp.add_argument("--level", default=None,
                    help="global level: novice|beginner|intermediate|advanced|expert")
    for dim in Dimension:
        sp.add_argument(f"--{dim.value}", default=None,
                        help=f"{dim.label} level (0-4 or label)")


def run_diagnose(profile: UserProfile, *, interactive: bool = True,
                 answers: list[str] | None = None) -> UserProfile:
    quiz = QuizAgent()
    questions = quiz.diagnostic(profile)
    results = []
    print("Quick diagnostic — answer honestly; this sets your levels.\n")
    for i, q in enumerate(questions, 1):
        print(f"[{q.dimension.label}] {q.question}")
        for j, opt in enumerate(q.options):
            print(f"  {chr(97 + j)}) {opt}")
        if answers is not None:
            ans = answers[i - 1] if i - 1 < len(answers) else ""
            picked = ans if ans in q.options else q.options[0]
        elif interactive:
            raw = input("Your answer (a/b/c/d): ").strip().lower()
            idx = ord(raw) - 97 if raw else 0
            picked = q.options[idx] if 0 <= idx < len(q.options) else q.options[0]
        else:
            picked = q.options[0]
        print(f"  -> {picked}\n")
        results.append(quiz.evaluate(q, picked, profile))
    return quiz.apply_diagnostic(profile, results)


def _load_dotenv() -> None:
    """Load KEY=VALUE from ./.env into the environment (never overrides)."""
    import os
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def cmd_explain(args) -> int:
    raw = "--demo" if getattr(args, "demo", False) else args.paper
    if not raw:
        print("error: provide a paper (arXiv id, file, URL) or --demo",
              file=sys.stderr)
        return 2
    paper = load_paper(raw)
    profile = load_profile(args)
    if args.diagnose:
        profile = run_diagnose(profile)
    try:
        llm = _build_llm(args)
        # Links stored in the profile are reused free; missing links for
        # weak hard-requirements are fetched by default (DDGS is free,
        # capped) unless --no-resources.
        search = None if getattr(args, "no_resources", False) else get_search(
            args.search, max_calls=args.max_resource_searches,
            use_cache=not args.no_cache)
    except (LLMError, SearchError) as exc:
        print(f"error: {exc}\nSet MODEL_API_KEY in your environment or a "
              "local .env file (see .env.example). Never commit the key.",
              file=sys.stderr)
        return 2
    result = ArXplainPipeline(
        llm, llm_mode=args.llm_mode, search=search,
        max_resource_searches=args.max_resource_searches,
        max_llm_concepts=args.max_llm_concepts).run(paper, profile)
    slug = re.sub(r"[^a-z0-9]+", "-", paper.title.lower()).strip("-")[:48] or "paper"
    out = Path(args.output) if args.output else Path(f"{slug}.explained.html")
    if out.suffix.lower() != ".html":
        out = out.with_suffix(".html")
    out.write_text(render_html(result), encoding="utf-8")
    written = [str(out)]
    print(f"Paper : {paper.title} ({len(paper.concepts)} concepts, "
          f"{len(paper.sections)} sections)")
    print(f"For   : {profile.name} — "
          + ", ".join(f"{d.label.split(' ')[0]}={Level.from_any(profile.level(d)).label}"
                      for d in Dimension))
    for t in result.trace:
        print(f"  [{t.agent}] {t.action}: {t.detail}")
    u = result.llm_usage or {}
    print(f"LLM   : provider={result.llm_provider} mode={args.llm_mode} "
          f"paid_calls={u.get('calls', 0)} cache_hits={u.get('cache_hits', 0)} "
          f"blocked={u.get('blocked', 0)} est_tokens={u.get('est_total_tokens', 0)}")
    print("Wrote : " + ", ".join(written))
    print("\nOpen the file above in your browser.")
    return 0


def collect_ratings(topics: list[Topic], *, input_fn=input,
                    ratings: dict[str, int] | None = None,
                    default: int | None = None) -> list[Topic]:
    """Ask 1-10 per topic, Hard Requirements first. Enter skips (unrated).

    Non-interactive paths: an explicit ratings dict (fuzzy name match),
    or a single default applied to every topic.
    """
    if ratings:
        for t in topics:
            for k, v in ratings.items():
                if k.lower() in t.name.lower() or t.name.lower() in k.lower():
                    t.familiarity = max(1, min(10, int(v)))
        return topics
    if default is not None:
        for t in topics:
            t.familiarity = max(1, min(10, int(default)))
        return topics
    print("\nFor each topic you will see where THIS paper uses it and a "
          "self-check question taken from that passage. Judge yourself "
          "against the question, then rate your familiarity from 1 "
          "(never seen it) to 10 (could teach it). Enter skips a topic.\n")
    for i, t in enumerate(topics, 1):
        tag = "HARD REQUIREMENT" if t.is_hard else "Good To Know (Optional)"
        print(f"{i}/{len(topics)}  [{tag}] {t.name}  ({t.dimension.label})")
        if t.paper_context:
            ctx = t.paper_context.strip()
            if len(ctx) > 340:
                ctx = ctx[:340].rsplit(" ", 1)[0] + "…"
            print(f"    In this paper: \"{ctx}\"")
        if t.self_check:
            print(f"    Self-check: {t.self_check}")
        elif t.why:
            print(f"    Why it matters: {t.why}")
        for _try in range(3):
            raw = input_fn("    Your familiarity (1-10, Enter to skip): ").strip()
            if not raw:
                break
            try:
                val = int(raw)
            except ValueError:
                print("    Please enter a whole number from 1 to 10.")
                continue
            if 1 <= val <= 10:
                t.familiarity = val
                break
            print("    Please enter a number between 1 and 10.")
    return topics


def _build_llm(args):
    return MockLLM() if args.llm_mode == "off" else get_llm(
        args.llm, max_calls=args.max_llm_calls, use_cache=not args.no_cache,
        reasoning_effort=args.reasoning_effort)


def cmd_profile_paper(args) -> int:
    raw = "--demo" if getattr(args, "demo", False) else args.paper
    if not raw:
        print("error: provide a paper (arXiv id, file, URL) or --demo", file=sys.stderr)
        return 2
    paper = load_paper(raw)
    try:
        llm = _build_llm(args)
        search = get_search(args.search, max_calls=args.max_searches,
                            use_cache=not args.no_cache)
    except (LLMError, SearchError) as exc:
        print(f"error: {exc}\nCheck the keys in your .env (see .env.example). "
              "Never commit a key.", file=sys.stderr)
        return 2
    profiler = ProfilerAgent(llm, search, max_searches=args.max_searches,
                             max_optional=args.max_optional)
    topics, trace = profiler.run(paper)
    print(f"Paper : {paper.title}")
    for t in trace:
        print(f"  [{t.agent}] {t.action}: {t.detail}")
    print(f"LLM   : provider={llm.name} paid_calls={llm.usage.calls} "
          f"cache_hits={llm.usage.cache_hits} est_tokens={llm.usage.est_total_tokens}")
    ratings = None
    if args.ratings:
        ratings = {}
        for part in args.ratings.split(","):
            if "=" in part:
                k, _, v = part.partition("=")
                try:
                    ratings[k.strip()] = int(v.strip())
                except ValueError:
                    pass
    topics = collect_ratings(topics, ratings=ratings,
                             default=args.default_rating)
    profile = UserProfile(name=args.name or "learner", topics=topics,
                          paper_title=paper.title, paper_id=paper.source_id)
    profile.sync_levels_from_topics()
    out = Path(args.save)
    out.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    rated = sum(1 for t in topics if t.familiarity is not None)
    low = [t.name for t in topics if t.low_confidence]
    print(f"\nSaved : {out}  ({rated}/{len(topics)} topics rated"
          + (f"; low confidence: {', '.join(low)}" if low else "") + ")")
    print(f"Next  : arxplain explain {args.paper or '--demo'} --profile {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="arxplain",
        description="Explain a research paper at YOUR level — mathematics, "
                    "experimentation, keywords, frameworks, language and tools "
                    "are each adapted to what you already know.")
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("explain", help="explain a paper to an .html file")
    ex.add_argument("paper", nargs="?", default=None,
                    help="arXiv id/URL, PDF/.txt/.md path, or URL")
    ex.add_argument("--demo", action="store_true",
                    help="explain the bundled demo paper (Attention Is All You Need)")
    ex.add_argument("-o", "--output", default=None, help="output .html file path")
    ex.add_argument("--llm", default=None,
                    help="mock|meta|openai|anthropic|ollama (default: ARXPLAIN_LLM or mock)")
    ex.add_argument("--llm-mode", choices=["off", "economy", "quality"],
                    default="economy",
                    help="off=0 LLM calls, economy=1 batched call (default), "
                         "quality=concepts + paper sections rewritten by LLM (2 calls)")
    ex.add_argument("--max-llm-calls", type=int, default=None,
                    help="hard cap on paid LLM calls this run (default 2; cache hits are free)")
    ex.add_argument("--max-llm-concepts", type=int, default=8,
                    help="max gap concepts the LLM writes (default 8, largest gaps first)")
    ex.add_argument("--no-cache", action="store_true",
                    help="ignore the on-disk LLM response cache")
    ex.add_argument("--reasoning-effort",
                    choices=["minimal", "low", "medium", "high"], default=None,
                    help="Meta reasoning effort (default low; minimal is cheapest)")
    ex.add_argument("--with-resources", action="store_true",
                    help="(default) fetch missing article/blog links for weak "
                         "hard-requirements; kept for compatibility")
    ex.add_argument("--no-resources", action="store_true",
                    help="do not fetch any new resource links at explain time")
    ex.add_argument("--search", default="auto",
                    help="search provider: auto|tavily|ddgs|off")
    ex.add_argument("--max-resource-searches", type=int, default=5,
                    help="cap on resource searches at explain time (default 5)")
    ex.add_argument("--diagnose", action="store_true",
                    help="run the adaptive diagnostic quiz first")
    _add_profile_args(ex)
    ex.set_defaults(fn=cmd_explain)

    pp = sub.add_parser(
        "profile-paper",
        help="analyze a paper (LLM + web search), rate topics 1-10, save profile.json")
    pp.add_argument("paper", nargs="?", default=None,
                    help="arXiv id/URL, PDF/.txt/.md path, or URL")
    pp.add_argument("--demo", action="store_true",
                    help="profile the bundled demo paper")
    pp.add_argument("--save", default="profile.json", help="where to write profile.json")
    pp.add_argument("--name", default=None)
    pp.add_argument("--llm", default=None, help="mock|meta|openai|anthropic|ollama")
    pp.add_argument("--llm-mode", choices=["off", "economy", "quality"],
                    default="economy",
                    help="LLM topic classification + self-check questions "
                         "(2 batched calls max, cached)")
    pp.add_argument("--max-llm-calls", type=int, default=None)
    pp.add_argument("--max-llm-concepts", type=int, default=8)
    pp.add_argument("--no-cache", action="store_true")
    pp.add_argument("--reasoning-effort",
                    choices=["minimal", "low", "medium", "high"], default=None)
    pp.add_argument("--search", default="auto",
                    help="auto|tavily|ddgs|off (auto: Tavily if keyed, else free DDGS)")
    pp.add_argument("--max-searches", type=int, default=16,
                    help="cap on profiling searches incl. resource links "
                         "(default 16: overview + one per topic; cached, so repeats are free)")
    pp.add_argument("--max-optional", type=int, default=8,
                    help="max Good-To-Know topics to rate (default 8)")
    pp.add_argument("--ratings", default=None,
                    help='non-interactive ratings, e.g. "Attention=3,PyTorch=8"')
    pp.add_argument("--default-rating", type=int, default=None,
                    help="non-interactive: give every topic this 1-10 rating")
    pp.set_defaults(fn=cmd_profile_paper)
    return ap


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
