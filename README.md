# ArXplain

**Explain a technological research paper at *your* level.**

ArXplain is a multi-step agentic system. It reads a paper, works out what
the paper assumes you already know — mathematics, experimentation,
keywords & terminology, results, assumptions and limitations —
domain knowledge and theory — compares that against *your* profile on
each of those dimensions, and then writes an explanation whose depth
adapts dimension by dimension: from-zero with an analogy where you have
a large gap, a guided pass where you are close, and a one-line recap
where you already know it.

The interface is a **CLI**. The explanation is written to a standalone
**`.html`** page you open in a browser.

```
paper ──▶ Ingestion ──▶ Concept extraction ──▶ Gap analysis (paper vs. you)
                                                      │
        Explanation (.html) ◀── Explainer ◀── Planner (learning path)
                                                      │
                                                 Quiz (adaptive check)
```

## Install

```bash
uv sync            # or: pip install -e .
uv sync --extra pdf   # optional: PDF text extraction via pypdf
```

No API key is needed to try it — ArXplain runs offline with a built-in
heuristic explainer. For richer prose, configure a real LLM. The primary
supported provider is the **Meta Model API** (Muse Spark):

```bash
cp .env.example .env   # then put your key in .env — it is gitignored
# .env contains: ARXPLAIN_LLM=meta, MODEL_API_KEY=..., ARXPLAIN_MODEL=muse-spark-1.3
arxplain explain 1706.03762 --profile me.json --llm meta
```

Other providers work the same way (`openai` / `anthropic` / `ollama`).
**Never commit a key, and never paste one into a profile or paper file.**
If a key has been shared anywhere public, rotate it.

### How ArXplain avoids blowing up your credits

| Rule | Effect |
|---|---|
| Default `--llm-mode economy` | **1 paid call per paper**, however many concepts: all gap concepts are explained in one batched JSON request. `--llm-mode quality` adds one summary call (2 total); `--llm-mode off` spends 0 |
| Concepts you already know are never sent | An expert run can cost **0 calls** |
| Hard budget cap | `--max-llm-calls 2` (default 2) refuses further paid calls; free heuristic text is used instead. Cache hits never count |
| On-disk response cache | Re-running the same paper + profile costs **0 new calls** (verified: 1 paid call, then `cache_hits=1, paid_calls=0`). `--no-cache` bypasses it |
| Low reasoning effort (Meta) | Muse Spark bills hidden reasoning tokens; ArXplain defaults to `--reasoning-effort low` (`minimal` is cheapest) |
| Only gap concepts, largest first | `--max-llm-concepts 8` caps how many concepts the LLM writes; evidence is truncated per concept |
| Usage printed every run | e.g. `LLM: provider=meta mode=economy paid_calls=1 cache_hits=0 est_tokens=1227` — real token counts from the API, also stamped into the output file |

## Recommended flow: profile the paper first

```bash
# 1. Profiling: the paper is analyzed (LLM + web search) into topics,
#    split into Hard Requirement vs Good To Know (Optional). For every
#    topic the prompt shows the passage of THIS paper where it appears
#    and a self-check question drawn from that passage (e.g. "Do you
#    understand why the attention scores are divided by sqrt(d_k)?"),
#    so you can judge yourself accurately before rating 1-10 (Hard
#    first, Good-To-Know capped at 8). The result is saved as a
#    per-paper profile.json, with learning links attached.
arxplain profile-paper 1706.03762 --llm meta --save profile.json

#    Non-interactive equivalent (scripting / testing):
arxplain profile-paper --demo --ratings "attention=2,PyTorch=8" --save profile.json

# 2. Explain: depth follows your 1-10 ratings topic by topic. Where your
#    confidence is low (<=4/10, or a real gap) the explanation carries
#    inline article/blog links, plus a Further Reading section. Links
#    saved in profile.json are reused — no new searches, no new cost.
arxplain explain 1706.03762 --profile profile.json --llm meta
arxplain explain paper.pdf   --profile profile.json -o paper.html
```

Search is Tavily (`TAVILY_API_KEY`) when a key is present, otherwise
DDGS (DuckDuckGo, free, no key — `uv sync --extra search`). Profiling search
is capped (`--max-searches 16`) and cached, like LLM calls.

## Quick start (without profiling)

```bash
arxplain explain --demo --level beginner
arxplain explain 1706.03762 --diagnose   # quiz sets coarse levels
arxplain explain 1706.03762 --profile examples/me.json   # hand-written profile file
```

Per-dimension flags work without a profile file too:

```bash
arxplain explain paper.pdf --mathematics 1 --theory advanced --limitations 0
# levels accept 0-4 or novice|beginner|intermediate|advanced|expert
```

## What the output contains

| Section | What it does |
|---|---|
| In one paragraph | The paper, summarised for you |
| Your profile | The levels and 1–10 topic ratings the explanation was adapted to |
| **The paper, explained section by section** | **The main body: EVERY section of the paper (Abstract, Background, Experiments, Conclusion, References, …), in the paper's own order. Each section says what the authors say, in plain language at your level, followed by that section's Hard Requirements with your rating and resource links where you are weak** |
| Your learning path | Concepts ordered biggest-gap-first, with time estimate |
| Background topics, explained at your level | The ideas behind the paper, at foundation / guided / brief depth, with analogies, paper evidence, self-check questions and links |
| Gaps at a glance | Required level vs. your level for every concept |

## The agents

| Agent | Responsibility |
|---|---|
| Ingestion | Full paper from arXiv (HTML, then PDF; abstract-only is a flagged fallback, never silent), PDF / URL / text → the paper's real sections |
| Extractor | Finds core ideas, maths, datasets & metrics, results, assumptions, limitations and terminology; framework/language/tool mentions are demoted to one experimental-setup note, never a topic |
| Gap analysis | Required level − your level, per concept |
| Planner | Orders a learning path (largest gap first), estimates time |
| Explainer | Writes each background-topic explanation at the depth your gap implies; uses the configured LLM when present, heuristics offline |
| Paper explainer | **Explains the paper itself, section by section**, adapted to your ratings, with Hard-Requirement links inline where you are weak (LLM rewrite in `--llm-mode quality`, extractive and free otherwise) |
| Profiler | LLM + web search classify paper topics as Hard Requirement vs Good To Know; your 1–10 ratings become `profile.json` |
| Quiz | Diagnostic quiz used by `--diagnose` to set coarse levels |

The agent trace (which agent did what, LLM usage) is backend metadata: it
is printed in the terminal and printed in the terminal, but it is **not** rendered into the
`.html` file the reader sees.

ArXplain is a CLI tool only — there is no library/Python API.
Everything is done through the `arxplain` command shown above.

## Development

```bash
uv run pytest     # test suite: extraction, adaptation, rendering, quiz, CLI
```
