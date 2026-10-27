"""Render an ExplainResult to Markdown or a standalone HTML page."""
from __future__ import annotations

import html as _html
import re

from .models import Dimension, ExplainResult, Level

_DEPTH_BADGE = {"foundation": "From zero", "guided": "Guided",
                "concise": "Brief", "skip-known": "You know this"}


def _esc(s: str) -> str:
    return _html.escape(s or "")


def _bold(s: str) -> str:
    """Escape, then render **bold** spans (from heuristic bodies) as <strong>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _esc(s))


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "topic"


class _ResourceIndex:
    """Every external link lives ONCE, in the Resources sidebar/panel.

    Built in reading order (paper sections first, then background
    topics). Each topic gets one sidebar entry (id=res-<slug>) holding
    its links; sections do not throw links inline — a topic name under
    a section is a jump-link to its sidebar entry, where the reader
    clicks through to the actual article. A URL shared by two topics
    is listed under the first topic only.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._by_topic: dict[str, dict] = {}
        self._seen_urls: set[str] = set()
        self._used_slugs: set[str] = set()

    def add(self, topic: str, resources) -> None:
        if not topic:
            return
        fresh = [r for r in (resources or [])
                 if r.url and r.url not in self._seen_urls]
        if not fresh:
            return
        self._seen_urls.update(r.url for r in fresh)
        key = topic.lower()
        if key in self._by_topic:
            self._by_topic[key]["resources"].extend(fresh)
            return
        slug = _slug(topic)
        base, n = slug, 2
        while slug in self._used_slugs:
            slug = f"{base}-{n}"
            n += 1
        self._used_slugs.add(slug)
        entry = {"name": topic, "slug": slug, "resources": list(fresh)}
        self._by_topic[key] = entry
        self.entries.append(entry)

    def entry_for(self, topic: str) -> dict | None:
        return self._by_topic.get((topic or "").lower())


def _resource_index(result: ExplainResult) -> _ResourceIndex:
    idx = _ResourceIndex()
    for s in result.paper_explanation:
        for t in s.hard_requirements:
            idx.add(t.name, t.resources)
        if s.resources:
            owner = (s.hard_requirements[0].name if s.hard_requirements
                     else (s.concepts_used[0] if s.concepts_used
                           else s.section_title))
            idx.add(owner, s.resources)
    for e in result.explanations:
        idx.add(e.concept_name, e.resources)
    return idx


def render_markdown(result: ExplainResult) -> str:
    p, prof = result.paper, result.profile
    lines: list[str] = [f"# {p.title}", ""]
    if p.authors:
        lines.append(f"*{', '.join(p.authors)}*  ")
    lines.append(f"Source: {p.source} {p.source_id}  ")
    lines.append(f"Explained for: **{prof.name}**"
                 + (f" — goal: {prof.goal}" if prof.goal else ""))
    lines += ["", "## Your profile", "",
              "| Dimension | Your level |", "|---|---|"]
    for dim in Dimension:
        lines.append(f"| {dim.label} | {Level.from_any(prof.level(dim)).label} |")
    if prof.topics:
        lines += ["", "### Topics you rated (1–10)", "",
                  "| Topic | Requirement | Your rating |", "|---|---|---|"]
        for t in prof.topics:
            req = "Hard Requirement" if t.is_hard else "Good To Know (Optional)"
            rating = str(t.familiarity) if t.familiarity is not None else "—"
            lines.append(f"| {t.name} | {req} | {rating}/10 |")
    lines += ["", "## In one paragraph", "", result.summary, ""]
    idx = _resource_index(result)
    if result.paper_explanation:
        lines += ["## The paper, explained section by section",
                  "_What the authors actually say, in plain language, slowed "
                  "down where your ratings are low and brief where they are high. "
                  "Topics under a section link to the Resources list at the end, "
                  "where every external link lives once._", ""]
        for s in result.paper_explanation:
            lines += [f"### {s.section_title}", "", s.body, ""]
            for f in s.figures:
                cap = f.caption or f.alt or f.label
                lines += [f"![{cap}]({f.src})", "", f"*{cap}*" if cap else "", ""]
            if s.hard_requirements:
                lines += ["**Hard requirements to understand this section:**", ""]
                for t in s.hard_requirements:
                    status = (f"you rated this {t.familiarity}/10"
                              if t.familiarity is not None
                              else "not rated in your profile")
                    entry = idx.entry_for(t.name)
                    name = (f"[{t.name}](#resources)" if entry
                            else t.name)
                    lines.append(f"- **{name}** — {status}"
                                 + (" · resources at the end ↓" if entry else ""))
                lines.append("")
    if result.plan:
        lines += ["", "## Your learning path",
                  f"_{result.plan.overview} (~{result.plan.estimated_minutes} min)_", ""]
        for s in result.plan.steps:
            lines.append(f"{s.order}. **{s.title}** "
                         f"[{s.dimension.label} · {_DEPTH_BADGE.get(s.target_depth, s.target_depth)}] — {s.why}")
    lines += ["", "## Background topics, explained at your level",
              "_The ideas the paper leans on, in the order worth learning them. "
              "Use these, and the links, when a section above feels shaky._", ""]
    by_dim: dict[Dimension, list] = {}
    for e in result.explanations:
        by_dim.setdefault(e.dimension, []).append(e)
    gap_by_name = {g.concept.name: g for g in result.gaps}
    for dim, exps in by_dim.items():
        lines += [f"### {dim.label}", ""]
        for e in exps:
            g = gap_by_name.get(e.concept_name)
            need = (f"needs {Level.from_any(g.concept.required_level).label}, "
                    f"you: {Level.from_any(g.user_level).label}"
                    if g else "")
            lines.append(f"#### {e.title}  ·  {_DEPTH_BADGE.get(e.depth, e.depth)}"
                         + (f"  ·  {need}" if need else ""))
            lines += ["", e.body]
            if e.analogy:
                lines += ["", f"> **Picture it:** {e.analogy}"]
            if e.example:
                lines += ["", f"_{e.example}_"]
            if e.check_question:
                lines += ["", f"**Check yourself:** {e.check_question}"]
            entry = idx.entry_for(e.concept_name)
            if entry:
                fam = (f" — you rated this {e.familiarity}/10"
                       if e.familiarity is not None else "")
                lines += ["", f"**Resources for this topic:** "
                          f"[see Resources](#resources){fam}"]
            lines.append("")
    if idx.entries:
        lines += ["## Resources", "",
                  "_Every external link in this document, grouped by topic. "
                  "Click a topic under any section above to jump here in the "
                  "HTML version._", ""]
        for entry in idx.entries:
            lines += [f"### {entry['name']}", ""]
            for r in entry["resources"]:
                lines.append(f"- [{r.title}]({r.url})")
            lines.append("")
    lines += ["## Gaps at a glance", "",
              "| Concept | Dimension | Needs | You | Gap |", "|---|---|---|---|---|"]
    for g in result.gaps:
        lines.append(f"| {g.concept.name} | {g.concept.dimension.label} | "
                     f"{Level.from_any(g.concept.required_level).label} | "
                     f"{Level.from_any(g.user_level).label} | {g.severity} |")
    _u = result.llm_usage or {}
    lines += ["", "---",
              f"_LLM: provider={result.llm_provider}, "
              f"{_u.get('calls', 0)} paid call(s), "
              f"{_u.get('cache_hits', 0)} cache hit(s), "
              f"~{_u.get('est_total_tokens', 0)} est. tokens._", ""]
    lines += ["", "---", "_Generated by ArXplain — explanations adapt to the "
              "profile above. Re-run with a different `--profile` to change depth._", ""]
    return "\n".join(lines)


_CSS = """
:root{--ink:#1a1d21;--muted:#5b6470;--line:#d9dde3;--accent:#0f766e;--paper:#fff;--soft:#f3f5f7}
*{box-sizing:border-box} body{margin:0;background:var(--soft);color:var(--ink);
 font-family:Georgia,'Times New Roman',serif;line-height:1.65}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px}
header{border-bottom:3px solid var(--ink);padding-bottom:16px;margin-bottom:8px}
h1{font-size:34px;line-height:1.15;margin:0 0 8px} h2{font-size:22px;margin:36px 0 10px;
 border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:24px 0 6px} h4{font-size:16px;margin:18px 0 4px}
.meta,.small{color:var(--muted);font-size:14px;font-family:ui-monospace,Menlo,monospace}
.card{background:var(--paper);border:1px solid var(--line);padding:18px 20px;margin:14px 0}
.badge{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:11px;
 letter-spacing:.04em;border:1px solid var(--ink);padding:1px 7px;margin-left:8px;vertical-align:2px}
.badge.foundation{background:var(--ink);color:#fff}.badge.skip-known{color:var(--muted);border-color:var(--line)}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--paper)}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left} th{background:var(--soft)}
blockquote{margin:10px 0;padding:10px 14px;background:var(--soft);border-left:3px solid var(--accent)}
a{color:var(--accent)} .check{font-weight:bold} footer{margin-top:40px;color:var(--muted);font-size:13px}
ol.path li{margin:6px 0} .dim{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted)}
.wrap{max-width:1180px} .layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:28px;align-items:start}
aside.resources{position:sticky;top:20px;background:var(--paper);border:1px solid var(--line);padding:16px 18px;max-height:calc(100vh - 40px);overflow:auto}
aside.resources h2{margin-top:0;font-size:19px} aside.resources h4{font-size:14px;margin:14px 0 4px}
aside.resources ul{margin:4px 0 10px;padding-left:18px;font-size:13px}
a.topic-link{font-weight:bold;text-decoration:underline;text-underline-offset:2px}
figure{margin:14px 0} figure img{max-width:100%;height:auto;border:1px solid var(--line)}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
.res-topic{scroll-margin-top:20px;border-top:1px solid var(--line);padding-top:8px}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside.resources{position:static;max-height:none;order:2}}
@media print{body{background:#fff}.wrap{max-width:none}aside.resources{position:static;max-height:none}}
"""


def render_html(result: ExplainResult) -> str:
    p, prof = result.paper, result.profile
    gap_by_name = {g.concept.name: g for g in result.gaps}
    profile_rows = "".join(
        f"<tr><td>{_esc(d.label)}</td><td>{_esc(Level.from_any(prof.level(d)).label)}</td></tr>"
        for d in Dimension)
    idx = _resource_index(result)
    sidebar = ""
    if idx.entries:
        blocks = ""
        for entry in idx.entries:
            lis = "".join(
                f"<li><a href=\"{_esc(r.url)}\" target=\"_blank\" "
                f"rel=\"noopener\">{_esc(r.title or r.url)}</a>"
                + (f" <span class='small'>({_esc(r.source)})</span>"
                   if r.source else "") + "</li>"
                for r in entry["resources"])
            blocks += (f"<div class='res-topic' id='res-{_esc(entry['slug'])}'>"
                       f"<h4>{_esc(entry['name'])}</h4><ul>{lis}</ul></div>")
        sidebar = ("<aside class='resources' id='resources'>"
                   "<h2>Resources</h2><p class='small'>Every link, grouped "
                   "by topic. Click a topic under any section to jump here, "
                   "then click through to learn it.</p>"
                   f"{blocks}</aside>")
    paper_html = ""
    if result.paper_explanation:
        parts = []
        for s in result.paper_explanation:
            body = "".join(f"<p>{_bold(p)}</p>"
                           for p in s.body.split("\n\n") if p.strip())
            figs_html = ""
            for f in s.figures:
                cap = f.caption or f.alt or f.label
                figs_html += (
                    f"<figure><img src=\"{_esc(f.src)}\" "
                    f"alt=\"{_esc(f.alt or cap)}\" loading=\"lazy\">"
                    + (f"<figcaption>{_esc(cap)}</figcaption>" if cap else "")
                    + "</figure>")
            req_html = ""
            if s.hard_requirements:
                items = ""
                for t in s.hard_requirements:
                    status = (f"you rated this {_esc(str(t.familiarity))}/10"
                              if t.familiarity is not None
                              else "not rated in your profile")
                    entry = idx.entry_for(t.name)
                    if entry:  # click the topic -> its Resources sidebar entry
                        name = (f"<a class='topic-link' "
                                f"href=\"#res-{_esc(entry['slug'])}\">"
                                f"{_esc(t.name)}</a>")
                    else:
                        name = f"<strong>{_esc(t.name)}</strong>"
                    items += f"<li>{name} — {status}</li>"
                req_html = ("<p><strong>Hard requirements to understand "
                            f"this section:</strong></p><ul>{items}</ul>")
            parts.append(f"<div class='card'><h3>{_esc(s.section_title)}</h3>"
                         f"{body}{figs_html}{req_html}</div>")
        paper_html = ("<h2>The paper, explained section by section</h2>"
                      "<p><em>What the authors actually say, in plain language — "
                      "slowed down where your ratings are low, brief where they "
                      "are high.</em></p>" + "".join(parts))
    path = ""
    if result.plan:
        items = "".join(
            f"<li><strong>{_esc(s.title)}</strong> <span class='dim'>[{_esc(s.dimension.label)}]</span>"
            f"<span class='badge {_esc(s.target_depth)}'>{_esc(_DEPTH_BADGE.get(s.target_depth, s.target_depth))}</span>"
            f"<br><span class='small'>{_esc(s.why)}</span></li>"
            for s in result.plan.steps)
        path = (f"<h2>Your learning path</h2><p><em>{_esc(result.plan.overview)} "
                f"(~{result.plan.estimated_minutes} min)</em></p>"
                f"<ol class='path'>{items}</ol>")
    by_dim: dict[Dimension, list] = {}
    for e in result.explanations:
        by_dim.setdefault(e.dimension, []).append(e)
    concept_html = ""
    for dim, exps in by_dim.items():
        concept_html += f"<h3>{_esc(dim.label)}</h3>"
        for e in exps:
            g = gap_by_name.get(e.concept_name)
            need = (f"needs {_esc(Level.from_any(g.concept.required_level).label)}, "
                    f"you: {_esc(Level.from_any(g.user_level).label)}" if g else "")
            concept_html += f"<div class='card'><h4>{_esc(e.title)}"
            concept_html += (f"<span class='badge {_esc(e.depth)}'>"
                             f"{_esc(_DEPTH_BADGE.get(e.depth, e.depth))}</span></h4>")
            if need:
                concept_html += f"<p class='small'>{need}</p>"
            concept_html += f"<p>{_esc(e.body)}</p>"
            if e.analogy:
                concept_html += f"<blockquote><strong>Picture it:</strong> {_esc(e.analogy)}</blockquote>"
            if e.example:
                concept_html += f"<p><em>{_esc(e.example)}</em></p>"
            if e.check_question:
                concept_html += f"<p class='check'>Check yourself: {_esc(e.check_question)}</p>"
            entry = idx.entry_for(e.concept_name)
            if entry:
                fam = (f" — you rated this {_esc(str(e.familiarity))}/10"
                       if e.familiarity is not None else "")
                concept_html += (
                    f"<p><strong>Resources:</strong> <a class='topic-link' "
                    f"href=\"#res-{_esc(entry['slug'])}\">"
                    f"{_esc(e.concept_name)} in the sidebar</a>{fam}</p>")
            concept_html += "</div>"
    further = ""
    topics_html = ""
    if prof.topics:
        rows = "".join(
            f"<tr><td>{_esc(t.name)}</td>"
            f"<td>{'Hard Requirement' if t.is_hard else 'Good To Know (Optional)'}</td>"
            f"<td>{_esc(str(t.familiarity) + '/10' if t.familiarity is not None else '—')}</td></tr>"
            for t in prof.topics)
        topics_html = ("<h3>Topics you rated (1–10)</h3><table>"
                       "<tr><th>Topic</th><th>Requirement</th><th>Your rating</th></tr>"
                       f"{rows}</table>")
    gap_rows = "".join(
        f"<tr><td>{_esc(g.concept.name)}</td><td>{_esc(g.concept.dimension.label)}</td>"
        f"<td>{_esc(Level.from_any(g.concept.required_level).label)}</td>"
        f"<td>{_esc(Level.from_any(g.user_level).label)}</td><td>{_esc(g.severity)}</td></tr>"
        for g in result.gaps)
    authors = f"<p class='meta'>{_esc(', '.join(p.authors))}</p>" if p.authors else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(p.title)} — explained by ArXplain</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header><p class="meta">ARXPLAIN · EXPLAINED FOR {_esc(prof.name.upper())}</p>
<h1>{_esc(p.title)}</h1>{authors}
<p class="meta">Source: {_esc(p.source)} {_esc(p.source_id)}</p></header>
<div class="layout"><main>
<h2>In one paragraph</h2><p>{_esc(result.summary)}</p>
<h2>Your profile</h2><table><tr><th>Dimension</th><th>Your level</th></tr>{profile_rows}</table>{topics_html}
{paper_html}
{path}
<h2>Background topics, explained at your level</h2>{concept_html}
{further}
<h2>Gaps at a glance</h2>
<table><tr><th>Concept</th><th>Dimension</th><th>Needs</th><th>You</th><th>Gap</th></tr>{gap_rows}</table>
<footer>Generated by ArXplain. LLM: {_esc(result.llm_provider)}, {_esc(str((result.llm_usage or {}).get('calls', 0)))} paid call(s), {_esc(str((result.llm_usage or {}).get('cache_hits', 0)))} cache hit(s). Re-run with a different profile to change the depth.</footer>
</main>{sidebar}</div>
</div></body></html>"""
