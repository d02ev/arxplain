"""Paper ingestion: arXiv id/URL, PDF file/bytes, URL, or raw text.

For arXiv, the FULL paper is fetched (HTML version first, then the PDF);
the abstract alone is only a last-resort fallback and is flagged with
Paper.complete=False so callers can warn instead of silently explaining
one section as if it were the whole paper.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from .models import Figure, Paper, Reference, Section


def resolve_reference_url(text: str, title: str = "",
                          existing: str = "") -> tuple[str, str]:
    """Best real link for a citation: publisher/DOI > arXiv > Scholar."""
    if existing and existing.startswith("http"):
        kind = "doi" if "doi.org" in existing else "publisher"
        return existing, kind
    m = re.search(r"10\.\d{4,9}/[^\s,;\]]+", text)
    if m:
        return "https://doi.org/" + m.group(0).rstrip("."), "doi"
    m = re.search(r"(?:arXiv:|abs/)(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.I)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}", "arxiv"
    m = re.search(r"(?:arXiv:|abs/)([a-z\-]+/\d{7})", text, re.I)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}", "arxiv"
    if title:
        q = urllib.parse.quote_plus(title.strip().rstrip("."))
        return f"https://scholar.google.com/scholar?q={q}", "scholar"
    return "", ""

_HEADING = re.compile(
    r"^(#{1,4}\s+.+|\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,80}|"
    r"(?:Abstract|Introduction|Related Work|Background|Method(?:ology)?|"
    r"Approach|Experiments?|Evaluation|Results|Discussion|Conclusion|"
    r"Conclusions|Limitations|Future Work|Acknowledg(?:e)?ments|Notes|"
    r"Broader Impact|Ethics|Data Availability|Code Availability|"
    r"References|Appendix|Supplementary(?: Material)?)\s*)$", re.M)


def split_sections(text: str) -> list[Section]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [Section(title="Paper", text=text.strip(), index=0)] if text.strip() else []
    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = re.sub(r"^#+\s*", "", m.group(0)).strip()
        sections.append(Section(title=title, text=text[start:end].strip(),
                                level=1, index=i))
    # Text before the first heading (title page, licence boilerplate)
    # is not a paper section and is never emitted as "Preamble".
    return sections


def _guess_title(text: str, fallback: str = "Untitled paper") -> str:
    for line in text.splitlines()[:12]:
        line = line.strip().lstrip("#").strip()
        if 8 < len(line) < 160 and not line.lower().startswith(("arxiv:", "abstract")):
            return line
    return fallback


def _guess_abstract(text: str) -> str:
    m = re.search(r"abstract[:\s]*\n?(.{60,1800}?)(?:\n\s*\n|introduction)", text,
                  re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else text[:600].strip()


def parse_references_from_text(section_text: str) -> list[Reference]:
    """Bibliography entries from plain text (PDF/text papers)."""
    chunks = re.split(r"\n(?=\[\d+\])", section_text)
    if len(chunks) < 2:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", section_text)
                  if len(c.strip()) > 40]
    refs: list[Reference] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk).strip()
        if len(text) < 25:
            continue
        label = ""
        m = re.match(r"(\[\d+\])\s*", text)
        if m:
            label, text_body = m.group(1), text[m.end():]
        else:
            text_body = text
        sentences = [s.strip() for s in re.split(r"(?<=[.])\s+", text_body)
                     if s.strip()]
        title = sentences[1] if len(sentences) > 1 else ""
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        url, kind = resolve_reference_url(text, title)
        refs.append(Reference(text=text_body, title=title,
                              authors=sentences[0] if sentences else "",
                              year=years[-1] if years else "",
                              label=label, url=url, url_kind=kind))
    return refs


def paper_from_text(text: str, *, title: str = "", source: str = "text",
                    source_id: str = "") -> Paper:
    text = text.replace("\r\n", "\n")
    paper = Paper(title=title or _guess_title(text), text=text,
                  abstract=_guess_abstract(text), source=source,
                  source_id=source_id, sections=split_sections(text))
    for s in paper.sections:
        if s.title.strip().lower() == "references":
            paper.references = parse_references_from_text(s.text)
    return paper


def extract_pdf_text(data: bytes) -> str:
    """Extract text with pypdf/PyPDF2 if installed; else a plain decode."""
    for mod in ("pypdf", "PyPDF2"):
        try:
            import io
            module = __import__(mod)
            reader = module.PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:  # noqa: BLE001
            continue
    return data.decode("utf-8", "ignore")


def paper_from_pdf_bytes(data: bytes, *, source_id: str = "") -> Paper:
    return paper_from_text(extract_pdf_text(data), source="pdf", source_id=source_id)


def normalize_arxiv_id(raw: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
    if m:
        return m.group(1)
    m = re.search(r"([a-z\-]+/\d{7})", raw, re.I)
    return m.group(1) if m else raw.strip()


class _ArxivHTMLParser(HTMLParser):
    """Collect (heading, paragraphs, figures) in document order.

    Figures: <figure> graphics (<img src> or <object data>) are captured
    with their <figcaption>, resolved against the page URL, and tagged
    with the section they appear in. Captions are kept on the Figure
    rather than duplicated into the section text. Table-figures with
    no graphic are skipped.
    """

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.sections: list[tuple[str, list[str]]] = []
        self.figures: list[Figure] = []
        self._heading: str | None = None
        self._buf: list[str] = []
        self._mode: str | None = None  # heading | para
        self._skip = 0
        self._fig: dict | None = None
        self._fig_depth = 0
        self._cap_buf: list[str] | None = None
        self.references: list[Reference] = []
        self._bib: dict | None = None
        self._bib_buf: list[str] | None = None
        self._bib_kind = ""  # block | label
        self._bib_span_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip += 1
        if self._skip:
            return
        cls = a.get("class") or ""
        if tag == "li" and "ltx_bibitem" in cls:
            self._bib = {"blocks": [], "url": "", "label": ""}
            return
        if self._bib is not None:
            if tag == "span" and self._bib_buf is not None:
                self._bib_span_depth += 1  # nested span inside a bib block
            elif tag == "span" and "ltx_bibblock" in cls:
                self._bib_buf, self._bib_kind = [], "block"
            elif tag == "span" and "ltx_tag_bibitem" in cls:
                self._bib_buf, self._bib_kind = [], "label"
            elif tag == "a":
                href = a.get("href") or ""
                if href.startswith("http") and not self._bib["url"]:
                    self._bib["url"] = href
            return
        if "ltx_abstract" in (a.get("class") or ""):
            # arXiv puts the abstract under the article title with no
            # usable heading; it is the Abstract section, by name.
            self._flush_para()
            if not self.sections or self.sections[-1][0] != "Abstract":
                self._heading = "Abstract"
                self.sections.append(("Abstract", []))
            return
        if tag == "figure":
            if self._fig is None:
                self._fig = {"src": "", "alt": "", "caption": ""}
            self._fig_depth += 1
            return
        if self._fig is not None:
            if tag == "img" and not self._fig["src"]:
                src = a.get("src") or (a.get("srcset") or "").split(",")[0].split(" ")[0]
                if src:
                    self._fig["src"] = urllib.parse.urljoin(self.base_url, src)
                    self._fig["alt"] = a.get("alt", "")
            elif tag == "object" and a.get("data") and not self._fig["src"]:
                if "image" in (a.get("type") or "image"):
                    self._fig["src"] = urllib.parse.urljoin(self.base_url,
                                                            a["data"])
            elif tag == "figcaption":
                self._cap_buf = []
            return
        if tag in ("h1", "h2", "h3", "h4"):
            self._flush_para()
            self._mode, self._buf = "heading", []
        elif tag in ("p", "li", "figcaption"):
            self._mode, self._buf = "para", []
        elif tag == "math" and a.get("alttext"):
            self._buf.append(f" ${a['alttext']}$ ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer") and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag == "span" and self._bib is not None and self._bib_buf is not None:
            if self._bib_span_depth > 0:
                self._bib_span_depth -= 1
                return
            text = re.sub(r"\s+", " ", "".join(self._bib_buf)).strip()
            if self._bib_kind == "block" and text:
                self._bib["blocks"].append(text)
            elif self._bib_kind == "label" and text:
                self._bib["label"] = text
            self._bib_buf, self._bib_kind = None, ""
            return
        if tag == "li" and self._bib is not None:
            bib, self._bib = self._bib, None
            blocks = bib["blocks"]
            full = " ".join(blocks).strip()
            if full:
                title = blocks[1] if len(blocks) > 1 else ""
                years = re.findall(r"\b(?:19|20)\d{2}\b", full)
                url, kind = resolve_reference_url(full, title, bib["url"])
                self.references.append(Reference(
                    text=full, title=title,
                    authors=blocks[0] if blocks else "",
                    year=years[-1] if years else "",
                    label=bib["label"], url=url, url_kind=kind))
            return
        if self._bib is not None:
            return
        if tag == "figcaption" and self._cap_buf is not None:
            if self._fig is not None:
                self._fig["caption"] = re.sub(
                    r"\s+", " ", "".join(self._cap_buf)).strip()
            self._cap_buf = None
            return
        if tag == "figure" and self._fig is not None:
            self._fig_depth -= 1
            if self._fig_depth == 0:
                fig, self._fig = self._fig, None
                if fig["src"]:
                    m = re.search(r"(Figure|Table)\s*\d+", fig["caption"])
                    self.figures.append(Figure(
                        src=fig["src"], caption=fig["caption"],
                        alt=fig["alt"], section_title=self._heading or "",
                        label=m.group(0) if m else "Figure"))
            return
        if self._fig is not None:
            return
        if tag in ("h1", "h2", "h3", "h4") and self._mode == "heading":
            title = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
            if title and title.lower() not in ("references",):
                self._heading = title
                self.sections.append((title, []))
            elif title:
                self._heading = title
                self.sections.append((title, []))
            self._mode, self._buf = None, []
        elif tag in ("p", "li", "figcaption") and self._mode == "para":
            self._flush_para()

    def handle_data(self, data):
        if self._bib_buf is not None:
            self._bib_buf.append(data)
        elif self._cap_buf is not None:
            self._cap_buf.append(data)
        elif not self._skip and self._mode and self._bib is None:
            self._buf.append(data)

    def _flush_para(self):
        if self._mode != "para":
            return
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._mode, self._buf = None, []
        if len(text) < 15:
            return
        if self._heading is None:
            self._heading = "Abstract" if "abstract" in text.lower()[:40] else "Preamble"
            self.sections.append((self._heading, []))
        self.sections[-1][1].append(text)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def normalize_sections(sections: list[Section],
                       abstract: str = "") -> list[Section]:
    """Drop non-sections and name the abstract correctly.

    - 'Preamble' (title-page / licence text before the first heading)
      is not a paper section and is removed.
    - The abstract often sits under the article-title heading; if its
      text matches the known abstract and no section is already called
      Abstract, that section is renamed to 'Abstract'.
    """
    sections = [s for s in sections if s.title.lower() != "preamble"]
    if abstract and not any(s.title.lower() == "abstract" for s in sections):
        needle = _norm_ws(abstract)[:90]
        if needle:
            for s in sections:
                if needle in _norm_ws(s.text):
                    s.title = "Abstract"
                    break
    for i, s in enumerate(sections):
        s.index = i
    return sections


def parse_html_document(html_text: str, base_url: str = "",
                        abstract: str = "") -> tuple[list[Section], list[Figure]]:
    """Parse structured HTML into (sections, figures), in document order."""
    parser = _ArxivHTMLParser(base_url)
    try:
        parser.feed(html_text)
    except Exception:  # noqa: BLE001
        return [], []
    out: list[Section] = []
    for i, (title, paras) in enumerate(parser.sections):
        body = "\n\n".join(paras)
        if not body.strip():
            # A section made only of figures (e.g. Attention
            # Visualizations) still exists: its captions are its text.
            caps = [f.caption for f in parser.figures
                    if f.section_title == title and f.caption]
            body = " ".join(caps)
        if not body.strip() and title.strip().lower() == "references":
            body = "\n".join(r.text for r in parser.references)
        if body.strip():
            out.append(Section(title=title, text=body, index=i))
    return normalize_sections(out, abstract), parser.figures


def parse_html_parts(html_text: str, base_url: str = "",
                     abstract: str = ""):
    """(sections, figures, references) from a single parse pass."""
    parser = _ArxivHTMLParser(base_url)
    try:
        parser.feed(html_text)
    except Exception:  # noqa: BLE001
        return [], [], []
    out: list[Section] = []
    for title, paras in parser.sections:
        body = "\n\n".join(paras)
        if not body.strip():
            caps = [f.caption for f in parser.figures
                    if f.section_title == title and f.caption]
            body = " ".join(caps)
        if not body.strip() and title.strip().lower() == "references":
            body = "\n".join(r.text for r in parser.references)
        if body.strip():
            out.append(Section(title=title, text=body))
    return (normalize_sections(out, abstract), parser.figures,
            parser.references)


def parse_arxiv_html(html_text: str, base_url: str = "") -> list[Section]:
    return parse_html_document(html_text, base_url)[0]


def _paper_cache_path(key: str) -> Path:
    import hashlib as _h
    base = os.getenv("ARXPLAIN_CACHE_DIR") or str(
        Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "arxplain")
    return Path(base) / "papers" / f"{_h.sha256(key.encode()).hexdigest()}.json"


def load_cached_paper(key: str) -> Paper | None:
    """A previously parsed PaperDocument for this source, if any."""
    if os.getenv("ARXPLAIN_PAPER_CACHE", "1") == "0":
        return None
    path = _paper_cache_path(key)
    if not path.exists():
        return None
    try:
        return Paper.from_document(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001 - a bad cache entry is just a miss
        return None


def save_cached_paper(key: str, paper: Paper) -> None:
    """Cache a fully parsed paper. Abstract-only results are NOT cached:
    a transient failure must not become permanent."""
    if os.getenv("ARXPLAIN_PAPER_CACHE", "1") == "0" or not paper.complete:
        return
    try:
        path = _paper_cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(paper.to_document()))
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass


def _get(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "arxplain/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def fetch_arxiv(arxiv_id: str, *, timeout: int = 30,
                use_cache: bool = True) -> Paper:
    """Fetch the FULL paper from arXiv: metadata, then HTML, then PDF.

    A previously parsed PaperDocument is served from the on-disk cache
    (profile-paper then explain on the same paper parses it once).
    Only if both full-text routes fail is the abstract returned alone,
    flagged complete=False (and never cached).
    """
    aid = normalize_arxiv_id(arxiv_id)
    cache_key = f"arxiv:{aid}"
    if use_cache:
        cached = load_cached_paper(cache_key)
        if cached is not None:
            cached.source = "arxiv (cached)"
            return cached
    xml = _get(f"https://export.arxiv.org/api/query?id_list={aid}", timeout).decode()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    if entry is None:
        raise ValueError(f"arXiv paper not found: {aid}")

    def _t(tag: str) -> str:
        el = entry.find(f"a:{tag}", ns)
        return re.sub(r"\s+", " ", el.text).strip() if el is not None and el.text else ""

    title, abstract = _t("title"), _t("summary")
    authors = [re.sub(r"\s+", " ", a.find("a:name", ns).text).strip()
               for a in entry.findall("a:author", ns)
               if a.find("a:name", ns) is not None]
    # 1) HTML full text (keeps the paper's real section structure).
    for html_url in (f"https://arxiv.org/html/{aid}",):
        try:
            sections, figures, references = parse_html_parts(
                _get(html_url, timeout).decode("utf-8", "ignore"),
                base_url=html_url, abstract=abstract)
        except Exception:  # noqa: BLE001
            sections, figures, references = [], [], []
        if len(sections) >= 3:
            text = "\n\n".join(f"{s.title}\n{s.text}" for s in sections)
            paper = Paper(title=title or aid, text=text, authors=authors,
                          abstract=abstract, source="arxiv", source_id=aid,
                          sections=sections, figures=figures,
                          references=references, complete=True)
            if use_cache:
                save_cached_paper(cache_key, paper)
            return paper
    # 2) PDF full text.
    try:
        paper = paper_from_pdf_bytes(_get(f"https://arxiv.org/pdf/{aid}", timeout),
                                     source_id=aid)
        if len(paper.sections) >= 3:
            paper.title, paper.authors = title or paper.title, authors
            paper.abstract, paper.source = abstract, "arxiv"
            if use_cache:
                save_cached_paper(cache_key, paper)
            return paper
    except Exception:  # noqa: BLE001
        pass
    # 3) Abstract only — flagged, never silent.
    text = f"{title}\n\nAbstract\n{abstract}\n"
    return Paper(title=title or aid, text=text, authors=authors,
                 abstract=abstract, source="arxiv", source_id=aid,
                 sections=split_sections(text), complete=False)


def fetch_url_text(url: str, *, timeout: int = 30,
                   use_cache: bool = True) -> Paper:
    cache_key = f"url:{url}"
    if use_cache:
        cached = load_cached_paper(cache_key)
        if cached is not None:
            return cached
    req = urllib.request.Request(url, headers={"User-Agent": "arxplain/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        data = r.read()
    if url.lower().endswith(".pdf") or data[:4] == b"%PDF":
        return paper_from_pdf_bytes(data, source_id=url)
    html = data.decode("utf-8", "ignore")
    sections, figures, references = parse_html_parts(html, base_url=url)
    if len(sections) >= 2:  # structured HTML (arXiv HTML and similar)
        text = "\n\n".join(f"{s.title}\n{s.text}" for s in sections)
        paper = paper_from_text(text, source="url", source_id=url)
        paper.sections, paper.figures = sections, figures
        paper.references = references
        if use_cache:
            save_cached_paper(cache_key, paper)
        return paper
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return paper_from_text(text, source="url", source_id=url)
