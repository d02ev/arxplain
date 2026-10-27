from arxplain.ingestion import paper_from_text, parse_arxiv_html
from arxplain.llm import MockLLM
from arxplain.models import Dimension, Resource, Topic, UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html, render_markdown

FULL = """Attention Study

Abstract
We study attention mechanisms in depth and propose a model.

1. Background
Earlier work used recurrent networks and scaled dot-product attention
with softmax, which readers must understand before continuing here.

2. Method
The Transformer uses scaled dot-product attention and embeddings.

3. Experiments
We run an ablation study on WMT with BLEU accuracy against baselines.

4. Conclusion
Attention alone works well and future work will scale it further.

Notes
Short note.

References
Vaswani et al. 2017. Attention Is All You Need. NeurIPS.
"""

HTML = """<html><body>
<h1>Attention Is All You Need</h1>
<h2>Abstract</h2><p>We propose the Transformer based on attention.</p>
<h2>1 Introduction</h2><p>Recurrent networks have dominated sequence tasks.</p>
<h2>2 Model Architecture</h2><p>The Transformer uses attention and softmax.</p>
<h2>3 Experiments</h2><p>We evaluate on WMT 2014 translation with BLEU.</p>
<h2>4 Conclusion</h2><p>Attention alone is sufficient for transduction.</p>
</body></html>"""


def test_arxiv_html_parser_keeps_every_section():
    sections = parse_arxiv_html(HTML)
    titles = [s.title for s in sections]
    assert "Abstract" in titles and "Conclusion" in titles
    assert len(sections) >= 5
    assert all(s.text for s in sections)


def test_every_paper_section_is_explained_including_short_and_references():
    result = explain_paper(paper_from_text(FULL, source="text"),
                           UserProfile(), llm=MockLLM())
    paper_titles = [s.title for s in result.paper.sections]
    explained = [s.section_title for s in result.paper_explanation]
    assert explained == paper_titles  # each and every section, in order
    assert "References" in explained and "Notes" in explained


def test_hard_requirements_live_inside_their_section_with_links():
    profile = UserProfile(name="Ada")
    profile.topics = [Topic(
        name="Scaled dot-product attention & softmax",
        dimension=Dimension.MATHEMATICS, requirement="hard", familiarity=1,
        resources=[Resource(title="Attention from scratch",
                            url="https://example.com/attn", source="ddgs")])]
    profile.sync_levels_from_topics()
    result = explain_paper(paper_from_text(FULL, source="text"),
                           profile, llm=MockLLM())
    background = next(s for s in result.paper_explanation
                      if s.section_title == "1. Background")
    assert any(t.name.startswith("Scaled dot-product")
               for t in background.hard_requirements)
    md = render_markdown(result)
    seg = md.split("### 1. Background")[1].split("###")[0]
    assert "Hard requirements to understand this section" in seg
    # the topic in the section points to Resources; the external link
    # itself lives once, in the Resources list/sidebar
    assert "(#resources)" in seg and "example.com/attn" not in seg
    assert "https://example.com/attn" in md.split("## Resources")[1]
    html = render_html(result)
    assert "Hard requirements to understand this section" in html
    assert "href=\"#res-" in html.split("1. Background")[1].split("</div>")[0] \
        or "href=\"#res-" in html
