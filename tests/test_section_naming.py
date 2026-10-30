from arxplain.ingestion import normalize_sections, parse_html_document
from arxplain.llm import MockLLM
from arxplain.models import Paper, Section, UserProfile
from arxplain.pipeline import explain_paper

HTML = """<html><body>
<p>Provided proper attribution is provided, Google hereby grants permission.</p>
<h1>Attention Is All You Need</h1>
<div class="ltx_abstract"><h6>Abstract</h6>
<p>The dominant sequence transduction models are based on recurrent networks.</p></div>
<h2>1 Introduction</h2><p>Recurrent networks have dominated sequence work.</p>
<h2>2 Background</h2><p>Convolutional models also compute in parallel here.</p>
</body></html>"""

ABSTRACT = "The dominant sequence transduction models are based on recurrent networks."


def test_preamble_dropped_and_abstract_named():
    sections, _ = parse_html_document(HTML, abstract=ABSTRACT)
    titles = [s.title for s in sections]
    assert "Preamble" not in titles
    assert titles[0] == "Abstract"
    assert all("grants permission" not in s.text for s in sections)


def test_title_heading_holding_abstract_is_renamed():
    secs = [Section(title="Attention Is All You Need", text=ABSTRACT + " More text here."),
            Section(title="Introduction", text="Recurrent networks have dominated.")]
    out = normalize_sections(secs, ABSTRACT)
    assert out[0].title == "Abstract"


def test_explainer_never_emits_preamble():
    paper = Paper(title="T", text="x", sections=[
        Section(title="Preamble", text="Licence boilerplate text that is not a section."),
        Section(title="Abstract", text="We propose a model in this abstract text here."),
        Section(title="Conclusion", text="We conclude the model works well here today.")])
    result = explain_paper(paper, UserProfile(), llm=MockLLM())
    assert [s.section_title for s in result.paper_explanation] == [
        "Abstract", "Conclusion"]
