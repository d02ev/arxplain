from arxplain.ingestion import (parse_html_parts,
                                resolve_reference_url)
from arxplain.llm import MockLLM
from arxplain.models import UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html

BIB_HTML = """<html><body>
<h2>Abstract</h2><p>We propose a model based on attention mechanisms.</p>
<h2>References</h2>
<ul class="ltx_biblist">
<li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[1]</span>
<span class="ltx_bibblock">Jimmy Lei Ba and Geoffrey Hinton.</span>
<span class="ltx_bibblock">Layer normalization.</span>
<span class="ltx_bibblock">arXiv preprint arXiv:1607.06450, 2016.</span></li>
<li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[2]</span>
<span class="ltx_bibblock">Jane Doe and John Roe.</span>
<span class="ltx_bibblock">A study of things.</span>
<span class="ltx_bibblock">Journal of Examples, doi:10.1000/xyz123, 2020.</span></li>
<li class="ltx_bibitem"><span class="ltx_tag ltx_tag_bibitem">[3]</span>
<span class="ltx_bibblock">Alan Smith.</span>
<span class="ltx_bibblock">Obscure workshop findings.</span>
<span class="ltx_bibblock">Workshop on Examples, 2019.</span></li>
</ul></body></html>"""


def test_bibliography_parsed_with_real_links():
    _, _, refs = parse_html_parts(BIB_HTML)
    assert len(refs) == 3
    by_label = {r.label: r for r in refs}
    assert by_label["[1]"].url == "https://arxiv.org/abs/1607.06450"
    assert by_label["[1]"].title == "Layer normalization."
    assert by_label["[2]"].url == "https://doi.org/10.1000/xyz123"
    assert by_label["[3]"].url_kind == "scholar"
    assert "Obscure+workshop+findings" in by_label["[3]"].url


def test_url_resolution_priority():
    assert resolve_reference_url("x doi:10.1109/ABC.2020.1 y")[1] == "doi"
    assert resolve_reference_url("arXiv preprint arXiv:1706.03762, 2017") == (
        "https://arxiv.org/abs/1706.03762", "arxiv")


def test_references_section_links_each_cited_paper():
    from arxplain.ingestion import parse_html_document
    sections, figures, refs = parse_html_parts(BIB_HTML)
    from arxplain.models import Paper
    paper = Paper(title="T", text="x", abstract="We propose a model.",
                  sections=sections, figures=figures, references=refs)
    result = explain_paper(paper, UserProfile(), llm=MockLLM())
    ref_sec = next(s for s in result.paper_explanation
                   if s.section_title == "References")
    assert len(ref_sec.references) == 3
    html = render_html(result)
    assert "https://arxiv.org/abs/1607.06450" in html
    assert "https://doi.org/10.1000/xyz123" in html
    assert "Layer normalization" in html
