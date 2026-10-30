from arxplain.ingestion import parse_html_document, paper_from_text
from arxplain.llm import MockLLM
from arxplain.models import Figure, UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html

HTML = """<html><body>
<h2>Abstract</h2><p>We propose a model architecture in this work.</p>
<h2>Model Architecture</h2>
<figure id="S3.F1"><img src="1706.03762v7/Figures/ModalNet-21.png" alt="Refer to caption">
<figcaption><span>Figure 1: </span>The Transformer - model architecture.</figcaption></figure>
<p>The Transformer follows this overall architecture in Figure 1.</p>
<figure id="Sx1.F3"><object type="image/svg+xml" data="figs/attn.svg"></object>
<figcaption>Figure 3: Attention heads attending to distant words.</figcaption></figure>
<figure class="ltx_table"><table><tr><td>1</td></tr></table></figure>
</body></html>"""


def test_parser_extracts_figures_with_absolute_urls_and_captions():
    sections, figures = parse_html_document(
        HTML, base_url="https://arxiv.org/html/1706.03762")
    assert len(figures) == 2  # table-only figure has no graphic, skipped
    f1 = figures[0]
    assert f1.src == "https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png"
    assert f1.label == "Figure 1"
    assert "Transformer - model architecture" in f1.caption
    assert f1.section_title == "Model Architecture"
    assert figures[1].src.endswith("figs/attn.svg")


def test_figures_appear_in_their_section_in_both_formats():
    text = ("T\n\nAbstract\nWe study attention mechanisms in this paper.\n\n"
            "1. Model\nThe model follows Figure 1 as described in this section "
            "of the paper, which readers should inspect carefully alongside.\n")
    paper = paper_from_text(text, source="text")
    paper.figures = [Figure(src="https://example.com/fig1.png",
                            caption="Figure 1: The model architecture.",
                            label="Figure 1", section_title="1. Model")]
    result = explain_paper(paper, UserProfile(), llm=MockLLM())
    model = next(s for s in result.paper_explanation
                 if s.section_title == "1. Model")
    assert model.figures and model.figures[0].label == "Figure 1"
    html = render_html(result)
    assert '<img src="https://example.com/fig1.png"' in html
    assert "Figure 1: The model architecture." in html
