from arxplain.cli import DEMO_TEXT
from arxplain.ingestion import paper_from_text
from arxplain.llm import MockLLM
from arxplain.models import Dimension, Resource, Topic, UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html, render_markdown


def profile_with_weak_attention():
    p = UserProfile(name="Ada")
    p.topics = [Topic(
        name="Scaled dot-product attention & softmax",
        dimension=Dimension.MATHEMATICS, requirement="hard", familiarity=2,
        resources=[Resource(title="Attention, explained simply",
                            url="https://example.com/attn", source="ddgs")])]
    p.sync_levels_from_topics()
    return p


def test_paper_itself_is_explained_section_by_section():
    result = explain_paper(paper_from_text(DEMO_TEXT, source="demo"),
                           profile_with_weak_attention(), llm=MockLLM())
    assert result.paper_explanation, "paper sections must be explained"
    titles = [s.section_title for s in result.paper_explanation]
    assert any("Model" in t or "Architecture" in t for t in titles)
    joined = " ".join(s.body for s in result.paper_explanation)
    assert "Transformer" in joined  # the paper's actual content, not just topics
    # weak hard-requirement topic attaches its link inside the section text flow
    linked = [s for s in result.paper_explanation if s.resources]
    assert linked and any(r.url == "https://example.com/attn"
                          for s in linked for r in s.resources)


def test_user_facing_files_hide_backend_trace():
    result = explain_paper(paper_from_text(DEMO_TEXT, source="demo"),
                           profile_with_weak_attention(), llm=MockLLM())
    assert result.trace  # trace still exists as backend metadata
    for out in (render_markdown(result), render_html(result)):
        assert "How this explanation was built" not in out
        assert "gap-analysis" not in out and "[profiler]" not in out
        assert "The paper, explained section by section" in out
        assert "https://example.com/attn" in out
    assert result.to_dict()["trace"]  # ...but remains available programmatically
