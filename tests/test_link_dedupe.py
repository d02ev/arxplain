import re

from arxplain.ingestion import paper_from_text
from arxplain.llm import MockLLM
from arxplain.models import Dimension, Resource, Topic, UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html

TEXT = """Attention Study

Abstract
We study scaled dot-product attention and softmax in this paper fully.

1. Introduction
Scaled dot-product attention and softmax are introduced here in detail.

2. Background
Scaled dot-product attention and softmax return as prior work context.
"""


def result_with_shared_link():
    profile = UserProfile(name="Ada")
    profile.topics = [Topic(
        name="Scaled dot-product attention & softmax",
        dimension=Dimension.MATHEMATICS, requirement="hard", familiarity=1,
        resources=[Resource(title="Attention explained",
                            url="https://example.com/attn", source="ddgs"),
                   Resource(title="Attention by hand",
                            url="https://example.com/attn2", source="ddgs")])]
    profile.sync_levels_from_topics()
    return explain_paper(paper_from_text(TEXT, source="text"),
                         profile, llm=MockLLM())


def test_no_link_repeats_anywhere_in_either_format():
    result = result_with_shared_link()
    for out in (render_html(result),):
        urls = re.findall(r"https://example\.com/[a-z0-9]+", out)
        assert urls, "expected the weak topic's links in Resources"
        assert len(urls) == len(set(urls)), f"repeated links: {urls}"
        assert (out.count("(https://example.com/attn)")
                + out.count('href="https://example.com/attn"')) == 1


def test_resources_live_in_one_sidebar_not_thrown_into_sections():
    result = result_with_shared_link()
    html = render_html(result)
    assert "<aside class='resources'" in html and "<h2>Resources</h2>" in html
    assert "Still unclear?" not in html
    assert "Further reading" not in html
    # external links occur inside the sidebar, not inside a section card
    sidebar = html.split("<aside class='resources'")[1]
    assert "https://example.com/attn" in sidebar
    main = html.split("<aside class='resources'")[0]
    assert "https://example.com/attn" not in main


def test_section_topics_jump_to_their_sidebar_entry():
    result = result_with_shared_link()
    html = render_html(result)
    # topic under a section is a jump-link to its sidebar entry...
    assert "href=\"#res-scaled-dot-product-attention-softmax\"" in html
    # ...and that entry exists exactly once, holding the external links
    assert html.count("id='res-scaled-dot-product-attention-softmax'") == 1
