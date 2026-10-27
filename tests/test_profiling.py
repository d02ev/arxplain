import json

from arxplain.cli import DEMO_TEXT, collect_ratings, main
from arxplain.ingestion import paper_from_text
from arxplain.llm import MockLLM
from arxplain.models import (Dimension, Resource, Topic, UserProfile,
                             familiarity_to_level)
from arxplain.pipeline import explain_paper
from arxplain.profiler import ProfilerAgent, heuristic_topics
from arxplain.render import render_html, render_markdown
from arxplain.search import BaseSearch, CachedSearch, NoSearch


class FakeSearch(BaseSearch):
    name = "fake"

    def search(self, query, *, max_results=3):
        self.calls += 1
        return [Resource(title=f"Learn: {query}", url="https://example.com/learn",
                         source="fake")]


def demo_paper():
    return paper_from_text(DEMO_TEXT, source="demo")


def test_familiarity_mapping():
    assert familiarity_to_level(1) == 0
    assert familiarity_to_level(10) == 4
    assert 0 <= familiarity_to_level(5) <= 2


def test_heuristic_topics_split_hard_and_optional():
    paper = demo_paper()
    from arxplain.extractor import extract_concepts
    extract_concepts(paper)
    topics = heuristic_topics(paper)
    assert any(t.is_hard for t in topics)
    assert topics[0].is_hard  # hard requirements come first
    assert all(t.search_query for t in topics)


def test_profiler_offline_costs_no_llm_calls_and_caps_search():
    search = CachedSearch(FakeSearch(), max_calls=3, use_cache=False)
    topics, trace = ProfilerAgent(MockLLM(), search, max_searches=3).run(demo_paper())
    assert topics and search.calls <= 3
    assert any(t.resources for t in topics)  # hard topics linked first


def test_collect_ratings_interactive_and_dict():
    topics = [Topic(name="Attention", requirement="hard"),
              Topic(name="Docker", requirement="optional")]
    answers = iter(["3", ""])  # rate first, skip second
    collect_ratings(topics, input_fn=lambda _p: next(answers))
    assert topics[0].familiarity == 3 and topics[1].familiarity is None
    topics2 = [Topic(name="Scaled dot-product attention & softmax")]
    collect_ratings(topics2, ratings={"attention": 9})
    assert topics2[0].familiarity == 9


def test_topic_rating_drives_gap_and_links_in_output():
    profile = UserProfile(name="Ada")
    profile.topics = [Topic(
        name="Scaled dot-product attention & softmax",
        dimension=Dimension.MATHEMATICS, requirement="hard", familiarity=2,
        resources=[Resource(title="Attention explained",
                            url="https://example.com/attn", source="fake")])]
    profile.sync_levels_from_topics()
    result = explain_paper(demo_paper(), profile, llm=MockLLM())
    attn = next(e for e in result.explanations
                if "attention" in e.concept_name.lower())
    assert attn.familiarity == 2 and attn.resources  # weak -> linked
    md, html = render_markdown(result), render_html(result)
    # each link appears exactly once in the whole document (no repeats in
    # Background topics / Further reading once a section has shown it)
    assert md.count("https://example.com/attn") == 1
    assert html.count("https://example.com/attn") == 1
    assert "Topics you rated" in html
    # round-trip through profile.json format keeps topics + resources
    again = UserProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert again.topics[0].familiarity == 2
    assert again.topics[0].resources[0].url == "https://example.com/attn"


def test_cli_profile_paper_writes_profile_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["profile-paper", "--demo", "--llm-mode", "off",
               "--search", "off", "--default-rating", "4",
               "--save", "profile.json"])
    assert rc == 0
    data = json.loads((tmp_path / "profile.json").read_text())
    assert data["topics"] and all(t["familiarity"] == 4 for t in data["topics"])
    assert any(t["requirement"] == "hard" for t in data["topics"])
    # and explain consumes it without any new profiling
    assert main(["explain", "--demo", "--profile", "profile.json",
                 "--format", "md", "-o", "out.md", "--no-resources"]) == 0
    assert "Topics you rated" in (tmp_path / "out.md").read_text()
