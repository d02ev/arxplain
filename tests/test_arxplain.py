import json

from arxplain.agents import QuizAgent
from arxplain.cli import DEMO_TEXT, main
from arxplain.extractor import extract_concepts
from arxplain.ingestion import normalize_arxiv_id, paper_from_text
from arxplain.llm import MockLLM, get_llm
from arxplain.models import Dimension, Level, UserProfile
from arxplain.pipeline import explain_paper
from arxplain.render import render_html


def demo_paper():
    return paper_from_text(DEMO_TEXT, source="demo")


def test_extraction_covers_all_dimensions():
    paper = demo_paper()
    concepts = extract_concepts(paper)
    dims = {c.dimension for c in concepts}
    assert Dimension.MATHEMATICS in dims
    assert Dimension.EXPERIMENTATION in dims
    assert Dimension.FRAMEWORKS in dims  # PyTorch
    assert Dimension.LANGUAGE in dims    # Python
    assert Dimension.TOOLS in dims       # Docker / CUDA


def test_explanation_depth_adapts_to_profile():
    novice = UserProfile()
    for d in Dimension:
        novice.set_level(d, Level.NOVICE)
    expert = UserProfile()
    for d in Dimension:
        expert.set_level(d, Level.EXPERT)
    n = explain_paper(demo_paper(), novice, llm=MockLLM())
    e = explain_paper(demo_paper(), expert, llm=MockLLM())
    assert any(x.depth == "foundation" for x in n.explanations)
    assert all(x.depth == "skip-known" for x in e.explanations)
    assert n.plan.estimated_minutes > e.plan.estimated_minutes


def test_renderers_contain_title_and_profile():
    profile = UserProfile(name="Ada")
    result = explain_paper(demo_paper(), profile, llm=MockLLM())
    html = render_html(result)
    assert "<!DOCTYPE html>" in html and "ADA" in html and "Attention Is All You Need" in html
    assert "Your learning path" in html and "Gaps at a glance" in html


def test_quiz_generate_and_evaluate():
    profile = UserProfile()
    paper = demo_paper()
    extract_concepts(paper)
    quiz = QuizAgent()
    qs = quiz.generate(paper, profile, count=3)
    assert len(qs) == 3 and all(q.options for q in qs)
    assert quiz.evaluate(qs[0], qs[0].answer, profile).correct is True
    assert quiz.evaluate(qs[0], "wrong", profile).correct is False
    # public view must not leak the answer key
    assert "answer" not in qs[0].public_dict()


def test_diagnostic_updates_profile():
    profile = UserProfile()
    quiz = QuizAgent()
    qs = quiz.diagnostic(profile)
    results = [quiz.evaluate(q, q.answer, profile) for q in qs]
    updated = quiz.apply_diagnostic(profile, results)
    assert all(int(updated.level(q.dimension)) >= 1 for q in qs)


def test_arxiv_id_normalization():
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762v5") == "1706.03762"
    assert normalize_arxiv_id("1706.03762") == "1706.03762"


def test_profile_roundtrip():
    p = UserProfile.from_dict({"name": "x", "levels": {"mathematics": "advanced"}})
    assert p.level(Dimension.MATHEMATICS) == Level.ADVANCED
    assert json.loads(json.dumps(p.to_dict()))["name"] == "x"


def test_mock_llm_is_default_and_offline():
    assert isinstance(get_llm(), MockLLM)


def test_cli_explain_writes_md_and_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["explain", "--demo", "--level", "beginner",
                 "--no-resources"]) == 0
    outs = list(tmp_path.glob("*.explained.*"))
    assert {p.suffix for p in outs} == {".html"}  # HTML is the only output
    html = outs[0].read_text()
    assert "Check yourself" in html or "You know this" in html
