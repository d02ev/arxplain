import json

from arxplain.cli import DEMO_TEXT, collect_ratings
from arxplain.ingestion import paper_from_text
from arxplain.llm import BaseLLM, MockLLM
from arxplain.models import Topic, UserProfile
from arxplain.profiler import ProfilerAgent, find_context
from arxplain.search import NoSearch


class QuestionLLM(BaseLLM):
    name = "questioner"
    model = "q-1"

    def complete(self, prompt, *, system="", max_tokens=1200):
        self.usage.calls += 1
        if "self-check question" in prompt:
            return json.dumps([
                {"n": "Scaled dot-product attention & softmax",
                 "q": "Do you understand why attention scores are divided "
                      "by sqrt(d_k) before the softmax?"},
                {"n": "PyTorch", "q": "Do you understand how PyTorch "
                                      "computes the attention matrix here?"}])
        return ""  # classification unusable -> heuristic topics kept


def test_context_is_the_papers_own_passage():
    ctx = find_context(DEMO_TEXT, "Scaled dot-product attention & softmax")
    assert "Transformer" in ctx or "attention" in ctx.lower()


GARBLED = """Study of Things

Abstract
We propose a clear approach in this paper for sequence modelling tasks.

1. Method
The value $i$ ii denotes the position [1, 2, 9] in $d_{k}$ dkd_{k} terms.
Scaled dot-product attention lets each position attend to all positions
in the previous layer, which is how the model mixes information [31].
We compute the dot products of queries with keys [4, 5] in parallel.
"""


def test_context_avoids_citation_and_formula_garble():
    from arxplain.profiler import clean_snippet, find_context as fc
    paper = paper_from_text(GARBLED, source="text")
    ctx = fc(paper, "Scaled dot-product attention & softmax")
    assert "attend to all positions" in ctx  # explanatory sentence wins
    assert "[31]" not in ctx and "[1, 2, 9]" not in ctx
    assert "dkd" not in ctx and "$i$ ii" not in ctx
    assert clean_snippet("scores $d_{k}$ dkd_{k} from [1, 2, 9] papers") == \
        "scores $d_k$ from papers"


def test_every_topic_gets_context_and_self_check_offline():
    topics, _ = ProfilerAgent(MockLLM(), NoSearch()).run(
        paper_from_text(DEMO_TEXT, source="demo"))
    assert topics
    for t in topics:
        assert t.self_check.startswith("Do you understand")
    assert any(t.paper_context for t in topics)


def test_llm_writes_specific_questions_from_the_passage():
    topics, _ = ProfilerAgent(QuestionLLM(), NoSearch()).run(
        paper_from_text(DEMO_TEXT, source="demo"))
    attn = next(t for t in topics if "attention" in t.name.lower())
    assert "sqrt(d_k)" in attn.self_check
    # persisted in profile.json and restored intact
    prof = UserProfile(topics=topics)
    again = UserProfile.from_dict(json.loads(json.dumps(prof.to_dict())))
    a2 = next(t for t in again.topics if "attention" in t.name.lower())
    assert a2.self_check == attn.self_check and a2.paper_context == attn.paper_context


def test_rating_prompt_shows_passage_and_question(capsys):
    topics = [Topic(name="Scaled dot-product attention & softmax",
                    requirement="hard",
                    paper_context="The Transformer uses scaled dot-product "
                                  "attention and softmax over embeddings.",
                    self_check="Do you understand why scores are scaled?")]
    collect_ratings(topics, input_fn=lambda _p: "4")
    out = capsys.readouterr().out
    assert "In this paper:" in out and "scaled dot-product" in out
    assert "Self-check: Do you understand why scores are scaled?" in out
    assert topics[0].familiarity == 4
