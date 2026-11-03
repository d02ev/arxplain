"""Keywords is not a dimension: terminology is per-topic, not a level."""
import json

from arxplain.extractor import extract_concepts
from arxplain.ingestion import paper_from_text
from arxplain.models import Dimension, Paper, UserProfile


def test_dimension_set_has_no_keywords():
    assert [d.value for d in Dimension] == [
        "theory", "mathematics", "experimentation", "results",
        "assumptions", "limitations", "domain"]


def test_old_profile_keywords_level_ignored_topic_becomes_theory():
    p = UserProfile.from_dict({
        "levels": {"keywords": 4, "mathematics": 3},
        "topics": [{"name": "Scaled Dot-Product Attention",
                    "dimension": "keywords", "familiarity": 9}],
    })
    dumped = json.loads(json.dumps(p.to_dict()))
    assert "keywords" not in dumped["levels"]
    assert "keywords" not in dumped["level_labels"]
    assert dumped["levels"]["mathematics"] == 3
    assert p.topics[0].dimension is Dimension.THEORY


def test_old_document_keywords_concept_becomes_theory():
    paper = paper_from_text('T\n\nAbstract\nWe use "Zebra Normalisation" here.\n',
                            source="text")
    doc = paper.to_document()
    doc["concepts"] = [{"name": "Zebra Normalisation",
                        "dimension": "keywords"}]
    q = Paper.from_document(doc)
    assert q.concepts[0].dimension is Dimension.THEORY


def test_extractor_terms_are_theory_not_keywords():
    paper = paper_from_text(
        'T\n\nAbstract\nWe introduce "Zebra Normalisation" for sequences.\n'
        '"Zebra Normalisation" rescales activations in every layer.\n',
        source="text")
    extract_concepts(paper)
    terms = [c for c in paper.concepts if c.name == "Zebra Normalisation"]
    assert terms and terms[0].dimension is Dimension.THEORY
