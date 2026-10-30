import json

import pytest

from arxplain.ingestion import paper_from_text
from arxplain.models import (PAPER_SCHEMA_VERSION, Dimension, Figure,
                             Paper, Reference, Section)


def sample_paper():
    p = paper_from_text(
        "T\n\nAbstract\nWe propose a model for sequences in this work.\n\n"
        "1. Method\nThe model uses attention over all positions in parallel.\n",
        source="text")
    p.figures = [Figure(src="https://x/fig1.png", caption="Figure 1: Model.",
                        label="Figure 1", section_title="1. Method")]
    p.references = [Reference(text="Doe. Layer normalization. arXiv:1607.06450",
                              title="Layer normalization.", label="[1]",
                              url="https://arxiv.org/abs/1607.06450",
                              url_kind="arxiv")]
    return p


def test_document_roundtrip_preserves_everything():
    p = sample_paper()
    doc = json.loads(json.dumps(p.to_document()))  # must be JSON-safe
    assert doc["schema_version"] == PAPER_SCHEMA_VERSION
    q = Paper.from_document(doc)
    assert q.title == p.title and q.id == p.id
    assert [s.title for s in q.sections] == [s.title for s in p.sections]
    assert q.sections[1].text == p.sections[1].text
    assert q.figures[0].src.endswith("fig1.png")
    assert q.references[0].url_kind == "arxiv"


def test_newer_schema_is_rejected_loudly():
    doc = sample_paper().to_document()
    doc["schema_version"] = PAPER_SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        Paper.from_document(doc)
