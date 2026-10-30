from arxplain.ingestion import (fetch_arxiv, load_cached_paper,
                                paper_from_text, save_cached_paper)

XML = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>Attention Is All You Need</title>
<summary>We propose the Transformer based on attention mechanisms.</summary>
<author><name>A Vaswani</name></author></entry></feed>"""

HTML = b"""<html><body>
<h2>Abstract</h2><p>We propose the Transformer based on attention.</p>
<h2>Introduction</h2><p>Recurrent networks have dominated sequence tasks.</p>
<h2>Conclusion</h2><p>Attention alone is sufficient for transduction.</p>
</body></html>"""


def fake_get_factory(calls):
    def fake_get(url, timeout):
        calls.append(url)
        return XML if "export.arxiv.org" in url else HTML
    return fake_get


def test_fetch_arxiv_parses_once_then_serves_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARXPLAIN_CACHE_DIR", str(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr("arxplain.ingestion._get", fake_get_factory(calls))
    p1 = fetch_arxiv("1706.03762")
    n_after_first = len(calls)
    assert p1.complete and len(p1.sections) >= 3
    p2 = fetch_arxiv("1706.03762")
    assert len(calls) == n_after_first  # zero new network on second fetch
    assert p2.title == p1.title and "cached" in p2.source
    assert [s.title for s in p2.sections] == [s.title for s in p1.sections]


def test_cache_roundtrip_and_no_cache_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("ARXPLAIN_CACHE_DIR", str(tmp_path))
    paper = paper_from_text(
        "T\n\nAbstract\nWe study attention in this abstract here.\n\n"
        "1. Method\nThe method uses attention over all positions here.")
    save_cached_paper("arxiv:0000.00001", paper)
    loaded = load_cached_paper("arxiv:0000.00001")
    assert loaded is not None and loaded.title == paper.title
    monkeypatch.setenv("ARXPLAIN_PAPER_CACHE", "0")
    assert load_cached_paper("arxiv:0000.00001") is None
