import sys
import types

from arxplain.ingestion import convert_with_markitdown


def test_absent_markitdown_is_graceful(monkeypatch):
    monkeypatch.setitem(sys.modules, "markitdown", None)
    # None in sys.modules makes __import__ raise ImportError -> fallback
    assert convert_with_markitdown(b"<html/>", ".html") is None


def test_markitdown_result_is_used_when_installed(monkeypatch):
    class Result:
        text_content = "# Study\n\n## Method\nWe use attention here in this work."

    class FakeMarkItDown:
        def convert_stream(self, stream, file_extension=""):
            assert file_extension == ".docx"
            return Result()

    fake = types.ModuleType("markitdown")
    fake.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake)
    out = convert_with_markitdown(b"binary", ".docx")
    assert out is not None and "## Method" in out
