import pytest

from arxplain.search import DDGSSearch, SearchError, TavilySearch, get_search


def test_only_tavily_and_ddgs_exist(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("ARXPLAIN_SEARCH", raising=False)
    assert get_search("auto").name == "ddgs"      # free fallback, no key
    assert get_search("ddgs").name == "ddgs"
    assert get_search("duckduckgo").name == "ddgs"  # alias still works
    with pytest.raises(SearchError):
        get_search("some-removed-provider")        # unknown names are loud
    with pytest.raises(SearchError):
        get_search("wikipedia")                    # not a provider anymore
    with pytest.raises(SearchError):
        get_search("tavily")                       # keyed, no key -> loud


def test_tavily_is_primary_when_keyed(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    assert get_search("auto").name == "tavily"
    assert isinstance(get_search("tavily").inner, TavilySearch)


def test_ddgs_parses_library_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=3):
            return [{"title": "Attention, explained", "href": "https://ex.com/a",
                     "body": "A friendly intro."}]

    monkeypatch.setattr("arxplain.search._ddgs_client", lambda: FakeDDGS)
    out = DDGSSearch().search("attention explained", max_results=2)
    assert out[0].url == "https://ex.com/a" and out[0].source == "ddgs"
