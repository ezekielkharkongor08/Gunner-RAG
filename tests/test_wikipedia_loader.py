"""Unit tests for wikipedia_loader.py.

All MediaWiki API calls are mocked so the suite does not depend on live
Wikipedia availability.  Covers: prose-only extraction, section hierarchy,
no parent/child duplication, missing/empty pages, API errors, retries,
timeouts, redirects, deduplication, and language/endpoint configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from langchain_core.documents import Document

import src.ingestion.wikipedia_loader as wl
from src.ingestion.wikipedia_loader import (
    EmptyArticleError,
    PageNotFoundError,
    WikipediaAPIError,
    fetch_wikipedia_documents,
    _parse_sections,
)


@pytest.fixture(autouse=True)
def _disable_fetch_cache(monkeypatch):
    """Tests use fake titles that may exist in the real on-disk cache."""
    from dataclasses import replace

    patched = replace(
        wl.settings,
        cache=replace(wl.settings.cache, enabled=False),
    )
    monkeypatch.setattr(wl, "settings", patched)

# ---------------------------------------------------------------------------
# Fake HTTP response
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, json_data, status_code=200, raise_for_status=None):
        self._json = json_data
        self.status_code = status_code
        self._raise_for_status = raise_for_status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self._raise_for_status is not None:
            raise self._raise_for_status


def _extract_response(title, extract, pageid=1, lastrevid=2, missing=False, language="en"):
    """Build a MediaWiki API response for ``action=query&prop=extracts|info``."""
    page = {
        "pageid": pageid,
        "title": title,
        "lastrevid": lastrevid,
        "canonicalurl": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "extract": extract,
    }
    if missing:
        page["missing"] = ""
    return {"query": {"pages": {str(pageid): page}}}


def _session_get(responses_by_title):
    """Return a fake ``session.get`` that answers from a title -> response dict."""
    def _get(url, *, params, timeout=None, headers=None):
        title = params.get("titles")
        data = responses_by_title.get(title)
        if data is None:
            # Unknown title -> missing page
            return _FakeResponse(_extract_response(title, "", missing=True))
        return _FakeResponse(data)
    return _get


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------
def test_parse_sections_intro_and_nested_hierarchy():
    text = (
        "Lead intro sentence about the topic.\n\n"
        "== History ==\n"
        "History body paragraph.\n\n"
        "=== Early years ===\n"
        "Early years body paragraph.\n\n"
        "== References ==\n"
        "Reference list content."
    )
    sections = _parse_sections(text)
    assert [s["heading"] for s in sections] == ["Introduction", "History", "Early years", "References"]
    assert sections[0]["path"] == ["Introduction"]
    assert sections[1]["path"] == ["History"]
    assert sections[2]["path"] == ["History", "Early years"]
    assert sections[2]["level"] == 3
    assert sections[0]["text"].startswith("Lead intro sentence")
    assert sections[1]["text"] == "History body paragraph."
    assert sections[2]["text"] == "Early years body paragraph."


def test_parse_sections_no_parent_child_duplication():
    text = (
        "Intro.\n\n"
        "== History ==\n"
        "Direct history content.\n\n"
        "=== Early years ===\n"
        "Early years content.\n\n"
        "More history after the subsection.\n\n"
        "== Honours ==\n"
        "Honours content."
    )
    sections = _parse_sections(text)
    history = next(s for s in sections if s["heading"] == "History")
    early = next(s for s in sections if s["heading"] == "Early years")
    # Early years content must not appear inside History.
    assert "Early years content." not in history["text"]
    # History keeps its direct content.
    assert "Direct history content." in history["text"]
    # Each paragraph appears in exactly one section (no duplication).
    all_text = "\n".join(s["text"] for s in sections)
    assert all_text.count("Direct history content.") == 1
    assert all_text.count("Early years content.") == 1
    assert all_text.count("More history after the subsection.") == 1
    assert all_text.count("Honours content.") == 1


def test_parse_sections_empty_dropped():
    text = "== History ==\n\n== Honours ==\nReal content."
    sections = _parse_sections(text)
    assert [s["heading"] for s in sections] == ["Honours"]


def test_parse_sections_intro_only():
    sections = _parse_sections("Just a lead paragraph, no headings.")
    assert len(sections) == 1
    assert sections[0]["heading"] == "Introduction"
    assert sections[0]["path"] == ["Introduction"]


# ---------------------------------------------------------------------------
# Loader: happy path -> prose-only Documents
# ---------------------------------------------------------------------------
def test_fetch_article_prose_only_documents():
    extract = (
        "Arsenal Football Club is a professional club.[1]\n\n"
        "== History ==\n"
        "Founded in 1886 as Dial Square.[2]\n\n"
        "=== Early years ===\n"
        "Turned professional in 1891.[3]\n\n"
        "== References ==\n"
        "[1] Source one\n[2] Source two\n"
    )
    responses = {
        "Arsenal F.C.": _extract_response("Arsenal F.C.", extract, pageid=101, lastrevid=5050),
    }
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])

    assert len(docs) == 4  # Introduction, History, Early years, References (loader keeps all sections; the cleaner drops junk sections)
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.metadata["content_type"] == "prose"
        assert doc.metadata["source"] == "wikipedia"
        assert doc.metadata["language"] == "en"
        assert doc.metadata["title"] == "Arsenal F.C."
        assert doc.metadata["url"].startswith("https://en.wikipedia.org/wiki/")
        assert doc.metadata["pageid"] == 101
        assert doc.metadata["revision_id"] == 5050
        assert "fetched_at" in doc.metadata
    # No table-specific metadata anywhere.
    for doc in docs:
        for key in ("table_caption", "table_header", "table_rows", "content_type"):
            if key in ("table_caption", "table_header", "table_rows"):
                assert key not in doc.metadata
    assert docs[0].metadata["section"] == "Introduction"
    assert docs[1].metadata["section"] == "History"
    assert docs[1].metadata["section_path"] == "History"
    assert docs[2].metadata["section_path"] == "History > Early years"


def test_introduction_preserved_as_own_section():
    extract = "Lead paragraph with no headings at all."
    responses = {"X": _extract_response("X", extract)}
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["X"])
    assert len(docs) == 1
    assert docs[0].metadata["section"] == "Introduction"
    assert docs[0].page_content == "Lead paragraph with no headings at all."


def test_no_duplicated_paragraphs_across_sections():
    extract = (
        "Intro line.\n\n"
        "== History ==\n"
        "History only.\n\n"
        "=== Sub ===\n"
        "Sub only.\n\n"
        "== See also ==\n"
        "See also line."
    )
    responses = {"X": _extract_response("X", extract)}
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["X"])
    texts = [d.page_content for d in docs]
    assert texts.count("History only.") == 1
    assert texts.count("Sub only.") == 1
    # The loader does not drop junk sections; the cleaner does.  All four
    # sections are present as prose documents.
    assert len(docs) == 4


# ---------------------------------------------------------------------------
# Loader: errors are never silent
# ---------------------------------------------------------------------------
def test_missing_page_raises_page_not_found():
    responses = {"Ghost": _extract_response("Ghost", "", missing=True)}
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        with pytest.raises(PageNotFoundError):
            wl._loader.fetch_article("Ghost")
    # fetch_wikipedia_documents skips it rather than returning an empty doc.
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["Ghost"])
    assert docs == []


def test_empty_extract_raises_empty_article():
    responses = {"Empty": _extract_response("Empty", "")}
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        with pytest.raises(EmptyArticleError):
            wl._loader.fetch_article("Empty")
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["Empty"])
    assert docs == []


def test_api_error_raises():
    def _get(url, *, params, timeout=None, headers=None):
        return _FakeResponse({"error": {"code": "missingtitle", "info": "Page does not exist"}}, status_code=200)
    with patch.object(wl._loader._session, "get", side_effect=_get):
        with pytest.raises(WikipediaAPIError):
            wl._loader.fetch_article("Whatever")


def test_network_error_retries_then_succeeds():
    import requests
    call_count = {"n": 0}

    def _get(url, *, params, timeout=None, headers=None):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise requests.ConnectionError("temporary failure")
        return _FakeResponse(_extract_response("Retry", "Retry content."))

    with patch.object(wl._loader._session, "get", side_effect=_get):
        article = wl._loader.fetch_article("Retry")
    assert article["title"] == "Retry"
    assert call_count["n"] == 3


def test_timeout_is_propagated():
    import requests

    def _get(url, *, params, timeout=None, headers=None):
        raise requests.Timeout("timed out")

    with patch.object(wl._loader._session, "get", side_effect=_get):
        with pytest.raises(WikipediaAPIError):
            wl._loader.fetch_article("Timeout")


def test_titles_deduplicated_within_run():
    responses = {"A": _extract_response("A", "Content A.")}
    calls = {"n": 0}

    def _get(url, *, params, timeout=None, headers=None):
        calls["n"] += 1
        return _FakeResponse(_extract_response(params.get("titles"), "x"))

    with patch.object(wl._loader._session, "get", side_effect=_get):
        fetch_wikipedia_documents(titles=["A", "A", "A"])
    # "A" should be fetched exactly once despite being requested three times.
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Language edition / endpoint configuration
# ---------------------------------------------------------------------------
def test_language_edition_used_in_metadata_and_url(monkeypatch):
    from dataclasses import replace
    new_wiki = replace(wl.settings.wikipedia, lang="fr")
    new_settings = replace(wl.settings, wikipedia=new_wiki)
    monkeypatch.setattr(wl, "settings", new_settings)
    monkeypatch.setattr(wl, "_loader", wl.WikipediaLoader())
    assert wl._loader.api_url() == "https://fr.wikipedia.org/w/api.php"
    extract = "Contenu en fran\u00e7ais.\n\n== Histoire ==\nSuite."
    responses = {"Arsenal F.C.": _extract_response("Arsenal F.C.", extract, language="fr")}
    with patch.object(wl._loader._session, "get", side_effect=_session_get(responses)):
        docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])
    assert all(d.metadata["language"] == "fr" for d in docs)
    assert all(d.metadata["url"].startswith("https://fr.wikipedia.org/wiki/") for d in docs)


def test_custom_api_url_is_used(monkeypatch):
    from dataclasses import replace
    new_wiki = replace(wl.settings.wikipedia, api_url="https://custom.example.org/w/api.php")
    new_settings = replace(wl.settings, wikipedia=new_wiki)
    monkeypatch.setattr(wl, "settings", new_settings)
    monkeypatch.setattr(wl, "_loader", wl.WikipediaLoader())
    assert wl._loader.api_url() == "https://custom.example.org/w/api.php"

    def _get(url, *, params, timeout=None, headers=None):
        assert url == "https://custom.example.org/w/api.php"
        return _FakeResponse(_extract_response("X", "Custom endpoint content."))

    with patch.object(wl._loader._session, "get", side_effect=_get):
        docs = wl.fetch_wikipedia_documents(titles=["X"])
    assert len(docs) == 1
    assert docs[0].page_content == "Custom endpoint content."
