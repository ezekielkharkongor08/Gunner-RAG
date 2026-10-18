"""Optional integration test against live Wikipedia.

Skipped unless ARSENAL_WEKI_INTEGRATION=1 is set, so the ordinary test suite
never depends on network availability.
"""

from __future__ import annotations

import os
import re

import pytest

from src.ingestion.document_cleaner import clean_documents
from src.ingestion.text_splitter import split_documents
from src.ingestion.wikipedia_loader import fetch_wikipedia_documents

pytestmark = pytest.mark.skipif(
    os.getenv("ARSENAL_WEKI_INTEGRATION") != "1",
    reason="set ARSENAL_WEKI_INTEGRATION=1 to run live Wikipedia integration tests",
)


def test_live_english_article_prose_sections():
    docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])
    assert docs
    assert all(d.metadata["content_type"] == "prose" for d in docs)
    assert all(d.metadata["language"] == "en" for d in docs)
    assert all(d.metadata["url"].startswith("https://en.wikipedia.org/wiki/") for d in docs)
    assert any(d.metadata["section"] == "Introduction" for d in docs)
    # No table-specific metadata anywhere.
    for d in docs:
        assert "table_caption" not in d.metadata
        assert "table_header" not in d.metadata
        assert "table_rows" not in d.metadata


def test_live_no_parent_child_duplication():
    docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])
    texts = [d.page_content for d in docs]
    # Each section's content is distinct (no exact duplication).
    assert len(texts) == len(set(texts))
    # The introduction (first section) is not repeated in any later section.
    intro = texts[0]
    for other in texts[1:]:
        assert intro not in other


def test_live_missing_page_skipped_not_empty():
    docs = fetch_wikipedia_documents(titles=["This Page Does Not Exist Xyz123"])
    assert docs == []


def test_live_redirect_followed():
    # "AI" is a redirect to "Artificial intelligence" on English Wikipedia.
    docs = fetch_wikipedia_documents(titles=["AI"])
    assert docs
    assert docs[0].metadata["title"] == "Artificial intelligence"


def test_live_cleaner_removes_citations_keeps_facts():
    docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])
    cleaned = clean_documents(docs)
    assert cleaned
    for d in cleaned:
        assert not re.search(r"\[\d+\]", d.page_content)
        assert not re.search(r"\[[a-z]\]", d.page_content)


def test_live_splitter_preserves_metadata():
    docs = fetch_wikipedia_documents(titles=["Arsenal F.C."])
    chunks = split_documents(clean_documents(docs))
    assert chunks
    for c in chunks:
        assert c.metadata["content_type"] == "prose"
        assert c.metadata["title"]
        assert c.metadata["url"]
        assert c.metadata["section"]
        assert c.metadata["chunk_id"]
