"""Unit tests for document_cleaner.py (prose only, no tables)."""

from __future__ import annotations

import pytest

from langchain_core.documents import Document

from src.ingestion.document_cleaner import (
    clean_document,
    clean_documents,
    clean_text,
    MIN_CONTENT_LENGTH,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------
def test_clean_text_removes_citation_markers_but_keeps_meaningful_content():
    text = (
        "Arsenal won the league in 1998 (their first since 1991).[1][2] "
        "The manager was Ars\u00e8ne Wenger.[a] "
        "Revenue was \u00a347.8m (after tax). "
        "Highbury closed in 2006.[3]"
    )
    cleaned = clean_text(text)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "[3]" not in cleaned
    assert "[a]" not in cleaned
    # Meaningful numbers, dates, currency and parentheses are preserved.
    assert "1998" in cleaned
    assert "1991" in cleaned
    assert "2006" in cleaned
    assert "\u00a347.8m" in cleaned
    assert "(their first since 1991)" in cleaned
    assert "(after tax)" in cleaned
    assert "Ars\u00e8ne Wenger" in cleaned


def test_clean_text_removes_templates_and_html_tags():
    text = "Lead {{Infobox club}} content <div>with</div> HTML <span>tags</span>."
    cleaned = clean_text(text)
    assert "{{" not in cleaned
    assert "}}" not in cleaned
    assert "<div>" not in cleaned
    assert "</span>" not in cleaned
    assert "Lead  content with HTML tags." in cleaned or "Lead content with HTML tags." in cleaned


def test_clean_text_decodes_html_entities():
    assert "Tom & Jerry" in clean_text("Tom &amp; Jerry")


def test_clean_text_preserves_paragraph_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    cleaned = clean_text(text)
    parts = cleaned.split("\n\n")
    assert "First paragraph." in parts[0]
    assert "Second paragraph." in parts[1]
    assert "Third paragraph." in parts[-1]


def test_clean_text_normalises_whitespace():
    text = "Too   many    spaces\tand\ttabs."
    cleaned = clean_text(text)
    assert "   " not in cleaned
    assert "\t" not in cleaned
    assert "Too many spaces and tabs." in cleaned


# ---------------------------------------------------------------------------
# clean_document
# ---------------------------------------------------------------------------
def test_clean_document_adds_audit_metadata_and_preserves_source_metadata():
    doc = Document(
        page_content="Hello [1] world.",
        metadata={
            "source": "wikipedia",
            "title": "Arsenal F.C.",
            "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
            "section": "History",
            "section_path": "History",
            "language": "en",
            "content_type": "prose",
        },
    )
    result = clean_document(doc)
    assert result.page_content == "Hello world."
    assert result.metadata["cleaned"] is True
    assert result.metadata["chars_before"] == len("Hello [1] world.")
    assert result.metadata["chars_after"] == len("Hello world.")
    # Source metadata preserved.
    assert result.metadata["title"] == "Arsenal F.C."
    assert result.metadata["section"] == "History"
    assert result.metadata["content_type"] == "prose"


# ---------------------------------------------------------------------------
# clean_documents
# ---------------------------------------------------------------------------
def _make_doc(title, section, text, **extra):
    base = {
        "source": "wikipedia",
        "title": title,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "section": section,
        "section_path": section,
        "language": "en",
        "content_type": "prose",
    }
    base.update(extra)
    return Document(page_content=text, metadata=base)


def test_clean_documents_drops_junk_sections():
    docs = [
        _make_doc("A", "History", "Real history content here."),
        _make_doc("A", "References", "[1] Reference one.\n[2] Reference two."),
        _make_doc("A", "See also", "See also links."),
    ]
    cleaned = clean_documents(docs)
    assert len(cleaned) == 1
    assert cleaned[0].metadata["section"] == "History"


def test_clean_documents_drops_short_empty_docs():
    docs = [
        _make_doc("A", "History", "Good content with enough length here."),
        _make_doc("A", "Stub", "x"),  # below MIN_CONTENT_LENGTH
        _make_doc("A", "Empty", ""),
    ]
    cleaned = clean_documents(docs)
    assert len(cleaned) == 1
    assert cleaned[0].metadata["section"] == "History"


def test_clean_documents_deduplicates_by_content():
    docs = [
        _make_doc("A", "History", "Duplicate content here that is long enough."),
        _make_doc("B", "History", "Duplicate content here that is long enough."),
        _make_doc("C", "History", "Unique content here that is long enough."),
    ]
    cleaned = clean_documents(docs)
    assert len(cleaned) == 2
    texts = {d.page_content for d in cleaned}
    assert texts == {
        "Duplicate content here that is long enough.",
        "Unique content here that is long enough.",
    }


def test_clean_documents_preserves_metadata_and_prose_type():
    docs = [
        _make_doc(
            "Arsenal F.C.", "History > Early years",
            "Turned professional in 1891.",
            section_path="History > Early years",
            pageid=123,
            revision_id=999,
        )
    ]
    cleaned = clean_documents(docs)
    assert len(cleaned) == 1
    d = cleaned[0]
    assert d.metadata["content_type"] == "prose"
    assert d.metadata["title"] == "Arsenal F.C."
    assert d.metadata["section_path"] == "History > Early years"
    assert d.metadata["pageid"] == 123
    assert d.metadata["revision_id"] == 999


def test_min_content_length_is_positive():
    assert MIN_CONTENT_LENGTH > 0
