"""Unit tests for text_splitter.py (prose only, no tables)."""

from __future__ import annotations

from langchain_core.documents import Document

import src.ingestion.text_splitter as ts
from src.ingestion.text_splitter import split_documents, describe_chunks, CONTENT_PROSE


def _make_section(title, section, section_path, text, language="en"):
    return Document(
        page_content=text,
        metadata={
            "source": "wikipedia",
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "section": section,
            "section_path": section_path,
            "language": language,
            "content_type": "prose",
        },
    )


def test_split_respects_chunk_size_and_overlap(monkeypatch):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=40, chunk_overlap=5, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )

    long_text = (
        "Sentence one about Arsenal. "
        "Sentence two about the club's history. "
        "Sentence three about Highbury stadium. "
        "Sentence four about the move to Emirates. "
    )
    doc = _make_section("Arsenal F.C.", "History", "History", long_text)
    chunks = split_documents([doc])

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert len(chunk.page_content) <= 40 + 5  # within size + overlap allowance
    # Metadata inherited.
    assert all(c.metadata["title"] == "Arsenal F.C." for c in chunks)
    assert all(c.metadata["section"] == "History" for c in chunks)
    assert all(c.metadata["content_type"] == CONTENT_PROSE for c in chunks)


def test_split_keeps_sections_separate_no_cross_contamination(monkeypatch):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )

    doc1 = _make_section("A", "History", "History", "History section content here. ")
    doc2 = _make_section("A", "Honours", "Honours", "Honours section content here. ")
    chunks = split_documents([doc1, doc2])

    history_chunks = [c for c in chunks if c.metadata["section"] == "History"]
    honours_chunks = [c for c in chunks if c.metadata["section"] == "Honours"]
    assert history_chunks and honours_chunks
    # No chunk mixes the two sections.
    for c in history_chunks:
        assert "Honours section content" not in c.page_content
    for c in honours_chunks:
        assert "History section content" not in c.page_content


def test_split_no_empty_chunks(monkeypatch):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )
    doc = _make_section("A", "History", "History", "Just one short paragraph.")
    chunks = split_documents([doc])
    assert chunks
    assert all(c.page_content.strip() for c in chunks)


def test_split_chunk_ids_unique(monkeypatch):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=30, chunk_overlap=5, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )
    text = "Alpha beta gamma delta. " * 8
    doc = _make_section("A", "History", "History", text)
    chunks = split_documents([doc])
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(ids)


def test_split_preserves_all_required_metadata(monkeypatch):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )
    doc = _make_section("Arsenal F.C.", "History > Early years", "History > Early years",
                        "Early years content.", language="fr")
    chunks = split_documents([doc])
    c = chunks[0]
    for field in ("source", "title", "url", "section", "section_path", "language", "content_type"):
        assert field in c.metadata and c.metadata[field] not in (None, "")
    assert c.metadata["content_type"] == "prose"
    assert c.metadata["language"] == "fr"
    assert "chunk_index" in c.metadata
    assert "chunk_count" in c.metadata
    assert "chunk_chars" in c.metadata


def test_split_skips_non_prose_defensively(monkeypatch, caplog):
    monkeypatch.setattr(ts, "MIN_CHUNK_SIZE", 1)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    monkeypatch.setattr(
        ts,
        "_build_splitter",
        lambda: RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""], add_start_index=False
        ),
    )
    doc = Document(
        page_content="should not be split",
        metadata={"content_type": "table", "title": "X"},
    )
    chunks = split_documents([doc])
    assert chunks == []


def test_describe_chunks():
    assert describe_chunks([]) == {"chunks": 0, "min": 0, "max": 0, "avg": 0}
    docs = [
        Document(page_content="a" * 100, metadata={}),
        Document(page_content="b" * 300, metadata={}),
    ]
    stats = describe_chunks(docs)
    assert stats == {"chunks": 2, "min": 100, "max": 300, "avg": 200}
