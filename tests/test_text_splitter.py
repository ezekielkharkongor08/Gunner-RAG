"""Unit tests for text_splitter.py (prose only, no tables)."""

from __future__ import annotations

import dataclasses

from langchain_core.documents import Document

import src.ingestion.text_splitter as ts
from src.ingestion.text_splitter import (
    CONTENT_PROSE,
    build_header,
    describe_chunks,
    split_documents,
)


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


def _with_chunk_settings(monkeypatch, *, size, overlap, min_size,
                         header_enabled=True):
    """Temporarily use small chunk settings (settings is frozen)."""
    chunk = dataclasses.replace(
        ts.settings.chunk,
        size=size,
        overlap=overlap,
        min_size=min_size,
        max_warn=size + 200,
        header_enabled=header_enabled,
    )
    monkeypatch.setattr(
        ts, "settings", dataclasses.replace(ts.settings, chunk=chunk)
    )


def test_split_respects_body_budget(monkeypatch):
    _with_chunk_settings(monkeypatch, size=60, overlap=5, min_size=1)

    long_text = (
        "Sentence one about Arsenal. "
        "Sentence two about the club's history. "
        "Sentence three about Highbury stadium. "
        "Sentence four about the move to Emirates. "
        "Sentence five about the current squad. "
    )
    doc = _make_section("Arsenal F.C.", "History", "History", long_text)
    chunks = split_documents([doc])

    assert len(chunks) > 1, "expected the text to be split"
    for chunk in chunks:
        assert chunk.metadata["body_chars"] <= 60
        assert chunk.metadata["chunk_chars"] == len(chunk.page_content)
    assert all(c.metadata["title"] == "Arsenal F.C." for c in chunks)
    assert all(c.metadata["section"] == "History" for c in chunks)
    assert all(c.metadata["content_type"] == CONTENT_PROSE for c in chunks)


def test_section_that_fits_stays_one_chunk(monkeypatch):
    _with_chunk_settings(monkeypatch, size=1000, overlap=100, min_size=150)
    text = "One paragraph. " * 15
    doc = _make_section("A", "History", "History", text)

    chunks = split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_count"] == 1
    assert chunks[0].metadata["body_chars"] == len(text.strip())


def test_header_is_added_once(monkeypatch):
    _with_chunk_settings(monkeypatch, size=1000, overlap=100, min_size=150)
    doc = _make_section("Arsenal F.C.", "May", "Review > May", "Body text.")

    chunk = split_documents([doc])[0]

    expected = "Arsenal F.C.\nSection: Review > May\n\nBody text."
    assert chunk.page_content == expected
    assert chunk.metadata["header_chars"] == len(
        "Arsenal F.C.\nSection: Review > May\n\n"
    )
    assert chunk.page_content.count("Section:") == 1


def test_header_can_be_disabled(monkeypatch):
    _with_chunk_settings(
        monkeypatch, size=1000, overlap=100, min_size=150,
        header_enabled=False,
    )
    doc = _make_section("A", "History", "History", "Body text.")

    chunk = split_documents([doc])[0]

    assert chunk.page_content == "Body text."
    assert chunk.metadata["header_chars"] == 0


def test_short_tail_folds_back_when_it_fits():
    pieces = ["a" * 200, "b" * 100]
    merged = ts._fold_short_chunks(pieces, min_size=150, max_body=400)
    assert merged == ["a" * 200 + "\n\n" + "b" * 100]


def test_short_tail_is_kept_when_folding_would_exceed_limit():
    pieces = ["a" * 300, "b" * 100]
    merged = ts._fold_short_chunks(pieces, min_size=150, max_body=350)
    assert len(merged) == 2


def test_no_chunk_starts_with_a_dot(monkeypatch):
    _with_chunk_settings(monkeypatch, size=80, overlap=10, min_size=1)
    text = (
        "Arsenal won the league. They also won the cup. "
        "The manager was happy. The fans celebrated."
    )
    chunks = split_documents([_make_section("A", "History", "History", text)])

    for chunk in chunks:
        body = chunk.page_content.split("\n\n", 1)[1]
        assert not body.startswith("."), body[:40]


def test_chunk_ids_unique(monkeypatch):
    _with_chunk_settings(monkeypatch, size=40, overlap=5, min_size=1)
    text = "Alpha beta gamma delta. " * 8
    chunks = split_documents([_make_section("A", "History", "History", text)])

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(ids)


def test_preserves_all_required_metadata(monkeypatch):
    _with_chunk_settings(monkeypatch, size=1000, overlap=100, min_size=150)
    doc = _make_section(
        "Arsenal F.C.", "History > Early years", "History > Early years",
        "Early years content.", language="fr",
    )
    chunk = split_documents([doc])[0]

    for field in ("source", "title", "url", "section", "section_path",
                  "language", "content_type"):
        assert field in chunk.metadata and chunk.metadata[field] not in (None, "")
    assert chunk.metadata["content_type"] == "prose"
    assert chunk.metadata["language"] == "fr"
    for field in ("chunk_index", "chunk_count", "chunk_chars", "body_chars",
                  "header_chars", "content_hash"):
        assert field in chunk.metadata


def test_duplicate_chunks_are_removed(monkeypatch):
    _with_chunk_settings(monkeypatch, size=30, overlap=0, min_size=1)
    text = "Same sentence here. " * 8
    chunks = split_documents([_make_section("A", "History", "History", text)])

    hashes = [c.metadata["content_hash"] for c in chunks]
    assert chunks
    assert len(hashes) == len(set(hashes))


def test_split_skips_non_prose_defensively():
    doc = Document(
        page_content="should not be split",
        metadata={"content_type": "table", "title": "X"},
    )
    assert split_documents([doc]) == []


def test_build_header():
    doc = _make_section("A", "May", "Review > May", "x")
    assert build_header(doc) == "A\nSection: Review > May\n\n"


def test_describe_chunks():
    assert describe_chunks([]) == {"chunks": 0, "min": 0, "max": 0, "avg": 0}
    docs = [
        Document(page_content="a" * 100, metadata={}),
        Document(page_content="b" * 300, metadata={}),
    ]
    assert describe_chunks(docs) == {"chunks": 2, "min": 100, "max": 300, "avg": 200}
