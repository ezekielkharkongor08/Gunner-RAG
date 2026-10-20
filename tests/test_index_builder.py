"""Unit tests for index_builder.py (validation only; no PostgreSQL needed)."""

from __future__ import annotations

import pytest

from langchain_core.documents import Document

from src.ingestion.index_builder import (
    PrefixedEmbeddings,
    build_index,
    stable_chunk_id,
    REQUIRED_METADATA_FIELDS,
)


def _make_chunk(text="Some prose content.", **meta):
    base = {
        "source": "wikipedia",
        "title": "Arsenal F.C.",
        "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
        "section": "History",
        "section_path": "History",
        "language": "en",
        "content_type": "prose",
    }
    base.update(meta)
    return Document(page_content=text, metadata=base)


def test_build_index_rejects_none():
    with pytest.raises(ValueError, match="non-empty list"):
        build_index(None)


def test_build_index_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        build_index([])


def test_build_index_rejects_empty_chunk_text():
    with pytest.raises(ValueError, match="empty chunk"):
        build_index([_make_chunk(text="")])


def test_build_index_rejects_missing_required_metadata():
    for field in REQUIRED_METADATA_FIELDS:
        meta = {
            "source": "wikipedia",
            "title": "Arsenal F.C.",
            "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
            "section": "History",
            "section_path": "History",
            "language": "en",
            "content_type": "prose",
        }
        del meta[field]
        with pytest.raises(ValueError, match="missing required metadata"):
            build_index([Document(page_content="Some prose content.", metadata=meta)])


def test_build_index_rejects_non_prose_content_type():
    with pytest.raises(ValueError, match="non-prose"):
        build_index([_make_chunk(content_type="table")])


def test_stable_chunk_id_is_deterministic_and_text_sensitive():
    chunk = _make_chunk()
    id1 = stable_chunk_id(chunk)
    id2 = stable_chunk_id(chunk)
    assert id1 == id2
    assert len(id1) == 32

    # Changing the text changes the id.
    chunk2 = _make_chunk(text="Different prose content.")
    assert stable_chunk_id(chunk2) != id1


def test_stable_chunk_id_is_scoped_to_collection():
    chunk = _make_chunk()
    assert stable_chunk_id(chunk, "arsenal_eval_a") != stable_chunk_id(
        chunk, "arsenal_eval_b"
    )


def test_required_metadata_fields():
    assert "title" in REQUIRED_METADATA_FIELDS
    assert "url" in REQUIRED_METADATA_FIELDS
    assert "section" in REQUIRED_METADATA_FIELDS
    assert "content_type" in REQUIRED_METADATA_FIELDS


# ---------------------------------------------------------------------------
# 2.2 Qwen3 query instruction: queries only, documents untouched
# ---------------------------------------------------------------------------
class _FakeBaseEmbeddings:
    """Stand-in for OllamaEmbeddings that records what it was asked to embed."""

    def __init__(self):
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [0.0]


def test_query_instruction_is_applied_to_queries_only():
    base = _FakeBaseEmbeddings()
    embeddings = PrefixedEmbeddings(
        base,
        doc_prefix="",
        query_prefix="",
        query_instruction="Given a question about Arsenal F.C., retrieve passages",
        query_instruction_enabled=True,
    )

    embeddings.embed_documents(["passage text"])
    embeddings.embed_query("who won the league?")

    assert base.documents == ["passage text"]
    assert base.queries == [
        "Instruct: Given a question about Arsenal F.C., retrieve passages\n"
        "Query:who won the league?"
    ]


def test_query_instruction_off_leaves_the_query_plain():
    base = _FakeBaseEmbeddings()
    embeddings = PrefixedEmbeddings(
        base,
        doc_prefix="",
        query_prefix="",
        query_instruction="ignored",
        query_instruction_enabled=False,
    )

    embeddings.embed_query("who won the league?")
    assert base.queries == ["who won the league?"]


def test_doc_prefix_still_applies_when_configured():
    base = _FakeBaseEmbeddings()
    embeddings = PrefixedEmbeddings(
        base, doc_prefix="search_document: ", query_prefix="search_query: ",
    )
    embeddings.embed_documents(["d"])
    embeddings.embed_query("q")
    assert base.documents == ["search_document: d"]
    assert base.queries == ["search_query: q"]
