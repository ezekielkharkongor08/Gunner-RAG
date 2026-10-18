"""Unit tests for retrieval/retriever.py (prose-only, no table logic)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from langchain_core.documents import Document

import src.retrieval.retriever as retr
from src.retrieval.retriever import RetrievedChunk, _doc_to_chunk, rewrite_queries


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        return _FakeLLMResponse("\n".join(self.variants))


# ---------------------------------------------------------------------------
# _doc_to_chunk / RetrievedChunk / citation
# ---------------------------------------------------------------------------
def test_doc_to_chunk_maps_metadata_including_url():
    doc = Document(
        page_content="Arsenal moved to Emirates in 2006.",
        metadata={
            "chunk_id": "https://en.wikipedia.org/wiki/Arsenal_F.C.#History#0",
            "title": "Arsenal F.C.",
            "section": "History",
            "section_path": "History",
            "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
            "language": "en",
            "content_type": "prose",
            "chunk_index": 0,
            "chunk_count": 2,
        },
    )
    chunk = _doc_to_chunk(doc, 0.25)
    assert chunk.chunk_id == "https://en.wikipedia.org/wiki/Arsenal_F.C.#History#0"
    assert chunk.title == "Arsenal F.C."
    assert chunk.section == "History"
    assert chunk.url == "https://en.wikipedia.org/wiki/Arsenal_F.C."
    assert chunk.content_type == "prose"
    assert chunk.text == "Arsenal moved to Emirates in 2006."
    assert chunk.score == 0.25


def test_citation_includes_title_section_and_url():
    chunk = RetrievedChunk(
        chunk_id="id",
        title="Arsenal F.C.",
        section="History > Early years",
        url="https://en.wikipedia.org/wiki/Arsenal_F.C.",
        content_type="prose",
        text="...",
        score=0.1,
    )
    cited = chunk.citation()
    assert "Arsenal F.C." in cited
    assert "section: History > Early years" in cited
    assert "source: https://en.wikipedia.org/wiki/Arsenal_F.C." in cited


def test_retrieved_chunk_has_no_table_specific_fields():
    # RetrievedChunk should not carry table_name / row_index / table_caption /
    # is_table() -- those concepts were removed from the prose-only pipeline.
    chunk = RetrievedChunk(
        chunk_id="id", title="T", section="S", url="U", content_type="prose",
        text="...", score=0.1,
    )
    assert not hasattr(chunk, "table_name")
    assert not hasattr(chunk, "row_index")
    assert not hasattr(chunk, "table_caption")
    assert not hasattr(chunk, "is_table")
    # Frozen dataclass: cannot add stray fields.
    with pytest.raises(FrozenInstanceError):
        chunk.table_name = "x"


# ---------------------------------------------------------------------------
# rewrite_queries
# ---------------------------------------------------------------------------
def test_rewrite_queries_includes_original_first(monkeypatch):
    class _FakeLLM:
        def invoke(self, prompt):
            return _FakeLLMResponse(
                "Who won the league in 1998?\nArsenal 1998 manager"
            )

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _FakeLLM())

    queries = rewrite_queries("Who managed Arsenal in 1998?")
    assert queries[0] == "Who managed Arsenal in 1998?"
    assert "Who won the league in 1998?" in queries
    assert len(queries) <= 1 + retr.settings.retrieval.query_rewrite_count


def test_rewrite_queries_falls_back_to_original_on_llm_failure(monkeypatch):
    class _BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _BrokenLLM())
    queries = rewrite_queries("Any question?")
    assert queries == ["Any question?"]


def test_rewrite_prompt_is_generic_not_arsenal_specific(monkeypatch):
    captured = {}

    class _CaptureLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _FakeLLMResponse("")

    monkeypatch.setattr(retr, "ChatOllama", lambda *a, **k: _CaptureLLM())
    rewrite_queries("Who was Arsenal's top scorer?")
    prompt = captured["prompt"]
    assert "Arsenal F.C. documents" not in prompt
    assert "Wikipedia articles" in prompt
    assert "Who was Arsenal's top scorer?" in prompt
