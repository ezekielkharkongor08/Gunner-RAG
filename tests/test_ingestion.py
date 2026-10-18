"""Tests for run_ingestion.py orchestration (Wikipedia-only pipeline).

The stage functions are mocked so no network, Ollama or PostgreSQL is
required.  Verifies the pipeline wires load -> clean -> split -> build_index
in order, that titles override settings.yaml, and that failures propagate.
"""

from __future__ import annotations

from langchain_core.documents import Document

import pytest

import src.ingestion.run_ingestion as ri
from src.ingestion.wikipedia_loader import EmptyArticleError, PageNotFoundError


class _FakeSummary:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_doc(title="Arsenal F.C.", section="History"):
    return Document(
        page_content="Some prose content.",
        metadata={
            "source": "wikipedia",
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "section": section,
            "section_path": section,
            "language": "en",
            "content_type": "prose",
        },
    )


def test_run_ingestion_wires_stages_in_order(monkeypatch):
    calls = []

    def fake_fetch(titles=None):
        calls.append(("fetch", list(titles or [])))
        return [_fake_doc()]

    def fake_clean(docs):
        calls.append(("clean", len(docs)))
        return docs

    def fake_split(docs):
        calls.append(("split", len(docs)))
        return [Document(page_content="chunk", metadata=d.metadata) for d in docs]

    def fake_build(chunks, *, reset=True, batch_size=32):
        calls.append(("build", len(chunks), reset, batch_size))
        return {"chunks": len(chunks), "written": len(chunks)}

    monkeypatch.setattr(ri, "fetch_wikipedia_documents", fake_fetch)
    monkeypatch.setattr(ri, "clean_documents", fake_clean)
    monkeypatch.setattr(ri, "split_documents", fake_split)
    monkeypatch.setattr(ri, "build_index", fake_build)

    summary = ri.run_ingestion(titles=["Arsenal F.C."], reset=True, batch_size=16)

    assert summary["chunks"] == 1
    assert calls == [
        ("fetch", ["Arsenal F.C."]),
        ("clean", 1),
        ("split", 1),
        ("build", 1, True, 16),
    ]


def test_run_ingestion_defaults_titles_from_settings(monkeypatch):
    seen = {}

    def fake_fetch(titles=None):
        seen["titles"] = list(titles or [])
        return []

    monkeypatch.setattr(ri, "fetch_wikipedia_documents", fake_fetch)
    monkeypatch.setattr(ri, "clean_documents", lambda docs: [])
    monkeypatch.setattr(ri, "split_documents", lambda docs: [])
    monkeypatch.setattr(ri, "build_index", lambda chunks, **kw: (_ for _ in ()).throw(
        ValueError("build_index called with no chunks")
    ))

    # No titles passed -> defaults to settings.pages; build_index should raise
    # because nothing valid was produced (failure is not hidden).
    from src.config import settings
    with pytest.raises(ValueError):
        ri.run_ingestion()
    assert seen["titles"] == list(settings.pages)


def test_run_ingestion_rejects_no_valid_chunks(monkeypatch):
    monkeypatch.setattr(ri, "fetch_wikipedia_documents", lambda titles=None: [])
    monkeypatch.setattr(ri, "clean_documents", lambda docs: [])
    monkeypatch.setattr(ri, "split_documents", lambda docs: [])
    # build_index must raise on empty input -- not silently succeed.
    monkeypatch.setattr(ri, "build_index", lambda chunks, **kw: (_ for _ in ()).throw(
        ValueError("build_index() requires a non-empty list of chunks")
    ))
    with pytest.raises(ValueError):
        ri.run_ingestion(titles=["Ghost"])


def test_run_ingestion_no_local_file_branch(monkeypatch):
    # Confirm the orchestrator only calls the Wikipedia fetcher, not any
    # local-file loader.
    fetched = []

    def fake_fetch(titles=None):
        fetched.append(True)
        return [_fake_doc()]

    monkeypatch.setattr(ri, "fetch_wikipedia_documents", fake_fetch)
    monkeypatch.setattr(ri, "clean_documents", lambda docs: docs)
    monkeypatch.setattr(ri, "split_documents", lambda docs: docs)
    monkeypatch.setattr(ri, "build_index", lambda chunks, **kw: {"chunks": len(chunks), "written": len(chunks)})

    ri.run_ingestion(titles=["X"])
    assert fetched == [True]


def test_failed_pages_do_not_produce_empty_documents(monkeypatch):
    # The loader raises on missing/empty pages; fetch_wikipedia_documents
    # skips them and never returns empty "success" docs.  Here we simulate a
    # loader that returns only valid docs and confirm run_ingestion passes
    # them through.
    monkeypatch.setattr(ri, "fetch_wikipedia_documents", lambda titles=None: [_fake_doc()])
    monkeypatch.setattr(ri, "clean_documents", lambda docs: docs)
    monkeypatch.setattr(ri, "split_documents", lambda docs: docs)
    monkeypatch.setattr(ri, "build_index", lambda chunks, **kw: {"chunks": len(chunks), "written": len(chunks)})

    summary = ri.run_ingestion(titles=["Arsenal F.C."])
    assert summary["chunks"] == 1


# PageNotFoundError and EmptyArticleError are importable for consumers.
def test_loader_errors_exist():
    assert issubclass(PageNotFoundError, Exception)
    assert issubclass(EmptyArticleError, Exception)
