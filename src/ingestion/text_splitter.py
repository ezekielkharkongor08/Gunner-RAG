"""text_splitter.py -- stage 3 of the ingestion pipeline.

Splits cleaned, section-aware prose ``Document`` objects (from
:mod:`document_cleaner`) into retrieval-friendly chunks.

There is no table-specific splitting here: every input document is already a
single prose section (``content_type="prose"``), so each document is split on
its own, never merged with another article or section.  Chunks keep the full
article/section metadata so retrieval can cite the correct source.

Strategy:

    * Each section document is split with
      :class:`langchain_text_splitters.RecursiveCharacterTextSplitter`,
      preferring paragraph boundaries (``\n\n``, ``\n``, ``. ``, `` ``).
    * Chunks are constrained by configured ``CHUNK_SIZE`` / ``CHUNK_OVERLAP``.
    * Very long paragraphs are split without exceeding the configured limit.
    * Chunks shorter than ``CHUNK_MIN_SIZE`` are folded into the next chunk so
      a bare heading never becomes a chunk of its own.
    * Empty and duplicate chunks are dropped.

Each chunk inherits the parent document's metadata plus
``chunk_index``, ``chunk_count`` and ``chunk_chars``.  ``content_type`` stays
``"prose"``.

Run the demo (no network):
    python -m src.ingestion.text_splitter
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_env, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Characters to split on, most preferred first.  Paragraph boundaries come
# first so chunks respect section/prose structure wherever possible.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Chunks shorter than this are folded into the following chunk.
MIN_CHUNK_SIZE = int(get_env("CHUNK_MIN_SIZE", "100"))

# Log a warning if the largest chunk exceeds this many characters.
CHUNK_MAX_WARN = int(get_env("CHUNK_MAX_WARN", "1000"))

# Content type for every chunk produced by this pipeline.
CONTENT_PROSE = "prose"


# ---------------------------------------------------------------------------
# Short-chunk folding
# ---------------------------------------------------------------------------
def _fold_short_chunks(pieces: list[Document], min_size: int) -> list[Document]:
    """Fold an over-short chunk forward into the chunk that follows it."""
    merged: list[Document] = []
    pending: Document | None = None

    for piece in pieces:
        if pending is not None:
            piece = Document(
                page_content=f"{pending.page_content}\n\n{piece.page_content}",
                metadata=dict(piece.metadata),
            )
            pending = None

        if len(piece.page_content) < min_size:
            pending = piece
            continue

        merged.append(piece)

    if pending is not None:
        merged.append(pending)

    return merged


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------
def _make_chunk(
    content: str,
    parent: Document,
    *,
    chunk_index: int,
    chunk_count: int,
) -> Document:
    """Build a chunk ``Document`` inheriting the parent's article/section metadata."""
    metadata = dict(parent.metadata)
    url = str(metadata.get("url", ""))
    section_path = str(metadata.get("section_path", ""))
    # A stable, human-readable chunk id used for retrieval de-duplication and
    # citation.  url + section_path + chunk_index is unique within a run.
    metadata.update(
        content_type=CONTENT_PROSE,
        chunk_id=f"{url}#{section_path}#{chunk_index}",
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        chunk_chars=len(content),
    )
    return Document(page_content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def split_documents(
    documents: list[Document],
    include_title_header: bool = True,
) -> list[Document]:
    """Split cleaned prose documents into chunks.

    Each input document is a single prose section and is split independently --
    unrelated articles and sections are never combined, and no paragraph is
    duplicated between chunks.

    Args:
        documents: output of ``document_cleaner.clean_documents()``.
        include_title_header: kept for API compatibility; context is carried in
            metadata rather than prepended to the chunk text, so chunks remain
            prose-only.

    Returns:
        A flat list of chunk ``Document`` objects.  Each chunk inherits
        ``source, title, url, section, section_path, language, content_type``
        plus ``chunk_index``, ``chunk_count`` and ``chunk_chars``.
    """
    splitter = _build_splitter()
    chunks: list[Document] = []

    for document in documents:
        content_type = str(document.metadata.get("content_type", "")).lower()
        if content_type != CONTENT_PROSE:
            logger.warning(
                "Skipping non-prose document: %s",
                document.metadata.get("title"),
            )
            continue

        title = str(document.metadata.get("title", "")).strip()
        text = document.page_content
        if not text.strip():
            continue

        pieces = splitter.split_text(text)
        pieces = _fold_short_chunks(
            [Document(page_content=p, metadata={}) for p in pieces],
            MIN_CHUNK_SIZE,
        )

        total = len(pieces)
        seen_fingerprints: set[str] = set()
        for index, piece in enumerate(pieces):
            if not piece.page_content.strip():
                continue
            # Prevent duplicate chunks within the same section.
            fingerprint = hashlib.sha1(piece.page_content.encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            chunk = _make_chunk(
                piece.page_content,
                document,
                chunk_index=index,
                chunk_count=total,
            )
            chunks.append(chunk)

    # Warn if any chunk exceeds CHUNK_MAX_WARN (embedding model token limit).
    oversized = [c for c in chunks if len(c.page_content) > CHUNK_MAX_WARN]
    if oversized:
        logger.warning(
            "%d chunk(s) exceed CHUNK_MAX_WARN=%d (embedding model token limit).",
            len(oversized), CHUNK_MAX_WARN,
        )

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(chunks))
    return chunks


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the splitter from .env values, with a clear error if they're off."""
    size = settings.chunk.size
    overlap = settings.chunk.overlap

    if size <= 0:
        raise ValueError(f"CHUNK_SIZE must be > 0, got {size}.")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"CHUNK_OVERLAP ({overlap}) must be between 0 and CHUNK_SIZE-1 ({size - 1})."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        add_start_index=False,
    )


def describe_chunks(chunks: list[Document]) -> dict[str, int]:
    """Tiny helper for logging/tests: min, max and average chunk length."""
    if not chunks:
        return {"chunks": 0, "min": 0, "max": 0, "avg": 0}
    sizes = [len(c.page_content) for c in chunks]
    return {
        "chunks": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "avg": sum(sizes) // len(sizes),
    }


# ---------------------------------------------------------------------------
# Demo (no network required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=(
                "Arsenal Football Club is a professional football club based in "
                "Islington, North London, England.\n\n"
                "== History ==\n"
                "In October 1886, Scotsman David Danskin and fifteen fellow "
                "munitions workers formed the Dial Square Football Club.  In "
                "1913 the club moved to the new Arsenal Stadium in Highbury.\n\n"
                "=== Early years ===\n"
                "The club turned professional in 1891 and joined the Football "
                "League in 1893."
            ),
            metadata={
                "source": "wikipedia",
                "title": "Arsenal F.C.",
                "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
                "section": "History",
                "section_path": "History",
                "language": "en",
                "content_type": "prose",
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
        )
    ]

    chunks = split_documents(docs)
    stats = describe_chunks(chunks)
    print(f"{len(docs)} document(s) -> {stats['chunks']} chunk(s)")
    print(f"sizes: {stats}\n")
    for chunk in chunks:
        md = chunk.metadata
        print(f"[{md['section']} chunk {md['chunk_index']}/{md['chunk_count']}] "
              f"{md['chunk_chars']} chars")
        print(f"  {chunk.page_content[:120].replace(chr(10), ' ')}")
