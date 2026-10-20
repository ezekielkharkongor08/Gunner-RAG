"""text_splitter.py -- stage 3 of the ingestion pipeline.

Splits cleaned, assembled section documents into retrieval-friendly chunks.

Every chunk starts with a small header so the embedding knows which page and
section it came from::

    <page title>
    Section: <full section path>

    <body text>

The header is added **after** splitting, so it is never duplicated. It does
not count toward ``CHUNK_SIZE``: ``CHUNK_SIZE`` is the body budget, and the
header is extra. ``CHUNK_MAX_WARN`` warns about the total chunk length.

Split strategy:

    * separators come from ``split_separators`` in settings.yaml, most
      preferred first (``\\n\\n`` before ``\\n`` before ``. `` ...)
    * ``keep_separator="end"`` means a chunk never starts with ". "
    * a section body that fits in CHUNK_SIZE stays one chunk
    * a very short tail is folded into the previous chunk (never dropped)
    * duplicate chunks inside one section are removed by SHA-1

Metadata added here: ``chunk_id``, ``chunk_index``, ``chunk_count``,
``chunk_chars``, ``body_chars``, ``header_chars``, ``content_hash``.

Run the demo (no network):
    python -m src.ingestion.text_splitter
"""

from __future__ import annotations

import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

# Content type for every chunk produced by this pipeline.
CONTENT_PROSE = "prose"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def build_header(document: Document) -> str:
    """Return the title + section header for a chunk, or "" when disabled."""
    if not settings.chunk.header_enabled:
        return ""
    title = str(document.metadata.get("title", "")).strip()
    section_path = str(document.metadata.get("section_path", "")).strip()
    return f"{title}\nSection: {section_path}\n\n"


# ---------------------------------------------------------------------------
# Short-chunk folding
# ---------------------------------------------------------------------------
def _fold_short_chunks(
    pieces: list[str], min_size: int, max_body: int
) -> list[str]:
    """Fold a too-short piece into its neighbour.

    A short piece is carried forward into the next piece. A short *last*
    piece is folded back into the previous chunk when it fits. Text is never
    dropped.
    """
    merged: list[str] = []
    carry = ""

    for index, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        if carry:
            piece = f"{carry}\n\n{piece}"
            carry = ""

        if len(piece) >= min_size:
            merged.append(piece)
            continue

        if index == len(pieces) - 1:
            # Last piece: fold backwards so the tail keeps its facts.
            if merged and len(f"{merged[-1]}\n\n{piece}") <= max_body:
                merged[-1] = f"{merged[-1]}\n\n{piece}"
            else:
                merged.append(piece)
                logger.warning(
                    "Kept a short final chunk (%d < %d chars) because folding "
                    "it back would exceed %d chars.",
                    len(piece), min_size, max_body,
                )
        else:
            carry = piece

    if carry:
        # Only reachable if every following piece was blank.
        if merged and len(f"{merged[-1]}\n\n{carry}") <= max_body:
            merged[-1] = f"{merged[-1]}\n\n{carry}"
        else:
            merged.append(carry)

    return merged


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------
def _make_chunk(
    body: str,
    header: str,
    parent: Document,
    *,
    chunk_index: int,
    chunk_count: int,
) -> Document:
    """Build a chunk Document with the header + inherited metadata."""
    content = f"{header}{body}".strip()
    metadata = dict(parent.metadata)
    url = str(metadata.get("url", ""))
    section_path = str(metadata.get("section_path", ""))
    metadata.update(
        content_type=CONTENT_PROSE,
        chunk_id=f"{url}#{section_path}#{chunk_index}",
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        chunk_chars=len(content),
        body_chars=len(body),
        header_chars=len(header),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )
    return Document(page_content=content, metadata=metadata)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def split_documents(documents: list[Document]) -> list[Document]:
    """Split assembled prose sections into chunks.

    Each input document is one section and is split independently. The
    title/section header is prepended to every piece after splitting.
    """
    splitter = _build_splitter()
    chunks: list[Document] = []
    body_limit = settings.chunk.size
    min_size = settings.chunk.min_size
    max_body = settings.chunk.size + settings.chunk.overlap

    for document in documents:
        content_type = str(document.metadata.get("content_type", "")).lower()
        if content_type != CONTENT_PROSE:
            logger.warning(
                "Skipping non-prose document: %s",
                document.metadata.get("title"),
            )
            continue

        body = document.page_content.strip()
        if not body:
            continue

        header = build_header(document)
        pieces = (
            [body] if len(body) <= body_limit else splitter.split_text(body)
        )
        pieces = _fold_short_chunks(
            [piece.strip() for piece in pieces if piece.strip()],
            min_size,
            max_body,
        )

        total = len(pieces)
        seen_fingerprints: set[str] = set()
        for index, piece in enumerate(pieces):
            # Prevent duplicate chunks within the same section.
            fingerprint = hashlib.sha1(piece.encode("utf-8")).hexdigest()
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            chunks.append(
                _make_chunk(
                    piece,
                    header,
                    document,
                    chunk_index=index,
                    chunk_count=total,
                )
            )

    oversized = [
        chunk for chunk in chunks
        if len(chunk.page_content) > settings.chunk.max_warn
    ]
    if oversized:
        logger.warning(
            "%d chunk(s) exceed CHUNK_MAX_WARN=%d (embedding model limit).",
            len(oversized), settings.chunk.max_warn,
        )

    logger.info(
        "Split %d document(s) into %d chunk(s).", len(documents), len(chunks)
    )
    return chunks


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the splitter from config, with a clear error if values are off."""
    size = settings.chunk.size
    overlap = settings.chunk.overlap

    if size <= 0:
        raise ValueError(f"CHUNK_SIZE must be > 0, got {size}.")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"CHUNK_OVERLAP ({overlap}) must be between 0 and CHUNK_SIZE-1 "
            f"({size - 1})."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=list(settings.chunk.separators),
        keep_separator="end",
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

    docs = [
        Document(
            page_content=(
                "Arsenal Football Club is a professional football club based in "
                "Islington, North London, England.\n\n"
                "In October 1886, Scotsman David Danskin and fifteen fellow "
                "munitions workers formed the Dial Square Football Club."
            ),
            metadata={
                "source": "wikipedia",
                "title": "Arsenal F.C.",
                "url": "https://en.wikipedia.org/wiki/Arsenal_F.C.",
                "section": "Introduction",
                "section_path": "Introduction",
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
        print(f"[{md['section_path']} chunk {md['chunk_index']}/{md['chunk_count']}] "
              f"{md['chunk_chars']} chars")
        print(chunk.page_content[:200])
