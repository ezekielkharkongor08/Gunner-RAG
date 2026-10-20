"""Route a question that names a section straight to that section.

"Why does this exist?" A vector query such as "describe the things that happen
in the month of November" is weak: the month word is drowned out by the article
title, so the correct section can land at 0.71 while an unrelated chunk scores
0.28. When the question names a section that exists exactly once in the
collection (e.g. "November"), we skip similarity search and read that section
in document order. Nothing is hard-coded: the token list lives in
``config/settings.yaml`` under ``section_tokens``.

Matching rules:

* case and punctuation insensitive (``Pre-season`` == ``preseason``);
* a single-word token must appear as a whole word in the question, so "March"
  does not match "marching";
* a multi-word token may match with spaces or hyphens (``summer transfer
  window``);
* the token must equal any segment of ``section_path`` (``Kits`` matches
  ``Kits > Kit information``), or be the tail of the last segment
  (``First team transfers (summer transfer window)`` ends with the token);
* if more than one section matches, the router declines and normal vector
  search runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from sqlalchemy import text

from src.config import settings
from src.ingestion.index_builder import get_engine

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_SEPARATOR = ">"


@dataclass(frozen=True)
class SectionMatch:
    """One resolved section: its stored path and how many chunks it has."""

    path: str
    chunks: int


def normalise(value: str) -> str:
    """Lowercase and keep letters/digits only: 'Pre-season' -> 'preseason'."""
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def token_in_question(token: str, question: str) -> bool:
    """True when the token appears in the question as a word or phrase."""
    token_norm = normalise(token)
    if not token_norm:
        return False

    # A token with no space or hyphen is a single word: require a whole word.
    if " " not in token.strip() and "-" not in token:
        return token_norm in _WORD.findall(question.casefold())

    # A phrase may be written with spaces or hyphens: compare squeezed text.
    return token_norm in normalise(question)


def segment_matches(path: str, token: str) -> bool:
    """True when a segment of section_path equals the token, or the last one ends with it.

    Parent equality lets "Kits" match ``Kits > Kit information``; the suffix
    rule lets "summer transfer window" match a long leaf name.
    """
    parts = [part.strip() for part in path.split(_SEPARATOR)]
    token_norm = normalise(token)
    if not token_norm or not parts:
        return False
    if any(normalise(part) == token_norm for part in parts):
        return True
    return normalise(parts[-1]).endswith(token_norm)


def match_section(
    question: str,
    paths: Sequence[str],
    tokens: Sequence[str],
) -> str | None:
    """Return the single section path the question names, or None.

    The longest matching token wins, so "summer transfer window" is preferred
    over a shorter overlapping token. Several matching paths means ambiguity ->
    None (fall back to vector search).
    """
    for token in sorted(tokens, key=len, reverse=True):
        if not token_in_question(token, question):
            continue
        matches = [path for path in paths if segment_matches(path, token)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.info(
                "Section token %r is ambiguous (%d paths); using vector search.",
                token, len(matches),
            )
            return None
    return None


def list_sections(collection: str | None = None) -> list[SectionMatch]:
    """Every stored section_path with its chunk count, best-effort."""
    name = collection or settings.database.collection
    sql = text(
        """
        SELECT e.cmetadata->>'section_path' AS path, count(*) AS chunks
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :name
          AND e.cmetadata->>'section_path' IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(sql, {"name": name}).mappings().all()
    except Exception as exc:  # noqa: BLE001 - routing is an optimisation
        logger.warning("Could not list sections in %r: %s", name, exc)
        return []
    return [SectionMatch(path=str(row["path"]), chunks=int(row["chunks"])) for row in rows]


def fetch_section(
    path: str,
    collection: str | None = None,
    limit: int | None = None,
) -> list[Document]:
    """All chunks of one section, in document order, capped at ``limit``."""
    name = collection or settings.database.collection
    cap = limit if limit is not None else settings.retrieval.max_section_chunks
    sql = text(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :name AND e.cmetadata->>'section_path' = :path
        ORDER BY (e.cmetadata->>'chunk_index')::int NULLS LAST
        LIMIT :limit
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(
            sql, {"name": name, "path": path, "limit": cap}
        ).mappings().all()

    documents: list[Document] = []
    for row in rows:
        metadata = row["cmetadata"] or {}
        if not isinstance(metadata, dict):
            metadata = {}
        documents.append(
            Document(page_content=row["document"] or "", metadata=metadata)
        )
    return documents


def route(
    question: str,
    collection: str | None = None,
) -> tuple[str, list[Document]] | None:
    """Resolve a question to one section. Returns (path, documents) or None."""
    if not settings.retrieval.section_routing_enabled:
        return None

    tokens = settings.section_tokens
    if not tokens:
        return None

    sections = list_sections(collection)
    if not sections:
        return None

    path = match_section(question, [s.path for s in sections], tokens)
    if path is None:
        return None

    documents = fetch_section(path, collection=collection)
    if not documents:
        logger.info("Section %r matched but returned no chunks.", path)
        return None

    logger.info("Section route: %r -> %s (%d chunks)", question, path, len(documents))
    return path, documents
