"""index_builder.py -- stage 4 of the ingestion pipeline.

Takes the prose chunks produced by :mod:`text_splitter`, embeds them with
Ollama and stores them in PostgreSQL + pgvector so the retriever can search
them.

This stage does **only** embedding and storing.  It does not load, clean or
split documents -- those are upstream stages
(wikipedia_loader -> document_cleaner -> text_splitter).  Pass pre-split
chunks to :func:`build_index`; if none are given it raises a clear error
instead of silently running the other stages.

Only valid prose chunks are accepted.  Required metadata
(``title``, ``url``, ``section``, ``content_type="prose"``) is validated and
empty or malformed documents are rejected -- nothing is silently indexed.

Database access goes through **SQLAlchemy**: ``langchain-postgres`` already
depends on it and accepts a SQLAlchemy ``Engine`` as its ``connection``, so the
same engine drives the vector store *and* our own health/status queries.

Input  -> list[Document] from text_splitter.split_documents()
Output -> rows in settings.database.collection (table: langchain_pg_embedding)

Run the orchestrator (runs all 4 stages in order):
    python src/ingestion/run_ingestion.py

Run this stage alone (you must supply chunks):
    python -c "
    from src.ingestion.index_builder import build_index
    from src.ingestion.text_splitter import split_documents
    from src.ingestion.document_cleaner import clean_documents
    from src.ingestion.wikipedia_loader import fetch_wikipedia_documents
    chunks = split_documents(clean_documents(fetch_wikipedia_documents()))
    summary = build_index(chunks, reset=True)
    "
"""

from __future__ import annotations

import hashlib
import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.config import settings

logger = logging.getLogger(__name__)

# Required metadata fields on every chunk indexed into pgvector.
REQUIRED_METADATA_FIELDS = ("title", "url", "section", "content_type")

# How many chunks are embedded per request. Everything about the database,
# the models and the collection comes from .env -- this is only batch plumbing.
BATCH_SIZE = 32

# nomic-embed-text always returns the same vector length, so measure it once
# per run instead of hard-coding 768.
_embedding_dim: int | None = None


# --- Embedding wrapper with prefixes ---------------------------------------
class PrefixedEmbeddings(Embeddings):
    """Wraps OllamaEmbeddings and prepends a prefix to each text before embedding.

    EMBEDDING_DOC_PREFIX is added in embed_documents (indexing).
    EMBEDDING_QUERY_PREFIX is added in embed_query (retrieval).
    Both default to empty. Example for nomic-embed-text:
        search_document:  /  search_query:

    Changing the model or prefixes requires reset=True on re-index, because
    old vectors were computed with different text.
    """

    def __init__(self, base: OllamaEmbeddings, doc_prefix: str, query_prefix: str):
        self._base = base
        self._doc_prefix = doc_prefix or ""
        self._query_prefix = query_prefix or ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._doc_prefix + t for t in texts]
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._query_prefix + text)


# --- Connections -------------------------------------------------------------
def get_embeddings() -> PrefixedEmbeddings:
    """Embedding model, from OLLAMA_BASE_URL + EMBEDDING_MODEL in .env.

    Wrapped in PrefixedEmbeddings so EMBEDDING_DOC_PREFIX /
    EMBEDDING_QUERY_PREFIX are applied.
    """
    base = OllamaEmbeddings(
        model=settings.ollama.embedding_model,
        base_url=settings.ollama.base_url,
    )
    return PrefixedEmbeddings(
        base=base,
        doc_prefix=settings.ollama.embedding_doc_prefix,
        query_prefix=settings.ollama.embedding_query_prefix,
    )


def get_engine() -> Engine:
    """One SQLAlchemy engine built from the POSTGRES_* values in .env."""
    return create_engine(settings.database.url, pool_pre_ping=True, future=True)


def ensure_vector_extension(engine: Engine) -> str:
    """Make sure the pgvector extension exists; report what happened.

    Creating an extension needs superuser rights, so a permission error is
    turned into an actionable message rather than a raw traceback.
    """
    with engine.connect() as conn:
        if conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first():
            return "already installed"

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "pgvector is missing and this role cannot create it. Run this once as a "
            "superuser (e.g. 'postgres'):\n"
            f"    psql -U postgres -d {settings.database.name} "
            '-c "CREATE EXTENSION vector;"'
        ) from exc

    return "created"


def detect_embedding_dimension(embeddings: PrefixedEmbeddings) -> int:
    """Ask the model for one vector and read off its length."""
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = len(embeddings.embed_query("dimension probe"))
        logger.info("Embedding dimension: %d", _embedding_dim)
    return _embedding_dim


def get_vector_store(reset: bool = False) -> PGVector:
    """Create (or recreate) the pgvector store from .env settings."""
    engine = get_engine()
    extension = ensure_vector_extension(engine)

    store = PGVector(
        embeddings=get_embeddings(),
        connection=engine,                      # <-- SQLAlchemy Engine
        collection_name=settings.database.collection,
        embedding_length=detect_embedding_dimension(get_embeddings()),
        pre_delete_collection=reset,            # reset=True wipes it first
        create_extension=False,                 # we already handled it above
    )

    if reset:
        logger.warning("Collection '%s' was deleted and recreated.",
                       settings.database.collection)
    logger.info("pgvector %s; store ready for collection '%s'.",
                extension, settings.database.collection)
    return store


# --- Helpers for delete-before-insert --------------------------------------
def _source_identity(chunk: Document) -> str:
    """Return the source identifier for a chunk: url, then source_url, then origin."""
    return str(
        chunk.metadata.get("url")
        or chunk.metadata.get("source_url")
        or chunk.metadata.get("origin", "")
    )


def delete_existing_for_source(engine: Engine, source_url: str) -> int:
    """Delete all existing rows for a given source (url / source_url / origin).

    Returns the number of rows deleted. Used before re-indexing a document so
    that changed pages don't leave stale chunks behind.  Checks all three
    metadata keys for compatibility with both current (``url``) and older
    (``source_url`` / ``origin``) chunks.
    """
    sql = text(
        """
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE c.uuid = e.collection_id
          AND c.name = :collection
          AND (e.metadata->>'url' = :source
               OR e.metadata->>'source_url' = :source
               OR e.metadata->>'origin' = :source)
        """
    )
    with engine.begin() as conn:
        result = conn.execute(sql, {
            "collection": settings.database.collection,
            "source": source_url,
        })
    return result.rowcount or 0


def stable_chunk_id(chunk: Document) -> str:
    """Deterministic id so re-indexing *updates* rows instead of duplicating them.

    Includes a short hash of the chunk text so that if the text changes, the
    id changes and the old row is replaced by delete-before-insert.
    """
    source = _source_identity(chunk)
    chunk_index = chunk.metadata.get("chunk_index", 0)
    text_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    identity = f"{source}|{chunk_index}|{text_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _validate_chunk(chunk: Document) -> None:
    """Raise ValueError if a chunk is empty or missing required metadata."""
    if not chunk.page_content or not chunk.page_content.strip():
        raise ValueError(
            "Refusing to index an empty chunk. "
            f"title={chunk.metadata.get('title')!r} "
            f"section={chunk.metadata.get('section')!r}"
        )
    missing = [
        field for field in REQUIRED_METADATA_FIELDS
        if not chunk.metadata.get(field)
    ]
    if missing:
        raise ValueError(
            "Refusing to index a chunk missing required metadata "
            f"{missing}. title={chunk.metadata.get('title')!r} "
            f"url={chunk.metadata.get('url')!r} section={chunk.metadata.get('section')!r}"
        )
    if str(chunk.metadata.get("content_type", "")) != "prose":
        raise ValueError(
            f"Refusing to index a non-prose chunk "
            f"(content_type={chunk.metadata.get('content_type')!r}). "
            f"title={chunk.metadata.get('title')!r}"
        )


# --- Pipeline: embed + store only ------------------------------------------
def build_index(
    chunks: list[Document],
    *,
    reset: bool = False,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Embed chunks and write them to pgvector. Returns a summary dict.

    Args:
        chunks: pre-split prose chunks from text_splitter.split_documents().
            Required -- this stage does NOT load, clean or split.
        reset: delete the whole collection first -- use this when you want a
            clean rebuild, otherwise stale chunks from an older run survive.
        batch_size: how many chunks to embed per request.

    Raises:
        ValueError: if ``chunks`` is empty or None, or if any chunk is empty
            or missing required metadata.  This stage must not silently run
            stages 1-3 or index malformed documents.
    """
    if chunks is None or len(chunks) == 0:
        raise ValueError(
            "build_index() requires a non-empty list of chunks. "
            "Run stages 1-3 first (wikipedia_loader -> document_cleaner -> "
            "text_splitter), or use src.ingestion.run_ingestion to run the "
            "full pipeline: load -> clean -> split -> build_index."
        )

    # Validate every chunk before touching the database.
    for chunk in chunks:
        _validate_chunk(chunk)

    store = get_vector_store(reset=reset)
    engine = get_engine()

    # Group chunks by source so we can delete-before-insert per source.
    by_source: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_source.setdefault(_source_identity(chunk), []).append(chunk)

    written = 0
    for source, source_chunks in by_source.items():
        # Delete any existing rows for this source (stale chunk cleanup).
        if not reset:
            deleted = delete_existing_for_source(engine, source)
            if deleted:
                logger.info("Deleted %d stale row(s) for %s", deleted, source[:80])

        ids = [stable_chunk_id(chunk) for chunk in source_chunks]

        for start in range(0, len(source_chunks), batch_size):
            batch = source_chunks[start:start + batch_size]
            store.add_documents(batch, ids=ids[start:start + batch_size])
            written += len(batch)
            logger.info("Embedded %d/%d chunks...", written, len(chunks))

    summary = {
        "chunks": len(chunks),
        "written": written,
        "collection": settings.database.collection,
        "embedding_model": settings.ollama.embedding_model,
        "reset": reset,
    }
    logger.info("Index build complete: %s", summary)
    return summary


# --- Reporting (plain SQLAlchemy) -------------------------------------------
def index_status() -> dict:
    """How many chunks are stored? Straight SQLAlchemy, no LangChain involved."""
    sql = text(
        """
        SELECT
            (SELECT count(*) FROM langchain_pg_collection WHERE name = :name)
                AS collections,
            (SELECT count(*) FROM langchain_pg_embedding e
               JOIN langchain_pg_collection c ON c.uuid = e.collection_id
              WHERE c.name = :name)
                AS chunks
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"name": settings.database.collection}).mappings().one()
    return dict(row)
