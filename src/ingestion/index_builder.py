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

# nomic-embed-text always returns the same vector length, so measure it once
# per run instead of hard-coding 768.
_embedding_dim: int | None = None


# --- Embedding wrapper with prefixes ---------------------------------------
class PrefixedEmbeddings(Embeddings):
    """Wraps OllamaEmbeddings and adds the configured prefixes.

    Documents get ``EMBEDDING_DOC_PREFIX`` (default empty).
    Queries get ``EMBEDDING_QUERY_PREFIX`` plus, when enabled, the Qwen3
    instruction in the form ``Instruct: <task>\\nQuery:<text>``.

    Changing the model or prefixes requires reset=True on re-index, because
    old vectors were computed with different text.
    """

    def __init__(
        self,
        base: OllamaEmbeddings,
        doc_prefix: str,
        query_prefix: str,
        query_instruction: str = "",
        query_instruction_enabled: bool = False,
    ):
        self._base = base
        self._doc_prefix = doc_prefix or ""
        self._query_prefix = query_prefix or ""
        self._query_instruction = query_instruction or ""
        self._query_instruction_enabled = query_instruction_enabled

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._doc_prefix + t for t in texts]
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        query = self._query_prefix + text
        if self._query_instruction_enabled and self._query_instruction:
            query = f"Instruct: {self._query_instruction}\nQuery:{query}"
        return self._base.embed_query(query)


# --- Connections -------------------------------------------------------------
def get_embeddings(instruction_enabled: bool | None = None) -> PrefixedEmbeddings:
    """Embedding model, from OLLAMA_BASE_URL + EMBEDDING_MODEL in .env.

    ``instruction_enabled`` overrides the config flag (used by the eval to
    compare query instruction on/off).
    """
    base = OllamaEmbeddings(
        model=settings.ollama.embedding_model,
        base_url=settings.ollama.base_url,
    )
    enabled = (
        settings.ollama.embedding_query_instruction_enabled
        if instruction_enabled is None
        else instruction_enabled
    )
    return PrefixedEmbeddings(
        base=base,
        doc_prefix=settings.ollama.embedding_doc_prefix,
        query_prefix=settings.ollama.embedding_query_prefix,
        query_instruction=settings.ollama.embedding_query_instruction,
        query_instruction_enabled=enabled,
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


def get_vector_store(
    reset: bool = False,
    collection: str | None = None,
    embeddings: PrefixedEmbeddings | None = None,
) -> PGVector:
    """Create (or recreate) the pgvector store from .env settings.

    ``collection`` overrides PGVECTOR_COLLECTION for eval runs. The eval only
    reads from ``arsenal_eval_*`` collections.
    """
    engine = get_engine()
    extension = ensure_vector_extension(engine)
    embedder = embeddings or get_embeddings()
    collection_name = collection or settings.database.collection

    store = PGVector(
        embeddings=embedder,
        connection=engine,                      # <-- SQLAlchemy Engine
        collection_name=collection_name,
        embedding_length=detect_embedding_dimension(embedder),
        pre_delete_collection=reset,            # reset=True wipes it first
        create_extension=False,                 # we already handled it above
    )

    if reset:
        logger.warning("Collection '%s' was deleted and recreated.",
                       collection_name)
    logger.info("pgvector %s; store ready for collection '%s'.",
                extension, collection_name)
    return store


# --- Helpers for delete-before-insert --------------------------------------
def _source_identity(chunk: Document) -> str:
    """Return the source identifier for a chunk: url, then source_url, then origin."""
    return str(
        chunk.metadata.get("url")
        or chunk.metadata.get("source_url")
        or chunk.metadata.get("origin", "")
    )


def delete_existing_for_source(
    engine: Engine, source_url: str, collection: str | None = None
) -> int:
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
          AND (e.cmetadata->>'url' = :source
               OR e.cmetadata->>'source_url' = :source
               OR e.cmetadata->>'origin' = :source)
        """
    )
    with engine.begin() as conn:
        result = conn.execute(sql, {
            "collection": collection or settings.database.collection,
            "source": source_url,
        })
    return result.rowcount or 0


def stable_chunk_id(chunk: Document, collection: str = "") -> str:
    """Deterministic id so re-indexing *updates* rows instead of duplicating them.

    Includes a short hash of the chunk text so that if the text changes, the
    id changes and the old row is replaced by delete-before-insert.

    ``collection`` is part of the id so the same chunk in two eval collections
    cannot overwrite the other collection's row. The database conflict target
    is only ``id``, not ``(collection_id, id)``.
    """
    source = _source_identity(chunk)
    chunk_index = chunk.metadata.get("chunk_index", 0)
    text_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    identity = f"{collection}|{source}|{chunk_index}|{text_hash}"
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
    batch_size: int | None = None,
    collection: str | None = None,
) -> dict:
    """Embed chunks and write them to pgvector. Returns a summary dict.

    Args:
        chunks: pre-split prose chunks from text_splitter.split_documents().
            Required -- this stage does NOT load, clean or split.
        reset: delete the whole collection first -- use this when you want a
            clean rebuild, otherwise stale chunks from an older run survive.
        batch_size: how many chunks to embed per request. Defaults to
            INDEX_BATCH_SIZE.
        collection: collection to write to. Defaults to PGVECTOR_COLLECTION.

    Raises:
        ValueError: if ``chunks`` is empty or None, or if any chunk is empty
            or missing required metadata.
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

    collection_name = collection or settings.database.collection
    batch = batch_size or settings.chunk.batch_size
    store = get_vector_store(reset=reset, collection=collection_name)
    engine = get_engine()

    # Group chunks by source so we can delete-before-insert per source.
    by_source: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_source.setdefault(_source_identity(chunk), []).append(chunk)

    written = 0
    for source, source_chunks in by_source.items():
        # Delete any existing rows for this source (stale chunk cleanup).
        if not reset:
            deleted = delete_existing_for_source(
                engine, source, collection=collection_name
            )
            if deleted:
                logger.info("Deleted %d stale row(s) for %s", deleted, source[:80])

        ids = [
            stable_chunk_id(chunk, collection=collection_name)
            for chunk in source_chunks
        ]

        for start in range(0, len(source_chunks), batch):
            batch_docs = source_chunks[start:start + batch]
            store.add_documents(batch_docs, ids=ids[start:start + batch])
            written += len(batch_docs)
            logger.info("Embedded %d/%d chunks...", written, len(chunks))

    summary = {
        "chunks": len(chunks),
        "written": written,
        "collection": collection_name,
        "embedding_model": settings.ollama.embedding_model,
        "reset": reset,
    }
    logger.info("Index build complete: %s", summary)
    return summary


# --- Reporting (plain SQLAlchemy) -------------------------------------------
def index_status(collection: str | None = None) -> dict:
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
    name = collection or settings.database.collection
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"name": name}).mappings().one()
    return dict(row)
