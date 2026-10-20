"""retriever.py -- retrieval stage for the prose-only Wikipedia RAG pipeline.

Takes a user question (plus conversation history), routes section questions
straight to that section, otherwise rewrites the question into query variants,
embeds them, searches pgvector, merges and de-duplicates results, applies the
configured gate, and returns the top chunks with citation metadata.

All settings come from ``.env`` / ``config/settings.yaml``:

    QUERY_REWRITE_COUNT  -- how many query variants to generate (default 3)
    RETRIEVE_K           -- how many chunks to fetch per query variant (default 15)
    FINAL_TOP_N          -- how many chunks to return after merging (default 8)
    RETRIEVAL_GATE       -- relative | absolute | none (default relative)
    RELATIVE_MARGIN      -- relative gate width (default 0.10)
    SCORE_THRESHOLD      -- absolute gate ceiling (default 0.7)

Pipeline:

    0. Route questions that name a section (tokens from settings.yaml).
    1. rewrite_queries(question, history) -> list of query strings
    2. Embed each query with the same Ollama model used at ingestion time
       (with EMBEDDING_QUERY_PREFIX applied).
    3. similarity_search for each query, collect (Document, score) pairs.
    4. Merge + de-duplicate by chunk_id, keeping the best score per chunk.
    5. Apply the gate (relative by default).
    6. Return the top FINAL_TOP_N chunks with citation metadata.

Chunk metadata (from ingestion):

    * title, section, section_path, url, language, content_type="prose"
    * chunk_index, chunk_count, chunk_chars, fetched_at

Run a quick smoke test:
    python -m src.retrieval.retriever "Who was Arsenal's top scorer?"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sqlalchemy import text

from src.config import settings
from src.ingestion.index_builder import get_embeddings, get_engine, get_vector_store
from src.retrieval.section_router import route as route_section
from src.text_utils import normalize_season_text

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (the usual 60 from the RRF paper).
RRF_K = 60

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the retriever, with citation metadata."""

    chunk_id: str
    title: str
    section: str
    url: str
    content_type: str
    text: str
    score: float
    header_chars: int = 0
    fetched_at: str = ""

    def citation(self) -> str:
        """Short citation string for displaying where the chunk came from."""
        parts = [self.title]
        if self.section:
            parts.append(f"section: {self.section}")
        if self.url:
            parts.append(f"source: {self.url}")
        if self.fetched_at:
            parts.append(f"fetched: {self.fetched_at}")
        return " | ".join(parts)

    def body_text(self) -> str:
        """Chunk text without the stored title/section header (when present)."""
        if self.header_chars and self.header_chars < len(self.text):
            return self.text[self.header_chars:].strip()
        return self.text.strip()


# ---------------------------------------------------------------------------
# Step 1: Query rewriting with history
# ---------------------------------------------------------------------------
def _history_pairs(
    history: Sequence[str] | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Normalise history to (question, answer) pairs.

    The chat loop stores pairs; older callers pass bare question strings. Both
    shapes are accepted so nothing breaks.
    """
    pairs: list[tuple[str, str]] = []
    for turn in history or ():
        if isinstance(turn, (tuple, list)) and len(turn) == 2:
            pairs.append((str(turn[0]), str(turn[1])))
        else:
            pairs.append((str(turn), ""))
    return pairs


def _format_history(
    history: Sequence[str] | Sequence[tuple[str, str]] | None,
) -> str:
    """Turn conversation history into a plain-text block for prompts."""
    pairs = _history_pairs(history)
    if not pairs:
        return "(none)"
    lines: list[str] = []
    for i, (question, answer) in enumerate(pairs, start=1):
        lines.append(f"Turn {i} user: {question}")
        if answer:
            lines.append(f"Turn {i} assistant: {answer}")
    return "\n".join(lines)


def is_followup(
    question: str,
    history: Sequence[str] | Sequence[tuple[str, str]] | None,
) -> bool:
    """True when the question probably depends on the previous turns.

    A question is a follow-up when there is history and it is short (at most
    FOLLOWUP_MAX_WORDS) or contains a word from ``followup_pronouns`` in
    settings.yaml ("they", "that", "it"...). Never true without history.
    """
    if not history:
        return False
    words = re.findall(r"[a-z']+", question.casefold())
    if not words:
        return False
    if len(words) <= settings.retrieval.followup_max_words:
        return True
    pronouns = set(settings.followup_pronouns)
    return any(word in pronouns for word in words)


def condense_question(
    question: str,
    history: Sequence[str] | Sequence[tuple[str, str]] | None,
) -> str:
    """Rewrite a follow-up into one standalone search question.

    Retrieval cannot resolve "who did they beat?" on its own, and rewriting
    alone does not see the conversation. The LLM is shown the recent turns and
    the latest message, and must keep every name, season and competition. On
    any failure the original question is returned, so retrieval still works.
    """
    if not settings.retrieval.condense_followups:
        return question
    if not _history_pairs(history):
        return question

    domain = ", ".join(settings.pages) if settings.pages else "the indexed articles"
    # Only the most recent turn: two months of history made the model merge
    # them ("... in November and March?"), which then routed to the wrong month.
    turns = max(1, settings.retrieval.condense_history_turns)
    recent = _history_pairs(history)[-turns:]
    prompt = (
        "Rewrite the user's latest message as ONE standalone search question "
        "for a retrieval system over these Wikipedia articles:\n"
        + domain
        + "\n\n"
        "Use the conversation only to resolve pronouns and missing subjects.\n"
        "Keep every name, season, month and competition from the conversation.\n"
        "If the message is already standalone, return it unchanged.\n"
        "Return only the question -- one line, no explanation.\n\n"
        "Conversation:\n"
        + _format_history(recent)
        + "\n\nLatest message: "
        + question
        + "\n\nStandalone question:"
    )

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.retrieval.query_rewrite_temperature,
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001 - never crash retrieval on LLM failure
        logger.warning(
            "Follow-up condensation failed (%s); using the original question.", exc
        )
        return question

    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        logger.warning("Follow-up condensation returned nothing; using the original.")
        return question

    condensed = normalize_season_text(lines[0].strip().strip('"').strip())
    logger.info("Condensed follow-up %r -> %r", question, condensed)
    return condensed


def rewrite_queries(question: str, history: Sequence[str] | None = None) -> list[str]:
    """Rewrite the user's question into QUERY_REWRITE_COUNT search variants.

    Uses the chat LLM (LLM_MODEL) at QUERY_REWRITE_TEMPERATURE (default 0.0) to
    rephrase the question in Wikipedia style. The original question is always
    kept first, seasons are normalised (``2025-2026`` -> ``2025–26``) and
    duplicates are removed. ``QUERY_REWRITE_MODE=off`` returns only the
    normalised original.

    The prompt names the indexed pages (from settings.yaml) so the model keeps
    the subject instead of drifting to the generic topic: without it, "November"
    became "November weather" and the correct section was pushed out of reach.
    """
    original = normalize_season_text(question)
    count = settings.retrieval.query_rewrite_count

    if not settings.retrieval.query_rewrite_enabled or count <= 0:
        logger.debug("Query rewriting is off; using the original question only.")
        return [original]

    history_text = _format_history(history)
    domain = ", ".join(settings.pages) if settings.pages else "the indexed articles"

    prompt = (
        "You rewrite a user question into search queries for a retrieval "
        "system over these Wikipedia articles:\n"
        + domain
        + "\n\n"
        "Rules:\n"
        "- Write exactly "
        + str(count)
        + " queries, one per line, no numbering.\n"
        "- Keep the subject of the question in every query. Do not broaden it "
        "to a general topic.\n"
        "- Keep any season, year, month or competition named in the question.\n"
        "- Prefer words that would appear in the article text.\n"
        "- Do not add explanations or extra text.\n\n"
        "Conversation history:\n"
        + history_text
        + "\n\nUser question: "
        + original
        + "\n\nRewritten queries:"
    )

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.retrieval.query_rewrite_temperature,
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001 - never crash retrieval on LLM failure
        logger.warning("Query rewriting failed (%s); falling back to original question only.", exc)
        return [original]

    # Parse lines, drop empties, strip whitespace, normalise seasons.
    variants = [
        normalize_season_text(line.strip())
        for line in raw.split("\n")
        if line.strip()
    ]

    # Always include the original question first.
    queries = [original]
    for variant in variants:
        if variant != original and variant not in queries:
            queries.append(variant)
        if len(queries) - 1 >= count:
            break

    logger.info("Rewrote query into %d variant(s).", len(queries))
    logger.debug("Query variants: %s", queries)
    return queries


# ---------------------------------------------------------------------------
# Steps 2-6: Embed, search, merge, threshold, return top-N
# ---------------------------------------------------------------------------
def search_chunks(
    question: str,
    k: int | None = None,
    collection: str | None = None,
    instruction_enabled: bool | None = None,
) -> list[tuple[Document, float]]:
    """Raw vector search, used by the retrieval eval.

    No query rewriting and no score threshold, so the eval measures chunking
    and embedding quality only. ``collection`` lets the eval read an
    ``arsenal_eval_*`` collection without touching production.
    """
    limit = k or settings.retrieval.retrieve_k
    embeddings = get_embeddings(instruction_enabled=instruction_enabled)
    store = get_vector_store(collection=collection, embeddings=embeddings)
    results = store.similarity_search_with_score(question, k=limit)
    _log_scores(question, results, limit)
    return results


def _log_scores(question: str, results: list[tuple[Document, float]], limit: int) -> None:
    """Log the distance range for one query (lower = closer, cosine distance)."""
    if not results:
        logger.info("Search %r: k=%d, no results.", question, limit)
        return
    distances = [score for _, score in results]
    logger.info(
        "Search %r: k=%d, distance %.4f..%.4f (threshold %.2f)",
        question, limit, min(distances), max(distances),
        settings.retrieval.score_threshold,
    )


def fuse_rankings(
    rankings: Sequence[Sequence[tuple[Document, float]]],
) -> list[tuple[Document, float]]:
    """Combine several ranked result lists with Reciprocal Rank Fusion.

    Cosine distances from differently-worded queries are not comparable, so a
    chunk gets ``1/(RRF_K + rank)`` from every list that returns it. The
    returned score stays the best raw distance, so the SCORE_THRESHOLD filter
    keeps working.
    """
    documents: dict[str, Document] = {}
    best_distance: dict[str, float] = {}
    fused: dict[str, float] = {}

    for results in rankings:
        for rank, (doc, score) in enumerate(results, start=1):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if chunk_id not in best_distance or score < best_distance[chunk_id]:
                documents[chunk_id] = doc
                best_distance[chunk_id] = score

    return sorted(
        ((documents[chunk_id], best_distance[chunk_id]) for chunk_id in documents),
        key=lambda pair: -fused[str(pair[0].metadata.get("chunk_id", ""))],
    )


def search_queries(
    queries: Sequence[str],
    k: int | None = None,
    collection: str | None = None,
    instruction_enabled: bool | None = None,
) -> list[tuple[Document, float]]:
    """Search every query variant and fuse the ranks by chunk_id.

    Used by the eval grid and by :func:`retrieve`, so both measure the same
    code path.
    """
    limit = k or settings.retrieval.retrieve_k
    embeddings = get_embeddings(instruction_enabled=instruction_enabled)
    store = get_vector_store(collection=collection, embeddings=embeddings)

    rankings: list[list[tuple[Document, float]]] = []
    for query in queries:
        try:
            results = store.similarity_search_with_score(query, k=limit)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            logger.error("Vector search failed for %r: %s", query, exc)
            continue
        _log_scores(query, results, limit)
        rankings.append(list(results))

    ordered = fuse_rankings(rankings)
    logger.debug(
        "Fused %d query variant(s) into %d unique chunk(s).",
        len(queries), len(ordered),
    )
    return ordered


def _best_distance(
    results: Sequence[tuple[Document, float]],
) -> float | None:
    """Lowest distance in a result set; None when there are no results."""
    return min((score for _, score in results), default=None)


def _append_new(
    base: Sequence[tuple[Document, float]],
    extra: Sequence[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Keep the base ranking and append only chunks it does not already have.

    Used by the fallback rewrite: the original ordering is never disturbed, so
    rewriting can only add recall, never push the right chunk down.
    """
    merged = list(base)
    seen = {str(doc.metadata.get("chunk_id", "")) for doc, _ in base}
    for doc, score in extra:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if chunk_id not in seen:
            merged.append((doc, score))
            seen.add(chunk_id)
    return merged


def apply_gate(
    chunks: Sequence[RetrievedChunk],
    mode: str | None = None,
    threshold: float | None = None,
    margin: float | None = None,
    k: int | None = None,
    fallback_on_empty: bool | None = None,
) -> list[RetrievedChunk]:
    """Cut weak results according to RETRIEVAL_GATE.

    relative  keep chunks within RELATIVE_MARGIN of the best score (default).
    absolute  keep chunks with score <= SCORE_THRESHOLD (the old behaviour).
    none      keep the first k chunks, no score filter.

    Input order is preserved: callers pass the ranking they trust (distance for
    a single query, RRF fusion for several), so the cap keeps the head of that
    ranking instead of re-sorting by raw distance. Returns at most ``k`` chunks.
    Pure function: no database, no settings lookup when all arguments are given
    (tests rely on this).
    """
    mode = mode or settings.retrieval.retrieval_gate
    threshold = (
        settings.retrieval.score_threshold if threshold is None else threshold
    )
    margin = settings.retrieval.relative_margin if margin is None else margin
    k = settings.retrieval.final_top_n if k is None else k
    if fallback_on_empty is None:
        fallback_on_empty = settings.retrieval.fallback_on_empty

    ordered = list(chunks)
    if not ordered:
        return []

    if mode == "none":
        return ordered[:k]

    if mode == "absolute":
        kept = [chunk for chunk in ordered if chunk.score <= threshold]
        if kept:
            return kept[:k]
        if fallback_on_empty:
            best = min(ordered, key=lambda chunk: chunk.score)
            logger.warning(
                "No chunk below SCORE_THRESHOLD=%.2f; returning the best one.",
                threshold,
            )
            return [best]
        return []

    # relative (default)
    best = min(chunk.score for chunk in ordered)
    kept = [chunk for chunk in ordered if chunk.score <= best + margin]
    return kept[:k]


def search_for_question(
    question: str,
    mode: str | None = None,
    history: Sequence[str] | None = None,
    k: int | None = None,
    collection: str | None = None,
    instruction_enabled: bool | None = None,
) -> list[tuple[Document, float]]:
    """Run the search stage according to QUERY_REWRITE_MODE.

    off       one search with the normalised original question.
    always    rewrite, then fuse every variant (measured worse in the grid).
    fallback  search the original first; rewrite only when the best distance is
              above QUERY_REWRITE_WEAK_SCORE, then append the new chunks.

    Shared by :func:`retrieve` and the retrieval eval, so the grid measures the
    exact code path the app uses. Returns raw (Document, score) pairs in ranking
    order; callers de-duplicate and apply the gate.
    """
    mode = mode or settings.retrieval.query_rewrite_mode
    options = {
        "k": k,
        "collection": collection,
        "instruction_enabled": instruction_enabled,
    }
    results = search_queries([normalize_season_text(question)], **options)

    if mode == "always":
        queries = rewrite_queries(question, history)
        if len(queries) > 1:
            logger.info(
                "Rewriting every query (mode=always): %d variants.", len(queries)
            )
            return search_queries(queries, **options)
        return results

    if mode != "fallback":
        return results

    best = _best_distance(results)
    weak_score = settings.retrieval.query_rewrite_weak_score
    if best is not None and best <= weak_score:
        logger.info("Rewrite not needed: best %.3f <= %.2f.", best, weak_score)
        return results

    variants = rewrite_queries(question, history)[1:]
    if not variants:
        return results

    before = len(results)
    results = _append_new(results, search_queries(variants, **options))
    logger.info(
        "Weak first search (best %.3f > %.2f): added %d chunk(s) from %d "
        "rewritten variant(s).",
        best if best is not None else -1.0,
        weak_score,
        len(results) - before,
        len(variants),
    )
    return results


def _section_chunks(question: str) -> list[RetrievedChunk] | None:
    """Return the named section in document order, or None if none is named."""
    routed = route_section(question)
    if routed is None:
        return None
    path, documents = routed
    chunks = [_doc_to_chunk(doc, 0.0) for doc in documents]
    logger.info(
        "Returning %d chunk(s) from section %r (vector search skipped).",
        len(chunks), path,
    )
    return chunks


def retrieve(
    question: str,
    history: Sequence[str] | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.

    Args:
        question: The user's question.
        history: Previous turns, as (question, answer) pairs or bare question
            strings. Used to condense follow-ups and to rewrite queries.
        k: How many chunks to return. Defaults to FINAL_TOP_N from .env.

    Returns:
        List of RetrievedChunk, sorted by score (best first), capped at k.
    """
    if k is None:
        k = settings.retrieval.final_top_n

    # Step 0: a question that names a known section ("November", "kits") is
    # routed to that section and returned in document order. Vector search is
    # not used, and the score gate does not apply.
    chunks = _section_chunks(question)
    if chunks is not None:
        return chunks

    # Step 0.5: a short follow-up ("who did they beat?") is condensed into a
    # standalone question using the conversation, so search has a subject. The
    # condensed form may name a section the follow-up left out ("... in March").
    search_question = question
    if settings.retrieval.condense_followups and is_followup(question, history):
        search_question = condense_question(question, history)
        chunks = _section_chunks(search_question)
        if chunks is not None:
            return chunks

    # Steps 1-4: search (original or rewritten, per QUERY_REWRITE_MODE).
    results = search_for_question(search_question, history=history)

    seen: dict[str, RetrievedChunk] = {}
    for doc, score in results:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        current = seen.get(chunk_id)
        if current is None or score < current.score:
            seen[chunk_id] = _doc_to_chunk(doc, score)

    # Step 5: apply the configured gate (relative / absolute / none).
    top = apply_gate(list(seen.values()), k=k)

    if not top:
        logger.info(
            "No chunk passed the gate (mode=%r); returning no context.",
            settings.retrieval.retrieval_gate,
        )
        return []

    logger.info("Retrieved %d chunk(s) for question: %r", len(top), question)
    return top


def retrieve_by_title(title: str, k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve all chunks for a given article title directly from pgvector.

    Unlike :func:`retrieve`, this does not use similarity search or apply a
    score threshold.  It fetches every stored chunk whose metadata ``title``
    matches, ordered by section path and chunk index.  This is used by the
    two-step retrieve-then-summarize flow so the summarizer sees the full
    article context.

    Args:
        title: The Wikipedia article title to match.
        k: Maximum number of chunks to return.  Defaults to FINAL_TOP_N.

    Returns:
        List of RetrievedChunk for the article, sorted by section and chunk
        index.
    """
    if k is None:
        k = settings.retrieval.final_top_n

    sql = text(
        """
        SELECT e.id, e.document, e.cmetadata, e.embedding
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :collection
          AND e.cmetadata->>'title' = :title
        ORDER BY e.cmetadata->>'section_path', (e.cmetadata->>'chunk_index')::int
        LIMIT :limit
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(
            sql,
            {
                "collection": settings.database.collection,
                "title": title,
                "limit": k,
            },
        ).mappings().all()

    chunks: list[RetrievedChunk] = []
    for row in rows:
        md = row["cmetadata"] if isinstance(row["cmetadata"], dict) else {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(md.get("chunk_id", "")),
                title=str(md.get("title", "")),
                section=str(md.get("section", "")),
                url=str(md.get("url", "")),
                content_type=str(md.get("content_type", "")),
                text=str(row["document"]),
                score=float(md.get("score", 0.0)) if md.get("score") is not None else 0.0,
                header_chars=int(md.get("header_chars", 0) or 0),
                fetched_at=str(md.get("fetched_at", "") or ""),
            )
        )

    logger.info("Retrieved %d chunk(s) for title: %r", len(chunks), title)
    return chunks


def _doc_to_chunk(doc: Document, score: float) -> RetrievedChunk:
    """Convert a LangChain Document + score into a RetrievedChunk."""
    md = doc.metadata
    return RetrievedChunk(
        chunk_id=str(md.get("chunk_id", "")),
        title=str(md.get("title", "")),
        section=str(md.get("section", "")),
        url=str(md.get("url", "")),
        content_type=str(md.get("content_type", "")),
        text=doc.page_content,
        score=float(score),
        header_chars=int(md.get("header_chars", 0) or 0),
        fetched_at=str(md.get("fetched_at", "") or ""),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    question = sys.argv[1] if len(sys.argv) > 1 else "What is Arsenal F.C. known for?"
    chunks = retrieve(question)

    print(f"\nQuestion: {question}")
    print(f"Top {len(chunks)} chunk(s):\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- {i}. {c.chunk_id} (score={c.score:.4f}) ---")
        print(f"    {c.citation()}")
        print(f"    {c.text[:200]}")
        print()
