"""retriever.py -- retrieval stage for Arsenal Weki RAG.

Takes a user question (plus conversation history), rewrites it into several
query variants, embeds each variant, searches pgvector, merges and
de-duplicates results, applies a score threshold, and returns the top chunks
with citation metadata.

All settings come from ``.env`` / ``config/settings.yaml``:

    QUERY_REWRITE_COUNT  -- how many query variants to generate (default 3)
    RETRIEVE_K           -- how many chunks to fetch per query variant (default 10)
    RERANK_TOP_N         -- how many chunks to return after merging (default 5)
    SCORE_THRESHOLD      -- minimum similarity score to keep a chunk (default 0.4)

Pipeline:

    1. rewrite_queries(question, history) -> list of query strings
    2. Embed each query with the same Ollama model used at ingestion time
       (with EMBEDDING_QUERY_PREFIX applied).
    3. similarity_search for each query, collect (Document, score) pairs.
    4. Merge + de-duplicate by chunk_id, keeping the best score per chunk.
    5. Filter out chunks below SCORE_THRESHOLD.
    6. Return the top RERANK_TOP_N chunks with citation metadata.

Run a quick smoke test:
    python -m src.retrieval.retriever "Who was Arsenal's top scorer?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from src.config import settings
from src.ingestion.index_builder import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the retriever, with citation metadata."""

    chunk_id: str
    title: str
    section: str
    source_url: str
    content_type: str
    text: str
    score: float

    def citation(self) -> str:
        """Short citation string for displaying where the chunk came from."""
        parts = [self.title]
        if self.section:
            parts.append(f"section: {self.section}")
        if self.source_url:
            parts.append(f"source: {self.source_url}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Step 1: Query rewriting with history
# ---------------------------------------------------------------------------
def _format_history(history: Sequence[str] | None) -> str:
    """Turn conversation history into a plain-text block for the prompt."""
    if not history:
        return "(none)"
    lines = []
    for i, turn in enumerate(history):
        lines.append(f"Turn {i + 1}: {turn}")
    return "\n".join(lines)


def rewrite_queries(question: str, history: Sequence[str] | None = None) -> list[str]:
    """Rewrite the user's question into QUERY_REWRITE_COUNT search variants.

    Uses the chat LLM (LLM_MODEL) to rephrase the question, taking the
    conversation history into account so pronouns like "he" or "it" are
    resolved against earlier turns.

    Returns a list that always includes the original question first, followed
    by the rewritten variants.
    """
    count = settings.retrieval.query_rewrite_count
    history_text = _format_history(history)

    prompt = (
        "You are helping a search system find Arsenal F.C. documents.\n"
        "Given the user's question and the conversation history, rewrite the "
        "question into exactly "
        + str(count)
        + " different search queries that would help find relevant documents.\n"
        "Keep each query short and focused on Arsenal facts.\n"
        "Return one query per line. Do not number them. Do not add extra text.\n\n"
        "Conversation history:\n"
        + history_text
        + "\n\nUser question: "
        + question
        + "\n\nRewritten queries:"
    )

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.ollama.temperature,
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001 - never crash retrieval on LLM failure
        logger.warning("Query rewriting failed (%s); falling back to original question only.", exc)
        return [question]

    # Parse lines, drop empties, strip whitespace.
    variants = [line.strip() for line in raw.split("\n") if line.strip()]

    # Always include the original question first.
    queries = [question]
    for variant in variants:
        if variant != question and variant not in queries:
            queries.append(variant)
        if len(queries) - 1 >= count:  # we already have the original + count variants
            break

    logger.info("Rewrote query into %d variant(s): %s", len(queries), queries)
    return queries


# ---------------------------------------------------------------------------
# Steps 2-6: Embed, search, merge, threshold, return top-N
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    history: Sequence[str] | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a question.

    Args:
        question: The user's question.
        history: Previous conversation turns (list of strings) for query rewriting.
        k: How many chunks to return. Defaults to RERANK_TOP_N from .env.

    Returns:
        List of RetrievedChunk, sorted by score (best first), capped at k.
    """
    if k is None:
        k = settings.retrieval.rerank_top_n

    # Step 1: rewrite the question into multiple variants.
    queries = rewrite_queries(question, history)

    # Step 2: get the embedding model (same one used at ingestion).
    embeddings = get_embeddings()

    # Step 3: get the vector store (read-only, no reset).
    store = get_vector_store()

    # Step 4: search for each query variant and collect results.
    seen: dict[str, RetrievedChunk] = {}

    for query in queries:
        logger.info("Retrieving for query: %r", query)
        try:
            results = store.similarity_search_with_score(query, k=settings.retrieval.retrieve_k)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            logger.error("Vector search failed for %r: %s", query, exc)
            continue

        for doc, score in results:
            md = doc.metadata
            chunk_id = str(md.get("chunk_id", ""))

            # pgvector returns cosine distance: lower = more similar.
            # We keep the best (lowest) score seen for each chunk_id.
            if chunk_id in seen:
                if score < seen[chunk_id].score:
                    seen[chunk_id] = _doc_to_chunk(doc, score)
            else:
                seen[chunk_id] = _doc_to_chunk(doc, score)

    # Step 5: apply score threshold.
    threshold = settings.retrieval.score_threshold
    filtered = [c for c in seen.values() if c.score <= threshold]

    # If nothing passed the threshold, fall back to the best chunk anyway
    # so the user is never left with zero results.
    if not filtered and seen:
        best = min(seen.values(), key=lambda c: c.score)
        logger.warning(
            "No chunks below SCORE_THRESHOLD=%.2f; returning best chunk (score=%.3f).",
            threshold, best.score,
        )
        filtered = [best]

    # Step 6: sort by score and return top-k.
    filtered.sort(key=lambda c: c.score)
    top = filtered[:k]

    logger.info("Retrieved %d chunk(s) for question: %r", len(top), question)
    return top


def _doc_to_chunk(doc: Document, score: float) -> RetrievedChunk:
    """Convert a LangChain Document + score into a RetrievedChunk."""
    md = doc.metadata
    return RetrievedChunk(
        chunk_id=str(md.get("chunk_id", "")),
        title=str(md.get("title", "")),
        section=str(md.get("section", "")),
        source_url=str(md.get("source_url", "")),
        content_type=str(md.get("content_type", "")),
        text=doc.page_content,
        score=float(score),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    question = sys.argv[1] if len(sys.argv) > 1 else "Who was Arsenal's top scorer?"
    chunks = retrieve(question)

    print(f"\nQuestion: {question}")
    print(f"Top {len(chunks)} chunk(s):\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- {i}. {c.chunk_id} (score={c.score:.4f}) ---")
        print(f"    {c.citation()}")
        print(f"    {c.text[:200]}")
        print()
