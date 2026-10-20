"""answer_chain.py -- generation stage for Arsenal Weki RAG.

Takes retrieved prose chunks from :mod:`src.retrieval.retriever` and generates
answers using Ollama's chat model with LangChain chat messages:

    * ``SystemMessage`` -- the RAG system prompt (synthesize from context,
      cite sources, note gaps honestly but do not refuse outright).
    * ``HumanMessage`` -- the user's question plus the retrieved context.
    * ``AIMessage``   -- prior assistant turns from conversation history so
      pronouns and references resolve across turns.

Two modes:

    * :func:`answer_question` -- Q&A over retrieved chunks (one-step retrieval).
    * :func:`summarize_article` -- two-step retrieve-then-summarize: fetch all
      chunks for a Wikipedia article, then generate a narrative summary.

Returns both the generated answer/summary and the source chunks used, so the
caller can display citations (title, section, URL).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import settings
from src.retrieval.retriever import RetrievedChunk, retrieve, retrieve_by_title

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """The output of the answer chain."""

    answer: str
    sources: list[RetrievedChunk]
    question: str


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Turn retrieved chunks into a numbered context block for the LLM.

    Each chunk is labelled ``[1]``, ``[2]``, ... The stored title/section
    header is stripped (``CONTEXT_STRIP_HEADER``) because the numbered Sources
    list is built in code; the model only writes ``[n]`` markers.
    """
    if not chunks:
        return "No relevant context was found."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        text = (
            chunk.body_text()
            if settings.retrieval.context_strip_header
            else chunk.text.strip()
        )
        lines.append(f"[{i}]\n{text}")
    return "\n\n".join(lines)


_SEASON_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–/]\s*(\d{2})\b")


def season_note(question: str) -> str:
    """Return a season instruction for the user turn, or an empty string.

    A question about "the 2024-25 quarter-final" used to be answered with the
    2025-26 quarter-final: the Background passage says "September 2024" and
    "January 2025", never "2024-25", so the model could not match the phrase.
    The note spells out both years and forbids borrowing another season's
    result. The years come from the question, not from a literal.
    """
    match = _SEASON_RANGE.search(question or "")
    if not match:
        return ""
    start, end = match.group(1), match.group(2)
    season = f"{start}-{end}"
    end_year = f"{start[:2]}{end}"
    return (
        f"Season asked: {season}. Answer from the passage that mentions "
        f"{start} or {end_year}; quote that passage's result. Do not use a "
        "result from another season, and do not attach this season to a "
        "result from another season.\n\n"
    )


def filter_to_season(
    chunks: Sequence[RetrievedChunk], question: str
) -> list[RetrievedChunk]:
    """Keep only passages that mention the season named in the question.

    With a larger FINAL_TOP_N the model saw more 2025-26 quarter-final
    passages and answered a 2024-25 question with the Sporting CP tie. The
    passage that holds the answer mentions the starting year ("September
    2024"); the others do not. If nothing matches, return all chunks so the
    model can still say the season is not covered (prompt rule 15).
    """
    match = _SEASON_RANGE.search(question or "")
    if not match:
        return list(chunks)
    start, end = match.group(1), match.group(2)
    season = f"{start}-{end}"
    season_en_dash = f"{start}\u2013{end}"
    kept = [
        chunk
        for chunk in chunks
        if start in chunk.text
        or season in chunk.text
        or season_en_dash in chunk.text
    ]
    if not kept:
        return list(chunks)
    if len(kept) != len(chunks):
        logger.info(
            "Season %s: kept %d of %d chunks for the answer.",
            season, len(kept), len(chunks),
        )
    return kept


def group_sources(
    chunks: Sequence[RetrievedChunk],
) -> list[tuple[list[int], RetrievedChunk]]:
    """Group chunks that cite the same article section.

    Retrieval usually returns several chunks from one section, which used to
    print the same source line once per chunk. Returns (passage numbers, first
    chunk) in order of first appearance; the numbers match the [n] markers.
    """
    order: list[tuple[str, str, str]] = []
    markers: dict[tuple[str, str, str], list[int]] = {}
    first: dict[tuple[str, str, str], RetrievedChunk] = {}
    for i, chunk in enumerate(chunks, 1):
        key = (chunk.title, chunk.section, chunk.url)
        if key not in markers:
            order.append(key)
            markers[key] = []
            first[key] = chunk
        markers[key].append(i)
    return [(markers[key], first[key]) for key in order]


def format_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered Sources block (title, section, URL, fetched_at) built in code.

    Chunks from the same section share one line, e.g. ``[1][2][3]``, so the
    list shows where the facts came from instead of repeating the same URL.
    """
    if not chunks:
        return "Sources: none"
    lines: list[str] = []
    for markers, chunk in group_sources(chunks):
        label = "".join(f"[{number}]" for number in markers)
        lines.append(f"  {label} {chunk.citation()}")
    return "\n".join(lines)


def _log_token_usage(response, role: str = "answer") -> None:
    """Log prompt_eval_count and warn above 90% of OLLAMA_NUM_CTX."""
    metadata = getattr(response, "response_metadata", None) or {}
    used = metadata.get("prompt_eval_count")
    num_ctx = settings.ollama.num_ctx
    if used is None:
        logger.debug("%s: no prompt_eval_count in the response.", role)
        return
    percent = (used / num_ctx * 100) if num_ctx else 0.0
    logger.info(
        "%s prompt tokens: %d (%.0f%% of OLLAMA_NUM_CTX=%d)",
        role, used, percent, num_ctx,
    )
    if percent > 90:
        logger.warning(
            "Prompt is %d tokens, above 90%% of OLLAMA_NUM_CTX=%d. Raise "
            "OLLAMA_NUM_CTX or send fewer/shorter chunks.",
            used, num_ctx,
        )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
# The season comes from settings.yaml, not from a literal: Background sections
# recap the previous season, and the model used to move those results into the
# current one ("finished third in the league phase in January 2025" was told as
# part of 2025-26).
_CURRENT_SEASON = settings.current_season or "the current season"

# Answer length is configuration, not a literal. The default in code stays
# "short" so an older settings.yaml keeps the old behaviour.
_ANSWER_LENGTH_RULES = {
    "short": "7. Keep answers short: a few sentences, or a short list.",
    "normal": (
        "7. Give a normal-length answer: one or two short paragraphs, about "
        "4-6 sentences in total, in flowing prose. Use a short list only when "
        "the question asks for several items. Do not write one sentence per "
        "line."
    ),
    "long": (
        "7. Give a detailed answer: one to three paragraphs, about 8-10 "
        "sentences in total, in flowing prose. Use a detailed list only when "
        "the question asks for several items. Do not repeat facts and do not "
        "write one sentence per line."
    ),
}
_ANSWER_STYLE = (
    settings.answer_style
    if settings.answer_style in _ANSWER_LENGTH_RULES
    else "short"
)

SYSTEM_PROMPT = f"""You answer questions about Arsenal F.C. using only the numbered Wikipedia passages you are given.

Rules:
1. Use only facts that appear in the passages. Do not use outside knowledge.
2. Copy names, scores and dates exactly as they are written in the passages. If a passage gives no date for an event, do not add one.
3. Add no details that are not in the passages.
4. If part of the question is not covered, answer the part that is covered and say which part is missing from the indexed pages.
5. If none of the passages answer the question, say that the indexed pages do not contain it. Do not guess.
6. Cite the passage number after each fact, like [1] or [2][3]. Write only the number in brackets -- no titles and no URLs.
{_ANSWER_LENGTH_RULES[_ANSWER_STYLE]}
8. Do not mention that you are an AI or that you were given context.
9. Answer only the latest question. Earlier turns are there so words like "they" or "that" make sense; do not repeat or merge earlier answers.
10. Never mix home and away: call a match "at home" only when the passage names Arsenal's own ground as the venue; a named opponent's ground is an away match.
11. The current season is {_CURRENT_SEASON}.
12. Keep every year and season exactly as the passage gives it. A result dated 2024 or 2025 belongs to that earlier season, even when it appears on the {_CURRENT_SEASON} page: Background sections recap the previous season.
13. Before you state a result, check the year next to it. Never move a previous-season result into the current season.
14. If the question names a season, answer for that season only, using only passages that mention that season or one of its years. A passage about a different season must not answer it.
15. If no passage covers the season the question names, say so. Do not answer with another season's result.
16. Write personal names exactly as the passage writes them. Do not expand a surname into a full name."""

SUMMARIZE_SYSTEM_PROMPT = """You write a short summary of a Wikipedia article from the numbered passages you are given.

Rules:
1. Use only facts that appear in the passages. Do not use outside knowledge.
2. Copy names, scores and dates exactly as they are written.
3. Add no details that are not in the passages.
4. Cite the passage number after each fact, like [1] or [2][3]. Write only the number in brackets -- no titles and no URLs.
5. If some aspects are not covered, say so instead of inventing them.
6. Do not mention that you are an AI or that you were given context."""


def build_history_messages(
    history: Sequence[str] | Sequence[tuple[str, str]] | None,
) -> list:
    """Turn history into HumanMessage/AIMessage pairs for the chat prompt.

    The chat loop stores (question, answer) pairs; older callers pass bare
    question strings. Both shapes are accepted. Only the latest question is
    answered -- earlier turns exist so pronouns resolve.
    """
    messages: list = []
    for turn in history or ():
        if isinstance(turn, (tuple, list)) and len(turn) == 2:
            question, answer = str(turn[0]), str(turn[1])
            messages.append(HumanMessage(content=question))
            if answer:
                messages.append(AIMessage(content=answer))
        else:
            messages.append(HumanMessage(content=str(turn)))
    return messages


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------
def build_answer_chain():
    """Build the LangChain answer chain.

    The chain takes a dict with keys ``question`` (str) and ``history``
    (list of (question, answer) pairs or strings, optional), retrieves relevant
    chunks, builds chat messages (SystemMessage + alternating history + the
    current turn), calls Ollama, and returns a GenerationResult with the answer
    and cited sources.
    """

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.ollama.answer_temperature,
        num_ctx=settings.ollama.num_ctx,
    )

    # Prompt with message placeholders so we can inject SystemMessage,
    # conversation history (AIMessage/HumanMessage pairs), and the current
    # user turn (HumanMessage with question + context).
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Question: {question}\n\n"
                "Numbered Wikipedia passages:\n"
                "{context}\n\n"
                "Answer (short, cite [n]):",
            ),
        ]
    )

    def _retrieve_and_format(inputs: dict) -> dict:
        question = inputs["question"]
        history = inputs.get("history", [])

        # Step 1: retrieve relevant chunks (prose-only, section-aware).
        chunks = retrieve(question, history=history)

        # Step 1b: a question that names a season must not be answered from
        # another season's passages (the model did this once given more chunks).
        chunks = filter_to_season(chunks, question)

        # Step 2: format context with citations.
        context = format_context(chunks)

        # Step 3: build LangChain chat messages. Earlier (question, answer)
        # pairs become HumanMessage/AIMessage so "they" resolves; only the
        # latest question is answered.
        langchain_history = build_history_messages(history)

        # Current user turn carries the question + retrieved context. A
        # question that names a season gets a season line before the passages.
        season_hint = season_note(question)
        current_user = HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Numbered Wikipedia passages:\n"
                f"{context}\n\n"
                f"{season_hint}"
                f"Answer (short, cite [n]):"
            )
        )

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + langchain_history + [current_user]

        return {
            "messages": messages,
            "chunks": chunks,
            "question": question,
        }

    def _generate(inputs: dict) -> GenerationResult:
        messages: list = inputs["messages"]
        chunks: list[RetrievedChunk] = inputs["chunks"]

        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        _log_token_usage(response, role="answer")

        return GenerationResult(
            answer=answer,
            sources=chunks,
            question=inputs["question"],
        )

    chain = (
        RunnablePassthrough()
        | _retrieve_and_format
        | _generate
    )
    return chain


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    history: Sequence[str] | Sequence[tuple[str, str]] | None = None,
) -> GenerationResult:
    """Ask a question and return the answer with cited sources.

    Args:
        question: The user's question.
        history: Previous turns as (question, answer) pairs (or bare strings).

    Returns:
        GenerationResult with the answer and the source chunks used.
    """
    chain = build_answer_chain()
    return chain.invoke({"question": question, "history": list(history or [])})


def summarize_article(
    title: str,
    history: Sequence[str] | Sequence[tuple[str, str]] | None = None,
    k: int | None = None,
) -> GenerationResult:
    """Two-step retrieve-then-summarize for a Wikipedia article.

    Step 1: retrieve *all* chunks for the given article title directly from
    pgvector (no similarity search, no score threshold).
    Step 2: generate a narrative summary using a dedicated summarization
    prompt that sees the full article context.

    Args:
        title: The Wikipedia article title to summarize (e.g.
            "2025–26 Arsenal F.C. season").
        history: Previous turns as (question, answer) pairs (or bare strings).
        k: Maximum number of chunks to retrieve.  Defaults to RERANK_TOP_N.

    Returns:
        GenerationResult with the summary and the source chunks used.
    """
    # Step 1: retrieve all chunks for the article.
    chunks = retrieve_by_title(title, k=k)

    if not chunks:
        return GenerationResult(
            answer=f"No Wikipedia article found with title: {title}",
            sources=[],
            question=f"Summarize: {title}",
        )

    # Step 2: format context with citations.
    context = format_context(chunks)

    # Step 3: build LangChain chat messages for summarization.
    langchain_history = build_history_messages(history)

    current_user = HumanMessage(
        content=(
            f"Article title: {title}\n\n"
            f"Numbered Wikipedia passages:\n"
            f"{context}\n\n"
            f"Write a short summary of this article (cite [n]):"
        )
    )

    messages = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
    ] + langchain_history + [current_user]

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.ollama.answer_temperature,
        num_ctx=settings.ollama.num_ctx,
    )

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    _log_token_usage(response, role="summary")

    return GenerationResult(
        answer=answer,
        sources=chunks,
        question=f"Summarize: {title}",
    )


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is Arsenal F.C. known for?"
    try:
        result = answer_question(question)
        print(f"\nQuestion: {result.question}")
        print(f"\nAnswer:\n{result.answer}")
        print("\nSources:")
        for i, src in enumerate(result.sources, 1):
            print(f"  [{i}] {src.citation()}")
    except Exception as exc:  # noqa: BLE001 - smoke test should not crash hard
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
