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

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import settings
from src.retrieval.retriever import RetrievedChunk, retrieve, retrieve_by_title

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
    """Turn retrieved chunks into a readable context block with citations."""
    if not chunks:
        return "No relevant context was found."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        cite = chunk.citation()  # "Title | section: ... | source: ..."
        lines.append(
            f"[{i}] {cite}\n"
            f"    {chunk.text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant answering questions using the provided Wikipedia context.

Rules:
1. Base your answer on the retrieved context chunks below.  Synthesize the
   information into a coherent response even if the context is partial.
2. Cite every factual claim using the source format: [Title > Section](URL).
3. If the context is incomplete, say what you *can* answer and note what
   information is missing (do not refuse outright).
4. Do not invent facts, player names, statistics, dates, or section titles
   that are not present in the context.
5. Do not mention that you are an AI or that you were given context; just answer
   directly.
6. Keep answers concise and grounded in the retrieved text."""

SUMMARIZE_SYSTEM_PROMPT = """You are a helpful assistant writing a season/article summary from Wikipedia context.

Rules:
1. Write a narrative summary of the article using the retrieved context chunks.
2. Organize by theme: introduction/overview, key events, results, trophies,
   manager, notable players, and any other relevant topics found in the text.
3. Cite every factual claim using the source format: [Title > Section](URL).
4. If some aspects are not covered by the context, say so explicitly rather than
   inventing details.
5. Do not invent facts, player names, statistics, dates, or section titles
   that are not present in the context.
6. Do not mention that you are an AI or that you were given context; just write
   the summary directly."""


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------
def build_answer_chain():
    """Build the LangChain answer chain.

    The chain takes a dict with keys ``question`` (str) and ``history``
    (list[str], optional), retrieves relevant chunks, builds chat messages
    (SystemMessage + HumanMessage + AIMessage history), calls Ollama, and
    returns a GenerationResult with the answer and cited sources.
    """

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.ollama.temperature,
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
                "Retrieved Wikipedia context (use these to answer and cite):\n"
                "{context}\n\n"
                "Answer:",
            ),
        ]
    )

    def _retrieve_and_format(inputs: dict) -> dict:
        question = inputs["question"]
        history = inputs.get("history", [])

        # Step 1: retrieve relevant chunks (prose-only, section-aware).
        chunks = retrieve(question, history=history)

        # Step 2: format context with citations.
        context = format_context(chunks)

        # Step 3: build LangChain chat messages.
        # SystemMessage is handled by the prompt template; history turns are
        # converted to AIMessage/HumanMessage so pronouns resolve across turns.
        langchain_history: list = []
        for turn in history:
            # Each history entry is a prior user question; we alternate
            # user/assistant. Since we only store user questions here, we treat
            # them as HumanMessage entries so the model sees the conversation
            # flow. Actual assistant answers are appended by the caller when
            # available.
            langchain_history.append(HumanMessage(content=turn))

        # Current user turn carries the question + retrieved context.
        current_user = HumanMessage(
            content=(
                f"Question: {question}\n\n"
                f"Retrieved Wikipedia context (use these to answer and cite):\n"
                f"{context}\n\n"
                f"Answer:"
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
def answer_question(question: str, history: Sequence[str] | None = None) -> GenerationResult:
    """Ask a question and return the answer with cited sources.

    Args:
        question: The user's question.
        history: Previous conversation turns (list of strings) for context.

    Returns:
        GenerationResult with the answer and the source chunks used.
    """
    chain = build_answer_chain()
    return chain.invoke({"question": question, "history": list(history or [])})


def summarize_article(
    title: str,
    history: Sequence[str] | None = None,
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
        history: Previous conversation turns (list of strings) for context.
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
    langchain_history: list = []
    for turn in history or []:
        langchain_history.append(HumanMessage(content=turn))

    current_user = HumanMessage(
        content=(
            f"Article title: {title}\n\n"
            f"Retrieved Wikipedia context (use all of it to write a summary):\n"
            f"{context}\n\n"
            f"Write a comprehensive summary of this article:"
        )
    )

    messages = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
    ] + langchain_history + [current_user]

    llm = ChatOllama(
        model=settings.ollama.chat_model,
        base_url=settings.ollama.base_url,
        temperature=settings.ollama.temperature,
    )

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

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
