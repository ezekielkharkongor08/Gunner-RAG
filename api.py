"""FastAPI Backend for Arsenal Wiki RAG.

Provides dedicated endpoints for:
- /api/status: System diagnostics (PostgreSQL, pgvector, Ollama, Index)
- /api/articles: Configured Wikipedia article list
- /api/ask: Single-turn Q&A with optional source grounding verification
- /api/chat: Multi-turn conversational Q&A with context memory
- /api/summarize: Two-step retrieve-and-summarize for an article
- /api/ingest: Fetch, clean, split, embed, and store a Wikipedia article
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from main import check_database, check_index, check_ollama
from src.config import settings
from src.generation.answer_chain import (
    GenerationResult,
    RetrievedChunk,
    SUMMARIZE_SYSTEM_PROMPT,
    answer_question,
    build_history_messages,
    format_context,
    format_sources,
    summarize_article,
)
from src.generation.grounding import check_answer
from src.ingestion.run_ingestion import run_ingestion
from src.retrieval.retriever import retrieve, retrieve_by_title

logger = logging.getLogger("arsenal_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Arsenal FC Wiki RAG API",
    description="FastAPI Backend for Gunner Wiki RAG Q&A, Chat, Summarization, and Wikipedia Ingestion",
    version="1.0.0",
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SourceModel(BaseModel):
    chunk_id: str
    title: str
    section: str
    url: str
    text: str
    score: float
    citation: str
    fetched_at: str = ""


def serialize_chunks(chunks: List[RetrievedChunk]) -> List[SourceModel]:
    result: List[SourceModel] = []
    for chunk in chunks:
        result.append(
            SourceModel(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section or "",
                url=chunk.url or "",
                text=chunk.body_text(),
                score=float(chunk.score),
                citation=chunk.citation(),
                fetched_at=chunk.fetched_at or "",
            )
        )
    return result


class StatusResponse(BaseModel):
    database: str
    ollama: str
    index: str
    configured_articles_count: int
    configured_articles: List[str]


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question about Arsenal FC")
    check_grounding: bool = Field(
        False, description="Whether to verify numbers and entities against sources"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceModel]
    formatted_sources: str
    grounding_report: Optional[dict[str, Any]] = None


class ChatMessageTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's current question/message")
    history: List[ChatMessageTurn] = Field(
        default_factory=list, description="Prior conversation turns for context resolution"
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceModel]
    formatted_sources: str


class SummarizeRequest(BaseModel):
    query: Optional[str] = Field(None, description="User query or topic to summarize from retrieved docs")
    title: Optional[str] = Field(None, description="Wikipedia article title or query")
    history: List[ChatMessageTurn] = Field(
        default_factory=list, description="Prior conversation turns for context resolution"
    )


class SummarizeResponse(BaseModel):
    title: str
    summary: str
    sources: List[SourceModel]
    formatted_sources: str


def summarize_docs_for_query(
    query: str,
    history: Optional[List[Tuple[str, str]]] = None,
) -> GenerationResult:
    """Retrieve relevant chunks for a user query and summarize them with citations."""
    # 1. Semantic retrieval matching the query across all pgvector chunks
    chunks = retrieve(query, history=history)

    # 2. If semantic search returned nothing, fallback to exact title search
    if not chunks:
        chunks = retrieve_by_title(query)

    if not chunks:
        return GenerationResult(
            answer=(
                f"No matching Wikipedia passages found in the indexed documents for: '{query}'. "
                "You can ingest this article or topic using the Ingestion Hub."
            ),
            sources=[],
            question=f"Summarize: {query}",
        )

    context = format_context(chunks)
    langchain_history = build_history_messages(history)

    current_user = HumanMessage(
        content=(
            f"User Topic / Query: {query}\n\n"
            f"Numbered Wikipedia passages:\n"
            f"{context}\n\n"
            f"Write a comprehensive, grounded summary of the retrieved passages above with respect to the user query (cite [n]):"
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

    return GenerationResult(
        answer=answer,
        sources=chunks,
        question=f"Summarize: {query}",
    )


class IngestRequest(BaseModel):
    title: str = Field(..., description="Wikipedia page title to ingest")
    reset: bool = Field(False, description="Whether to reset the pgvector collection before inserting")


class IngestResponse(BaseModel):
    success: bool
    title: str
    message: str
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Return live system health (PostgreSQL, Ollama, pgvector collection)."""
    return StatusResponse(
        database=check_database(),
        ollama=check_ollama(),
        index=check_index(),
        configured_articles_count=len(settings.pages),
        configured_articles=list(settings.pages),
    )


@app.get("/api/articles")
def get_articles() -> dict[str, Any]:
    """Return pre-configured Wikipedia article titles."""
    return {
        "articles": list(settings.pages),
        "current_season": settings.current_season,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Ask a single question with retrieved context and optional grounding verification."""
    cleaned_q = req.question.strip()
    if not cleaned_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(cleaned_q)
        grounding_report = None
        if req.check_grounding and result.sources:
            grounding_report = check_answer(
                result.answer or "",
                [chunk.body_text() for chunk in result.sources],
            )

        return AskResponse(
            question=cleaned_q,
            answer=result.answer or "No answer produced.",
            sources=serialize_chunks(result.sources),
            formatted_sources=format_sources(result.sources),
            grounding_report=grounding_report,
        )
    except Exception as exc:
        logger.exception("Error answering question: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Multi-turn conversation over indexed Arsenal Wikipedia articles."""
    cleaned_q = req.question.strip()
    if not cleaned_q:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history_tuples: List[Tuple[str, str]] = [
        (turn.question, turn.answer) for turn in req.history
    ]

    try:
        result = answer_question(cleaned_q, history=history_tuples)
        return ChatResponse(
            question=cleaned_q,
            answer=result.answer or "No answer produced.",
            sources=serialize_chunks(result.sources),
            formatted_sources=format_sources(result.sources),
        )
    except Exception as exc:
        logger.exception("Error during chat: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    """Retrieve relevant docs for a query/topic and summarize them with citations."""
    text_to_summarize = (req.query or req.title or "").strip()
    if not text_to_summarize:
        raise HTTPException(status_code=400, detail="Query or title cannot be empty.")

    history_tuples: List[Tuple[str, str]] = [
        (turn.question, turn.answer) for turn in req.history
    ]

    try:
        result = summarize_docs_for_query(text_to_summarize, history=history_tuples)
        return SummarizeResponse(
            title=text_to_summarize,
            summary=result.answer or "No summary generated.",
            sources=serialize_chunks(result.sources),
            formatted_sources=format_sources(result.sources),
        )
    except Exception as exc:
        logger.exception("Error summarizing '%s': %s", text_to_summarize, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest a Wikipedia page: load -> clean -> split -> embed -> store."""
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    try:
        summary = run_ingestion(titles=[title], reset=req.reset)
        written = summary.get("written", 0)
        chunks_count = summary.get("chunks", 0)
        return IngestResponse(
            success=True,
            title=title,
            message=f"Successfully ingested '{title}': produced {chunks_count} chunks and wrote {written} vectors.",
            summary=summary,
        )
    except Exception as exc:
        logger.exception("Error ingesting '%s': %s", title, exc)
        raise HTTPException(status_code=500, detail=str(exc))
