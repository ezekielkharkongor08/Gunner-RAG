"""Smoke check, chat, and summarize entry point for Arsenal Weki RAG.

Run:  python main.py                                          # settings, PostgreSQL, Ollama smoke check
      python main.py --chat                                   # interactive Q&A over ingested articles
      python main.py --chat "Your question here"              # single question, then exit
      python main.py --summarize "Article Title"              # two-step retrieve-then-summarize an article
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from src.config import ConfigError, settings
from src.generation.answer_chain import (
    answer_question,
    format_sources,
    summarize_article,
)


def check_database() -> str:
    try:
        import psycopg
    except ImportError:
        return "SKIP (psycopg not installed yet -- uv pip install -r requirements.txt)"

    try:
        with psycopg.connect(
            host=settings.database.host,
            port=settings.database.port,
            user=settings.database.user,
            password=settings.database.password,
            dbname=settings.database.name,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0].split(",")[0]
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                has_vector = cur.fetchone() is not None
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        return f"FAIL ({type(exc).__name__}: {exc})"

    return f"OK -- {version}, pgvector={'installed' if has_vector else 'MISSING (CREATE EXTENSION vector;)'}"


def check_ollama() -> str:
    import requests

    try:
        response = requests.get(f"{settings.ollama.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        tags = {model["name"] for model in response.json().get("models", [])}
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__}: {exc})"

    def present(name: str) -> str:
        return "OK" if any(t == name or t.split(":")[0] == name.split(":")[0] for t in tags) else "MISSING"

    return (
        f"chat={present(settings.ollama.chat_model)} "
        f"embeddings={present(settings.ollama.embedding_model)}"
    )


def check_index() -> str:
    """Quick check that the pgvector collection has indexed chunks."""
    try:
        from src.ingestion.index_builder import index_status
        status = index_status()
        return f"OK -- {status.get('chunks', 0)} chunk(s) in collection '{settings.database.collection}'"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__}: {exc})"


def chat_once(
    question: str,
    history: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Ask a single question and print the answer with citations."""
    result = answer_question(question, history=history)
    print(f"\nAnswer:\n{result.answer}")
    if result.sources:
        print("\nSources:")
        print(format_sources(result.sources))
    else:
        print("\nNo sources found.")


def summarize_once(title: str) -> None:
    """Summarize a Wikipedia article using two-step retrieve-then-summarize."""
    result = summarize_article(title)
    print(f"\nArticle: {title}")
    print(f"\nSummary:\n{result.answer}")
    if result.sources:
        print("\nSources:")
        print(format_sources(result.sources))
    else:
        print("\nNo sources found.")


def chat_loop() -> None:
    """Interactive chat loop over ingested Wikipedia articles."""
    print("Arsenal Weki — Chat over Wikipedia articles")
    print("Type your question and press Enter. Type 'exit' or 'quit' to leave.\n")
    print("Tip: make sure you've run ingestion first "
          "(python src/ingestion/run_ingestion.py).\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            return

        try:
            result = answer_question(question, history=history)
            print(f"\nAssistant: {result.answer}")
            if result.sources:
                print("Sources:")
                print(format_sources(result.sources))
            print()

            # Keep (question, answer) pairs so follow-ups resolve: the answer
            # tells the model what "they" or "that" was about.
            history.append((question, result.answer))
            if len(history) > settings.chat_history_turns:
                history = history[-settings.chat_history_turns:]

        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            print(f"\nAssistant: Sorry, I ran into an error: {exc}\n")


def main() -> int:
    try:
        print("Resolved settings (from .env):")
        print(json.dumps(settings.redacted(), indent=2, ensure_ascii=False))
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nPostgreSQL:", check_database())
    print("Ollama:    ", check_ollama())
    print("Index:     ", check_index())
    print("\nWikipedia pages queued:", len(settings.pages))
    for title in settings.pages:
        print("  -", title)

    # --- Chat mode -------------------------------------------------------
    if "--chat" in sys.argv:
        # Extract a single question if passed after --chat.
        single_question: Optional[str] = None
        args = sys.argv[1:]
        if "--chat" in args:
            idx = args.index("--chat")
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                single_question = args[idx + 1]

        if single_question:
            chat_once(single_question)
        else:
            chat_loop()
        return 0

    # --- Summarize mode --------------------------------------------------
    if "--summarize" in sys.argv:
        args = sys.argv[1:]
        if "--summarize" in args:
            idx = args.index("--summarize")
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                title = args[idx + 1]
            else:
                print("ERROR: --summarize requires an article title.", file=sys.stderr)
                return 1
        else:
            print("ERROR: --summarize requires an article title.", file=sys.stderr)
            return 1

        summarize_once(title)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
