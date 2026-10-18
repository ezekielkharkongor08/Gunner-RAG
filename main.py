"""Smoke check: verify .env resolves and that PostgreSQL + Ollama are reachable.

Run:  python main.py
"""

from __future__ import annotations

import json
import sys

from src.config import ConfigError, settings


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


def main() -> int:
    try:
        print("Resolved settings (from .env):")
        print(json.dumps(settings.redacted(), indent=2, ensure_ascii=False))
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nPostgreSQL:", check_database())
    print("Ollama:    ", check_ollama())
    print("\nWikipedia pages queued:", len(settings.pages))
    for title in settings.pages:
        print("  -", title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
