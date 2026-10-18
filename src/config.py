"""Central configuration for the whole project.

Every runtime value (database instance, LLM, embeddings, chunking, retrieval)
comes from environment variables loaded out of a ``.env`` file at the project
root.  Nothing in the codebase hard-codes a database, model or embedding
name -- change them in ``.env`` instead.

Usage:
    from src.config import settings
    print(settings.database.url, settings.ollama.chat_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"

# Loading is idempotent and safe to call from every module that imports us.
load_dotenv(ENV_FILE, override=False)


class ConfigError(RuntimeError):
    """Raised when a required setting is missing from ``.env``."""


# --- Reading helpers --------------------------------------------------------
def get_env(name: str, default: str | None = None) -> str | None:
    """Return a stripped env var, treating an empty string as "not set"."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def require_env(name: str) -> str:
    value = get_env(name)
    if value is None:
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            f"Set it in {ENV_FILE} (see {ENV_FILE.with_suffix('.env.example')})."
        )
    return value


def get_int(name: str, default: int) -> int:
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"'{name}' must be an integer, got {raw!r}.") from exc


def get_float(name: str, default: float) -> float:
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"'{name}' must be a number, got {raw!r}.") from exc


# --- Settings blocks --------------------------------------------------------
@dataclass(frozen=True)
class DatabaseSettings:
    """PostgreSQL instance + pgvector collection. All values from .env."""

    host: str
    port: int
    user: str
    password: str
    name: str
    collection: str
    url: str  # full connection string, either given or built from the parts

    @property
    def sqlalchemy_url(self) -> str:
        """Alias kept for readability -- langchain-postgres expects this form."""
        return self.url


@dataclass(frozen=True)
class OllamaSettings:
    """Local Ollama chat model + embedding model."""

    base_url: str
    chat_model: str
    embedding_model: str
    temperature: float
    embedding_doc_prefix: str = ""
    embedding_query_prefix: str = ""


@dataclass(frozen=True)
class WikipediaSettings:
    """Wikipedia fetch behaviour for ingestion Mode B."""

    lang: str
    user_agent: str
    load_max_docs: int
    max_chars: int


@dataclass(frozen=True)
class ChunkSettings:
    """Text splitting parameters."""

    size: int
    overlap: int
    min_size: int = 120
    max_warn: int = 1000


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval behaviour for answering questions."""

    top_k: int
    query_rewrite_count: int = 3
    retrieve_k: int = 10
    rerank_top_n: int = 5
    score_threshold: float = 0.4


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings
    ollama: OllamaSettings
    wikipedia: WikipediaSettings
    chunk: ChunkSettings
    retrieval: RetrievalSettings

    # From config/settings.yaml (non-secret pipeline config)
    pages: tuple[str, ...] = ()
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    junk_sections: tuple[str, ...] = ()
    csv_rows_per_chunk: int = 12
    csv_row_style: str = "pairs"

    # Convenience
    project_root: Path = PROJECT_ROOT

    def redacted(self) -> dict[str, object]:
        """Settings as a dict with the password masked -- safe to print/log."""
        return {
            "database": {
                "url": self.database.url.replace(self.database.password, "***"),
                "host": self.database.host,
                "port": self.database.port,
                "user": self.database.user,
                "name": self.database.name,
                "collection": self.database.collection,
            },
            "ollama": {
                "base_url": self.ollama.base_url,
                "chat_model": self.ollama.chat_model,
                "embedding_model": self.ollama.embedding_model,
                "temperature": self.ollama.temperature,
                "embedding_doc_prefix": self.ollama.embedding_doc_prefix,
                "embedding_query_prefix": self.ollama.embedding_query_prefix,
            },
            "wikipedia": self.wikipedia.__dict__,
            "chunk": self.chunk.__dict__,
            "retrieval": self.retrieval.__dict__,
            "pages": list(self.pages),
            "data_raw_dir": str(self.data_raw_dir),
            "data_processed_dir": str(self.data_processed_dir),
            "junk_sections": list(self.junk_sections),
            "csv_rows_per_chunk": self.csv_rows_per_chunk,
            "csv_row_style": self.csv_row_style,
        }


# --- Builder ----------------------------------------------------------------
def _build_database_url(user: str, password: str, host: str, port: int, name: str) -> str:
    """Build a psycopg connection string from the individual .env parts."""
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(name)}"


def _load_yaml() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{CONFIG_FILE} must contain a YAML mapping.")
    return data


def load_settings() -> Settings:
    """Read .env + config/settings.yaml and return fully validated settings."""
    # --- Database ---
    user = require_env("POSTGRES_USER")
    password = require_env("POSTGRES_PASSWORD")
    host = get_env("POSTGRES_HOST", "localhost")
    port = get_int("POSTGRES_PORT", 5432)
    name = require_env("POSTGRES_DB")

    # DATABASE_URL, if provided, wins over the individual parts.
    url = get_env("DATABASE_URL") or _build_database_url(user, password, host, port, name)
    database = DatabaseSettings(
        host=host, port=port, user=user, password=password, name=name,
        collection=get_env("PGVECTOR_COLLECTION", "arsenal_documents"),
        url=url,
    )

    # --- Ollama ---
    ollama = OllamaSettings(
        base_url=get_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=require_env("LLM_MODEL"),
        embedding_model=require_env("EMBEDDING_MODEL"),
        temperature=get_float("LLM_TEMPERATURE", 0.2),
        embedding_doc_prefix=get_env("EMBEDDING_DOC_PREFIX", ""),
        embedding_query_prefix=get_env("EMBEDDING_QUERY_PREFIX", ""),
    )

    # --- Wikipedia ---
    wikipedia = WikipediaSettings(
        lang=get_env("WIKIPEDIA_LANG", "en"),
        user_agent=get_env("WIKIPEDIA_USER_AGENT", "arsenal-weki-rag/0.1"),
        load_max_docs=get_int("WIKIPEDIA_LOAD_MAX_DOCS", 2),
        max_chars=get_int("WIKIPEDIA_MAX_CHARS", 200_000),
    )

    # --- Chunking / retrieval ---
    chunk = ChunkSettings(
        size=get_int("CHUNK_SIZE", 1000),
        overlap=get_int("CHUNK_OVERLAP", 150),
        min_size=get_int("CHUNK_MIN_SIZE", 120),
        max_warn=get_int("CHUNK_MAX_WARN", 1000),
    )
    retrieval = RetrievalSettings(
        top_k=get_int("TOP_K", 5),
        query_rewrite_count=get_int("QUERY_REWRITE_COUNT", 3),
        retrieve_k=get_int("RETRIEVE_K", 10),
        rerank_top_n=get_int("RERANK_TOP_N", 5),
        score_threshold=get_float("SCORE_THRESHOLD", 0.4),
    )

    # --- Non-secret pipeline config ---
    yaml_cfg = _load_yaml()
    raw_rel = yaml_cfg.get("data_dirs", {}).get("raw", "data/raw")
    processed_rel = yaml_cfg.get("data_dirs", {}).get("processed", "data/processed")
    pages = tuple(str(p) for p in (yaml_cfg.get("pages") or []))
    junk_sections = tuple(
        str(s) for s in (yaml_cfg.get("junk_sections") or [])
    )
    csv_rows_per_chunk = int(
        yaml_cfg.get("csv_rows_per_chunk", 12) or 12
    )
    csv_row_style = str(
        yaml_cfg.get("csv_row_style", "pairs") or "pairs"
    )

    return Settings(
        database=database,
        ollama=ollama,
        wikipedia=wikipedia,
        chunk=chunk,
        retrieval=retrieval,
        pages=pages,
        data_raw_dir=PROJECT_ROOT / raw_rel,
        data_processed_dir=PROJECT_ROOT / processed_rel,
        junk_sections=junk_sections,
        csv_rows_per_chunk=csv_rows_per_chunk,
        csv_row_style=csv_row_style,
    )


# Module-level singleton: `from src.config import settings`
settings: Settings = load_settings()
