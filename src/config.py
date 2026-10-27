"""Central configuration for the whole project.

Every runtime value (database instance, LLM, embeddings, chunking, retrieval,
Wikipedia API) comes from environment variables loaded out of a ``.env`` file
at the project root, plus non-secret pipeline config from
``config/settings.yaml``. Nothing in the codebase hard-codes a database, model
or embedding name -- change them in ``.env`` instead.

The pipeline is Wikipedia-only and prose-only: there is no local-file
ingestion and no table processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"

# Loading is idempotent and safe to call from every module that imports us.
load_dotenv(ENV_FILE, override=False)

# Used only if settings.yaml has no season_normalise_pattern.
# Matches 2025-2026, 2025-26, 2025/26 and the en-dash form.
DEFAULT_SEASON_PATTERN = r"\b((?:19|20)\d{2})\s*[–\-/]\s*(\d{2,4})\b"

# Allowed values for RETRIEVAL_GATE.
GATE_MODES = ("relative", "absolute", "none")

# Allowed values for QUERY_REWRITE_MODE.
REWRITE_MODES = ("off", "always", "fallback")
ANSWER_STYLES = ("short", "normal", "long")


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


def get_bool(name: str, default: bool) -> bool:
    """Read a true/false env var. Accepts 1/0, yes/no, on/off."""
    raw = get_env(name)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"'{name}' must be true or false, got {raw!r}.")


def get_path(name: str, default: str | Path) -> Path:
    """Read a path from .env; relative paths are relative to the project root."""
    raw = get_env(name)
    path = Path(raw) if raw else Path(default)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_patterns(raw: object) -> tuple[dict[str, object], ...]:
    """Validate a YAML list of rule dicts (pattern + extra keys)."""
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ConfigError("Expected a YAML list of rule mappings.")
    patterns: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or "pattern" not in item:
            raise ConfigError(f"Rule must be a mapping with 'pattern': {item!r}.")
        patterns.append(dict(item))
    return tuple(patterns)


def _parse_hit_at(raw: str) -> tuple[int, ...]:
    """Parse EVAL_HIT_AT=1,3,5,8 into a sorted tuple."""
    try:
        values = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    except ValueError as exc:
        raise ConfigError(f"EVAL_HIT_AT must be comma-separated integers, got {raw!r}.") from exc
    if not values:
        raise ConfigError("EVAL_HIT_AT must contain at least one number.")
    return values


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
    # Qwen3 works better with a query instruction, but it is off by default.
    # The eval decides whether it stays on.
    embedding_query_instruction: str = ""
    embedding_query_instruction_enabled: bool = False
    # Context window sent to the chat model (OLLAMA_NUM_CTX).
    num_ctx: int = 8192
    # Temperature for answer generation (ANSWER_TEMPERATURE); rewriting has
    # its own temperature in RetrievalSettings.
    answer_temperature: float = 0.2


@dataclass(frozen=True)
class WikipediaSettings:
    """Wikipedia fetch behaviour for the MediaWiki Action API.

    The API endpoint is configurable through ``WIKIPEDIA_API_URL``; when unset
    it defaults to ``https://{lang}.wikipedia.org/w/api.php`` so any language
    edition can be targeted.
    """

    lang: str
    user_agent: str
    api_url: Optional[str]
    timeout: int
    max_retries: int


@dataclass(frozen=True)
class CacheSettings:
    """Raw Wikipedia extract cache, used to keep experiments reproducible."""

    enabled: bool
    refresh: bool
    directory: Path


@dataclass(frozen=True)
class ChunkSettings:
    """Text splitting and section assembly parameters.

    ``size`` is the body budget. The title + section header is added on top,
    so the total chunk can be a little longer. ``max_warn`` is only a warning.
    """

    size: int
    overlap: int
    min_size: int = 150
    max_warn: int = 1200
    header_enabled: bool = True
    batch_size: int = 32
    min_section_chars: int = 250
    min_section_keep: int = 60
    min_section_max_passes: int = 3
    boilerplate_max_chars: int = 300
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", " ", "")


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval behaviour for answering questions."""

    final_top_n: int = 8
    query_rewrite_count: int = 3
    # off | always | fallback (rewrite only when the first search is weak).
    query_rewrite_mode: str = "fallback"
    # Derived from the mode; kept for callers and old configs.
    query_rewrite_enabled: bool = True
    # In fallback mode, rewrite when no chunk scores at least this close.
    query_rewrite_weak_score: float = 0.65
    retrieve_k: int = 15
    score_threshold: float = 0.7
    query_rewrite_temperature: float = 0.0
    # Strip the stored "title + Section:" header before sending a chunk to the
    # LLM (the numbered Sources list replaces it).
    context_strip_header: bool = True
    # When nothing passes SCORE_THRESHOLD: False -> return no chunks, so the
    # model can say the indexed pages do not contain the answer.
    fallback_on_empty: bool = False
    # Questions that name a section ("November", "kits") are routed straight to
    # that section instead of vector search. Tokens live in settings.yaml.
    section_routing_enabled: bool = True
    max_section_chunks: int = 16
    # Condense a short follow-up ("who did they beat?") into a standalone
    # question using the conversation, before searching.
    condense_followups: bool = True
    # A question with at most this many words counts as a possible follow-up.
    followup_max_words: int = 6
    # How many recent (question, answer) pairs the condenser may read. One is
    # deliberate: with two, "who did they beat at home?" merged November + March.
    condense_history_turns: int = 1
    # How to cut weak results: "relative" keeps chunks close to the best one,
    # "absolute" uses SCORE_THRESHOLD, "none" keeps the top k.
    retrieval_gate: str = "relative"
    relative_margin: float = 0.10


@dataclass(frozen=True)
class EvalSettings:
    """Retrieval-only eval settings (no LLM call)."""

    top_k: int = 8
    hit_at: tuple[int, ...] = (1, 3, 5, 8)
    collection_prefix: str = "arsenal_eval_"
    results_dir: Path = PROJECT_ROOT / "eval" / "results"
    report_path: Path = PROJECT_ROOT / "test_report.txt"


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings
    ollama: OllamaSettings
    wikipedia: WikipediaSettings
    chunk: ChunkSettings
    retrieval: RetrievalSettings
    cache: CacheSettings
    eval: EvalSettings

    # From config/settings.yaml (non-secret pipeline config)
    pages: tuple[str, ...] = ()
    current_season: str = ""
    section_tokens: tuple[str, ...] = ()
    followup_pronouns: tuple[str, ...] = ()
    junk_sections: tuple[str, ...] = ()
    boilerplate_patterns: tuple[dict[str, object], ...] = ()
    page_type_rules: tuple[dict[str, object], ...] = ()

    # Season metadata. Title only by default (the user decision).
    seasons_from_text: bool = False
    seasons_max: int = 8
    season_normalise_pattern: str = DEFAULT_SEASON_PATTERN

    # Convenience
    project_root: Path = PROJECT_ROOT

    # Chat: how many (question, answer) pairs to keep for follow-ups.
    chat_history_turns: int = 4

    # Chat: how much detail an answer should give (short | normal | long).
    answer_style: str = "short"

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
                "embedding_query_instruction": self.ollama.embedding_query_instruction,
                "embedding_query_instruction_enabled": (
                    self.ollama.embedding_query_instruction_enabled
                ),
                "num_ctx": self.ollama.num_ctx,
                "answer_temperature": self.ollama.answer_temperature,
            },
            "wikipedia": {
                "lang": self.wikipedia.lang,
                "api_url": self.wikipedia.api_url,
                "timeout": self.wikipedia.timeout,
                "max_retries": self.wikipedia.max_retries,
            },
            "chunk": {
                "size": self.chunk.size,
                "overlap": self.chunk.overlap,
                "min_size": self.chunk.min_size,
                "max_warn": self.chunk.max_warn,
                "header_enabled": self.chunk.header_enabled,
                "batch_size": self.chunk.batch_size,
                "min_section_chars": self.chunk.min_section_chars,
                "min_section_keep": self.chunk.min_section_keep,
                "min_section_max_passes": self.chunk.min_section_max_passes,
                "boilerplate_max_chars": self.chunk.boilerplate_max_chars,
                "separators": list(self.chunk.separators),
            },
            "retrieval": {
                "final_top_n": self.retrieval.final_top_n,
                "query_rewrite_count": self.retrieval.query_rewrite_count,
                "query_rewrite_mode": self.retrieval.query_rewrite_mode,
                "query_rewrite_enabled": self.retrieval.query_rewrite_enabled,
                "query_rewrite_weak_score": (
                    self.retrieval.query_rewrite_weak_score
                ),
                "retrieve_k": self.retrieval.retrieve_k,
                "score_threshold": self.retrieval.score_threshold,
                "query_rewrite_temperature": (
                    self.retrieval.query_rewrite_temperature
                ),
                "context_strip_header": self.retrieval.context_strip_header,
                "fallback_on_empty": self.retrieval.fallback_on_empty,
                "section_routing_enabled": self.retrieval.section_routing_enabled,
                "max_section_chunks": self.retrieval.max_section_chunks,
                "retrieval_gate": self.retrieval.retrieval_gate,
                "relative_margin": self.retrieval.relative_margin,
                "condense_followups": self.retrieval.condense_followups,
                "followup_max_words": self.retrieval.followup_max_words,
                "condense_history_turns": (
                    self.retrieval.condense_history_turns
                ),
            },
            "cache": {
                "enabled": self.cache.enabled,
                "refresh": self.cache.refresh,
                "directory": str(self.cache.directory),
            },
            "eval": {
                "top_k": self.eval.top_k,
                "hit_at": list(self.eval.hit_at),
                "collection_prefix": self.eval.collection_prefix,
                "results_dir": str(self.eval.results_dir),
                "report_path": str(self.eval.report_path),
            },
            "pages": list(self.pages),
            "current_season": self.current_season,
            "answer_style": self.answer_style,
            "section_tokens": list(self.section_tokens),
            "followup_pronouns": list(self.followup_pronouns),
            "junk_sections": list(self.junk_sections),
            "boilerplate_patterns": list(self.boilerplate_patterns),
            "page_type_rules": list(self.page_type_rules),
            "seasons_from_text": self.seasons_from_text,
            "seasons_max": self.seasons_max,
            "season_normalise_pattern": self.season_normalise_pattern,
            "chat_history_turns": self.chat_history_turns,
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
    yaml_cfg = _load_yaml()

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
    # LLM_MODEL is the old key; OLLAMA_MODEL is the current one in .env.
    chat_model = get_env("LLM_MODEL") or require_env("OLLAMA_MODEL")
    ollama = OllamaSettings(
        base_url=get_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=chat_model,
        embedding_model=require_env("EMBEDDING_MODEL"),
        temperature=get_float("LLM_TEMPERATURE", 0.2),
        embedding_doc_prefix=get_env("EMBEDDING_DOC_PREFIX", ""),
        embedding_query_prefix=get_env("EMBEDDING_QUERY_PREFIX", ""),
        embedding_query_instruction=get_env("EMBEDDING_QUERY_INSTRUCTION", ""),
        embedding_query_instruction_enabled=get_bool(
            "EMBEDDING_QUERY_INSTRUCTION_ENABLED", False
        ),
        num_ctx=get_int("OLLAMA_NUM_CTX", 8192),
        answer_temperature=get_float(
            "ANSWER_TEMPERATURE", get_float("LLM_TEMPERATURE", 0.1)
        ),
    )

    # --- Wikipedia (MediaWiki Action API) ---
    wikipedia = WikipediaSettings(
        lang=get_env("WIKIPEDIA_LANGUAGE") or get_env("WIKIPEDIA_LANG", "en"),
        user_agent=get_env("WIKIPEDIA_USER_AGENT", "arsenal-weki-rag/0.1"),
        api_url=get_env("WIKIPEDIA_API_URL"),
        timeout=get_int("WIKIPEDIA_TIMEOUT", 30),
        max_retries=get_int("WIKIPEDIA_MAX_RETRIES", 3),
    )

    # --- Chunking ---
    default_separators = ["\n\n", "\n", ". ", "; ", " ", ""]
    separators = tuple(
        str(item) for item in (yaml_cfg.get("split_separators") or default_separators)
    )
    chunk = ChunkSettings(
        size=get_int("CHUNK_SIZE", 900),
        overlap=get_int("CHUNK_OVERLAP", 120),
        min_size=get_int("CHUNK_MIN_SIZE", 150),
        max_warn=get_int("CHUNK_MAX_WARN", 1200),
        header_enabled=get_bool("CHUNK_HEADER_ENABLED", True),
        batch_size=get_int("INDEX_BATCH_SIZE", 32),
        min_section_chars=get_int("MIN_SECTION_CHARS", 250),
        min_section_keep=get_int("MIN_SECTION_KEEP", 60),
        min_section_max_passes=get_int("MIN_SECTION_MAX_PASSES", 3),
        boilerplate_max_chars=get_int("BOILERPLATE_MAX_CHARS", 300),
        separators=separators,
    )

    # --- Retrieval ---
    gate = get_env("RETRIEVAL_GATE", "relative").strip().lower()
    if gate not in GATE_MODES:
        raise ConfigError(
            f"RETRIEVAL_GATE must be one of {GATE_MODES}, got {gate!r}"
        )

    # QUERY_REWRITE_MODE is the switch; the old boolean stays as a fallback.
    legacy_rewrite = get_bool("QUERY_REWRITE_ENABLED", False)
    rewrite_mode = (
        get_env("QUERY_REWRITE_MODE") or ("always" if legacy_rewrite else "fallback")
    ).strip().lower()
    if rewrite_mode not in REWRITE_MODES:
        raise ConfigError(
            f"QUERY_REWRITE_MODE must be one of {REWRITE_MODES}, got {rewrite_mode!r}"
        )

    retrieval = RetrievalSettings(
        final_top_n=get_int("FINAL_TOP_N", 8),
        query_rewrite_count=get_int("QUERY_REWRITE_COUNT", 3),
        query_rewrite_mode=rewrite_mode,
        query_rewrite_enabled=rewrite_mode != "off",
        query_rewrite_weak_score=get_float("QUERY_REWRITE_WEAK_SCORE", 0.65),
        retrieve_k=get_int("RETRIEVE_K", 15),
        score_threshold=get_float("SCORE_THRESHOLD", 0.7),
        query_rewrite_temperature=get_float("QUERY_REWRITE_TEMPERATURE", 0.0),
        context_strip_header=get_bool("CONTEXT_STRIP_HEADER", True),
        fallback_on_empty=get_bool("RETRIEVAL_FALLBACK_ON_EMPTY", False),
        section_routing_enabled=get_bool("SECTION_ROUTING_ENABLED", True),
        max_section_chunks=get_int("MAX_SECTION_CHUNKS", 16),
        condense_followups=get_bool("QUERY_CONDENSE_ENABLED", True),
        followup_max_words=get_int("FOLLOWUP_MAX_WORDS", 6),
        condense_history_turns=get_int("CONDENSE_HISTORY_TURNS", 1),
        retrieval_gate=gate,
        relative_margin=get_float("RELATIVE_MARGIN", 0.10),
    )

    # --- Cache ---
    cache = CacheSettings(
        enabled=get_bool("USE_FETCH_CACHE", True),
        refresh=get_bool("REFRESH_FETCH_CACHE", False),
        directory=get_path("FETCH_CACHE_DIR", "data/cache"),
    )

    # --- Eval ---
    eval_settings = EvalSettings(
        top_k=get_int("EVAL_TOP_K", 8),
        hit_at=_parse_hit_at(get_env("EVAL_HIT_AT", "1,3,5,8")),
        collection_prefix=get_env("EVAL_COLLECTION_PREFIX", "arsenal_eval_"),
        results_dir=get_path("EVAL_RESULTS_DIR", "eval/results"),
        report_path=get_path("TEST_REPORT_PATH", "test_report.txt"),
    )

    # --- Non-secret pipeline config ---
    pages = tuple(str(p) for p in (yaml_cfg.get("pages") or []))
    current_season = str(yaml_cfg.get("current_season") or "").strip()
    answer_style = str(yaml_cfg.get("answer_style") or "short").strip().lower()
    if answer_style not in ANSWER_STYLES:
        raise ConfigError(
            f"answer_style must be one of {ANSWER_STYLES}, got {answer_style!r}"
        )
    section_tokens = tuple(
        str(t) for t in (yaml_cfg.get("section_tokens") or [])
    )
    followup_pronouns = tuple(
        str(word).casefold() for word in (yaml_cfg.get("followup_pronouns") or [])
    )
    junk_sections = tuple(
        str(s) for s in (yaml_cfg.get("junk_sections") or [])
    )
    boilerplate_patterns = _load_patterns(yaml_cfg.get("boilerplate_patterns"))
    page_type_rules = _load_patterns(yaml_cfg.get("page_type_rules"))
    season_normalise_pattern = str(
        yaml_cfg.get("season_normalise_pattern") or DEFAULT_SEASON_PATTERN
    )

    return Settings(
        database=database,
        ollama=ollama,
        wikipedia=wikipedia,
        chunk=chunk,
        retrieval=retrieval,
        cache=cache,
        eval=eval_settings,
        pages=pages,
        current_season=current_season,
        answer_style=answer_style,
        section_tokens=section_tokens,
        followup_pronouns=followup_pronouns,
        junk_sections=junk_sections,
        boilerplate_patterns=boilerplate_patterns,
        page_type_rules=page_type_rules,
        seasons_from_text=get_bool("SEASONS_FROM_TEXT", False),
        seasons_max=get_int("SEASONS_MAX", 8),
        chat_history_turns=get_int("CHAT_HISTORY_TURNS", 4),
        season_normalise_pattern=season_normalise_pattern,
    )


# Module-level singleton: `from src.config import settings`
settings: Settings = load_settings()
