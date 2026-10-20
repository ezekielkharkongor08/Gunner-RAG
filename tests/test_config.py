"""Settings audit: keep src/config.py, the code and .env.example in sync.

Three rules:

    * every env var the code reads is documented in .env.example
    * every key in .env.example is actually read somewhere in src/
    * the cache settings carry the ``refresh`` flag (loader reads it)

These tests are static: they read files, they never touch the database or the
network.
"""

from __future__ import annotations

import re

import pytest

from src.config import PROJECT_ROOT, settings

READERS = ("get_env", "get_int", "get_float", "get_bool", "get_path", "require_env")
# get_env("NAME", ...) / get_int("NAME", ...) with the name in double quotes.
CALL = re.compile(
    r"(?:" + "|".join(READERS) + r")\(\s*\"([A-Z][A-Z0-9_]*)\""
)
# KEY=value lines in .env.example (comments are ignored on purpose here).
ASSIGNMENT = re.compile(r"(?m)^\s*([A-Z][A-Z0-9_]*)=")

EXAMPLE_FILE = PROJECT_ROOT / ".env.example"


def _example_text() -> str:
    return EXAMPLE_FILE.read_text(encoding="utf-8")


def _read_env_names() -> set[str]:
    """Every env var name read through the config helpers, anywhere in src/."""
    names: set[str] = set()
    for path in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        names |= set(CALL.findall(path.read_text(encoding="utf-8")))
    return names


def test_env_example_exists():
    assert EXAMPLE_FILE.exists(), f"{EXAMPLE_FILE} is missing"


def test_every_env_var_the_code_reads_is_documented():
    example = _example_text()
    read = _read_env_names()
    assert read, "no env var names found in src/ -- the regex or the code changed"
    # The whole file counts, so a commented-out optional key is still "documented".
    missing = sorted(name for name in read if name not in example)
    assert not missing, (
        ".env.example does not document these keys read by the code: "
        f"{missing}"
    )


def test_every_env_example_key_is_read_by_the_code():
    read = _read_env_names()
    # Keys used by tooling only (not read through src.config helpers).
    ignored = {"ARSENAL_WEKI_INTEGRATION"}
    keys = set(ASSIGNMENT.findall(_example_text())) - ignored
    unread = sorted(key for key in keys if key not in read)
    assert not unread, (
        ".env.example documents keys that nothing reads "
        f"(delete them or wire them up): {unread}"
    )


def test_required_keys_are_in_env_example():
    example = _example_text()
    required = (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
    )
    missing = [name for name in required if name not in example]
    assert not missing, f"required keys missing from .env.example: {missing}"


def test_cache_settings_expose_refresh():
    """wikipedia_loader reads settings.cache.refresh -- it must exist."""
    assert hasattr(settings.cache, "refresh")
    assert isinstance(settings.cache.refresh, bool)
    assert settings.cache.directory.parts[-2:] == ("data", "cache") or (
        settings.cache.directory.is_absolute()
    )


@pytest.mark.parametrize(
    "name",
    ["size", "overlap", "min_size", "max_warn", "header_enabled", "batch_size",
     "min_section_chars", "min_section_keep", "min_section_max_passes",
     "separators"],
)
def test_chunk_settings_are_loaded(name):
    """Phase 1 chunk settings must exist on the settings object."""
    assert hasattr(settings.chunk, name), f"chunk.{name} is missing"
