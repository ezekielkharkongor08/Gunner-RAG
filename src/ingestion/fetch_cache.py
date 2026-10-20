"""fetch_cache.py -- local cache for Wikipedia extracts.

The season page changes often. Every experiment must use exactly the same
text, so the first fetch is saved to disk and later runs read it back.

Layout::

    data/cache/index.json          title -> cache file
    data/cache/<pageid>_<rev>.json raw extract + metadata

The raw extract is stored (not parsed sections), so a parser bug can be
re-checked without another network call. ``REFRESH_FETCH_CACHE=true`` forces
a fresh fetch and updates the pointer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

INDEX_NAME = "index.json"


def cache_dir(path: Path | None = None) -> Path:
    """Return the cache directory (created if missing)."""
    directory = Path(path) if path is not None else settings.cache.directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _index_path(directory: Path) -> Path:
    return directory / INDEX_NAME


def _read_index(directory: Path) -> dict:
    path = _index_path(directory)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read fetch cache index %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_index(directory: Path, index: dict) -> None:
    path = _index_path(directory)
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cache_file_name(article: dict) -> str:
    """Stable file name for one page revision."""
    pageid = article.get("pageid") or "nopage"
    revision = article.get("revision_id") or "norev"
    return f"{pageid}_{revision}.json"


def load_cached(title: str, path: Path | None = None) -> dict | None:
    """Return the cached article for ``title``, or None on a miss."""
    directory = cache_dir(path)
    entry = _read_index(directory).get(title)
    if not entry:
        return None
    file_path = directory / str(entry.get("file", ""))
    if not file_path.exists():
        logger.warning("Cache pointer for %r is missing: %s", title, file_path)
        return None
    try:
        article = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read cached article %s: %s", file_path, exc)
        return None
    logger.info(
        "Fetch cache hit for %r (revision %s).",
        title, article.get("revision_id"),
    )
    return article


def save_cached(title: str, article: dict, path: Path | None = None) -> Path:
    """Write one article to the cache and point ``title`` at it."""
    directory = cache_dir(path)
    file_name = cache_file_name(article)
    file_path = directory / file_name
    file_path.write_text(
        json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    index = _read_index(directory)
    index[title] = {
        "file": file_name,
        "pageid": article.get("pageid"),
        "revision_id": article.get("revision_id"),
        "fetched_at": article.get("fetched_at"),
    }
    _write_index(directory, index)
    logger.info("Cached %r as %s", title, file_name)
    return file_path
