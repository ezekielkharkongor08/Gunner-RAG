"""wikipedia_loader.py -- stage 1 of the ingestion pipeline.

Turns *content* into LangChain ``Document`` objects. It does not clean, split
or embed anything; that happens downstream (document_cleaner -> text_splitter
-> index_builder).

There are two modes, matching ingestion.md:

    Mode A -- load_local_documents()   read local .txt / .md / .csv files
    Mode B -- fetch_wikipedia_documents() fetch pages through the Wikipedia API

Both return the same thing: a list of ``Document`` objects whose metadata
always contains at least ``title`` and ``source_url``.

Nothing here is hard-coded -- every value comes from ``.env`` (and
``config/settings.yaml``) through ``src.config``.

Run a quick demo with:
    python -m src.ingestion.wikipedia_loader
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote

from langchain_core.documents import Document

from src.config import get_env, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-type constants
# ---------------------------------------------------------------------------
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
CSV_SUFFIXES = {".csv"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | CSV_SUFFIXES

# A header line looks like "KEY: value". These are the keys we care about,
# normalised (lowercase + spaces turned into underscores).
# "SOURCE URL: https://..." -> source_url
# "# TABLE_TOPIC: Transfers in" -> title
HEADER_KEY_ALIASES = {
    "source_url": "source_url",
    "url": "source_url",
    "canonical_url": "source_url",
    "table_topic": "title",
    "page_title": "title",
    "subject": "title",
    "topic": "title",
    "title": "title",
    # CSV-specific headers captured for the splitter.
    "column_notes": "column_notes",
    "row_template": "row_template",
    "summary_columns": "summary_columns",
}

# Lines made only of these characters are decorative separators, e.g.
# "======================================================================"
SEPARATOR_CHARS = {"=", "-", "#", "*", "_", "~"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _normalise_key(raw_key: str) -> str:
    """'SOURCE URL' -> 'source_url'"""
    return re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")


def _split_header(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Peel the metadata block off the top of a file.

    Returns ``(metadata, body_lines)``. Header lines look like "KEY: value";
    decorative separators and blank lines are skipped. Parsing stops at the
    first real content line.
    """
    metadata: dict[str, str] = {}
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()

        if not stripped or set(stripped) <= SEPARATOR_CHARS:
            index += 1
            continue

        # "# KEY: value" or "KEY: value"
        candidate = stripped.lstrip("#").strip()
        match = re.match(r"^([A-Za-z][A-Za-z _]{1,30})\s*:\s*(.+)$", candidate)
        if match:
            key = _normalise_key(match.group(1))
            mapped = HEADER_KEY_ALIASES.get(key)
            if mapped:
                metadata[mapped] = match.group(2).strip()
            index += 1
            continue

        break  # reached real content

    return metadata, lines[index:]


def _title_from_path(path: Path) -> str:
    """Fall back to the filename when a file carries no title metadata."""
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _page_label_from_url(url: str) -> str:
    """Human-readable label taken from a Wikipedia URL, or "" for anything else.

    '.../wiki/List_of_Arsenal_F.C._records_and_statistics'
        -> 'List of Arsenal F.C. records and statistics'

    Tables need this context: a title like "Top goalscorers" never mentions
    Arsenal, so keyword search would not connect the query "top scorer" to it.
    """
    if "wikipedia.org" not in url:
        return ""
    tail = url.rstrip("/").split("/")[-1]
    if tail.lower().endswith(".html"):
        tail = tail[: -len(".html")]
    return unquote(tail).replace("_", " ").strip()


def _utc_now() -> str:
    """Current UTC time in ISO 8601 format, for ``fetched_at`` metadata."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Mode A: local files
# ---------------------------------------------------------------------------
def _document_from_text_file(path: Path) -> Document:
    """Read a .txt / .md file and turn it into a Document."""
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata, body_lines = _split_header(text.splitlines())
    page_content = "\n".join(body_lines).strip()

    title = metadata.get("title") or _title_from_path(path)
    source_url = metadata.get("source_url", "")

    return Document(
        page_content=page_content,
        metadata={
            "title": title,
            "source_url": source_url,          # canonical URL if the file had one
            "source": "local",
            "origin": str(path.relative_to(settings.project_root)),
            "file_type": path.suffix.lstrip(".").lower(),
            "fetched_at": _utc_now(),
        },
    )


def _csv_to_row_lines(rows: list[list[str]]) -> str:
    """Render a CSV as one self-describing line per row.

    A pipe table such as "Rank | Player | 558 (11) | 70 (1)" carries almost no
    semantic signal -- an embedding model sees mostly digits and separators,
    so a long prose chunk merely mentioning "records" outranks it.

    Writing each row as "- Player: Thierry Henry, Total Games/Goals: 377 / 228"
    makes the names and numbers searchable, while staying in file order so the
    table still reads top-to-bottom for the LLM.
    """
    header, *data_rows = rows
    header = [cell.strip() for cell in header]
    lines: list[str] = []

    for row in data_rows:
        cells = (list(row) + [""] * len(header))[:len(header)]
        pairs = [
            f"{name}: {value.strip()}"
            for name, value in zip(header, cells)
            if name and value and value.strip()
        ]
        if pairs:
            lines.append("- " + ", ".join(pairs))

    return "\n".join(lines)


def _document_from_csv_file(path: Path) -> Document:
    """Read a .csv file (with an optional '# KEY: value' header block).

    Captures special CSV headers used by the splitter:
      # COLUMN_NOTES: human-readable column explanations
      # ROW_TEMPLATE: sentence template for sentence-style rows
      # SUMMARY_COLUMNS: columns to build a top-3 summary chunk for
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata, body_lines = _split_header(text.splitlines())

    # Rejoin and parse with the csv module, dropping completely blank rows.
    body = "\n".join(body_lines)
    rows = [row for row in csv.reader(io.StringIO(body)) if any(c.strip() for c in row)]

    header_row = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []

    if not data_rows:
        logger.warning("CSV file has no data rows: %s", path)
        table = ""
    else:
        table = _csv_to_row_lines(rows)

    title = metadata.get("title") or _title_from_path(path)
    source_url = metadata.get("source_url", "")
    page_context = _page_label_from_url(source_url)

    # Single newline: a blank line here would make the splitter treat the
    # title as its own tiny chunk instead of grouping it with the table.
    # The context line says which page the table came from.
    lines = [title]
    if page_context:
        lines.append(page_context)
    if table:
        lines.append(table)
    page_content = "\n".join(lines)

    import json as _json

    return Document(
        page_content=page_content,
        metadata={
            "title": title,
            "source_url": source_url,
            "page_context": page_context,
            "source": "local",
            "origin": str(path.relative_to(settings.project_root)),
            "file_type": "csv",
            "columns": len(header_row) if header_row else 0,
            "csv_header_row": [c.strip() for c in header_row],
            "csv_rows": _json.dumps(data_rows, ensure_ascii=False),
            "csv_column_notes": metadata.get("column_notes", ""),
            "csv_row_template": metadata.get("row_template", ""),
            "csv_summary_columns": metadata.get("summary_columns", ""),
            "fetched_at": _utc_now(),
        },
    )


def load_local_documents(
    roots: Sequence[Path] | None = None,
    suffixes: set[str] | None = None,
) -> list[Document]:
    """Mode A -- load every supported file under the given folders.

    Args:
        roots: folders to walk. Defaults to data/raw + data/processed.
        suffixes: which file extensions to accept.

    Returns:
        One Document per file.
    """
    if roots is None:
        roots = [settings.data_raw_dir, settings.data_processed_dir]
    if suffixes is None:
        suffixes = SUPPORTED_SUFFIXES

    documents: list[Document] = []
    seen: set[Path] = set()

    for root in roots:
        root = Path(root)
        if not root.exists():
            logger.info("Skipping missing folder: %s", root)
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path in seen:
                continue
            seen.add(path)

            try:
                if path.suffix.lower() in CSV_SUFFIXES:
                    documents.append(_document_from_csv_file(path))
                else:
                    documents.append(_document_from_text_file(path))
            except OSError as exc:
                logger.error("Could not read %s: %s", path, exc)

    logger.info("Mode A loaded %d local file(s).", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Mode B: Wikipedia API
# ---------------------------------------------------------------------------
def _build_wiki_client():
    """Create a wikipediaapi client from .env settings."""
    import wikipediaapi  # imported late so Mode A works without it installed

    kwargs = {
        "user_agent": settings.wikipedia.user_agent,
        "language": settings.wikipedia.lang,
        "extract_format": wikipediaapi.ExtractFormat.WIKI,
    }
    try:
        # Wikipedia-API >= 0.11 supports retry configuration.
        return wikipediaapi.Wikipedia(max_retries=3, retry_wait=1.0, **kwargs)
    except TypeError:
        return wikipediaapi.Wikipedia(**kwargs)


def _wiki_revision(page) -> str | None:
    """Best effort: return the Wikipedia revision id, else pageid, else None.

    The wikipediaapi library does not always expose a revision id, so we fall
    back to the internal pageid. We never invent a value.
    """
    revision = getattr(page, "revision_id", None)
    if revision is not None:
        return str(revision)
    pageid = getattr(page, "pageid", None)
    if pageid is not None:
        return str(pageid)
    return None


def fetch_with_wikipedia_api(titles: Sequence[str]) -> list[Document]:
    """Fetch pages with the `Wikipedia-API` package (the robust option).

    Handles the two failure cases ingestion.md calls out:
      * missing pages   -> page.exists() is False, we log and skip
      * API failures    -> we catch WikipediaException, log, and keep going
    """
    import wikipediaapi

    wiki = _build_wiki_client()
    max_chars = settings.wikipedia.max_chars
    documents: list[Document] = []

    for title in titles:
        try:
            page = wiki.page(title)

            if not page.exists():
                logger.warning("No Wikipedia page found for %r -- skipped.", title)
                continue

            text = page.text
            if not text.strip():
                logger.warning("Wikipedia page %r is empty -- skipped.", title)
                continue

            truncated = len(text) > max_chars
            if truncated:
                logger.warning(
                    "Page %r is %d chars, truncating to %d (WIKIPEDIA_MAX_CHARS).",
                    title, len(text), max_chars,
                )
                text = text[:max_chars]

            # canonicalurl resolves redirects; fullurl is the fallback.
            source_url = getattr(page, "canonicalurl", "") or page.fullurl

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "title": page.title,
                        "source_url": source_url,
                        "source": "wikipedia",
                        "lang": settings.wikipedia.lang,
                        "truncated": truncated,
                        "fetched_at": _utc_now(),
                        "revision_id": _wiki_revision(page),
                    },
                )
            )

        except wikipediaapi.WikipediaException as exc:
            logger.error("Wikipedia API error fetching %r: %s", title, exc)
        except Exception as exc:  # network/JSON issues, bad titles, ...
            logger.error("Unexpected error fetching %r: %s", title, exc)

    logger.info("Mode B fetched %d/%d page(s) via Wikipedia-API.", len(documents), len(titles))
    return documents


def _best_title_match(results: list[Document], wanted: str) -> Document | None:
    """LangChain's loader searches, so pick the result that actually matches."""
    if not results:
        return None
    wanted_normalised = wanted.strip().lower()
    for doc in results:
        if doc.metadata.get("title", "").strip().lower() == wanted_normalised:
            return doc
    return results[0]


def fetch_with_langchain_loader(titles: Sequence[str]) -> list[Document]:
    """Fetch pages with LangChain's built-in ``WikipediaLoader``.

    Simpler, but it searches instead of fetching an exact title and caps the
    text length, so results are guarded with a title check.
    """
    from langchain_community.document_loaders import WikipediaLoader

    documents: list[Document] = []
    fallback_url = (
        f"https://{settings.wikipedia.lang}.wikipedia.org/wiki/"
        + titles[0].replace(" ", "_") if titles else ""
    )

    for title in titles:
        try:
            results = (
                WikipediaLoader(
                    query=title,
                    lang=settings.wikipedia.lang,
                    load_max_docs=settings.wikipedia.load_max_docs,
                    doc_content_chars_max=settings.wikipedia.max_chars,
                )
                .load()
            )
        except ImportError as exc:
            logger.error("LangChain WikipediaLoader needs the 'wikipedia' package: %s", exc)
            return documents
        except Exception as exc:
            logger.error("WikipediaLoader failed for %r: %s", title, exc)
            continue

        doc = _best_title_match(results, title)
        if doc is None:
            logger.warning("No Wikipedia result for %r -- skipped.", title)
            continue

        metadata = dict(doc.metadata)
        metadata["title"] = metadata.get("title") or title
        metadata["source_url"] = metadata.get("source") or fallback_url
        metadata["source"] = "wikipedia"
        metadata["lang"] = settings.wikipedia.lang
        metadata["truncated"] = len(doc.page_content) >= settings.wikipedia.max_chars
        metadata["fetched_at"] = _utc_now()
        # LangChain WikipediaLoader does not expose revision/pageid reliably.

        documents.append(Document(page_content=doc.page_content, metadata=metadata))

    logger.info("Mode B fetched %d/%d page(s) via WikipediaLoader.", len(documents), len(titles))
    return documents


def fetch_wikipedia_documents(
    titles: Sequence[str] | None = None,
    fetcher: str | None = None,
) -> list[Document]:
    """Mode B -- public entry point.

    Args:
        titles: page titles. Defaults to the list in config/settings.yaml.
        fetcher: "wikiapi" (Wikipedia-API package) or "langchain"
                 (built-in WikipediaLoader). Defaults to WIKIPEDIA_FETCHER
                 in .env, falling back to "wikiapi".
    """
    if titles is None:
        titles = list(settings.pages)
    if not titles:
        logger.warning("No Wikipedia titles configured -- nothing to fetch.")
        return []

    fetcher = (fetcher or get_env("WIKIPEDIA_FETCHER", "wikiapi")).strip().lower()
    if fetcher == "langchain":
        return fetch_with_langchain_loader(titles)
    if fetcher == "wikiapi":
        return fetch_with_wikipedia_api(titles)

    logger.error("Unknown WIKIPEDIA_FETCHER %r (use 'wikiapi' or 'langchain').", fetcher)
    return []


# ---------------------------------------------------------------------------
# Convenience: run both modes in one go
# ---------------------------------------------------------------------------
def load_documents(
    include_local: bool = True,
    include_wikipedia: bool = True,
    titles: Sequence[str] | None = None,
) -> list[Document]:
    """Load everything the pipeline needs, ready for the cleaner/splitter."""
    documents: list[Document] = []

    if include_local:
        documents.extend(load_local_documents())
    if include_wikipedia:
        documents.extend(fetch_wikipedia_documents(titles))

    logger.info("Loader produced %d document(s) total.", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"Fetch mode: {get_env('WIKIPEDIA_FETCHER', 'wikiapi')}")
    print(f"Raw dir:   {settings.data_raw_dir}\n")

    print("--- Mode A: local files ---")
    local_docs = load_local_documents()
    for doc in local_docs:
        snippet = doc.page_content[:60].replace("\n", " ")
        print(f"[{doc.metadata['file_type']:>4}] {doc.metadata['title']:<45} {snippet}...")

    print(f"\n{len(local_docs)} local document(s) loaded.")

    print("\n--- Mode B: Wikipedia (title from settings.yaml, first only) ---")
    if settings.pages:
        wiki_docs = fetch_wikipedia_documents(titles=[settings.pages[0]])
        for doc in wiki_docs:
            print(f"{doc.metadata['title']}\n  url : {doc.metadata['source_url']}\n  size: {len(doc.page_content)} chars")
    else:
        print("No pages configured in config/settings.yaml.")
