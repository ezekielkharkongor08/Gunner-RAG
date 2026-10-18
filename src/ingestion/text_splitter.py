"""text_splitter.py -- stage 3 of the ingestion pipeline.

Splits each cleaned ``Document`` into chunks small enough to embed and to feed
into the LLM's context window.

Two strategies:

    * Prose documents (.txt / .md / Wikipedia) -- split by section headings
      first (== History ==, === Wenger era ===), then run
      RecursiveCharacterTextSplitter inside each section only if the section
      is longer than CHUNK_SIZE.  Each chunk gets a breadcrumb line
      ("Arsenal F.C. > History > Wenger era") and a ``section`` metadata field.

    * CSV documents (file_type == "csv") -- skip the generic text splitter.
      Chunk by rows instead: CSV_ROWS_PER_CHUNK rows per chunk, never cutting
      a row in the middle.  The context (title, page context, Columns: line)
      is repeated at the top of every chunk.

All numbers come from ``.env`` / ``config/settings.yaml`` through
``src.config``: CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_SIZE, CHUNK_MAX_WARN,
CSV_ROWS_PER_CHUNK, CSV_ROW_STYLE.

Input  -> list[Document] from document_cleaner.clean_documents()
Output -> list[Document], one per chunk, each carrying title, source_url,
          section, content_type, chunk position and chunk_chars so retrieval
          can cite where an answer came from.

Run the demo (Mode A only, no network):
    python -m src.ingestion.text_splitter
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_env, settings
from src.ingestion.document_cleaner import clean_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Wikipedia section headings: == History ==, === Wenger era ===, ==== x ====
HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)

# Characters to look for, most preferred first. The splitter keeps trying the
# next separator until a piece fits chunk_size.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Chunks shorter than this are folded into the next one. Without this, a
# heading like "### BACKGROUND ###" sitting in front of a paragraph longer
# than CHUNK_SIZE ends up as a chunk of its own -- useless for retrieval.
MIN_CHUNK_SIZE = int(get_env("CHUNK_MIN_SIZE", "120"))

# If the largest chunk exceeds this, log a warning.
CHUNK_MAX_WARN = int(get_env("CHUNK_MAX_WARN", "1000"))

# CSV rows per chunk (a row is never split across chunks).
CSV_ROWS_PER_CHUNK = int(get_env("CSV_ROWS_PER_CHUNK", "12"))

# Junk section headings (case-insensitive) dropped completely.
JUNK_SECTIONS = {s.strip().lower() for s in settings.junk_sections if s.strip()}

# Content types.
CONTENT_PROSE = "prose"
CONTENT_TABLE = "table"
CONTENT_TABLE_SUMMARY = "table_summary"


# ---------------------------------------------------------------------------
# Section detection (prose documents)
# ---------------------------------------------------------------------------
def _heading_level(line: str) -> int | None:
    """Return the heading level (number of = chars) if line is a heading."""
    m = HEADING_RE.match(line.strip())
    if m:
        return len(m.group(1))
    return None


def _heading_text(line: str) -> str | None:
    """Return the heading text if line is a heading, else None."""
    m = HEADING_RE.match(line.strip())
    if m:
        return m.group(2).strip()
    return None


def _split_into_sections(text: str) -> list[dict]:
    """Split text into sections based on ``== Heading ==`` markers.

    Returns a list of dicts: {"heading": str|None, "level": int, "text": str}.
    The first section (heading=None) holds everything before the first
    heading -- usually the lead paragraph(s).
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current: dict | None = None

    for line in lines:
        level = _heading_level(line)
        if level is not None:
            # Save the previous section before starting a new one.
            if current is not None:
                sections.append(current)
            current = {
                "heading": _heading_text(line),
                "level": level,
                "lines": [line],
            }
        else:
            if current is None:
                current = {
                    "heading": None,
                    "level": 0,
                    "lines": [],
                }
            current["lines"].append(line)

    if current is not None:
        sections.append(current)

    # Join lines back into text per section.
    for section in sections:
        section["text"] = "\n".join(section["lines"]).strip()

    return sections


def _is_junk_section(heading: str | None) -> bool:
    """True if this section heading is in the junk list (case-insensitive)."""
    if not heading:
        return False
    return heading.strip().lower() in JUNK_SECTIONS


def _breadcrumb_for(heading_path: list[str], title: str) -> str:
    """Build a breadcrumb line: "Title > Heading1 > Heading2"."""
    parts = [title] + [h for h in heading_path if h]
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Prose splitting
# ---------------------------------------------------------------------------
def _split_prose_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split a prose document by section, then by size within each section.

    Junk sections (References, External links, ...) are dropped and logged.
    Each chunk gets a breadcrumb line and ``section`` metadata.
    """
    title = str(document.metadata.get("title", "")).strip()
    text = document.page_content

    sections = _split_into_sections(text)

    # Track the heading hierarchy so we can build breadcrumbs.
    # Stack of (level, heading_text) for ancestors of the current section.
    ancestor_stack: list[tuple[int, str]] = []
    chunks: list[Document] = []
    junk_dropped = 0

    for section in sections:
        heading = section["heading"]
        level = section["level"]
        section_text = section["text"]

        # Pop ancestors whose level is >= current level (siblings/uncles).
        while ancestor_stack and ancestor_stack[-1][0] >= level:
            ancestor_stack.pop()

        # Build the heading path for this section (ancestors + self).
        heading_path = [h for _, h in ancestor_stack]
        if heading:
            heading_path = heading_path + [heading]
            ancestor_stack.append((level, heading))

        # Drop junk sections entirely.
        if _is_junk_section(heading):
            junk_dropped += 1
            logger.info("Dropping junk section: %r (doc: %s)", heading, title)
            continue

        if not section_text.strip():
            continue

        breadcrumb = _breadcrumb_for(heading_path, title)

        if len(section_text) <= settings.chunk.size:
            # Section is short enough: one chunk, no sub-splitting.
            chunk = _make_chunk(
                section_text,
                document,
                breadcrumb=breadcrumb,
                section=" > ".join(heading_path) if heading_path else title,
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
        else:
            # Section too long: split inside the section only.
            pieces = splitter.split_documents(
                [Document(page_content=section_text, metadata={})]
            )
            pieces = _fold_short_chunks(pieces, MIN_CHUNK_SIZE)
            for index, piece in enumerate(pieces):
                chunk = _make_chunk(
                    piece.page_content,
                    document,
                    breadcrumb=breadcrumb,
                    section=" > ".join(heading_path) if heading_path else title,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

    if junk_dropped:
        logger.info("Dropped %d junk section(s) from %s.", junk_dropped, title)

    return chunks


def _make_chunk(
    content: str,
    parent: Document,
    *,
    breadcrumb: str,
    section: str,
    chunk_index: int,
) -> Document:
    """Build a chunk Document with breadcrumb prepended and rich metadata."""
    # Prepend the breadcrumb line so every chunk knows where it came from.
    chunk_text = f"{breadcrumb}\n\n{content}"

    metadata = dict(parent.metadata)
    metadata.update(
        section=section,
        content_type=CONTENT_PROSE,
        chunk_index=chunk_index,
        chunk_chars=len(chunk_text),
    )

    return Document(page_content=chunk_text, metadata=metadata)


# ---------------------------------------------------------------------------
# CSV row-based chunking
# ---------------------------------------------------------------------------
def _first_number(value: str) -> int | None:
    """Return the first integer found in a cell, or None."""
    m = re.search(r"\d+", value)
    return int(m.group()) if m else None


def _build_summary_chunk(
    title: str,
    page_context: str,
    column_notes: str,
    header_row: list[str],
    data_rows: list[list[str]],
    summary_columns: list[str],
    parent_metadata: dict,
) -> Document | None:
    """Build one summary chunk per SUMMARY_COLUMN: top 3 rows by first number."""
    lines: list[str] = [title]
    if page_context:
        lines.append(page_context)
    lines.append(f"Columns: {', '.join(header_row)}")
    if column_notes:
        lines.append(f"Notes: {column_notes}")

    any_summary = False
    for col_name in summary_columns:
        try:
            col_index = header_row.index(col_name)
        except ValueError:
            logger.warning(
                "SUMMARY_COLUMNS: column %r not found in CSV header, skipping.",
                col_name,
            )
            continue

        # Collect (first_number, original_row) for rows that have a number.
        scored: list[tuple[int, list[str]]] = []
        for row in data_rows:
            if col_index < len(row):
                num = _first_number(row[col_index])
                if num is not None:
                    scored.append((num, row))

        if not scored:
            logger.warning(
                "SUMMARY_COLUMNS: column %r has no numeric values, skipping.",
                col_name,
            )
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top3 = scored[:3]

        parts = []
        for num, row in top3:
            cell = row[col_index] if col_index < len(row) else ""
            parts.append(f"{row[0] if row else '?'} ({cell})")

        lines.append(
            f"Summary: The highest value of {col_name} in this table is "
            + ", then ".join(parts) + "."
        )
        any_summary = True

    if not any_summary:
        return None

    chunk_text = "\n".join(lines)
    metadata = dict(parent_metadata)
    metadata.update(
        content_type=CONTENT_TABLE_SUMMARY,
        section=str(parent_metadata.get("title", "")),
        chunk_chars=len(chunk_text),
    )
    return Document(page_content=chunk_text, metadata=metadata)


def _format_row_pairs(
    header_row: list[str],
    row: list[str],
) -> str:
    """Format one CSV row as '- Key: value, Key: value' (pairs style)."""
    cells = (list(row) + [""] * len(header_row))[: len(header_row)]
    pairs = [
        f"{name}: {value.strip()}"
        for name, value in zip(header_row, cells)
        if name and value and value.strip()
    ]
    return "- " + ", ".join(pairs)


def _format_row_sentence(
    header_row: list[str],
    row: list[str],
    row_template: str,
) -> str:
    """Format one CSV row using a template (sentence style), falling back to
    pairs if a value is missing."""
    cells = (list(row) + [""] * len(header_row))[: len(header_row)]
    values = dict(zip(header_row, cells))

    # Check all placeholders in the template have values.
    placeholders = re.findall(r"\{(\w+)\}", row_template)
    missing = [p for p in placeholders if not values.get(p, "").strip()]
    if missing:
        return _format_row_pairs(header_row, row)

    try:
        text = row_template.format(**values)
    except (KeyError, IndexError):
        return _format_row_pairs(header_row, row)

    return "- " + text


def _split_csv_document(
    document: Document,
) -> list[Document]:
    """Chunk a CSV document by rows, never splitting a row.

    Every chunk repeats the context: title, page context, Columns: line,
    optional Notes: line, then the row lines for this chunk.
    """
    title = str(document.metadata.get("title", "")).strip()
    page_context = str(document.metadata.get("page_context", "")).strip()
    column_notes = str(document.metadata.get("csv_column_notes", "")).strip()
    row_template = str(document.metadata.get("csv_row_template", "")).strip()
    summary_columns_raw = str(
        document.metadata.get("csv_summary_columns", "")
    ).strip()
    header_row = document.metadata.get("csv_header_row", []) or []
    csv_rows_raw = document.metadata.get("csv_rows", "[]") or "[]"

    try:
        data_rows = json.loads(csv_rows_raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse csv_rows for %s; falling back to text.", title)
        return _fallback_csv_chunk(document)

    if not header_row:
        return _fallback_csv_chunk(document)

    if not data_rows:
        return _fallback_csv_chunk(document)

    # Build context lines repeated at the top of every chunk.
    context_lines: list[str] = [title]
    if page_context:
        context_lines.append(page_context)
    context_lines.append(f"Columns: {', '.join(header_row)}")
    if column_notes:
        context_lines.append(f"Notes: {column_notes}")
    context_prefix = "\n".join(context_lines) + "\n\n"

    # Format each row line once, then clean noise (citations like [475] in
    # CSV cell values are removed here so they don't survive into chunks).
    row_style = settings.csv_row_style
    row_lines: list[str] = []
    for row in data_rows:
        if row_style == "sentence" and row_template:
            raw_line = _format_row_sentence(header_row, row, row_template)
        else:
            raw_line = _format_row_pairs(header_row, row)
        row_lines.append(clean_text(raw_line))

    # Group rows into chunks.
    chunks: list[Document] = []
    chunk_index = 0

    for start in range(0, len(row_lines), CSV_ROWS_PER_CHUNK):
        batch = row_lines[start:start + CSV_ROWS_PER_CHUNK]
        row_start = start + 1  # 1-based data row numbers
        row_end = start + len(batch)

        chunk_text = context_prefix + "\n".join(batch)
        metadata = dict(document.metadata)
        metadata.update(
            content_type=CONTENT_TABLE,
            table_name=title,
            section=title,
            row_start=row_start,
            row_end=row_end,
            chunk_index=chunk_index,
            chunk_chars=len(chunk_text),
        )
        chunks.append(Document(page_content=chunk_text, metadata=metadata))
        chunk_index += 1

    # Optional summary chunk(s) for SUMMARY_COLUMNS.
    if summary_columns_raw:
        summary_columns = [c.strip() for c in summary_columns_raw.split(",") if c.strip()]
        if summary_columns:
            parent_metadata = dict(document.metadata)
            summary_doc = _build_summary_chunk(
                title=title,
                page_context=page_context,
                column_notes=column_notes,
                header_row=header_row,
                data_rows=data_rows,
                summary_columns=summary_columns,
                parent_metadata=parent_metadata,
            )
            if summary_doc is not None:
                summary_doc.metadata["chunk_index"] = chunk_index
                summary_doc.metadata["chunk_chars"] = len(summary_doc.page_content)
                chunks.append(summary_doc)

    logger.info(
        "CSV %s: %d data row(s) -> %d chunk(s).",
        title, len(data_rows), len(chunks),
    )
    return chunks


def _fallback_csv_chunk(document: Document) -> list[Document]:
    """If CSV metadata is missing, treat the document as a single table chunk."""
    metadata = dict(document.metadata)
    metadata.update(
        content_type=CONTENT_TABLE,
        table_name=str(document.metadata.get("title", "")),
        section=str(document.metadata.get("title", "")),
        row_start=1,
        row_end=1,
        chunk_index=0,
        chunk_chars=len(document.page_content),
    )
    return [Document(page_content=document.page_content, metadata=metadata)]


# ---------------------------------------------------------------------------
# Short-chunk folding
# ---------------------------------------------------------------------------
def _fold_short_chunks(pieces: list[Document], min_size: int) -> list[Document]:
    """Push an over-short chunk forward into the chunk that follows it.

    Only ever one pending piece is carried at a time, so a merged chunk can
    grow by at most ``min_size`` characters -- a fair trade for never handing
    the retriever a chunk that is just a heading.
    """
    merged: list[Document] = []
    pending: Document | None = None

    for piece in pieces:
        if pending is not None:
            piece = Document(
                page_content=f"{pending.page_content}\n\n{piece.page_content}",
                metadata={**piece.metadata, "start_index": pending.metadata.get("start_index", 0)},
            )
            pending = None

        if len(piece.page_content) < min_size:
            pending = piece
            continue

        merged.append(piece)

    if pending is not None:
        merged.append(pending)   # short tail at the end of a document

    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def split_documents(
    documents: list[Document],
    include_title_header: bool = True,
) -> list[Document]:
    """Split cleaned documents into chunks.

    Prose documents are split by section headings first, then by size within
    each section.  CSV documents are chunked by rows.

    Args:
        documents: output of ``clean_documents()``.
        include_title_header: kept for API compatibility; the breadcrumb is
            always added for prose chunks.

    Returns:
        Flat list of chunk Documents. Metadata is inherited from the parent
        plus ``chunk_index`` / ``chunk_count`` / ``chunk_id`` /
        ``start_index`` / ``section`` / ``content_type`` / ``chunk_chars``.
    """
    splitter = _build_splitter()
    chunks: list[Document] = []

    for document in documents:
        file_type = str(document.metadata.get("file_type", "")).lower()

        if file_type == "csv":
            csv_chunks = _split_csv_document(document)
            total_csv = len(csv_chunks)
            # Re-number chunk_index / chunk_count across the CSV chunks.
            for index, chunk in enumerate(csv_chunks):
                metadata = dict(chunk.metadata)
                metadata["chunk_index"] = index
                metadata["chunk_count"] = total_csv
                metadata["chunk_id"] = f"{chunk.metadata.get('table_name', 'untitled')}#{index}"
                chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
            continue

        # --- Prose ---
        section_chunks = _split_prose_document(document, splitter)
        title = str(document.metadata.get("title", "")).strip()
        total = len(section_chunks)

        for index, piece in enumerate(section_chunks):
            metadata = dict(piece.metadata)
            metadata.update(
                chunk_count=total,
                chunk_id=f"{title or 'untitled'}#{index}",
                start_index=metadata.get("start_index", 0),
            )
            chunks.append(Document(page_content=piece.page_content, metadata=metadata))

    # Chunk count per source document (prose): set chunk_count now that we know totals.
    # (CSV chunks already have chunk_index set; chunk_count left as-is per table.)

    # Warn if any chunk exceeds CHUNK_MAX_WARN.
    oversized = [c for c in chunks if len(c.page_content) > CHUNK_MAX_WARN]
    if oversized:
        logger.warning(
            "%d chunk(s) exceed CHUNK_MAX_WARN=%d (embedding model token limit).",
            len(oversized), CHUNK_MAX_WARN,
        )

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(chunks))
    return chunks


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the splitter from .env values, with a clear error if they're off."""
    size = settings.chunk.size
    overlap = settings.chunk.overlap

    if size <= 0:
        raise ValueError(f"CHUNK_SIZE must be > 0, got {size}.")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"CHUNK_OVERLAP ({overlap}) must be between 0 and CHUNK_SIZE-1 ({size - 1})."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        add_start_index=True,   # records where each chunk began in the parent doc
    )


def describe_chunks(chunks: list[Document]) -> dict[str, int]:
    """Tiny helper for logging/tests: min, max and average chunk length."""
    if not chunks:
        return {"chunks": 0, "min": 0, "max": 0, "avg": 0}
    sizes = [len(c.page_content) for c in chunks]
    return {
        "chunks": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "avg": sum(sizes) // len(sizes),
    }


# ---------------------------------------------------------------------------
# Demo -- Mode A only (local files, no network)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from src.ingestion.document_cleaner import clean_documents
    from src.ingestion.wikipedia_loader import load_local_documents

    docs = clean_documents(load_local_documents())
    chunks = split_documents(docs)

    stats = describe_chunks(chunks)
    print(f"CHUNK_SIZE={settings.chunk.size}  CHUNK_OVERLAP={settings.chunk.overlap}  "
          f"CHUNK_MIN_SIZE={MIN_CHUNK_SIZE}  CHUNK_MAX_WARN={CHUNK_MAX_WARN}")
    print(f"{len(docs)} document(s) -> {stats['chunks']} chunk(s)")
    print(f"sizes: {stats}\n")

    print("--- first 8 chunks ---")
    for chunk in chunks[:8]:
        size = len(chunk.page_content)
        print(f"{chunk.metadata.get('chunk_id',''):<45} {size:>5} chars  "
              f"type={chunk.metadata.get('content_type',''):<8}  "
              f"section={chunk.metadata.get('section','')[:40]}")
        print(f"  {chunk.page_content[:120].replace(chr(10), ' ')}")

    print("\n--- chunks above CHUNK_MAX_WARN (should be empty) ---")
    for chunk in chunks:
        if len(chunk.page_content) > CHUNK_MAX_WARN:
            print(f"  {len(chunk.page_content):>5}  {chunk.metadata.get('chunk_id','')}")
