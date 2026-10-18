"""document_cleaner.py -- stage 2 of the ingestion pipeline.

Cleans the prose-only ``Document`` objects produced by
:wmod:`wikipedia_loader`.  The loader already excludes HTML tables, infoboxes
and navigation chrome, so this cleaner never needs to process tables.

Cleaning rules (regex-based, not configuration):

    * citation markers left over from extraction, e.g. ``[12]`` / ``[228]`` /
      ``[1a]``
    * single-letter footnote markers, e.g. ``[a]`` / ``[b]``
    * wiki-link remnants ``[[A|B]]`` / ``[[A]]`` (display text is kept)
    * residual template markup ``{{...}}``
    * any residual HTML tags
    * HTML entities (decoded, e.g. ``&amp;`` -> ``&``)
    * decorative separator lines made of ``= - # * ~ _``
    * tabs, multi-spaces, trailing spaces and runs of blank lines

Meaningful punctuation, numbers, dates, abbreviations, names and parenthetical
factual information are preserved.  Citation markup is distinguished from
genuine prose: only citation-shaped brackets (digits, optionally a letter) are
removed; ordinary parentheses and brackets in factual text are untouched.

Junk sections (``References``, ``External links``, ...) are dropped entirely,
as are empty / too-short documents.  Duplicate documents (by SHA-1 fingerprint
of the cleaned text) are removed.

Input  -> list[Document] from wikipedia_loader
Output -> list[Document], same metadata plus audit fields
"""

from __future__ import annotations

import hashlib
import html
import logging
import re

from langchain_core.documents import Document

from src.config import get_env, settings

logger = logging.getLogger(__name__)

# --- Cleaning rules (regex, not configuration) ------------------------------
CITATION = re.compile(r"\[\d+[a-z]?\]")   # [12]  [228]  [401]  [1a]
FOOTNOTE = re.compile(r"\[[a-z]\]")        # [a] [b] [c]
WIKILINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")  # [[A|B]] -> B ; [[A]] -> A
TEMPLATE = re.compile(r"\{\{.*?\}\}", re.DOTALL)          # {{...}} template remnants
HTML_TAG = re.compile(r"<[^>]+>")                         # any residual HTML tags
SEPARATOR_LINE = re.compile(r"^[=\-#*~_]{3,}\s*$", re.MULTILINE)
WHITESPACE_RUNS = re.compile(r"[ \t]{2,}")
TRAILING_SPACES = re.compile(r" *\n")
MULTI_BLANK = re.compile(r"\n{3,}")

# Files shorter than this after cleaning are considered empty and dropped.
MIN_CONTENT_LENGTH = int(get_env("CLEAN_MIN_LENGTH", "25"))

# Junk section headings (case-insensitive), from config/settings.yaml.
JUNK_SECTIONS = {s.strip().lower() for s in settings.junk_sections if s.strip()}


def clean_text(text: str) -> str:
    """Return ``text`` with markup noise removed and whitespace normalised.

    Meaningful punctuation, numbers, dates, abbreviations, names and
    parenthetical factual information are preserved.  Only citation-shaped
    brackets are removed; ordinary brackets/parentheses in factual prose are
    left alone.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Wiki-link remnants: keep the display text, drop the link target.
    text = WIKILINK.sub(lambda m: m.group(1), text)

    # Citation markers and footnotes (citation-shaped brackets only).
    text = CITATION.sub("", text)
    text = FOOTNOTE.sub("", text)

    # Residual template markup and HTML tags.
    text = TEMPLATE.sub("", text)
    text = HTML_TAG.sub("", text)

    # Decode HTML entities (e.g. &amp; -> &).
    text = html.unescape(text)

    # Decorative separator lines.
    text = SEPARATOR_LINE.sub("", text)

    # Normalise whitespace while preserving paragraph breaks.
    text = text.replace("\t", " ")
    text = WHITESPACE_RUNS.sub(" ", text)
    text = TRAILING_SPACES.sub("\n", text)
    text = MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def clean_document(document: Document) -> Document:
    """Clean one ``Document`` and record before/after sizes in its metadata."""
    original = document.page_content
    cleaned = clean_text(original)

    metadata = dict(document.metadata)
    metadata.update(
        chars_before=len(original),
        chars_after=len(cleaned),
        cleaned=True,
    )
    return Document(page_content=cleaned, metadata=metadata)


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean a batch, dropping junk sections, empty/short docs and duplicates.

    * Documents whose ``section`` is a junk section are dropped.
    * Anything under ``CLEAN_MIN_LENGTH`` chars after cleaning is dropped.
    * Deduplication is by SHA-1 fingerprint of the cleaned text.
    """
    cleaned: list[Document] = []
    seen: dict[str, str] = {}

    for document in documents:
        result = clean_document(document)

        section = str(result.metadata.get("section", "")).strip().lower()
        if section in JUNK_SECTIONS:
            logger.info(
                "Dropping junk section: %s (%s)",
                result.metadata.get("title"), section,
            )
            continue

        if len(result.page_content) < MIN_CONTENT_LENGTH:
            logger.warning("Dropping short/empty doc: %s", result.metadata.get("title"))
            continue

        fingerprint = hashlib.sha1(result.page_content.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            logger.info("Skipping duplicate: %s", result.metadata.get("title"))
            continue
        seen[fingerprint] = result.metadata.get("title", "")

        cleaned.append(result)

    logger.info("Cleaned %d -> %d document(s).", len(documents), len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Demo (no network required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample = """Lead paragraph about Arsenal F.C.

== History ==
Arsenal was formed in 1886 as Dial Square.[1][2]

=== Early years ===
The club turned professional in 1891.[3]

== References ==
[1] Some source
[2] Another source
"""

    cleaned = clean_text(sample)
    print("--- cleaned ---")
    print(cleaned)
