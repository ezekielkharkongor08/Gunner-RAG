"""document_cleaner.py -- stage 2 of the ingestion pipeline.

Takes the ``Document`` objects produced by wikipedia_loader.py and strips
textual noise so the splitter and the LLM only see meaningful content:

    * citation markers left over from Wikipedia markup, e.g. ``[12]`` / ``[228]``
    * single-letter footnote markers, e.g. ``[a]`` / ``[b]``
    * wiki links ``[[Land | the land]]`` -> plain words
    * decorative separator lines made of ``=`` / ``-`` / ``#``
    * stray blank lines, tabs, trailing whitespace

Input  -> list[Document] from load_local_documents() / fetch_wikipedia_documents()
Output -> list[Document], same metadata plus a couple of audit fields.

Run the demo (Mode A only, no network):
    python -m src.ingestion.document_cleaner
"""

from __future__ import annotations

import hashlib
import logging
import re

from langchain_core.documents import Document

from src.config import get_env

logger = logging.getLogger(__name__)

# --- Cleaning rules (regex, not configuration) ------------------------------
CITATION = re.compile(r"\[\d+[a-z]?\]")          # [12]  [228]  [401]  [1a]
FOOTNOTE = re.compile(r"\[[a-z]\]")              # [a] [b] [c] -- table footnotes
WIKI_LINK = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")   # [[Target | shown]]
WIKI_PLAIN = re.compile(r"\[\[([^\]]+)\]\]")     # [[Target]]
SEPARATOR_LINE = re.compile(r"^[=\-#*~_]{3,}\s*$", re.MULTILINE)

# Files shorter than this after cleaning are considered empty and dropped.
MIN_CONTENT_LENGTH = int(get_env("CLEAN_MIN_LENGTH", "25"))


def clean_text(text: str) -> str:
    """Return `text` with markup noise and excess whitespace removed.

    Each step is independent, so it is easy to add or drop one later.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = WIKI_LINK.sub(r"\2", text)   # show the readable part of the link
    text = WIKI_PLAIN.sub(r"\1", text)
    text = CITATION.sub("", text)       # citations carry no meaning for us
    text = FOOTNOTE.sub("", text)

    text = SEPARATOR_LINE.sub("", text)  # "=======" / "------"

    # Normalise whitespace: fix tabs and trailing spaces per line, then
    # collapse any run of blank lines down to a single blank line.
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_document(document: Document) -> Document:
    """Clean one Document and record before/after sizes in its metadata."""
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
    """Clean a batch, dropping anything empty, too short, or duplicated.

    Deduplication matters here: several files point at the same Wikipedia
    article and would otherwise pollute retrieval with identical chunks.
    """
    cleaned: list[Document] = []
    seen: dict[str, str] = {}

    for document in documents:
        result = clean_document(document)

        if len(result.page_content) < MIN_CONTENT_LENGTH:
            logger.warning("Dropping short/empty doc: %s", result.metadata.get("title"))
            continue

        fingerprint = hashlib.sha1(result.page_content.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            logger.info("Skipping duplicate: %s (%s)",
                        result.metadata.get("title"), result.metadata.get("origin", "?"))
            continue
        seen[fingerprint] = result.metadata.get("title", "")

        cleaned.append(result)

    logger.info("Cleaned %d -> %d document(s).", len(documents), len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Demo -- Mode A only (local files, no network)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    from src.ingestion.wikipedia_loader import load_local_documents

    raw_docs = load_local_documents()
    clean_docs = clean_documents(raw_docs)

    print(f"{len(raw_docs)} raw  ->  {len(clean_docs)} cleaned\n")
    for doc in clean_docs:
        saved = doc.metadata["chars_before"] - doc.metadata["chars_after"]
        print(f"{doc.metadata['title'][:42]:<44} "
              f"{doc.metadata['chars_before']:>6} -> {doc.metadata['chars_after']:>6}  "
              f"(-{saved})")

    sample = clean_docs[0]
    print(f"\n--- sample: {sample.metadata['title']} ---")
    print(sample.page_content[:500])
