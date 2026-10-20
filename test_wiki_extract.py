"""test_wiki_extract.py -- Standalone test script to inspect Wikipedia extract quality."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Standalone Configuration
# ---------------------------------------------------------------------------
TARGET_URL = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season"
OUTPUT_FILE = Path("extracted_2025_26_arsenal.txt")     

WIKIPEDIA_LANG = "en"
WIKIPEDIA_API_URL = f"https://{WIKIPEDIA_LANG}.wikipedia.org/w/api.php"
# Wikimedia requires a descriptive User-Agent header
USER_AGENT = "ArsenalWikiRAG/0.1 (local-extract-test; contact@example.com)"
TIMEOUT = 15
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class WikipediaError(Exception):
    """Base class for Wikipedia loader errors."""


class WikipediaAPIError(WikipediaError):
    """Raised on a transient/HTTP MediaWiki API problem (may be retried)."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.code = code


class PageNotFoundError(WikipediaError):
    """Raised when a requested page does not exist."""


class EmptyArticleError(WikipediaError):
    """Raised when a page exists but has no extractable prose."""


# ---------------------------------------------------------------------------
# Helpers & Section Parsing
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def _title_from_url(url: str) -> str:
    """Extract and URL-decode the Wikipedia page title from a full URL."""
    path = urlparse(url).path
    tail = path.rstrip("/").split("/")[-1]
    return unquote(tail).replace("_", " ").strip()


def _utc_now() -> str:
    """Current UTC time in ISO 8601 format, for ``fetched_at`` metadata."""
    return datetime.now(timezone.utc).isoformat()


def _parse_sections(text: str) -> list[dict]:
    """Split an explaintext extract into section dictionaries."""
    all_sections: list[dict] = []
    stack: list[dict] = []  # open sections, deepest last
    intro_paragraphs: list[str] = []
    pending: list[str] = []  # lines of the paragraph currently being built

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        para = " ".join(line.strip() for line in pending if line.strip()).strip()
        pending = []
        if not para:
            return
        target = stack[-1] if stack else None
        if target is None:
            intro_paragraphs.append(para)
        else:
            target["paragraphs"].append(para)

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        match = _HEADING_RE.match(line)
        if match:
            flush_pending()
            level = len(match.group(1))
            heading = match.group(2).strip()
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            path = [s["heading"] for s in stack] + [heading]
            section = {
                "heading": heading,
                "level": level,
                "path": path,
                "paragraphs": [],
            }
            stack.append(section)
            all_sections.append(section)
        else:
            pending.append(raw_line)

    flush_pending()
    for section in all_sections:
        section["text"] = "\n\n".join(section["paragraphs"]).strip()

    ordered: list[dict] = []
    intro_text = "\n\n".join(intro_paragraphs).strip()
    if intro_text:
        ordered.append({
            "heading": "Introduction",
            "level": 0,
            "path": ["Introduction"],
            "text": intro_text,
        })
    ordered.extend(section for section in all_sections if section["text"].strip())
    return ordered


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class WikipediaLoader:
    """Fetch Wikipedia articles via the MediaWiki Action API."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })

    def fetch_article(self, title: str) -> dict:
        """Fetch one article's parsed prose extract and metadata."""
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
            "format": "json",
        }

        last_error: Optional[WikipediaAPIError] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._session.get(
                    WIKIPEDIA_API_URL,
                    params=params,
                    timeout=TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = WikipediaAPIError(
                    f"Network error fetching Wikipedia page {title!r}: {exc}",
                    retryable=True,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise last_error

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = WikipediaAPIError(
                    f"HTTP {response.status_code} fetching Wikipedia page {title!r}",
                    status_code=response.status_code,
                    retryable=True,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise last_error

            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise WikipediaAPIError(
                    f"HTTP error fetching Wikipedia page {title!r}: {exc}"
                ) from exc

            try:
                data = response.json()
            except ValueError as exc:
                raise WikipediaAPIError(
                    f"Invalid JSON from Wikipedia API for {title!r}: {exc}"
                ) from exc

            break
        else:
            raise last_error or WikipediaAPIError(
                f"Failed to fetch Wikipedia page {title!r} after {MAX_RETRIES} attempts"
            )

        if "error" in data:
            err = data["error"]
            raise WikipediaAPIError(
                f"Wikipedia API error for {title!r}: "
                f"{err.get('info', 'unknown error')} (code={err.get('code', '')})",
                code=err.get("code"),
            )

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise WikipediaAPIError(f"No pages returned by Wikipedia API for {title!r}")

        page = next(iter(pages.values()))

        if "missing" in page:
            raise PageNotFoundError(f"Wikipedia page not found: {title!r}")

        extract = page.get("extract") or ""
        if not extract.strip():
            raise EmptyArticleError(
                f"Wikipedia page {title!r} has no extractable prose"
            )

        for redirect in data.get("redirects", []):
            logger.info(
                "Redirect followed: %r -> %r",
                redirect.get("from"),
                redirect.get("to"),
            )

        return {
            "title": page.get("title") or title,
            "url": page.get("canonicalurl") or page.get("fullurl") or "",
            "pageid": page.get("pageid"),
            "revision_id": page.get("lastrevid"),
            "language": WIKIPEDIA_LANG,
            "extract": extract,
        }

    def to_documents(self, article: dict) -> list[Document]:
        """Convert a fetched article into section-aware prose-only Documents."""
        documents: list[Document] = []
        sections = _parse_sections(article["extract"])
        for section in sections:
            documents.append(
                Document(
                    page_content=section["text"],
                    metadata={
                        "source": "wikipedia",
                        "title": article["title"],
                        "url": article["url"],
                        "section": section["heading"],
                        "section_path": " > ".join(section["path"]),
                        "language": article["language"],
                        "content_type": "prose",
                        "pageid": article.get("pageid"),
                        "revision_id": article.get("revision_id"),
                        "fetched_at": _utc_now(),
                    },
                )
            )
        return documents


# ---------------------------------------------------------------------------
# Run & Save to .txt
# ---------------------------------------------------------------------------
def main() -> None:
    title = _title_from_url(TARGET_URL)
    logger.info("Extracted title %r from URL", title)

    loader = WikipediaLoader()
    article = loader.fetch_article(title)
    docs = loader.to_documents(article)

    total_chars = sum(len(doc.page_content) for doc in docs)
    lines: list[str] = [
        f"TITLE       : {article['title']}",
        f"URL         : {article['url']}",
        f"PAGE ID     : {article['pageid']}",
        f"REVISION ID : {article['revision_id']}",
        f"SECTIONS    : {len(docs)}",
        f"TOTAL CHARS : {total_chars}",
        "=" * 80,
        "",
    ]

    for idx, doc in enumerate(docs, start=1):
        md = doc.metadata
        lines.append(f"[{idx}/{len(docs)}] SECTION: {md['section']}")
        lines.append(f"PATH : {md['section_path']}")
        lines.append(f"SIZE : {len(doc.page_content)} chars")
        lines.append("-" * 80)
        lines.append(doc.page_content)
        lines.append("\n" + "=" * 80 + "\n")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %d section document(s) (%d chars) to %s", len(docs), total_chars, OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()