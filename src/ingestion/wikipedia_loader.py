"""wikipedia_loader.py -- stage 1 of the ingestion pipeline.

Fetches Wikipedia articles exclusively through the official MediaWiki Action
API and turns them into LangChain ``Document`` objects containing **prose
only**, organised by article section.

There is no local-file loading, no HTML/web scraping (no BeautifulSoup), and
no table extraction here.  Every returned Document has ``content_type="prose"``
and carries stable metadata:

    source, title, url, section, section_path, language, content_type
    (+ pageid, revision_id, fetched_at for traceability)

The introduction is preserved as its own ``"Introduction"`` section.  Each
paragraph is assigned to exactly one section, so parent and child sections
never duplicate content.

Missing pages, redirects, disambiguation pages, API errors, rate limits, time
outs and empty responses are handled explicitly -- a failed extraction is
never silently reported as a successful empty article.

Official MediaWiki Action API: https://www.mediawiki.org/wiki/API:Main_page

We use ``action=query&prop=extracts|info`` with ``explaintext=1`` and
``exsectionformat=wiki``.  ``explaintext`` returns plain, rendered prose with
``== Section ==`` heading markers and inherently strips the HTML chrome
(infoboxes, tables, navigation boxes, references) that we must exclude.  A
small defence-in-depth cleanup pass in :mod:`document_cleaner` removes any
residual template/markup noise, so the loader never has to parse HTML.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional, Sequence

import requests
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class WikipediaError(Exception):
    """Base class for Wikipedia loader errors."""


class WikipediaAPIError(WikipediaError):
    """Raised on a transient/HTTP MediaWiki API problem (may be retried)."""

    def __init__(self, message: str, *, retryable: bool = False,
                 status_code: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.code = code


class PageNotFoundError(WikipediaError):
    """Raised when a requested page does not exist."""


class EmptyArticleError(WikipediaError):
    """Raised when a page exists but has no extractable prose."""


# ---------------------------------------------------------------------------
# Section parsing of the explaintext extract
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def _parse_sections(text: str) -> list[dict]:
    """Split an explaintext extract into section dictionaries.

    Each section: ``{"heading", "level", "path", "paragraphs", "text"}``.
    Content belongs to the deepest currently-open section, so a paragraph is
    assigned to exactly one section and parent/child sections never duplicate.
    The lead/intro (text before the first heading) becomes its own section.
    Empty sections are dropped.
    """
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
            # Close sections at this level or deeper; shallower ancestors stay open.
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            path = [s["heading"] for s in stack] + [heading]
            section = {"heading": heading, "level": level, "path": path,
                       "paragraphs": []}
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
# UTC helper
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    """Current UTC time in ISO 8601 format, for ``fetched_at`` metadata."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class WikipediaLoader:
    """Fetch Wikipedia articles via the MediaWiki Action API.

    A single :class:`requests.Session` is reused for connection pooling and a
    consistent User-Agent.  Titles are deduplicated within a run so the same
    page is never fetched twice.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": settings.wikipedia.user_agent,
            "Accept": "application/json",
        })

    def api_url(self) -> str:
        """Resolve the MediaWiki API endpoint.

        ``WIKIPEDIA_API_URL`` from settings takes precedence; otherwise the
        endpoint is derived from ``WIKIPEDIA_LANG`` (default English
        Wikipedia), so any language edition can be targeted.
        """
        if settings.wikipedia.api_url:
            return settings.wikipedia.api_url
        return f"https://{settings.wikipedia.lang}.wikipedia.org/w/api.php"

    def fetch_article(self, title: str) -> dict:
        """Fetch one article's parsed prose extract and metadata.

        Raises :class:`PageNotFoundError`, :class:`EmptyArticleError` or
        :class:`WikipediaAPIError` -- never returns an empty "success" result.
        Transient network/HTTP errors are retried with backoff.
        """
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
        for attempt in range(1, settings.wikipedia.max_retries + 1):
            # --- Network request (retried on failure) ---
            try:
                response = self._session.get(
                    self.api_url(), params=params,
                    timeout=settings.wikipedia.timeout,
                )
            except requests.RequestException as exc:
                last_error = WikipediaAPIError(
                    f"Network error fetching Wikipedia page {title!r}: {exc}",
                    retryable=True,
                )
                if attempt < settings.wikipedia.max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise last_error

            # --- Retryable HTTP status codes ---
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = WikipediaAPIError(
                    f"HTTP {response.status_code} fetching Wikipedia page {title!r}",
                    status_code=response.status_code, retryable=True,
                )
                if attempt < settings.wikipedia.max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise last_error

            # --- Non-retryable HTTP errors ---
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise WikipediaAPIError(
                    f"HTTP error fetching Wikipedia page {title!r}: {exc}"
                ) from exc

            # --- JSON parse ---
            try:
                data = response.json()
            except ValueError as exc:
                raise WikipediaAPIError(
                    f"Invalid JSON from Wikipedia API for {title!r}: {exc}"
                ) from exc

            break  # success
        else:
            raise last_error or WikipediaAPIError(
                f"Failed to fetch Wikipedia page {title!r} after "
                f"{settings.wikipedia.max_retries} attempts"
            )

        # --- MediaWiki API error in the response body ---
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

        # Only one title is requested at a time, so there is exactly one page.
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
                redirect.get("from"), redirect.get("to"),
            )

        return {
            "title": page.get("title") or title,
            "url": page.get("canonicalurl") or page.get("fullurl") or "",
            "pageid": page.get("pageid"),
            "revision_id": page.get("lastrevid"),
            "language": settings.wikipedia.lang,
            "extract": extract,
        }

    def fetch_articles(self, titles: Sequence[str]) -> list[dict]:
        """Fetch several articles, deduplicating titles within the run.

        Failed pages are logged and skipped -- one bad title never kills the
        batch, and failures are never returned as empty articles.
        """
        seen: set[str] = set()
        unique_titles: list[str] = []
        for title in titles:
            if title not in seen:
                seen.add(title)
                unique_titles.append(title)

        articles: list[dict] = []
        for title in unique_titles:
            try:
                articles.append(self.fetch_article(title))
            except PageNotFoundError as exc:
                logger.warning("Skipping missing Wikipedia page: %s", exc)
            except EmptyArticleError as exc:
                logger.warning("Skipping empty Wikipedia page: %s", exc)
            except WikipediaAPIError as exc:
                logger.error("Failed to fetch Wikipedia page %r: %s", title, exc)

        logger.info("Fetched %d/%d Wikipedia article(s).", len(articles), len(unique_titles))
        return articles

    def to_documents(self, articles: list[dict]) -> list[Document]:
        """Convert fetched articles into section-aware prose-only Documents."""
        documents: list[Document] = []
        for article in articles:
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
        logger.info("Produced %d prose document(s) from %d article(s).",
                    len(documents), len(articles))
        return documents


# Module-level loader reused across calls within a process.
_loader = WikipediaLoader()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fetch_wikipedia_documents(
    titles: Optional[Sequence[str]] = None,
) -> list[Document]:
    """Fetch Wikipedia articles and return section-aware prose Documents.

    Args:
        titles: page titles to fetch. Defaults to the ``pages`` list in
            ``config/settings.yaml``. Titles are deduplicated within the run.

    Returns:
        A list of LangChain ``Document`` objects, one per non-empty section of
        each article, each with ``content_type="prose"``.
    """
    if titles is None:
        titles = list(settings.pages)
    if not titles:
        logger.warning("No Wikipedia titles configured -- nothing to fetch.")
        return []

    articles = _loader.fetch_articles(titles)
    return _loader.to_documents(articles)


# Backward-compatible alias for callers that used ``load_documents``.
load_documents = fetch_wikipedia_documents


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"API endpoint: {_loader.api_url()}")
    print(f"Configured pages: {settings.pages}\n")

    if settings.pages:
        docs = fetch_wikipedia_documents(titles=[settings.pages[0]])
        for doc in docs:
            md = doc.metadata
            print(f"[{md['content_type']}] {md['title']} | "
                  f"section={md['section']} | path={md['section_path']} | "
                  f"{len(doc.page_content)} chars | url={md['url']}")
        print(f"\n{docs} document(s) produced.")
    else:
        print("No pages configured in config/settings.yaml.")
