# Arsenal Weki — Ingestion Pipeline (prose-only Wikipedia)

This document describes the ingestion pipeline (`src/ingestion/`), which now
fetches content **exclusively from Wikipedia via the MediaWiki Action API** and
extracts **prose only**.

Pipeline order:

```
load (wikipedia_loader.py)
    -> clean (document_cleaner.py)
    -> split (text_splitter.py)
    -> embed + store (index_builder.py)
```

All tunable values come from `.env` / `config/settings.yaml` through
`src.config`; nothing is hard-coded.

A separate orchestrator, `src/ingestion/run_ingestion.py`, runs all four
stages in order:

```
python src/ingestion/run_ingestion.py              # default pages from settings.yaml
python src/ingestion/run_ingestion.py --titles "A" "B"
python src/ingestion/run_ingestion.py --no-reset
```

---

## 1. Load / fetch — `wikipedia_loader.py`

There is a single source: Wikipedia.  Articles are fetched through the official
MediaWiki Action API
(`action=query&prop=extracts|info`, `explaintext=1`, `exsectionformat=wiki`).

`explaintext=1` returns rendered plain prose with `== Section ==` heading
markers and inherently strips the HTML chrome (infoboxes, tables, navigation
boxes, references).  There is **no BeautifulSoup, no HTML scraping, and no
table extraction**.

### What is extracted
For each article, one Document per section:

* **Introduction** — the lead text before the first heading, as its own
  `"Introduction"` section.
* **Section headings and their prose** — including nested subsections.
* Each Document carries `content_type="prose"` and metadata:
  `source, title, url, section, section_path, language, content_type`
  (plus `pageid`, `revision_id`, `fetched_at`).

### What is excluded
HTML tables, infoboxes, reference lists, citation markers at load time are kept
only insofar as the cleaner removes them later; navigation boxes, sidebars,
categories, edit controls, maintenance messages, raw HTML/CSS/JS, template
artifacts, image captions, and empty sections are all excluded.  Excluded
tables are **not** replaced with LLM-generated summaries.

### Error handling
Missing pages (`PageNotFoundError`), empty extracts (`EmptyArticleError`),
API errors (`WikipediaAPIError`), rate limits (HTTP 429) and timeouts are
handled explicitly.  A failed extraction is **never** silently reported as a
successful empty article.  Transient network/HTTP errors are retried with
backoff (`WIKIPEDIA_TIMEOUT`, `WIKIPEDIA_MAX_RETRIES`).  Titles are
deduplicated within a run so the same page is never fetched twice.

### Configuration
* `WIKIPEDIA_LANG` — language edition (default `en`).
* `WIKIPEDIA_API_URL` — optional full API URL; defaults to
  `https://{lang}.wikipedia.org/w/api.php`.
* `WIKIPEDIA_USER_AGENT`, `WIKIPEDIA_TIMEOUT`, `WIKIPEDIA_MAX_RETRIES`.

---

## 2. Clean — `document_cleaner.py`

Cleans extracted prose only.  The loader already excludes tables, so the
cleaner never processes tables.

Rules:
* Citation markers `[12]`, `[228]`, `[1a]` and footnotes `[a]` removed.
* Wiki-link remnants `[[A|B]]` / `[[A]]` converted to display text.
* Residual `{{...}}` template markup and HTML tags removed.
* HTML entities decoded (`&amp;` → `&`).
* Whitespace normalised; paragraph boundaries preserved.
* Meaningful punctuation, numbers, dates, abbreviations, names and
  parenthetical factual information preserved.

Junk sections (`References`, `External links`, `See also`, …, from
`config/settings.yaml`) are dropped entirely.  Empty / too-short documents
(`CLEAN_MIN_LENGTH`) and duplicates (SHA-1 fingerprint) are removed.

Audit fields `chars_before`, `chars_after`, `cleaned` are added to metadata.

---

## 3. Split — `text_splitter.py`

Splits cleaned prose into retrieval-friendly chunks.  There is **no
table-specific splitting**; every input document is one prose section.

* Each section document is split independently with
  `RecursiveCharacterTextSplitter`, preferring paragraph boundaries
  (`\n\n`, `\n`, `. `, ` `).
* Chunks are constrained by `CHUNK_SIZE` / `CHUNK_OVERLAP`.
* Very long paragraphs are split without exceeding the limit.
* Chunks shorter than `CHUNK_MIN_SIZE` are folded into the next chunk.
* Empty and duplicate chunks are dropped.
* Unrelated articles and sections are never combined; no paragraph is
  duplicated between chunks.

Each chunk inherits `source, title, url, section, section_path, language,
content_type` plus `chunk_id`, `chunk_index`, `chunk_count`, `chunk_chars`.
`content_type` stays `"prose"`.

---

## 4. Embed + store — `index_builder.py`

Embeds chunks with Ollama and stores them in PostgreSQL/pgvector.  This stage
does **only** embedding and storing — it does not load, clean or split.

* Requires a non-empty list of chunks (fail-fast; `ValueError` otherwise).
* Validates required metadata (`title`, `url`, `section`,
  `content_type="prose"`); rejects empty or malformed documents.
* Embedding model/prefixes from `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`,
  `EMBEDDING_DOC_PREFIX`, `EMBEDDING_QUERY_PREFIX`.
* Dimension is probed once per run (not hard-coded).
* pgvector extension ensured; collection from `PGVECTOR_COLLECTION`.
* Delete-before-insert per source prevents stale rows; stable chunk IDs keep
  re-indexing idempotent.

---

## 5. Configuration removed in this refactor

The following no longer exist (local-file ingestion and table processing were
removed): local `.txt/.md/.csv` loading, BeautifulSoup scraping, HTML table
extraction (rowspan/colspan), table row-chunking, table summary chunks,
`CSV_ROWS_PER_CHUNK`, `CSV_ROW_STYLE`, `TABLE_ROWS_PER_CHUNK`,
`WIKIPEDIA_LOAD_MAX_DOCS`, `WIKIPEDIA_MAX_CHARS`, `data_raw_dir`,
`data_processed_dir`.
