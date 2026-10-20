# Arsenal Wiki — Prose-Only Wikipedia RAG

Arsenal Wiki is a retrieval-augmented generation (RAG) pipeline that ingests
**Wikipedia articles exclusively through the official MediaWiki Action API**,
extracts **prose only** (no HTML tables, infoboxes, or scraping), splits the
prose into section-aware chunks, embeds them with Ollama, and stores them in
PostgreSQL/pgvector.

## Architecture

The pipeline has four stages, each in its own module under `src/ingestion/`:

```
1. wikipedia_loader.py   fetch Wikipedia articles via the MediaWiki Action API
                           -> section-aware prose Documents (content_type="prose")
2. document_cleaner.py   clean prose (citation markers, templates, whitespace)
3. text_splitter.py      split prose into retrieval-friendly chunks
4. index_builder.py      embed chunks (Ollama) and store in PostgreSQL/pgvector
```

`src/ingestion/run_ingestion.py` orchestrates the four stages.
`src/retrieval/retriever.py` performs pgvector similarity search and returns
chunks with title / section / URL citation metadata.

## Key design decisions

- **Wikipedia is the only content source.** Articles are fetched through the
  MediaWiki Action API
  (`action=query&prop=extracts|info`, `explaintext=1`, `exsectionformat=wiki`),
  which returns rendered plain prose with `== Section ==` heading markers and
  inherently excludes the HTML chrome (infoboxes, tables, navigation boxes,
  references). No BeautifulSoup, no HTML scraping, no third-party APIs.
- **Prose only.** Every Document has `content_type="prose"`. Tables, infoboxes,
  reference lists, navigation boxes and other non-prose elements are excluded.
  Excluded tables are **not** replaced with LLM-generated summaries.
- **Section-aware.** Each paragraph is assigned to exactly one section; the
  introduction is its own `"Introduction"` section. Metadata carries
  `source, title, url, section, section_path, language, content_type` so
  retrieval can cite the correct article and section. No paragraph is
  duplicated between parent and child sections.
- **Generic Wikipedia support.** Any language edition and any article type can
  be processed — no hard-coded titles, section names, or Arsenal-specific data.
  The API endpoint is configurable via `WIKIPEDIA_API_URL` (defaults to
  `https://{WIKIPEDIA_LANG}.wikipedia.org/w/api.php`).
- **No local-file ingestion.** There is no loading of `.txt/.md/.csv` files or
  any other local content.

## Configuration

Settings come from `.env` (see `.env.example`) and `config/settings.yaml`.

Key environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `WIKIPEDIA_LANG` | `en` | Wikipedia language edition |
| `WIKIPEDIA_API_URL` | (none) | Full MediaWiki API URL; defaults to `https://{lang}.wikipedia.org/w/api.php` |
| `WIKIPEDIA_USER_AGENT` | `arsenal-weki-rag/0.1` | User-Agent for API requests |
| `WIKIPEDIA_TIMEOUT` | `30` | HTTP request timeout (seconds) |
| `WIKIPEDIA_MAX_RETRIES` | `3` | Retries for transient API errors |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text splitting limits |
| `CLEAN_MIN_LENGTH` | `25` | Drop documents shorter than this after cleaning |
| `POSTGRES_*` / `PGVECTOR_COLLECTION` | — | PostgreSQL connection + collection name |
| `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL` | — | Ollama chat + embedding models |
| `QUERY_REWRITE_COUNT`, `RETRIEVE_K`, `RERANK_TOP_N`, `SCORE_THRESHOLD` | — | Retrieval behaviour |

`config/settings.yaml` holds non-secret pipeline config: the default `pages`
list and the `junk_sections` to drop during cleaning.

## Running

```bash
# Install dependencies (from the project root, with the venv activated)
uv pip install -r requirements.txt

# Smoke check: settings, PostgreSQL, Ollama
python main.py

# Ingest the default Wikipedia pages (from config/settings.yaml)
python src/ingestion/run_ingestion.py

# Ingest specific titles
python src/ingestion/run_ingestion.py --titles "Arsenal F.C." "History of Arsenal F.C."

# Re-insert without wiping the collection
python src/ingestion/run_ingestion.py --no-reset

# Run the unit test suite (mocked, no network or database required)
python -m pytest tests/

# Optional live integration test against Wikipedia
set ARSENAL_WEKI_INTEGRATION=1   (Windows) / export ARSENAL_WEKI_INTEGRATION=1
python -m pytest tests/test_integration.py
```

## Database

Chunks are stored in PostgreSQL/pgvector (collection
`langchain_pg_embedding`, name from `PGVECTOR_COLLECTION`). `build_index()`
requires a non-empty list of validated prose chunks and never silently indexes
empty or malformed documents. Existing tables are not dropped; re-indexing uses
delete-before-insert per source.

## Tests

`tests/` contains unit tests for every stage (mocked — no live Wikipedia or
database required) plus `tests/test_integration.py`, which is gated behind
`ARSENAL_WEKI_INTEGRATION=1` and runs against live Wikipedia.
