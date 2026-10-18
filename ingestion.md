# Arsenal Weki — Ingestion Pipeline

This document describes everything the ingestion pipeline (`src/ingestion/`) does, from fetching raw data to storing embedded chunks in PostgreSQL/pgvector.

Pipeline order:

```
load (wikipedia_loader.py)
    -> clean (document_cleaner.py)
    -> split (text_splitter.py)
    -> embed + store (index_builder.py)
```

All tunable values come from `.env` / `config/settings.yaml` through `src.config`; nothing is hard-coded.

---

## 1. How we fetch the data (Mode A vs Mode B)

Entry point: `wikipedia_loader.load_documents()` runs both modes and returns a list of LangChain `Document` objects (`page_content` + `metadata`). Every document always carries at least `title` and `source_url` in metadata.

### Mode A — load local files (`load_local_documents()`)

Walks the folders `data/raw` and `data/processed` (configurable) and reads every supported file:

| Extension | Handler | Becomes |
|-----------|---------|---------|
| `.txt`, `.md`, `.markdown` | `_document_from_text_file()` | one Document per file |
| `.csv` | `_document_from_csv_file()` | one Document per file |

For each file it:

1. Reads the file as UTF-8 (`errors="replace"` so bad bytes never crash the run).
2. Peels off a top metadata block — lines like `KEY: value` or `# KEY: value` (decorator separators and blank lines are skipped). Recognised aliases map `source_url`/`url`/`canonical_url` -> `source_url`, and `table_topic`/`page_title`/`subject`/`topic`/`title` -> `title`.
3. Uses the remaining body as `page_content`.
4. Falls back to the filename (underscores/hyphens turned to spaces) when no title metadata exists.
5. Records `source` = `"local"`, `origin` = path relative to project root, `file_type`.

Duplicate file paths are skipped (a `seen` set), and unreadable files are logged as errors without aborting the run.

### Mode B — fetch Wikipedia pages (`fetch_wikipedia_documents()`)

Takes a list of page titles (defaults to the list in `config/settings.yaml`, e.g. "Arsenal F.C.", "History of Arsenal F.C.", …) and fetches them over the network. Two fetchers, chosen via `WIKIPEDIA_FETCHER` in `.env` (default `wikiapi`):

**A) Wikipedia-API package** (`fetch_with_wikipedia_api()`)
- Builds a `wikipediaapi.Wikipedia` client with the configured user agent, language and `extract_format = WIKI` (raw wiki markup, so downstream cleaning has something to clean).
- `max_retries=3`, `retry_wait=1.0` for resilience.
- Skips pages that don't exist (`page.exists() is False`) and empty pages, logging a warning each time.
- Truncates text to `WIKIPEDIA_MAX_CHARS` when the page is longer, flagging `truncated=True` in metadata.
- Resolves the canonical URL via `page.canonicalurl` (redirects), falling back to `page.fullurl`.
- Catches `WikipediaException` and generic network/JSON errors per title, so one bad title never kills the whole batch.

**B) LangChain WikipediaLoader** (`fetch_with_langchain_loader()`)
- Simpler but searches instead of fetching an exact title, and caps text length itself.
- Results are guarded with a title match (`_best_title_match()` picks the exact title, otherwise the first result).
- Metadata is normalised afterwards so it matches the wikiapi shape (`title`, `source_url`, `source="wikipedia"`, `lang`, `truncated`).

Missing pages and API failures are always handled gracefully — logged, skipped, and execution continues.

---

## 2. How we handle .txt and .csv files

### .txt / .md files

Read whole-file, split header from body, keep body as page content. Metadata keys: `title`, `source_url`, `source`, `origin`, `file_type`.

### .csv files

A CSV is **not** stored as raw comma text, because a pipe table like `Rank | Player | 558 (11) | 70 (1)` carries almost no semantic signal — an embedding model sees mostly digits and separators.

Instead (`_csv_to_row_lines()`):

1. Split header block from the CSV body (same `# KEY: value` header handling).
2. Parse the body with Python's `csv` module, dropping completely blank rows.
3. Render **one self-describing line per data row**:

   ```
   - Player: Thierry Henry, Total Games/Goals: 377 / 228
   ```

   Each non-empty cell becomes `ColumnName: value`, joined with commas, prefixed with `- `. Empty cells and empty headers are skipped.
4. The Document's `page_content` is `title` (+ optional `page_context` derived from the source URL, e.g. "List of Arsenal F.C. records and statistics") followed by the table lines, so the title reads together with the table instead of becoming a tiny orphan chunk later.
5. Metadata gains `page_context`, `columns` (column count), `file_type = "csv"`.

---

## 3. How we convert raw files/pages into LangChain Documents

Both modes produce `langchain_core.documents.Document` objects:

```python
Document(
    page_content="...",
    metadata={
        "title": "...",
        "source_url": "...",
        "source": "local" | "wikipedia",
        ...
    },
)
```

This happens in `wikipedia_loader.py` only — no cleaning, splitting, or embedding happens there. Those are downstream stages.

---

## 4. How we remove noise / clean the LangChain Documents

Stage 2: `document_cleaner.py` (`clean_documents()` / `clean_document()` / `clean_text()`). Input = Documents from the loader; output = cleaned Documents with audit fields added to metadata.

### Noise removal rules (applied in `clean_text()`)

1. **Line endings** — `\r\n` and `\r` normalised to `\n`.
2. **Wiki links** — `[[Target | shown text]]` -> `shown text`; `[[Target]]` -> `Target`. Keeps readable words, drops markup.
3. **Citation markers** — `[12]`, `[228]`, `[401]`, `[1a]` removed (`\[\d+[a-z]?\]`).
4. **Footnotes** — `[a]`, `[b]`, `[c]` removed (`\[[a-z]\]`).
5. **Decorator separator lines** — lines made only of `= - # * ~ _` (3+ chars) removed.
6. **Whitespace** — tabs -> spaces; runs of 2+ spaces -> one space; trailing spaces per line removed; runs of 3+ blank lines collapsed to 2.

### After cleaning — audit + filtering

`clean_document()` records in metadata:
- `chars_before`, `chars_after`
- `cleaned = True`

`clean_documents()` then:

- **Drops empty/too-short documents** — anything under `CLEAN_MIN_LENGTH` chars (default 25, from `.env`) after cleaning is logged and dropped.
- **De-duplicates** — several local files can point at the same Wikipedia article. A SHA-1 fingerprint of the cleaned text is kept in a `seen` dict; duplicates are logged and skipped so retrieval isn't polluted by identical chunks.

Result: `Cleaned N -> M document(s)` logged, with metadata preserved plus the audit fields.

---

## 5. How we split the cleaned documents

Stage 3: `text_splitter.py` (`split_documents()`). Input = cleaned Documents; output = a flat list of chunk Documents.

### Splitting strategy

- Uses LangChain's `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]` — it tries to cut on paragraphs, then lines, then sentences, then words, then characters, so chunks stay semantically coherent.
- `chunk_size` and `chunk_overlap` come from `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) and are validated: size must be > 0, overlap must be `0 <= overlap < size`.
- `add_start_index=True` records where each chunk began in the parent document.

### Folding short chunks (`_fold_short_chunks()`)

A heading like `### BACKGROUND ###` sitting before a long paragraph would otherwise become a chunk of its own — useless for retrieval. Any chunk shorter than `CHUNK_MIN_SIZE` (default 120, from `.env`) is folded into the following chunk (`pending` carried forward at most one piece), keeping the earlier `start_index`.

### Chunk metadata

Every chunk inherits the parent's metadata and gains:
- `chunk_index` — position within the document
- `chunk_count` — total chunks for that document
- `chunk_id` — `"{title}#{index}"`
- `start_index` — character offset in the original document

### Title header in chunk text

By default the document `title` is prepended to each chunk's text (`"{title}\n\n{chunk}"`) so the retriever/LLM still knows which page a chunk came from even when it only sees the chunk text. Skipped when the chunk already starts with the title (CSV tables open with their title already).

Result: `Split N document(s) into M chunk(s)` logged, plus a `describe_chunks()` helper reporting min/max/avg chunk size.

---

## 6. How we convert chunks into embeddings

Stage 4: `index_builder.py` (`build_index()`, `get_embeddings()`).

- **Embedding model:** `OllamaEmbeddings` using `OLLAMA_BASE_URL` + `EMBEDDING_MODEL` from `.env` (e.g. `nomic-embed-text`).
- **Dimension detection:** instead of hard-coding 768, the code probes the model with `embed_query("dimension probe")` and measures the returned vector length once per run (`_embedding_dim` cached).
- **Batching:** chunks are embedded and written in batches of `BATCH_SIZE = 32` (`store.add_documents(batch, ids=...)`), with progress logging `Embedded X/Y chunks...`.

Embedding happens implicitly inside `PGVector.add_documents()` — LangChain sends the chunk texts to Ollama, gets vectors back, and writes them with the chunk text and metadata.

---

## 7. How we store the embeddings

Storage is **PostgreSQL + pgvector**, accessed through **SQLAlchemy** (`langchain-postgres` accepts a SQLAlchemy `Engine` as its connection, so one engine drives both the vector store and our own status queries).

### Database setup (`get_engine()`, `ensure_vector_extension()`)

- Engine created from `settings.database.url` (`POSTGRES_*` values in `.env`), `pool_pre_ping=True`.
- `ensure_vector_extension()` checks `pg_extension` for `vector`; if missing it runs `CREATE EXTENSION IF NOT EXISTS vector`. If the role lacks superuser rights it raises a clear, actionable error telling you to run `CREATE EXTENSION vector;` as `postgres` once.

### The vector store (`get_vector_store()`)

```python
PGVector(
    embeddings=get_embeddings(),
    connection=engine,                     # SQLAlchemy Engine
    collection_name=settings.database.collection,
    embedding_length=<detected dimension>, # e.g. 768 for nomic-embed-text
    pre_delete_collection=reset,           # wipe first when reset=True
    create_extension=False,                # already handled above
)
```

- Collection/table: `settings.database.collection` (LangChain tables `langchain_pg_collection` / `langchain_pg_embedding`).
- `reset=True` deletes and recreates the collection — use it for a clean rebuild; otherwise stale chunks from older runs survive.

### Stable chunk IDs (re-indexing without duplicates)

`stable_chunk_id()` builds a deterministic ID from `source_url` (or `origin`) + `chunk_id` + `start_index` — **not** from content. So re-indexing an unchanged document produces the same ID and the row is updated in place; changed text with the same identity is refreshed rather than duplicated.

### Full build (`build_index()`)

1. If no chunks passed in, runs stages 1–3: `load_documents()` -> `clean_documents()` -> `split_documents()`.
2. Aborts early with `{"chunks": 0, "written": 0}` when there is nothing to index.
3. `get_vector_store(reset=reset)`.
4. Computes stable IDs, embeds + writes in batches of 32.
5. Returns a summary dict: `chunks`, `written`, `collection`, `embedding_model`, `reset`.

### Checking what's stored (`index_status()`)

Plain SQLAlchemy (no LangChain) counts rows in `langchain_pg_embedding` joined to `langchain_pg_collection` for the configured collection — returns `collections` and `chunks`.

### Smoke test

The module's `__main__` builds the index from local files (`reset=True`), prints summary + status, and runs a similarity search (`"Who was Arsenal's top scorer?"`, k=3) against the store to verify retrieval works end-to-end.

---

## Summary of the four stages

| Stage | File | Input | Output |
|-------|------|-------|--------|
| 1. Load / fetch | `wikipedia_loader.py` | local `.txt/.md/.csv` + Wikipedia titles | LangChain `Document` objects |
| 2. Clean | `document_cleaner.py` | raw Documents | cleaned Documents (noise removed, deduped, short ones dropped) |
| 3. Split | `text_splitter.py` | cleaned Documents | overlapping chunks with chunk metadata |
| 4. Embed + store | `index_builder.py` | chunks | vectors + text in PostgreSQL/pgvector (`langchain_pg_embedding`) |
