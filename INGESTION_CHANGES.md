# Arsenal Weki — Ingestion Changes

This file lists every change made to the ingestion pipeline (`src/ingestion/`)
and the surrounding configuration, grouped as in INGESTION_IMPROVEMENTS.md.

## 1. General improvements (Wikipedia text and local .txt/.md files)

### 1.1 Split by section first, then by size
**File:** `src/ingestion/text_splitter.py` — `_split_into_sections()`, `_split_prose_document()`
**Why:** Wikipedia text has headings like `== History ==`. Splitting by section
first keeps related content together, then the size-based splitter only runs
inside a section when it exceeds `CHUNK_SIZE`.

- Added `HEADING_RE` to detect `== Heading ==` and `=== Subheading ===` markers.
- Documents are split into sections, junk sections are dropped, then each
  remaining section is split by `RecursiveCharacterTextSplitter` if it is
  longer than `CHUNK_SIZE`.
- **Breadcrumb line** added to every chunk: `Arsenal F.C. > History > Wenger era`
  (replaces the plain title header). Documents with no headings get just the
  title as the breadcrumb.
- `section` stored in chunk metadata (the full heading path).

**File:** `src/ingestion/wikipedia_loader.py` — `_utc_now()`, `_wiki_revision()`
**Why:** Headings must survive cleaning. The cleaner's separator-line rule only
matches lines made *entirely* of `= - # * _` characters, so `== History ==`
(with text inside) is preserved. No change to the cleaner was needed.

**File:** `src/ingestion/wikipedia_loader.py`
**Why:** Richer metadata.
- `fetched_at` (UTC ISO timestamp) added to every document (local + Wikipedia).
- `revision_id` added to Wikipedia documents (falls back to `pageid` if the
  library does not expose a revision id; never invents a value).

### 1.2 Remove junk sections
**File:** `src/ingestion/text_splitter.py` — `_is_junk_section()`, `_split_prose_document()`
**File:** `config/settings.yaml` — `junk_sections` list
**Why:** `References`, `External links`, `See also`, `Further reading`, `Notes`,
`Sources`, `Bibliography` sections add noise and are dropped completely.
- The list is configurable in `config/settings.yaml` under `junk_sections`.
- Dropped sections are logged.

### 1.3 Chunk size
**File:** `.env` — `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`
**File:** `src/config.py` — `ChunkSettings` now includes `min_size` and `max_warn`
**Why:** Defaults changed to `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150`.
`CHUNK_MIN_SIZE=120` folding is kept.
- Added `CHUNK_MAX_WARN=1000`: a log warning is emitted if any chunk exceeds
  this (embedding model token limit).

### 1.4 Fix the stable ID problem
**File:** `src/ingestion/index_builder.py` — `delete_existing_for_source()`,
`stable_chunk_id()`, `build_index()`
**Why:** The old ID used `start_index`, which shifts when a page changes and
leaves stale rows in the database.
- Before indexing, all existing rows for that `source_url` (or `origin` for
  local files) are deleted, then fresh chunks are inserted.
- `stable_chunk_id()` now includes a short hash of the chunk text, so changed
  text gets a new ID.
- Re-running on unchanged data gives the same row count (no growth).

### 1.5 Richer metadata
**File:** `src/ingestion/wikipedia_loader.py` — `fetched_at`, `revision_id`
**File:** `src/ingestion/text_splitter.py` — `section`, `content_type`,
`chunk_chars` on every chunk
**Why:** Every chunk now carries:
- `section` (heading path; table name for CSV)
- `content_type`: `"prose"` or `"table"` (or `"table_summary"`)
- `fetched_at`: UTC timestamp of when the page/file was loaded
- `revision_id`: Wikipedia revision id or pageid (Wikipedia docs only)
- `chunk_chars`: length of the chunk text

### 1.6 More pages
**File:** `config/settings.yaml` — `pages` list expanded
**Why:** Added 9 new titles (kept all existing ones):
- List of Arsenal F.C. seasons
- History of Arsenal F.C. (1886–1966)
- History of Arsenal F.C. (1966–1986)
- History of Arsenal F.C. (1986–2006)
- History of Arsenal F.C. (2006–present)
- Arsène Wenger
- Mikel Arteta
- Emirates Stadium (already present, kept)
- Highbury
- North London derby
- Arsenal F.C. in European football (already present, kept)

Missing pages are skipped with a warning (already handled). `WIKIPEDIA_MAX_CHARS`
default is 200,000 — long pages are truncated and the truncation is logged.

### 1.7 Embedding model and prefixes
**File:** `src/ingestion/index_builder.py` — `PrefixedEmbeddings` class
**File:** `src/config.py` — `OllamaSettings` gains `embedding_doc_prefix`,
`embedding_query_prefix`
**File:** `.env` — `EMBEDDING_DOC_PREFIX=`, `EMBEDDING_QUERY_PREFIX=`
**Why:** A wrapper class around `OllamaEmbeddings` adds a prefix to each text:
- `EMBEDDING_DOC_PREFIX` added in `embed_documents` (indexing).
- `EMBEDDING_QUERY_PREFIX` added in `embed_query` (retrieval).
- Both default to empty. Example for nomic-embed-text: `search_document: `
  and `search_query: `.
- The dimension probe uses the wrapper, so it still works.
- **Note:** changing the model or prefixes requires `reset=True` on re-index.

## 2. CSV improvements

### 2.1 Chunk CSVs by rows, not by characters
**File:** `src/ingestion/text_splitter.py` — `_split_csv_document()`
**File:** `src/ingestion/wikipedia_loader.py` — `_document_from_csv_file()`
**Why:** The generic text splitter could cut a table in the middle.
- CSV documents skip the generic text splitter entirely.
- Rows are grouped `CSV_ROWS_PER_CHUNK` (default 12) per chunk; a row is never
  split across two chunks.
- Small tables (rows <= `CSV_ROWS_PER_CHUNK`) stay as one chunk.
- Context is repeated at the top of every chunk: title, page context,
  `Columns:` line, then the row lines for that chunk.

### 2.2 Make columns easy to understand
**File:** `src/ingestion/wikipedia_loader.py` — captures `# COLUMN_NOTES`,
`# ROW_TEMPLATE`, `# SUMMARY_COLUMNS` headers
**File:** `src/ingestion/text_splitter.py` — `_format_row_pairs()`,
`_format_row_sentence()`
**Why:**
- `# COLUMN_NOTES: ...` header adds a `Notes: ...` line under `Columns:` in
  every chunk.
- `# ROW_TEMPLATE: ...` header enables `sentence` style (set `CSV_ROW_STYLE=sentence`).
  If a row has a missing template value, it falls back to `pairs` style —
  never crashes.
- `CSV_ROW_STYLE` values: `pairs` (default) or `sentence`.

### 2.3 Summary chunk per table
**File:** `src/ingestion/text_splitter.py` — `_build_summary_chunk()`
**Why:** Embeddings are weak at "most / highest / first / latest" questions.
- `# SUMMARY_COLUMNS: Total Games/Goals` header opts in.
- For each listed column, the first number in each cell is read
  (`377 / 228` → 377, `558 (11)` → 518). Top 3 rows are written as plain text,
  e.g. `Summary: The highest value of Total Games/Goals ... is Thierry Henry (377 / 228), then ...`
- Done in Python, not with the LLM.
- Columns with no numbers are logged and skipped.
- Summary chunks get `content_type="table_summary"`.

### 2.4 CSV metadata
**File:** `src/ingestion/text_splitter.py` — `_split_csv_document()`
Every CSV chunk gets: `content_type`, `table_name`, `row_start`, `row_end`,
`columns`, `file_type="csv"`, `source`, `origin`, `section`, `chunk_chars`.

### 2.5 Cleaner check for CSV
**File:** `src/ingestion/text_splitter.py` — `clean_text()` applied to each
CSV row line after formatting
**Why:** The cleaner removes `[12]` and `[a]` markers from CSV cell values
too (e.g. `Ref.: [475]` becomes `Ref.:`). The cleaner itself was not changed
for prose, as required. A test case covers this in `test.py`.

## 3. New settings

### .env / .env.example
| Setting | Default | Meaning |
|---------|---------|---------|
| `CHUNK_SIZE` | 1000 | Max chunk size in characters |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `CHUNK_MIN_SIZE` | 120 | Fold chunks shorter than this |
| `CHUNK_MAX_WARN` | 1000 | Warn if any chunk exceeds this |
| `EMBEDDING_DOC_PREFIX` | (empty) | Prefix for documents at embed time |
| `EMBEDDING_QUERY_PREFIX` | (empty) | Prefix for queries at embed time |
| `CSV_ROWS_PER_CHUNK` | 12 | Data rows per CSV chunk |
| `CSV_ROW_STYLE` | pairs | `pairs` or `sentence` |

### config/settings.yaml
| Setting | Default | Meaning |
|---------|---------|---------|
| `pages` | 17 titles | Wikipedia pages to fetch (Mode B) |
| `junk_sections` | 7 section names | Sections dropped during splitting |
| `csv_rows_per_chunk` | 12 | (mirrors .env `CSV_ROWS_PER_CHUNK`) |
| `csv_row_style` | pairs | (mirrors .env `CSV_ROW_STYLE`) |

## 4. How to run test.py

```
python test.py                     # stages 1-3 only (local files, no Ollama/DB)
python test.py --wikipedia         # also fetches Wikipedia pages (needs network)
python test.py --db                # also embeds + stores (needs Ollama + PostgreSQL)
```

- Prints PASS/FAIL for each check.
- Saves full report to `test_report.txt`.
- Exits with code 1 if any check failed.

## 5. How to rebuild the index

```
python -m src.ingestion.index_builder    # reset=True, local files only
```

Or from Python:
```python
from src.ingestion.index_builder import build_index
summary = build_index(reset=True)
```

Use `reset=True` for a clean rebuild (wipes the collection first). Without
`reset`, each source is deleted before re-inserting its fresh chunks, so
re-indexing never leaves stale rows.

## 6. Known limits
- Tables from Wikipedia pages are NOT scraped (out of scope per task rules).
  Tables enter only through local CSV files.
- Editorial `.txt` files using `### Markdown ###` headings are not split by
  section (only `== Wikipedia ==` style headings are detected). They are still
  split by size within the single section.
- Changing the embedding model or prefixes requires `reset=True`.

## 7. Problems found but not fixed
- None. All checks in `test.py` pass.