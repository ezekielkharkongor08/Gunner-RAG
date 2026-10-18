# Arsenal Weki: Ingestion Improvements (Task for OpenCode)

Read `ingestion.md` first. It explains how the pipeline works today:

```
load (wikipedia_loader.py) -> clean (document_cleaner.py) -> split (text_splitter.py) -> embed + store (index_builder.py)
```

Your job is to improve **only the ingestion part** (`src/ingestion/`) so that retrieval gives the best answers. Work through the tasks below in order.

---

## 0. Rules (please follow these)

1. **Scope.** Change only the ingestion pipeline. Do not touch the retrieval, LLM, or UI code. Do not add HNSW, hybrid search, or a reranker. That is a later step.
2. **No hard-coded values.** Every new setting goes into `.env` / `config/settings.yaml` and is read through `src.config`, like the current settings. Add each new setting to `.env.example` with a short comment.
3. **Tables from Wikipedia are NOT in scope.** Keep using `wikipediaapi` text extraction as it is. Do not add HTML table scraping. Tables enter the system only through the local CSV files.
4. **Keep what works.** Keep the current behavior for graceful error handling, deduplication, short-chunk folding, and batch embedding.
5. **Keep code simple and commented.** The owner is a student and will read this code. Use clear names and short docstrings.
6. **Do not move to any next step** after finishing. See Section 5 (the approval gate).

---

## 1. General improvements (Wikipedia text and local .txt/.md files)

### 1.1 Split by section first, then by size
- Wikipedia text has headings like `== History ==` and `=== Wenger era ===`.
- First split each document on these headings. Then run `RecursiveCharacterTextSplitter` inside each section only if the section is longer than `CHUNK_SIZE`.
- Add a **breadcrumb** line at the top of each chunk, for example:
  `Arsenal F.C. > History > Wenger era`
  This replaces the plain title header. For documents with no headings, the breadcrumb is just the title.
- Store `section` (the full heading path) in the chunk metadata.
- The heading text must be read **before** the cleaner removes anything that would break heading detection. Check that the cleaner does not remove the `==` markers too early. If it does, either detect the sections in the loader or change the order carefully. Explain your choice in the changes file.

### 1.2 Remove junk sections
Drop these sections completely (case-insensitive, list is configurable in `settings.yaml`):
`References`, `External links`, `See also`, `Further reading`, `Notes`, `Sources`, `Bibliography`.
Log how many sections were dropped.

### 1.3 Chunk size
- Make the defaults `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150` (still configurable in `.env`).
- Keep `CHUNK_MIN_SIZE` folding.
- Keep chunks smaller than the embedding model's token limit. Add a log warning if the largest chunk is above a configurable `CHUNK_MAX_WARN` value.

### 1.4 Fix the stable ID problem
Today the ID uses `start_index`. If a page changes, the offsets move and old chunks stay in the database as duplicates.
- Before indexing a document, **delete all existing rows for that `source_url`** (or `origin` for local files), then insert the fresh chunks.
- Keep IDs deterministic. Include a short hash of the chunk text in the ID in addition to the source and chunk index.
- Re-running the build on unchanged data must give the **same row count** (no growth).

### 1.5 Richer metadata
Add these fields to every chunk where possible:
- `section` (heading path)
- `content_type`: `"prose"` or `"table"`
- `fetched_at`: UTC timestamp (ISO format) of when the page or file was loaded
- `revision_id`: the Wikipedia revision id **if the library gives it**. If it does not, store `pageid` instead, or skip it. Do not invent values.
- `chunk_chars`: length of the chunk text

### 1.6 More pages
In `config/settings.yaml`, add these titles to the default page list (keep the existing ones):
- List of Arsenal F.C. managers
- List of Arsenal F.C. seasons
- List of Arsenal F.C. records and statistics
- History of Arsenal F.C. (1886–1966)
- History of Arsenal F.C. (1966–1986)
- History of Arsenal F.C. (1986–2006)
- History of Arsenal F.C. (2006–present)
- Arsène Wenger
- Mikel Arteta
- Emirates Stadium
- Highbury
- North London derby
- Arsenal F.C. in European football

Skip missing pages with a warning (already handled). Also check whether `WIKIPEDIA_MAX_CHARS` cuts long pages. If yes, raise the default or log clearly which pages were truncated.

### 1.7 Embedding model and prefixes
- Do **not** change the default embedding model. The owner will test models later.
- Make the model fully switchable by `EMBEDDING_MODEL` in `.env` (this already works).
- Add a small wrapper class around `OllamaEmbeddings` that adds a prefix to the text:
  - `EMBEDDING_DOC_PREFIX` is added in `embed_documents`
  - `EMBEDDING_QUERY_PREFIX` is added in `embed_query`
  - Both default to empty. Example for nomic-embed-text: `search_document: ` and `search_query: `.
- The dimension probe must still work. The probe should use the wrapper.
- Note in the changes file that **changing the model or prefixes needs `reset=True`**.

---

## 2. CSV improvements

Today a CSV becomes one Document, then the generic text splitter cuts it by characters. That can cut the table in the middle and later chunks lose the table context. Fix it like this.

### 2.1 Chunk CSVs by rows, not by characters
- CSV Documents must **skip** the generic text splitter. Give them `content_type="table"` and let `split_documents()` pass them through (or build the row chunks inside a dedicated CSV function).
- Group `CSV_ROWS_PER_CHUNK` rows per chunk (default 12, set in `.env`). Never cut a row in the middle.
- A small table (rows <= `CSV_ROWS_PER_CHUNK`) stays as one chunk.
- **Repeat the context at the top of every chunk**:

```
List of Arsenal F.C. records and statistics: Top appearances and goals
Columns: Player, Total Games/Goals

- Rank: 1, Player: Thierry Henry, Total Games/Goals: 377 / 228
- Rank: 2, Player: Ian Wright, Total Games/Goals: 288 / 185
```

  (title, optional page context, a `Columns:` line, then the row lines.)

### 2.2 Make columns easy to understand
- Support an optional header line in the CSV file: `# COLUMN_NOTES: Total Games/Goals = appearances / goals scored; Rank = position in the list`.
- If present, add one line `Notes: ...` under the `Columns:` line in every chunk.
- Add an optional mode `CSV_ROW_STYLE` with values `pairs` (default, the current `Key: value` format) and `sentence`. For `sentence`, use a simple template from an optional header `# ROW_TEMPLATE: {Player} made {Total Games/Goals} (games / goals) for Arsenal.` If a row has a missing value, fall back to the `pairs` style for that row. Do not crash.

### 2.3 Summary chunk per table
Embeddings are weak at "most / highest / first / latest" questions. Add **one extra chunk per table** with the answer written as plain text:
- Opt in with a header line: `# SUMMARY_COLUMNS: Total Games/Goals`.
- For each listed column, read the **first number** in each cell (so `377 / 228` gives 377 and `558 (11)` gives 558). Sort and write the top 3 rows, for example:
  `Summary: The highest value of Total Games/Goals in this table is Thierry Henry (377 / 228), then Ian Wright (288 / 185), then ...`
- Do this in code (Python), not with the LLM.
- If a column has no numbers, log a warning and skip it.
- Mark the chunk with `content_type="table_summary"`.

### 2.4 CSV metadata
Every CSV chunk gets: `content_type`, `table_name` (the title), `row_start`, `row_end` (row numbers in this chunk), `columns`, `file_type="csv"`, `source`, `origin`.

### 2.5 Cleaner check for CSV
The cleaner removes `[12]` and `[a]` style markers. Make sure it does not damage real CSV values. Write a test case for it (see `test.py` below). Do not change the cleaner for prose.

---

## 3. `test.py` (required)

Create `test.py` in the project root. The owner will run it **after your changes** to check the ingestion result with their own eyes. It must be easy to read and must print a clear report.

### 3.1 How it runs
- `python test.py` runs stages 1 to 3 (load, clean, split) **without** needing Ollama or PostgreSQL. It uses local files only by default.
- `python test.py --wikipedia` also fetches the Wikipedia pages.
- `python test.py --db` also runs the embed and store step (needs Ollama and PostgreSQL) and the database checks.
- Print `PASS` or `FAIL` for each check. At the end print a summary and exit with code 1 if any check failed.
- Save the full report to `test_report.txt` too.

### 3.2 Checks (no database)
1. **Counts:** documents loaded, after cleaning, number of chunks. Show counts per source (`local`, `wikipedia`) and per `content_type`.
2. **Chunk size:** min, max, average. FAIL if any chunk is empty. WARN if above `CHUNK_MAX_WARN`.
3. **Junk sections:** FAIL if any chunk breadcrumb or text still has a section named References, External links, See also, Further reading, or Notes.
4. **Noise:** FAIL if any chunk still contains `[[`, `]]`, or citation markers like `[12]` (except allowed CSV cases, explain them).
5. **Metadata:** FAIL if any chunk misses `title`, `source_url`, `source`, `chunk_id`, `section`, `content_type`, `fetched_at`.
6. **Breadcrumb:** every prose chunk starts with the breadcrumb line.
7. **Unique IDs:** all stable chunk IDs are unique.
8. **CSV checks:**
   - every CSV chunk starts with the title and has a `Columns:` line
   - no row is split across two chunks
   - the number of row lines in all chunks of a table equals the number of data rows in the CSV file
   - summary chunk exists for tables that have `SUMMARY_COLUMNS`, and the top value in it matches a manual calculation
   - a sample CSV with `[a]` in a cell is not damaged
9. **Deduplication:** loading the same file twice gives no duplicate chunks.
10. **Unit tests** (small, with fake input) for: section splitting, junk section removal, short-chunk folding, row grouping, summary chunk, prefix wrapper adds the prefix.

### 3.3 Checks (only with `--db`)
1. Build the index with `reset=True`. The number of rows in the database equals the number of chunks.
2. Run `build_index()` again **without reset**. The row count must stay the same (no duplicates).
3. Change one chunk's text in a copy of the data, re-index, and check that old rows for that `source_url` are gone (the stale chunk problem).
4. Run these sample questions with `k=3` and print the title, section, and the first 150 characters of each result:
   - "Who was Arsenal's top scorer?"
   - "Who managed Arsenal in 1998?"
   - "How many league titles has Arsenal won?"
   - "When was Highbury closed?"
   - "Where do Arsenal play their home games?"
   The test only prints these. The owner will judge the quality by reading them.
5. Print the database `index_status()`.

### 3.4 Sample output at the end
Print 5 random prose chunks and 3 random CSV chunks in full, so the owner can read them.

---

## 4. Write a new changes file (required)

Do **not** edit `ingestion.md`. Create a **new** file called `INGESTION_CHANGES.md` in the project root with:
1. A short list of every change made (file name, function name, and why), grouped as in Sections 1 and 2 above.
2. A list of all new `.env` and `settings.yaml` options, with default values and meaning.
3. Anything you did differently from this task file, and why.
4. How to run `test.py` and how to rebuild the index (`reset=True`).
5. Known limits (for example: tables from Wikipedia pages are still not included).
6. Any problem you found but did not fix.

Write this file in **simple English**, short sentences, easy to understand.

---

## 5. Approval gate (very important)

When you finish:
1. Run `python test.py` and show the result.
2. Tell the owner to review `test_report.txt` and `INGESTION_CHANGES.md`.
3. **STOP.** Do not start any next step (retrieval, hybrid search, reranker, LLM, UI) until the owner clearly says they are happy with the ingestion result.
4. If the owner finds a problem, fix it, re-run `test.py`, and wait again.
