# Ingestion Changes

This file records the ingestion work in phases.

**Phase 1 is done. Phase 2 has not started.**

---

## Phase 1 — Ingestion and chunking

Date: 2026-10-20

### Goal

The old pipeline had four problems:

1. Paragraphs were joined with spaces. The `\n\n` paragraph breaks were lost.
2. Boilerplate sections were indexed. Examples: captions, legends, "The following ..." stubs.
3. Short sections became tiny chunks. Some chunks were under 30 characters.
4. Chunks had no section context. The embedding could not tell which part of the page a chunk came from.

Phase 1 fixes these problems. It does not add table extraction. It does not change the answer generation prompt.

### Pipeline shape (unchanged)

```
load -> clean -> assemble sections -> split -> build_index(chunks)
```

`build_index(chunks)` only embeds and writes. It does not run stages 1-3.

### 1. Paragraph fix

MediaWiki `explaintext` uses **one** newline between paragraphs, not a blank line.

Old rule: join lines until a blank line.
New rule: every non-empty explaintext line is a paragraph. Join paragraphs with `\n\n`.

Evidence from the 2025-26 season page:

| Section | Chars | Paragraphs | `\n\n` | Old space-join `\n\n` |
|---|---:|---:|---:|---:|
| Introduction | 912 | 2 | 1 | 0 |
| Review > May | 4,547 | 5 | 4 | 0 |

Paragraph check: **1.0** (5/5 in May, 2/2 in Introduction).

### 2. Cleanup and section assembly

New stage: `src/ingestion/section_assembler.py`.

It runs after `clean_documents` and before `split_documents`.

Rules:

- Drop boilerplate sections. The rules are in `config/settings.yaml` under `boilerplate_patterns`.
- Drop a section if its length is under the rule's `max_chars` in `config/settings.yaml` (or `BOILERPLATE_MAX_CHARS` if the rule omits it) and its text matches the rule's pattern.
- A rule may also carry `keep_if_pattern`. When that regex matches anywhere in the body, the section is kept. The `^The following ` rule uses `keep_if_pattern: '[£€$]\s?\d'`, so short transfer sections that hold a money total (spend/income) survive.
- Merge a short parent into its first child. Example: `Kits` -> `Kits > Kit information`.
- Merge a short sibling into the previous sibling. Example: `Kent County FA` -> `London FA`.
- Merge a short child into its parent. Example: `Most appearances` -> `Player records > Appearances`.
- Drop a short section if it cannot be merged and is under `MIN_SECTION_KEEP`.
- Repeat the merge passes up to `MIN_SECTION_MAX_PASSES` times.

Live result on the three configured pages:

| Item | Count |
|---|---:|
| Sections in | 97 |
| Sections out | 74 |
| Dropped | 20 |
| Merged | 3 |

Dropped:

- 19 boilerplate/caption sections.
- 1 too-short section: `Competitions > Premier League > Matches` (51 chars).

Merged:

- `Kits` (67) -> `Kits > Kit information`
- `Kent County FA` (37) -> `London FA`
- `Most appearances` (104) -> `Player records > Appearances`

Kept sections under 150 chars: **0**.

Per page: records 21, season 27, Arsenal F.C. 26.

Full log: `eval/phase1_evidence.log`.

### 3. Chunk format

Every chunk now has a header:

```
<title>
Section: <section_path>

<body>
```

`CHUNK_SIZE` is the **body** budget. The header is added on top.
`CHUNK_MAX_WARN` is a warning for the **total** length.

This gives every chunk its own context. A chunk from `Review > May` now says so in the text.

### 4. Chunking rules

- Split the body only. The header is added after splitting.
- Use `keep_separator="end"` so a chunk does not start with `. `.
- Fold a short final piece back into the previous chunk if the merged body fits.
- Keep a short final piece separate only if folding would exceed `CHUNK_SIZE + CHUNK_OVERLAP`.
- Remove duplicate chunks inside one section with a body hash.
- `chunk_index` and `chunk_count` are per section.
- `chunk_id` is `url#section_path#chunk_index`.
- `content_hash` is the SHA-256 of the body.
- `section_hash` is the SHA-256 of the whole section body.

### 5. Metadata added

Every chunk has:

- `page_type` — `season`, `list`, `club`, or `other`
- `seasons` — up to `SEASONS_MAX` season strings, normalised to `2025–26`
- `heading_level` — 0 for the lead, otherwise path length + 1
- `parent_section`
- `is_lead`
- `merged_from` — section paths that were merged into this chunk
- `header_chars`
- `body_chars`
- `chunk_chars`
- `chunk_index`
- `chunk_count`
- `chunk_id`
- `content_hash`
- `section_hash`

The header stays in the stored chunk. Phase 2 (not done yet) will strip it before the text is sent to the LLM, build the Sources list in code, and have the LLM write only `[n]` markers.

### 6. Settings

New keys in `.env`:

```text
CHUNK_SIZE=900
CHUNK_OVERLAP=120
CHUNK_MIN_SIZE=150
CHUNK_MAX_WARN=1200
CHUNK_HEADER_ENABLED=true
INDEX_BATCH_SIZE=32

MIN_SECTION_CHARS=250
MIN_SECTION_KEEP=60
MIN_SECTION_MAX_PASSES=3
BOILERPLATE_MAX_CHARS=800

SEASONS_FROM_TEXT=false
SEASONS_MAX=8

USE_FETCH_CACHE=true
REFRESH_FETCH_CACHE=false
FETCH_CACHE_DIR=data/cache

EVAL_TOP_K=8
EVAL_HIT_AT=1,3,5,8
EVAL_COLLECTION_PREFIX=arsenal_eval_
EVAL_RESULTS_DIR=eval/results
TEST_REPORT_PATH=test_report.txt

EMBEDDING_QUERY_INSTRUCTION=Given a question about Arsenal F.C., retrieve Wikipedia passages that answer it
EMBEDDING_QUERY_INSTRUCTION_ENABLED=false

FINAL_TOP_N=8
QUERY_REWRITE_TEMPERATURE=0.0
RETRIEVE_K=15
SCORE_THRESHOLD=0.7
```

Removed:

- `TOP_K` — removed. The code uses `FINAL_TOP_N`.
- `RERANK_TOP_N` — renamed to `FINAL_TOP_N`.

New `config/settings.yaml` blocks:

- `split_separators`
- `boilerplate_patterns`
- `page_type_rules`
- `season_normalise_pattern`

All keys are documented in `.env.example`.

### 7. Cache

New file: `src/ingestion/fetch_cache.py`.

- Caches the raw MediaWiki extract and the title -> file pointer.
- `index.json` maps title to `{pageid}_{revid}.json`.
- A corrupt or missing file falls back to a live fetch.
- The second run uses the cache and gets identical text.
- The loader tests disable the cache with an autouse fixture.

### 8. Evaluation

The eval uses the low-level `search_chunks()` directly. It does **not** use query rewriting and does **not** apply the score threshold. This measures chunking and embedding quality only.

Two metrics:

- **Primary** — the chunk must be in the expected section and contain the expected keyword(s).
- **Secondary** — the chunk only has to contain the keyword(s).

22 questions: 20 positive, 2 negative (Q21, Q22). Negative questions are not scored.

Eval collections are isolated:

- `arsenal_eval_600_100`
- `arsenal_eval_900_120`
- `arsenal_eval_1200_150`
- `arsenal_eval_1500_220`

The production collection `arsenal_documents` was **not** rebuilt. It stays at 159 old chunks.

### 9. Results

`hit@N` = the answer chunk is in the top N.
`MRR` uses section + keyword matching. `MRR(kw)` uses keywords only.

| collection | chunks | median body | hit@1 | hit@3 | hit@5 | hit@8 | MRR | MRR(kw) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arsenal_documents` (old baseline) | 159 | 930* | 0.50 | 0.65 | 0.80 | 0.85 | 0.608 | 0.608 |
| `arsenal_eval_600_100` | 311 | 443 | 0.45 | 0.75 | 0.90 | 0.95 | 0.643 | 0.735 |
| `arsenal_eval_900_120` | 222 | 608 | 0.60 | 0.85 | 0.95 | 0.95 | 0.733 | 0.767 |
| `arsenal_eval_1200_150` | 179 | 743 | 0.70 | 0.80 | 0.90 | 0.90 | 0.778 | 0.778 |
| `arsenal_eval_1500_220` | 146 | 930 | 0.70 | 0.80 | 0.90 | 0.95 | 0.780 | 0.780 |

\* Old chunks have no `body_chars`; this is total chunk length.

What the numbers say:

- The old baseline missed Q10 and Q12. All new configs fix them.
- The old baseline had 61 chunks starting with `.` and 14 chunks under 150 chars. The new 900/120 config has 0 and 0.
- 1500/220 has the highest MRR (0.780). The gain comes from two records-list questions, Q19 and Q20.
- 900/120 has the best `hit@3` (0.85) and `hit@5` (0.95). It ties the best `hit@8` (0.95).
- 900/120 has 0 chunks over the 1200 warning. 1200/150 has 14. 1500/220 has 46.
- Q1 (three-part question) misses in every config. A single query cannot reach three distant sections in the top 8. Phase 2 query rewriting is the fix.

**Recommended default: 900 / 120.**

Reasons:

- Best precision at k = 3 and k = 5.
- Ties the best hit@8.
- Keeps chunks focused.
- No chunk over the warning limit.
- Ranks the season-page questions better than 1500/220.

If the goal is the highest raw MRR, use 1500/220. It is a valid alternative, but it is less focused and produces 46 chunks over the warning limit.

### 10. Top results for Q1, Q4, Q14, Q21 on 900/120

Full top-8 log: `eval/eval_900_show.log`.

- **Q1** — "what is arsenal season 2025-2026 and what did they achieved": not in the top 8. Best part rank is 22 at k=30. Top results: Introduction, UCL League phase, Review > October. Three facts, three distant sections, one query. Not a chunking bug.
- **Q4** — League Cup final: answer chunk `Final` at rank 2.
- **Q14** — Thierry Henry / top scorer: `Player records > Top goalscorers` at rank 1. Introduction at rank 3. `require_any_part` -> rank 1.
- **Q21** — average possession (negative question): not scored. Top results are Introduction and Review > May. No possession stat in the top 8, which is correct for the available pages.

### 11. Tests

- `pytest`: **101 passed, 6 skipped** (84 at the end of Phase 1, plus 15 settings-audit tests in `tests/test_config.py` and 2 currency-rule tests).
- The 6 skips are live Wikipedia integration tests. Run them with `ARSENAL_WEKI_INTEGRATION=1`.
- Quality checks on `arsenal_eval_900_120`: **14/14 PASS**. Re-run with `__pycache__` deleted on 2026-10-20: same numbers.
  - no body under 150 (shortest 166)
  - no body starts with `.`
  - every chunk has the title + section header
  - no junk sections
  - no boilerplate stubs
  - Kits merged into Kit information
  - Kent County FA not standalone
  - paragraph breaks survived
  - all metadata present
  - chunk ids unique
  - re-index without reset: 222 -> 222

Report: `test_report.txt`.

### 11b. Currency rule change (0.3)

`^The following ` sections under 300 chars were dropped, which also removed the
transfer spend/income totals. The rule now has
`keep_if_pattern: '[£€$]\s?\d'`.

Effect on the same cached text (no network):

| Item | Before | After |
|---|---:|---:|
| Sections in | 97 | 97 |
| Sections kept | 74 | 76 |
| Sections dropped | 20 | 16 |
| Merges | 3 | 5 |
| Chunks | 222 | 224 |

The totals still live in the index: `Total expenditure: £248.2 million`,
`Total income: £14.5 million` and `Total expenditure: £0.0 million` are now in
the `Contracts and transfers > Transfers out` chunk. Sections without a money
total (`New contracts`, `Released`, `Appearances`) are still dropped.

### 12. Known limits

- Q1 is a multi-part question. One embedding cannot retrieve all parts. Phase 2 rewriting should help.
- 600/100 keeps one 93-char final chunk because folding it back would exceed the 700-char limit.
- 1500/220 has 46 chunks over the 1200 warning. It works, but the context is larger.
- The currency rule was added in 0.3, after the results table above was measured. The table is the 222-chunk baseline; the 224-chunk corpus is the Phase 2 baseline.

### 13. Files changed

New:

- `src/ingestion/section_assembler.py`
- `src/ingestion/fetch_cache.py`
- `src/text_utils.py`
- `eval/questions.json`
- `eval/build_eval_collection.py`
- `eval/retrieval_eval.py`
- `eval/compare_results.py`
- `tests/test_section_assembler.py`
- `tests/test_text_utils.py`
- `tests/test_ingestion_quality.py`
- `tests/run_quality_checks.py`
- `.env.example`

Modified:

- `src/config.py`
- `config/settings.yaml`
- `src/ingestion/wikipedia_loader.py`
- `src/ingestion/document_cleaner.py`
- `src/ingestion/text_splitter.py`
- `src/ingestion/index_builder.py`
- `src/ingestion/run_ingestion.py`
- `src/retrieval/retriever.py`
- `tests/test_text_splitter.py`
- `tests/test_ingestion.py`
- `tests/test_index_builder.py`
- `tests/test_wikipedia_loader.py`
- `.env`

Evidence:

- `eval/phase1_evidence.log`
- `eval/eval_900_show.log`
- `test_report.txt`

### 14. Deviations from the plan

1. Paragraph rule is line-based, not blank-line-based. The data forced this.
2. Added child -> parent merge. The plan only had parent -> child and sibling -> sibling. This was needed for `Most appearances`.
3. `RERANK_TOP_N` -> `FINAL_TOP_N` was done in Phase 1. The retriever needed it once the old key was removed.
4. Added dormant Qwen3 query-instruction settings. Default is off.
5. Added `chunk_id` and per-chunk `content_hash`. `section_hash` holds the section hash.
6. Added `EVAL_HIT_AT` so the metric list is not hard-coded.
7. Fixed a pre-existing SQL bug: `delete_existing_for_source` used `e.metadata`. The column is `e.cmetadata`.
8. Fixed a new bug: chunk IDs collided across collections. The ID now includes the collection name.
9. `build_index`, `get_vector_store`, `delete_existing_for_source`, and `index_status` accept a `collection` argument.
10. `batch_size` comes from `INDEX_BATCH_SIZE`. The hard-coded `BATCH_SIZE = 32` was removed.

---

## Phase 2 — not started

Phase 2 covers answer generation, query rewriting, the query instruction, and prompt changes. It starts only after Phase 1 is approved.
