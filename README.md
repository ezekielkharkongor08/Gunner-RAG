# Arsenal Wiki

A local question-answering system for Arsenal F.C. content on Wikipedia.

It downloads Wikipedia pages through the official MediaWiki API, keeps only the
prose, splits it into section-aware chunks, stores the chunks as vectors in
PostgreSQL/pgvector, and answers questions with a local Ollama model. Every
answer cites the article and section it came from.

There is no HTML scraping, no local-file ingestion, and no table or infobox
extraction. Wikipedia is the only source.

---

## Features

- **Wikipedia only** — pages are fetched through the MediaWiki Action API with
  `action=query&prop=extracts|info`, `explaintext=1` and
  `exsectionformat=wiki`.
- **Prose only** — infoboxes, tables, navigation boxes and reference lists are
  never indexed. Excluded tables are not replaced with generated summaries.
- **Section aware** — every paragraph belongs to exactly one section. The
  introduction is its own section, and parent/child sections never duplicate
  text.
- **Clean chunks** — short sections are merged instead of becoming tiny
  chunks, boilerplate captions are dropped, and each chunk carries a
  `Title` + `Section:` header for context.
- **Reproducible ingestion** — raw Wikipedia extracts are cached in
  `data/cache/`, so experiments reuse the same text.
- **pgvector search** — embeddings come from Ollama and are stored in
  PostgreSQL with the pgvector extension.
- **Section routing** — a question that names a section (“November”, “kits”)
  is routed straight to that section instead of relying on similarity search.
- **Relative result gate** — keeps chunks close to the best match, so a correct
  passage is not dropped by a fixed score cut-off.
- **Follow-up questions** — the chat keeps real `(question, answer)` pairs and
  condenses short follow-ups such as “who did they beat?” into a standalone
  question.
- **Honest citations** — the model writes only `[n]` markers; the source list
  is built in code from stored metadata (title, section, URL, fetch time).
- **Grounding check** — `ask.py --check` reports numbers and names in the
  answer that do not appear in the sources.
- **Season guard** — questions that name a season are filtered to passages
  from that season, and the prompt forbids moving a previous-season result
  into the current season.
- **Evaluation harness** — retrieval-only metrics (hit@N and MRR), section
  routing checks, a gate report and a chunk-size comparison table.

---

## Architecture

```
Wikipedia (MediaWiki Action API)
        |
        v
src/ingestion/wikipedia_loader.py      fetch pages, split into sections
        |
        v
src/ingestion/document_cleaner.py      remove citations, templates, HTML
        |
        v
src/ingestion/section_assembler.py     drop boilerplate, merge short sections
        |
        v
src/ingestion/text_splitter.py         split into chunks with a header
        |
        v
src/ingestion/index_builder.py         embed with Ollama, write to pgvector
        |
        v
src/retrieval/section_router.py        route questions that name a section
src/retrieval/retriever.py             search, fuse, gate, return chunks
        |
        v
src/generation/answer_chain.py         build the prompt, call the model, cite
src/generation/grounding.py            check the answer against the sources
        |
        v
ask.py / main.py                       answer in the terminal
```

`src/ingestion/run_ingestion.py` runs all ingestion stages in order.

---

## Tech stack

| Part | Technology |
|---|---|
| Language | Python 3.13 |
| Framework | LangChain |
| Chat model | `llama3.1:8b` via Ollama |
| Embeddings | `dengcao/Qwen3-Embedding-0.6B:Q8_0` via Ollama |
| Vector store | PostgreSQL + pgvector |
| Content source | Wikipedia, through the MediaWiki Action API |

---

## Quick start

### 1. Prerequisites

- Python 3.13
- PostgreSQL with the `vector` extension
- Ollama running locally, with the chat and embedding models pulled

### 2. Install

```powershell
uv pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and fill in the database password and model
names. Non-secret pipeline settings live in `config/settings.yaml`
(article titles, section tokens, boilerplate rules, answer style).

### 4. Check the setup

```powershell
python main.py
```

This prints the resolved settings (password masked), the PostgreSQL status,
the Ollama models and the number of indexed chunks.

### 5. Ingest

```powershell
# fetch the pages listed in config/settings.yaml
python src/ingestion/run_ingestion.py

# fetch specific pages
python src/ingestion/run_ingestion.py --titles "Arsenal F.C." "History of Arsenal F.C."

# add to the existing collection instead of rebuilding it
python src/ingestion/run_ingestion.py --no-reset
```

### 6. Ask

```powershell
python ask.py "Who did Arsenal lose to in the 2025-26 EFL Cup final?"
python ask.py --check "what was Arsenal's average possession?"
```

`--check` also prints unsupported numbers and names. The exit code is `3` when
a number in the answer is not in the sources.

### 7. Chat or summarize

```powershell
python main.py --chat
python main.py --chat "Who was Arsenal's top scorer?"
python main.py --summarize "2025–26 Arsenal F.C. season"
```

### 8. Evaluate and check data quality

```powershell
python tests/run_quality_checks.py --collection arsenal_documents
python eval/retrieval_eval.py --collection arsenal_documents
python eval/compare_results.py
```

---

## How an answer is built

1. **Route** — if the question names a configured section, return that whole
   section in document order.
2. **Condense** — a short follow-up is rewritten into a standalone question
   using the most recent turn.
3. **Search** — embed the question and fetch the nearest chunks. In
   `fallback` mode the original search runs first; a rewrite is used only when
   the first search is weak.
4. **Merge** — combine results with Reciprocal Rank Fusion, keeping the best
   raw distance per chunk.
5. **Gate** — keep the chunks close to the best match, up to `FINAL_TOP_N`.
6. **Filter** — if the question names a season, keep only passages from that
   season.
7. **Prompt** — send numbered passages (`[1]`, `[2]`, …) with the answer rules
   to the chat model.
8. **Cite** — build the source list in code from title, section, URL and fetch
   time. Group chunks from the same section onto one line.

---

## Configuration

Settings come from `.env` (secrets and numbers) and `config/settings.yaml`
(page list and non-secret pipeline rules).

| Area | Main keys |
|---|---|
| Wikipedia | `WIKIPEDIA_LANG`, `WIKIPEDIA_API_URL`, `WIKIPEDIA_USER_AGENT`, `WIKIPEDIA_TIMEOUT`, `WIKIPEDIA_MAX_RETRIES` |
| Chunking | `CHUNK_SIZE=900`, `CHUNK_OVERLAP=120`, `CHUNK_MIN_SIZE=150`, `CHUNK_MAX_WARN=1200`, `CHUNK_HEADER_ENABLED=true` |
| Section cleanup | `MIN_SECTION_CHARS=250`, `MIN_SECTION_KEEP=60`, `MIN_SECTION_MAX_PASSES=3`, `BOILERPLATE_MAX_CHARS` |
| Cache | `USE_FETCH_CACHE=true`, `REFRESH_FETCH_CACHE=false`, `FETCH_CACHE_DIR=data/cache` |
| Database | `POSTGRES_*`, optional `DATABASE_URL`, `PGVECTOR_COLLECTION=arsenal_documents` |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `OLLAMA_NUM_CTX=8192`, `ANSWER_TEMPERATURE=0.1` |
| Retrieval | `RETRIEVE_K`, `FINAL_TOP_N`, `RETRIEVAL_GATE=relative`, `RELATIVE_MARGIN`, `SCORE_THRESHOLD` |
| Rewriting | `QUERY_REWRITE_MODE=fallback`, `QUERY_REWRITE_WEAK_SCORE=0.65`, `QUERY_REWRITE_COUNT=3` |
| Sections | `SECTION_ROUTING_ENABLED=true`, `MAX_SECTION_CHUNKS=16` |
| Follow-ups | `QUERY_CONDENSE_ENABLED=true`, `FOLLOWUP_MAX_WORDS=6`, `CONDENSE_HISTORY_TURNS=1`, `CHAT_HISTORY_TURNS=4` |
| Evaluation | `EVAL_TOP_K=8`, `EVAL_HIT_AT=1,3,5,8`, `EVAL_COLLECTION_PREFIX`, `EVAL_RESULTS_DIR` |

`config/settings.yaml` holds:

- `pages` — the 12 Wikipedia articles to ingest;
- `current_season: "2025–26"`;
- `answer_style: long` (`short`, `normal` or `long`);
- `section_tokens` — section names questions may target;
- `followup_pronouns` — words that mark a possible follow-up;
- `junk_sections`, `boilerplate_patterns`, `split_separators`,
  `page_type_rules`, `season_normalise_pattern`.

---

## Current state

| Item | Value |
|---|---:|
| Production collection | `arsenal_documents` |
| Chunk setting | 900 / 120 |
| Chunks | 224 |
| Median body | 608 chars |
| Quality checks | 13 / 13 pass |
| hit@1 / hit@3 / hit@5 / hit@8 | 0.60 / 0.85 / 0.95 / 0.95 |
| MRR | 0.733 |
| Section router | 8 / 8 expected paths |
| Season questions | 4 / 4 correct |

The main eval has 20 scored questions and 2 negative (not scored) questions.
The eval set is small, so treat differences under 0.10 as noise.

---

## Project layout

```
Arsenal_Weki/
├── README.md                    project summary
├── explanation.md               file-by-file and function-by-function reference
├── main.py                      smoke check, chat, summarize
├── ask.py                       ask one question, optional grounding check
├── test_wiki_extract.py         standalone Wikipedia extract inspector
├── config/
│   └── settings.yaml            pages and non-secret pipeline config
├── src/
│   ├── config.py                all settings, loaded and validated
│   ├── text_utils.py            season and page-type helpers
│   ├── ingestion/
│   │   ├── wikipedia_loader.py
│   │   ├── fetch_cache.py
│   │   ├── document_cleaner.py
│   │   ├── section_assembler.py
│   │   ├── text_splitter.py
│   │   ├── index_builder.py
│   │   └── run_ingestion.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── section_router.py
│   └── generation/
│       ├── answer_chain.py
│       └── grounding.py
├── eval/                        evaluation scripts, questions and results
├── data/cache/                  cached Wikipedia extracts
└── tests/                       offline test suite and quality checks
```

---

## Documentation

- **`explanation.md`** — what every file and function does, what goes in and
  what comes out, key features, settings reference, phase history, measured
  results and known limits.

---

## Known limits

- The model can still add a date that is not in the passages. The grounding
  check reports it but does not correct it.
- Multi-part questions can miss when the answer needs distant sections.
- The relative gate always keeps at least one chunk, so negative questions
  still retrieve passages. `ask.py` labels them as “not used for the answer”.
- A section is capped at 16 chunks (`MAX_SECTION_CHUNKS`).
- Season handling steers the model; it does not guarantee obedience.
- Settings are read once at import time. Restart after editing `.env` or
  `config/settings.yaml`.
- Some Windows terminals show the en dash (`–`) as a replacement glyph even
  though the stored text is correct.
