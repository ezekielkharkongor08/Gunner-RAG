# Phase 2 change report

> **Later work supersedes three decisions here.** See
> [`RELIABILITY_CHANGES.md`](RELIABILITY_CHANGES.md): the absolute `0.70` gate
> became a relative gate (P1), rewriting defaults to `fallback` instead of off
> (P7), and month questions are routed to their section (P2). Everything else
> in this report still applies.

This file lists everything that was added, changed or updated in Task 0
(0.1–0.4), Phase 2 (2.1–2.8) and the Final Readiness pass (R1–R5).

Simple English summary: retrieval and answering are finished, measured and
switched to the setting that scored best. The index is rebuilt at 900/120 with
224 chunks. All tests and quality checks pass.

---

## 1. Task 0 results

### 0.2 Re-verification (the hard stop)

| Check | Expected | Got | Match |
|---|---|---:|---|
| `pytest tests/` | 84 passed, 6 skipped (Phase 1 baseline) | **99 passed, 6 skipped** — the same 84 plus the 15 config tests added in 0.1 | yes |
| Quality checks, `arsenal_eval_900_120` | 14 of 14 | **14 of 14** | yes |
| Chunk count | 222 | **222** | yes |
| hit@1 / hit@3 / hit@5 / hit@8 | 0.60 / 0.85 / 0.95 / 0.95 | **0.60 / 0.85 / 0.95 / 0.95** | yes |
| MRR | 0.733 | **0.733** | yes |

`__pycache__` was deleted before the run, so the numbers came from a clean
import. The test count at the very end is higher only because 0.1 added 15
settings tests and later steps added more; the three Phase 1 numbers (14/14,
222, hit rates) matched exactly.

### 0.1 Settings audit

- `CacheSettings` now has `refresh`, so `wikipedia_loader.py` no longer crashes
  on `settings.cache.refresh`.
- The code reads the names that `.env` actually uses:
  `USE_FETCH_CACHE`, `FETCH_CACHE_DIR`, `REFRESH_FETCH_CACHE`.
- New file `tests/test_config.py` (15 tests) checks three rules:
  1. every env key the code reads is documented in `.env.example`;
  2. every key in `.env.example` is read by the code;
  3. the chunk and cache settings exist.
- Missing documentation added: `DATABASE_URL`, `CLEAN_MIN_LENGTH`, and the
  aliases `OLLAMA_MODEL` / `WIKIPEDIA_LANGUAGE`.

This audit found a real drift: `CLEAN_MIN_LENGTH` was read by the cleaner but
was not documented anywhere.

### 0.3 Currency exception

Problem: `^The following ` sections under 300 chars were dropped, and the
transfer spend/income totals live in exactly those sections.

Fix: a boilerplate rule may now carry `keep_if_pattern`. When it matches
anywhere in the body, the section is kept. The rule in `config/settings.yaml`:

```yaml
- pattern: '^The following '
  max_chars: 300
  keep_if_pattern: '[£€$]\s?\d'
```

Same check mirrored in `tests/run_quality_checks.py` (so the checker does not
flag a section that was kept on purpose).

Effect on the cached text (no network):

| Item | Before | After |
|---|---:|---:|
| Sections kept | 74 | 76 |
| Sections dropped | 20 | 16 |
| Merges | 3 | 5 |
| Chunks | 222 | **224** |

The totals are now in the index: `Total expenditure: £248.2 million`,
`Total income: £14.5 million`, `Total expenditure: £0.0 million`. Sections
without money (`New contracts`, `Released`, `Appearances`) are still dropped.

### 0.4 `INGESTION_CHANGES.md`

Phase 1 text corrected: the cache key names, the currency rule, the real
counts, the test count, and the fact that header stripping is a Phase 2
feature (it was wrongly described as already done).

---

## 2. Phase 2 (2.1–2.8)

### 2.1 Rebuild `arsenal_documents` (900/120)

- Rebuilt with `reset=True` from the fetch cache: **224 chunks written**.
- Quality checks on production: **13 of 13 PASS** (13 because the re-index
  row only runs on an `arsenal_eval_*` collection).
- All four old `arsenal_eval_*` collections were kept (no deletions).
- A new collection `arsenal_eval_900_120_v2` was built from the new corpus so
  the 14-check re-index test can run. Result: 224 chunks, 14 of 14 PASS,
  re-index 224 → 224.
- New baseline for the new corpus: 224 chunks, hit@1 0.60, hit@3 0.85,
  hit@5 0.95, hit@8 0.95, MRR 0.733.

### 2.2 Query instruction (Qwen3)

The instruction was already applied to queries only. It is now covered by
tests (`tests/test_index_builder.py`): documents get no instruction, queries
get `Instruct: <text>\nQuery:<question>`, and the on/off switch works.

### 2.3 Query rewriting + 2.3b score logging

- `QUERY_REWRITE_ENABLED` switch; when off, only the normalised original is
  searched (no LLM call).
- `normalize_season_text()` runs on the original and on every variant
  (`2025-2026` / `2025/26` → `2025–26`).
- The rewrite prompt asks for short Wikipedia-style queries and, for broad
  questions, covers league, FA Cup, EFL Cup and Champions League.
- Rewriting temperature is `QUERY_REWRITE_TEMPERATURE` (default 0.0), not the
  answer temperature.
- Multi-query results are merged with **Reciprocal Rank Fusion** (RRF_K = 60)
  in `fuse_rankings()`. Raw cosine distances from different wordings are not
  comparable; RRF uses ranks. The returned score is still the best raw
  distance, so `SCORE_THRESHOLD` keeps working.
- Every search logs `k`, the distance range and the threshold, e.g.
  `Search '…': k=15, distance 0.2385..0.3765 (threshold 0.70)`.

### 2.4 Context block and sources

- Context is now `[1]\n<text>`, `[2]\n<text>`, … with no repeated title /
  URL line per chunk (it wasted tokens and the model copied it).
- The stored `Title\nSection: path\n\n` header is stripped before the text is
  sent to the model (`CONTEXT_STRIP_HEADER=true`); `header_chars` in metadata
  makes the cut exact.
- `format_sources()` builds the numbered Sources list in code:
  `[1] Title | section: … | source: URL | fetched: timestamp`.
- `RetrievedChunk` gained `header_chars` and `fetched_at`, plus `body_text()`.

### 2.5 `num_ctx` and token logging

- `ChatOllama(..., num_ctx=OLLAMA_NUM_CTX)` for answers and summaries
  (default **8192**).
- Answer temperature is `ANSWER_TEMPERATURE` (default **0.1**) and falls back
  to `LLM_TEMPERATURE` if the new key is unset.
- `_log_token_usage()` logs `prompt_eval_count` and warns above 90 % of
  `num_ctx`.
- Measured on the transfer question: **1234 prompt tokens = 15 % of 8192**.

### 2.6 Answer prompt

The system prompt now has eight rules: facts only from the passages, copy
names/scores/dates exactly, no extra details, say which part is missing, say
when the indexed pages do not contain the answer, cite only `[n]`, keep it
short, never mention being an AI. The old `[Title > Section](URL)` citation
style is gone (URLs are built in code).

### 2.7 Experiment grid

Run on `arsenal_documents` (224 chunks, 20 positive questions, 2 negative not
scored). JSON files are in `eval/results/grid/`.

| Query instruction | Query rewrite | hit@1 | hit@3 | hit@5 | hit@8 | MRR | MRR (kw) |
|---|---|---:|---:|---:|---:|---:|---:|
| **off** | **off** | **0.60** | **0.85** | **0.95** | **0.95** | **0.733** | **0.767** |
| off | on | 0.20 | 0.55 | 0.75 | 0.85 | 0.440 | 0.492 |
| on | off | 0.40 | 0.80 | 0.85 | 0.95 | 0.609 | 0.642 |
| on | on | 0.40 | 0.70 | 0.80 | 0.90 | 0.576 | 0.620 |

**Decision: instruction off, rewriting off** (the current defaults). Both
features stay implemented and switchable, but they are not used by default.

Why rewriting lost: 20 of 20 positive questions ranked the right chunk at the
same rank or worse. The rewrites were good, but three near-identical variants
outvote the original query in any fusion scheme, and the original is the one
that finds the specific chunk.

### 2.8 New tests

| File | Tests now | What is covered |
|---|---:|---|
| `tests/test_config.py` | 15 | env/example sync, chunk + cache settings |
| `tests/test_answer_chain.py` | 11 | numbered context, header strip, sources, num_ctx, token warning, prompt rules |
| `tests/test_retrieval.py` | 12 | rewrite switch, season normalisation, RRF fusion |
| `tests/test_index_builder.py` | 11 | instruction on queries only, prefix behaviour |
| `tests/test_section_assembler.py` | 12 | currency sections kept, non-money stubs dropped |

---

## 3. Demo questions (final settings)

Command: `python ask.py "<question>"`

### D1 – single fact

> **Q:** Who did Arsenal lose to in the 2025-26 EFL Cup final?
>
> **A:** In the 2025-26 EFL Cup final, Arsenal lost to Manchester City. [1]
>
> **Sources:**
> [1] 2025–26 Arsenal F.C. season | section: EFL Cup | source: https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season | fetched: 2026-10-19T20:17:31Z
> [2] 2025–26 Arsenal F.C. season | section: March | (same article)

### D2 – multi-part question (the known hard one)

> **Q:** what is arsenal season 2025-2026 and what did they achieved
>
> **A:** Arsenal's 2025-2026 season was their 34th season in the Premier League
> and 109th season in the top flight overall. They achieved their 14th English
> league title, becoming the first team in Premier League history to complete a
> campaign without conceding a penalty or receiving a red card [3].
>
> **Sources:** [1] Introduction, [2] League phase, [3] May — all from
> *2025–26 Arsenal F.C. season*.

### D3 – transfer spend (the 0.3 fix end to end)

> **Q:** How much did Arsenal spend in the 2025-26 summer transfer window?
>
> **A:** Arsenal spent **£248.2 million** in the 2025-26 summer transfer
> window, excluding potential add-ons, bonuses, undisclosed figures and future
> transfers [2].
>
> **Sources:**
> [1] 2025–26 Arsenal F.C. season | section: First team transfers (summer transfer window)
> [2] 2025–26 Arsenal F.C. season | section: Transfers out

### D4 – negative question 1

> **Q:** What was Arsenal's average possession in the 2025-26 Premier League season?
>
> **A:** The indexed pages do not contain the average possession of Arsenal in
> the 2025-26 Premier League season.
>
> **Exit code:** 0. Eight chunks passed the 0.70 threshold but the model did
> not use them; `ask.py` labels them **"Retrieved chunks (not used for the
> answer)"**.

### D5 – negative question 2

> **Q:** Who won the 2022 FIFA World Cup?
>
> **A:** The indexed pages do not contain the answer to who won the 2022 FIFA
> World Cup.
>
> **Exit code:** 0. Same honest "not used" label for the retrieved chunks.

### Negative-question verdict

Both negative questions were refused with the exact wording asked for in 2.6,
and no player, country or number was invented. The only weakness is that
retrieval still returns 8 chunks above the threshold; they are labelled, not
hidden.

---

## 4. Tests and quality checks (final)

| Check | Result |
|---|---|
| `pytest tests/` (clean, no `__pycache__`) | **121 passed, 6 skipped** |
| Skips | 6 live-Wikipedia integration tests (`ARSENAL_WEKI_INTEGRATION=1` to run) |
| Quality `arsenal_documents` | **13 of 13 PASS**, 224 chunks, shortest body 166, median body 608 |
| Quality `arsenal_eval_900_120_v2 --reindex-check` | **14 of 14 PASS**, re-index 224 → 224 |
| Retrieval eval `arsenal_documents` | 0.60 / 0.85 / 0.95 / 0.95, MRR 0.733 |
| Empty `.py` files in `src`, `tests`, `eval` | **0** |
| Prompt size on a real question | 1234 tokens (15 % of 8192) |

---

## 5. New settings

### `.env` / `.env.example`

| Key | Default | Meaning |
|---|---|---|
| `USE_FETCH_CACHE` | `true` | read cached Wikipedia extracts (the name the code now uses) |
| `FETCH_CACHE_DIR` | `data/cache` | where the cache lives |
| `REFRESH_FETCH_CACHE` | `false` | force new network fetches (new `cache.refresh` field) |
| `OLLAMA_NUM_CTX` | `8192` | context window passed to `ChatOllama` |
| `ANSWER_TEMPERATURE` | `0.1` | answer + summary temperature (falls back to `LLM_TEMPERATURE`) |
| `QUERY_REWRITE_ENABLED` | `false` | multi-query rewriting (grid says off) |
| `CONTEXT_STRIP_HEADER` | `true` | strip `Title\nSection:` from chunks sent to the LLM |
| `RETRIEVAL_FALLBACK_ON_EMPTY` | `false` | return no context instead of the best weak chunk |

Also changed: `QUERY_REWRITE_TEMPERATURE` code default `0.2` → `0.0`.
Documented (not new): `DATABASE_URL`, `CLEAN_MIN_LENGTH`, alias names.

### Config file

| Item | Value | Meaning |
|---|---|---|
| `boilerplate_patterns[].keep_if_pattern` | `'[£€$]\s?\d'` on `^The following ` | keep short sections that hold a money total |
| `RRF_K` (constant in `retriever.py`) | `60` | Reciprocal Rank Fusion constant |

---

## 6. Files changed or added

### Code

| File | Change |
|---|---|
| `src/config.py` | cache `refresh`; `num_ctx`; `answer_temperature`; `query_rewrite_enabled`; `context_strip_header`; `fallback_on_empty`; redacted dump |
| `src/ingestion/section_assembler.py` | honours `keep_if_pattern` in `_is_boilerplate` |
| `src/retrieval/retriever.py` | `header_chars` / `fetched_at` / `body_text()`; rewrite switch + Wikipedia prompt + season normalisation + 0.0 temperature; `fuse_rankings()` (RRF); threshold behaviour; score logging |
| `src/generation/answer_chain.py` | numbered context, header strip, `format_sources()`, new prompt, `num_ctx`, answer temperature, token logging, summary prompt |
| `ask.py` | **new** one-shot terminal entry point with numbered sources and honest "not used" label |

### Eval and checks

| File | Change |
|---|---|
| `eval/retrieval_eval.py` | `--query-rewrite`, `--out-dir`, `--with-llm`; season normalisation; instruction/rewrite recorded in JSON; advisory fact check |
| `tests/run_quality_checks.py` | mirrors the `keep_if_pattern` exception |
| `eval/results/arsenal_documents.json` | new production baseline (224 chunks) |
| `eval/results/arsenal_eval_900_120_v2.json` | **new** eval baseline for the new corpus |
| `eval/results/grid/*.json` | **new** four grid rows |

### Tests

| File | Change |
|---|---|
| `tests/test_config.py` | **new**, 15 tests |
| `tests/test_answer_chain.py` | **new**, 11 tests |
| `tests/test_retrieval.py` | rewrite switch, season normalisation, RRF |
| `tests/test_index_builder.py` | instruction on queries only |
| `tests/test_section_assembler.py` | currency keep / non-money drop |

### Data and docs

| File | Change |
|---|---|
| `arsenal_documents` (PostgreSQL) | rebuilt, 224 chunks, 900/120 |
| `arsenal_eval_900_120_v2` (PostgreSQL) | **new** collection, 224 chunks |
| old `arsenal_eval_*` collections | untouched (600_100 = 311, 900_120 = 222, 1200_150 = 179, 1500_220 = 146) |
| `config/settings.yaml` | currency `keep_if_pattern` |
| `.env.example` | 20 lines: new Phase 2 keys + documented missing keys |
| `.env` | new keys only (not committed) |
| `INGESTION_CHANGES.md` | Phase 1 accuracy fixes + 0.3 section |
| `.gitignore` | removed a stray `.env.example` line (it must stay tracked) |
| `test_report.txt` | regenerated (production, 13 checks) |

---

## 7. Problems not fixed

1. **Q1 (multi-part) still misses.** Rewriting did not help; it made every
   question equal or worse. Possible next step: weight the original query much
   higher, or only rewrite follow-up questions with pronouns.
2. **Rewriting is a net loss.** Implemented, tested and switched off. A future
   version could rewrite only when the question has no explicit entity.
3. **D3 cites "Transfers out" for the spend total.** The short `Transfers in`
   section (214 chars, under `MIN_SECTION_CHARS=250`) is merged into its
   sibling, so the £248.2m text sits in a chunk labelled `Transfers out`. The
   fact is right; the section label is imprecise. Fix: never merge a section
   that was kept by `keep_if_pattern`.
4. **Negatives still retrieve 8 chunks.** `SCORE_THRESHOLD=0.7` is lenient
   (distance, so 0.7 ≈ 0.3 similarity). They are labelled "not used", not
   filtered. A relevance gate or a stricter threshold is future work.
5. **Fetch-cache junk pointers.** `data/cache/index.json` still has test
   entries `X`, `Retry`, `A` → `1_2.json`. Not deleted, because deleting data
   needs approval.
6. **D2's first sentence has no `[n]` marker.** The prompt asks for a citation
   after each fact; the model cites only the second sentence.
7. **Windows console mangles `–` when piped.** The stored text is correct
   (U+2013 confirmed). `ask.py` forces UTF-8 stdout; some terminals still show
   a replacement glyph.
8. **Ollama server default `num_ctx` was never probed.** We always send 8192,
   so the app does not depend on the server default.
9. **The eval set is small** (20 scored questions). One question is 0.05 of
   hit@3; treat differences under 0.10 as noise.
10. **Old eval collections hold the pre-0.3 corpus** (222 chunks). They were
    kept on purpose; `arsenal_eval_900_120_v2` is the new reference.

---

## 8. Small changes to the agreed plan

1. Multi-query merging uses **RRF** instead of "keep the best distance". Best
   distance per chunk is still returned for the threshold.
2. The zero-result fallback is controlled by a new setting
   `RETRIEVAL_FALLBACK_ON_EMPTY` (default `false`) instead of removing it.
3. Header stripping is controlled by `CONTEXT_STRIP_HEADER` (default `true`)
   instead of being unconditional.
4. `QUERY_REWRITE_ENABLED` ends up **false**, because the grid says so.
5. A new collection `arsenal_eval_900_120_v2` was created instead of
   rebuilding `arsenal_eval_900_120` (no collection was deleted).
6. `ask.py` was added rather than extending `main.py --chat`.
7. `.gitignore` lost a stray `.env.example` line that appeared mid-build.

Note: `src/config.py`, `src/ingestion/section_assembler.py` and `.gitignore`
were edited by another writer while this build was running (cache keys, the
`max_chars` fallback and that `.gitignore` line). Those edits were kept and
built on; nothing was reverted except the `.gitignore` line.

---

## 9. How to run

```powershell
# rebuild the index from the fetch cache (no network)
.venv\Scripts\python.exe -m src.ingestion.run_ingestion

# ask one question
.venv\Scripts\python.exe ask.py "Who did Arsenal lose to in the 2025-26 EFL Cup final?"

# interactive
.venv\Scripts\python.exe main.py --chat

# checks
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe tests/run_quality_checks.py --collection arsenal_documents
.venv\Scripts\python.exe eval/retrieval_eval.py --collection arsenal_documents

# one grid row (rewriting on, for comparison)
.venv\Scripts\python.exe eval/retrieval_eval.py --collection arsenal_documents `
    --query-instruction off --query-rewrite on --out-dir eval/results/grid
```
