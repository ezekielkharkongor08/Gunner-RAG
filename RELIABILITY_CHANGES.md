# Reliability changes: section routing, a usable gate, follow-ups, honest sources

This file documents the work done after `PHASE2_CHANGES.md`, in response to four
real chat failures:

1. "describe the things that happen in the month of November" returned **nothing**.
2. "give more details about november" returned **nothing**.
3. "give a brief review of March" produced a **rambling answer that mixed months**.
4. Negative questions (average possession, 2022 World Cup) were answered with a
   refusal but still listed sources as if they supported the answer.

Everything below was measured, not assumed. No re-ingestion was needed: all
metadata was already stored, and no database writes were made in P1–P7.

---

## 1. What was actually wrong

| Symptom | Measurement | Cause |
|---|---|---|
| November returns nothing | best chunk `Review > November` at distance **0.712**, `SCORE_THRESHOLD=0.70` | The absolute gate rejected the correct chunk by **0.012**. The gate was the bug, not the index. |
| November, second phrasing | **0.708**, same section | Same gate. Rewriting made it worse: variants drifted to "November weather" (0.787 `Background`), "November holidays" (0.715 `Crest`), "November (month)" (0.754 `October`). |
| March rambles / mixes months | March chunks do contain "Amex", "Hincapié" | Only *questions* were stored in history (`main.py:131`), and replayed as bare `HumanMessage`s; the model saw questions but never its own answers, and retrieval could not resolve "they". |
| April/May-style questions | March 0.681 and 0.632, both under 0.70 | Same gate: correct answers were one bad question away from returning nothing. |
| Negatives list sources | unanswerable "average possession" = **0.284**, "2022 FIFA World Cup" = **0.609** | Both *passed* 0.70 while the correct November chunk *failed*. An absolute threshold cannot separate them; and sources were printed as "Sources:" even on a refusal. |

**Conclusion:** the 0.70 absolute gate was unusable in both directions. Month
questions need deterministic section routing, not a better paraphrase.

---

## 2. Decisions taken

The five open questions were answered with the recommended defaults:

| Decision | Chosen |
|---|---|
| Rewrite mode | `fallback` (rewrite only when the first search is weak) |
| Keep an `always` grid row | Yes, it is still measured |
| Whole month section or cap | Whole section, capped at `MAX_SECTION_CHUNKS=16` (largest section is 14) |
| Separate eval file for new questions | Yes, `eval/questions_sections.json` |
| Hide or collapse unused sources | Collapse under "Retrieved chunks (not used for the answer)" |

One small setting was added beyond the plan: `CONDENSE_HISTORY_TURNS=1`
(see P3). It is listed with the other settings in §6.

---

## 3. What changed, step by step

### P2 — Section routing (`src/retrieval/section_router.py`, new)

- `normalise` (line 52), `token_in_question` (57), `segment_matches` (71),
  `match_section` (86), `list_sections` (112), `fetch_section` (135),
  `route` (169).
- `retriever.retrieve` (line 541) calls `route_section` first via
  `_section_chunks` (527). A match returns the whole section in document order
  and **skips vector search and the gate**.
- Matching is case/punctuation insensitive, whole-word for single tokens, and
  requires a **unique** matching `section_path`. Tokens live in
  `config/settings.yaml` (`section_tokens`), not in code.
- Real paths corrected during testing: the kit section is
  `Kits > Kit information`, so parent segments are matched too.

### P1 — Configurable gate (`retriever.apply_gate`, line 410)

- `RETRIEVAL_GATE=relative|absolute|none`, default `relative`.
- `relative`: keep chunks within `RELATIVE_MARGIN=0.10` of the best distance,
  capped at `FINAL_TOP_N`. The best chunk is always kept, so a weak answer is
  answered or refused by the model, not silently dropped.
- `absolute` keeps the old `SCORE_THRESHOLD` behaviour (including
  `RETRIEVAL_FALLBACK_ON_EMPTY`); `none` keeps the head of the ranking.
- The gate **filters but does not re-rank** (`apply_gate` docstring), so RRF
  order survives and the cap keeps the fusion's head.

### P7 — Rewriting modes (`retriever.search_for_question`, line 465)

- `QUERY_REWRITE_MODE=off|always|fallback`, default `fallback`.
- `fallback`: search the original first; if the best distance is above
  `QUERY_REWRITE_WEAK_SCORE=0.65`, rewrite and **append** the new chunks.
  The original ranking is never disturbed, so rewriting can only add recall.
- The rewrite prompt now lists the indexed pages (from `settings.pages`) and
  requires keeping the subject, season, month and competition. This is why the
  variants no longer drift to "November weather".
- `QUERY_REWRITE_ENABLED` remains as a legacy alias: it is only consulted when
  `QUERY_REWRITE_MODE` is absent.

### P3 — Real conversation memory

- `main.py:136` stores `(question, answer)` pairs, capped at
  `CHAT_HISTORY_TURNS=4` (was 10 bare strings).
- `answer_chain.build_history_messages` (line 160) turns pairs into alternating
  `HumanMessage`/`AIMessage`; bare strings still work.
- `SYSTEM_PROMPT` rule 9: answer only the latest question.
- `retriever.is_followup` (128) and `condense_question` (149): a short question
  or one containing a `followup_pronouns` word is condensed into a standalone
  question using the **most recent** turn, then section routing runs again on
  the condensed text. This is what turns "who did they beat at home?" into
  "Who did Arsenal beat at home in March?" and routes it to March.

### P4 — Grouped sources (`answer_chain.group_sources`, line 74)

- `format_sources` (96) prints one line per `(title, section, url)` with all
  passage numbers: `[1][2][3] ... section: March ...`.
- `main.py` and `ask.py` all use it; the source list no longer repeats the same
  URL up to 14 times.

### P5 — Eval for sections and the gate (`eval/retrieval_eval.py`)

- `eval/questions_sections.json`: 8 questions that name a section.
- `--questions-file`, `--gate-report`, expected-section scoring in
  `question_rank` (line 117), and result filenames that include the questions
  file so rows do not overwrite each other.

### P6 — Answer rules and grounding checks

- `SYSTEM_PROMPT` rule 2 now forbids inventing dates; rule 10 forbids mixing
  home and away ("a named opponent's ground is an away match").
- New `src/generation/grounding.py`: `numbers_in` (41), `capitalised_terms`
  (49), `unsupported_numbers` (72), `unsupported_terms` (82),
  `check_answer` (90).
- `ask.py --check` prints unsupported numbers/names and exits **3** when a
  number in the answer is not in the sources. The eval's `--with-llm` check now
  imports the same helpers.

---

## 4. Results

### 4.1 Grid, main questions (20 scored, 2 negative, collection `arsenal_eval_900_120_v2`, 224 chunks, k=8)

| Query instruction | Rewrite | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|---|
| off | off | 0.60 | 0.85 | 0.95 | 0.95 | 0.733 |
| **off** | **fallback** | **0.60** | **0.85** | **0.95** | **0.95** | **0.733** |
| on | off | 0.40 | 0.80 | 0.85 | 0.95 | 0.609 |
| off | always | 0.20 | 0.55 | 0.75 | 0.85 | 0.440 |
| on | always | 0.40 | 0.70 | 0.80 | 0.90 | 0.576 |

The fallback row equals `off` because **no** main question is weak: the worst
best-distance is 0.412, well under 0.65, so the rewrite never fires. Unconditional
rewriting (`always`) remains clearly worse and is not used.

### 4.2 Grid, section questions (8 questions, same collection, k=8)

| Rewrite | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|
| off | 0.875 | 0.875 | 1.00 | 1.00 | 0.906 |
| fallback | 0.875 | 0.875 | 1.00 | 1.00 | 0.906 |

The ranks are identical, but the **distances** are not: rewriting sharpens the
weak section questions, which means the relative gate keeps a tighter, more
relevant set. From `--gate-report`:

| Question | best (off) | kept | best (fallback) | kept |
|---|---|---|---|---|
| November, long | 0.712 | 8 | **0.247** | 2 |
| November, short | 0.708 | 8 | **0.342** | 7 |
| March | 0.632 | 3 | 0.632 | 3 |
| October | 0.685 | 8 | **0.236** | 2 |
| Transfer window | 0.384 | 1 | 0.384 | 1 |
| Kits | 0.254 | 4 | 0.254 | 4 |
| December | 0.751 | 8 | **0.246** | 3 |
| January | 0.632 | 7 | 0.632 | 7 |

And the router itself: **8/8** section questions resolve to the expected
`section_path` (after fixing the kit path).

### 4.3 The 5 demos

Run with the final settings (`ask.py`/`answer_question`). Sources are grouped;
`fetched` timestamps come from ingestion.

**Demo 1 — the question that used to return nothing**

> Q: describe the things that happen in the month of November
>
> Here are the events that occurred in the month of November:
>
> * Arsenal won 0-2 against Burnley on 1 November [1].
> * Arsenal won 3-0 against Slavia Prague on 4 November [2].
> * Arsenal drew 2-2 against Sunderland on 7 November [3].
> * Arsenal drew 2-2 against Sunderland ended their winning run and reduced their lead on 2nd-placed Manchester City to four points [4].
> * Arsenal won 4-1 against Tottenham Hotspur on 23 November [6].
> * Arsenal won 3-1 against Bayern Munich on 26 November [7].
> * Arsenal drew 1-1 against Chelsea on 30 November [9].
>
> Sources:
> `[1][2][3][4][5][6][7][8][9] 2025–26 Arsenal F.C. season | section: November | source: https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season | fetched: 2026-10-19T20:17:31.781486+00:00`
>
> Grounding: unsupported numbers = **['7']** (see §8), names none.

**Demo 2 — the March review**

> Q: give a brief review of March
>
> In March, Arsenal played several matches. They beat Chelsea 2-1 at home [1], drew 1-1 against Brighton at Brighton's ground [3], beat Mansfield Town 2-1 in the FA Cup 5th Round [4], drew 1-1 against Bayer Leverkusen in the Champions League round of 16 first leg in Germany [5], beat Everton 2-0 at home [7], beat Bayer Leverkusen 3-1 in the Champions League round of 16 second leg at home [9], and lost 2-0 to Manchester City in the EFL Cup final at Wembley Stadium [12].
>
> Sources:
> `[1][2][3][4][5][6][7][8][9][10][11][12][13][14] 2025–26 Arsenal F.C. season | section: March | source: https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season | fetched: 2026-10-19T20:17:31.781486+00:00`
>
> Grounding: unsupported numbers none, names none. Brighton is correctly away.

**Demo 3 — follow-up after November *and* March**

> Q: who did they beat at home?
>
> Arsenal beat Chelsea 2-1 at home [1], and they beat Everton 2-0 at home [7].
>
> Sources:
> `[1][2][3][4][5][6][7][8][9][10][11][12][13][14] 2025–26 Arsenal F.C. season | section: March | ...`
>
> Grounding: unsupported numbers none, names none.

The condensed question was "Who did Arsenal beat at home in March?" and the
section route went to **March**, not November. With two history turns the first
version merged both months; `CONDENSE_HISTORY_TURNS=1` fixed it.

**Demo 4 — negative (not in the pages)**

> Q: what was Arsenal average possession in the 2025-26 Premier League?
>
> The indexed pages do not contain the information about Arsenal's average possession in the 2025-26 Premier League.
>
> Sources: `[1] Introduction`, `[2] Background`, `[3][4] May` (same article).
>
> Grounding: none unsupported.

**Demo 5 — negative (outside the corpus)**

> Q: who won the 2022 FIFA World Cup?
>
> The indexed pages do not contain the answer to who won the 2022 FIFA World Cup.
>
> Sources: `[1] International`, `[2] Introduction`, `[3] November`, `[4] Quarter-final`, `[5] 1996–2018: Arsène Wenger revolution`, `[6] May`, `[7] Semi-final`, `[8] League phase`.
>
> Grounding: unsupported numbers **['2022']** — the year comes from the
> question, not the passages, so this flag is expected and harmless.

With `ask.py`, both negatives print the collapsed label
"Retrieved chunks (not used for the answer):" instead of "Sources:".

The full plain-text evidence is in `eval/results/final_demos.txt`.

---

## 5. Tests and checks

| Check | Result |
|---|---|
| `pytest tests/` | **174 passed, 6 skipped** (the 6 are integration tests) |
| `tests/run_quality_checks.py` | **13/13 PASS**, 224 chunks, median body 608 |
| Re-index `arsenal_eval_900_120_v2` without reset | **224 → 224** rows |
| Router on section questions | **8/8** matched the expected path |
| Grid rows | 5 main + 2 section rows, all reproducible from the commands below |

New tests: `tests/test_section_router.py` (13), gate and rewrite-mode tests and
follow-up condensation tests in `tests/test_retrieval.py`, history and source
grouping tests in `tests/test_answer_chain.py`, `tests/test_grounding.py` (13,
including `ask.py --check`).

---

## 6. New settings

| .env key | Default | Meaning |
|---|---|---|
| `SECTION_ROUTING_ENABLED` | `true` | Route questions that name a section |
| `MAX_SECTION_CHUNKS` | `16` | Cap on one section (largest is 14) |
| `RETRIEVAL_GATE` | `relative` | `relative`, `absolute` or `none` |
| `RELATIVE_MARGIN` | `0.10` | Keep chunks within this distance of the best |
| `QUERY_REWRITE_MODE` | `fallback` | `off`, `always` or `fallback` |
| `QUERY_REWRITE_WEAK_SCORE` | `0.65` | Fallback rewrites above this distance |
| `QUERY_CONDENSE_ENABLED` | `true` | Condense follow-ups before searching |
| `FOLLOWUP_MAX_WORDS` | `6` | Short question = possible follow-up |
| `CONDENSE_HISTORY_TURNS` | `1` | Recent pairs the condenser may read |
| `CHAT_HISTORY_TURNS` | `4` | Pairs kept by the chat loop |

`config/settings.yaml` gained `section_tokens` (21 values) and
`followup_pronouns` (17 values). No secret was added, changed or printed.
`QUERY_REWRITE_ENABLED=false` is kept for old configs; it is ignored while
`QUERY_REWRITE_MODE` is set.

Follow-up (season mixing) added one YAML key, `current_season: "2025–26"`
(no `.env` key). See section 11.

---

## 7. Files changed

| File | Change |
|---|---|
| `src/retrieval/section_router.py` | new: token matching, section listing/fetching, routing |
| `src/retrieval/retriever.py` | `search_for_question` 465, `apply_gate` 410, `_section_chunks` 527, `retrieve` 541, `is_followup` 128, `condense_question` 149, `_history_pairs` 96, `_format_history` 113, `_append_new` 391 |
| `src/config.py` | RetrievalSettings fields, `GATE_MODES`, `REWRITE_MODES`, loaders, redacted output, `Settings.section_tokens`, `followup_pronouns`, `chat_history_turns` |
| `src/generation/answer_chain.py` | `group_sources` 74, `format_sources` 96, `build_history_messages` 160, prompt rules 2/9/10, `answer_question` 276 |
| `src/generation/grounding.py` | new: number/name checks shared by ask and eval |
| `main.py` | `chat_loop` 104, history pairs 136–138, grouped sources |
| `ask.py` | `--check`, exit code 3, grouped sources, clearer empty message |
| `eval/retrieval_eval.py` | `--questions-file`, `--gate-report`, `--query-rewrite-mode`, expected-section scoring, per-file result names, shared grounding helpers |
| `eval/questions_sections.json` | new: 8 section questions with expected paths |
| `config/settings.yaml` | `section_tokens`, `followup_pronouns` |
| `.env`, `.env.example` | 10 new keys, documented |
| `tests/` | new router/grounding tests, extended retrieval/answer/config tests |

---

## 8. What could break

- **Wrong section match.** Routing bypasses similarity search. The uniqueness
  guard and the YAML token list limit the risk; removing a token or renaming a
  section disables routing for it (vector search then runs as before).
- **Relative gate accepts a weak best chunk.** It always keeps one chunk; the
  model is responsible for refusing. Use `RETRIEVAL_GATE=absolute` to restore
  the old strict behaviour.
- **Prompt growth.** A section can add up to 16 chunks and history up to 4
  pairs. Token usage is logged; the warning fires above 90% of `OLLAMA_NUM_CTX`.
- **Follow-ups cost one extra LLM call.** Only when history exists and the
  question is short or contains a pronoun; on failure the original question is
  used.
- **Source format changed** from one line per chunk to one line per section
  (`[1][2][3]`). Anything parsing the old shape must adapt.
- **Default rewrite mode changed** from off to fallback. Set
  `QUERY_REWRITE_MODE=off` to get the Phase 2 behaviour exactly.
- **Eval filenames** gain the questions-file stem when `--questions-file` is
  used; old grid files are unchanged.

---

## 9. Unfixed problems

1. **The model still adds a date that is not in the passages.** The November
   demo says "Sunderland on 7 November"; the section gives no date for that
   match, and the grounding check flags `7`. Rule 2 was strengthened, but the
   model does not always obey. The checker reports it; it does not correct it.
2. **Duplicate bullet.** The November answer repeats the Sunderland result
   because two chunks describe the same match. The model merges them badly.
3. **Follow-ups are not in the grid.** They are covered by unit tests and the
   demo, not by a ranked eval: the eval would need history-aware retrieval
   returning `section_path`, which `RetrievedChunk` does not carry.
4. **`--with-llm` was not re-run** after the prompt changes (it is slow: one LLM
   call per question). It now uses the shared grounding helpers, so its result
   would match `ask.py --check`.
5. **Section cap.** A future section longer than 16 chunks would be truncated.
   Today the largest is 14, so nothing is cut.
6. **Old eval collections are untouched** (`arsenal_eval_900_120` still has 222
   chunks). New artifacts use `arsenal_eval_900_120_v2` (224).

---

## 10. Reproduce

```powershell
# tests and data checks
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe tests\run_quality_checks.py
.venv\Scripts\python.exe tests\run_quality_checks.py --collection arsenal_eval_900_120_v2 --reindex-check

# grid rows (main questions)
.venv\Scripts\python.exe eval\retrieval_eval.py --collection arsenal_eval_900_120_v2 --out-dir eval\results\grid --query-rewrite-mode off
.venv\Scripts\python.exe eval\retrieval_eval.py --collection arsenal_eval_900_120_v2 --out-dir eval\results\grid --query-rewrite-mode fallback

# section questions + gate report + router check
.venv\Scripts\python.exe eval\retrieval_eval.py --collection arsenal_eval_900_120_v2 --questions-file eval\questions_sections.json --out-dir eval\results\grid --query-rewrite-mode off --gate-report

# ask with the grounding check
.venv\Scripts\python.exe ask.py --check "who did they beat at home in March?"
```

---

## 11. Follow-up: season mixing (2026-10-20)

### What the user saw

Asked for "a poetic summary of Arsenal's journey to the Champions League
final", the answer mixed two seasons: "finished third in the league phase in
January 2025", "beat Real Madrid 5-1", "lost to PSG in the semi-final" (all
2024–25) followed by "won the 14th title on 19 May 2026" and "lost the final
4-3 on penalties" (2025–26).

### Cause

One retrieved chunk caused it:
`2025–26 Arsenal F.C. season → Review > Background` recaps the 2024–25
Champions League run ("in September 2024", "in January 2025", "5–1 aggregate
triumph over ... Real Madrid", "3–1 aggregate loss to ... Paris
Saint-Germain"). The answer dropped the years and attached those results to
2025–26.

Not a retrieval bug: the question named no season, so returning both the
current campaign and its Background was correct. Not fixable by metadata
either: `seasons` is title-derived (`SEASONS_FROM_TEXT=false`), so that chunk
is tagged `2025–26` even though its content is 2024–25, and the text says bare
years ("September 2024"), not "2024–25".

### Fix (prompt only, no re-index, no DB writes)

1. `config/settings.yaml`: new `current_season: "2025–26"`, loaded into
   `settings.current_season` and shown by `settings.redacted()`.
2. `src/generation/answer_chain.py`: `SYSTEM_PROMPT` is now built from that
   setting and gained rules 11–16:
   - current season is announced;
   - a result dated 2024/2025 stays in that season even on the 2025–26 page,
     because Background sections recap the previous season;
   - check the year before stating a result; never move it forward;
   - if the question names a season, use only passages that mention it, and
     say so when that season is not covered;
   - do not expand a surname into a full name.
3. `season_note()` (new, `answer_chain.py`): when the question names a season
   (`2024-25`, `2023–24`, `2022/23`), a line with both years is placed after
   the passages and immediately before "Answer". This was the step that made
   the model obey: with the line only in the system prompt, the 2024–25
   quarter-final question still answered "Sporting CP" (2025–26).
4. `eval/questions_seasons.json`: four season questions, including the
   reported one, with expected sections.
5. `eval/retrieval_eval.py`: a custom question file now always gets its own
   result name. Before, `--questions-file` without `--out-dir` wrote
   `arsenal_documents.json` and replaced the 20-question baseline.

### Results

| Question | Before | After |
|---|---|---|
| poetic summary of the journey to the final | mixed 2024–25 into 2025–26 | 2025–26 only: 11 wins/3 draws, 1–0 v Atlético, lost final 4–3 on penalties; grounding clean |
| 2024–25 semi-final | — | lost 3–1 on aggregate to PSG |
| 2024–25 quarter-final | "beat Sporting CP 1–0" (2025–26) and labelled it 2024–25 | "5–1 aggregate triumph over Real Madrid" |

Season eval (`eval/questions_seasons.json --with-llm`): 4/4 answers correct,
0 grounding flags, retrieval 4/4 at rank 1, hit@1/3/5/8 = 1.00, MRR 1.000
(saved as `eval/results/arsenal_documents_questions_seasons.json`).

Main 20-question baseline re-run after the change: 0.60 / 0.85 / 0.95 / 0.95,
MRR 0.733 — unchanged. Tests: **179 passed, 6 skipped**; all data quality
checks pass (224 chunks).

### Files touched in this follow-up

- `config/settings.yaml` (new key)
- `src/config.py` (`current_season` field, loader, redacted)
- `src/generation/answer_chain.py` (prompt rules 11–16, `season_note`, hint in
  the user turn)
- `eval/questions_seasons.json` (new)
- `eval/retrieval_eval.py` (result naming)
- `tests/test_answer_chain.py` (4 new tests)

### Limits

- This steers the model; it does not guarantee obedience. A wrong answer is
  still possible on a new phrasing.
- The season is a single YAML value. When the next season page is indexed,
  change `current_season` (and `pages`); nothing in code needs editing.
- `season_note` only reads explicit ranges (`2024-25`). "Last season" or "the
  previous campaign" is not detected; those rely on the system prompt.
- A full-article summary still includes the Background section by design: it
  is part of the article, not a question about one season.

Commits for this work: P2 `cbec460`, P1 `4184c73`, P7 `207de00`, P3 `a8fc169`,
P4 `659dd01`, P5 `d910eb9`, P6 `ffcd186`, plus the final fixes commit.

---

## 12. More context and longer answers (2026-10-20)

### Settings changed

| Where | Key | Before | After |
|---|---|---|---|
| `.env` | `RETRIEVE_K` | 15 | **20** |
| `.env` | `FINAL_TOP_N` | 8 | **12** |
| `.env` | `RELATIVE_MARGIN` | 0.10 | **0.15** |
| `config/settings.yaml` | `answer_style` | (new) | **normal** |

`answer_style` accepts `short` (old behaviour), `normal` or `long`. It is
validated in `src/config.py` (41, 302, 539–542) and shown by `redacted()`.

### Prompt change

Rule 7 in `SYSTEM_PROMPT` is now chosen from a mapping
(`src/generation/answer_chain.py:204`):

- `short`: "Keep answers short: a few sentences, or a short list."
- `normal`: one or two short paragraphs, about 4–6 sentences in total, in
  flowing prose; a list only when the question asks for several items; no
  one-sentence-per-line.
- `long`: one to three paragraphs, about 8–10 sentences; no repetition.

The first `normal` wording produced one sentence per line, so it was tightened
to ask for flowing prose. The March demo now answers in two sentences; the
Champions League demo gives two paragraphs.

### A new safeguard: `filter_to_season`

With 12 chunks the season bug came back: the 2024–25 quarter-final question
answered "Sporting CP" again because the context held more 2025–26
quarter-final passages. Prompt rules alone were not enough at this size.

`filter_to_season` (`answer_chain.py:101`, called at 323) now keeps only
passages that mention the starting year of a season named in the question
(`2024` for `2024-25`, or the written range). If nothing matches, it keeps all
passages so the model can say the season is not covered. The years come from
the question, not from a literal.

Effect: normal questions still receive 12 chunks; explicit-season questions
receive only the matching ones (the 2024–25 quarter-final now gets 2). The
March question still receives all 14 chunks in its section.

### Results

- March question: 14 section chunks in the context and the sources.
- Champions League journey: 12 chunks, two paragraphs, no 2024–25 facts.
- 2024–25 quarter-final: correct "Real Madrid, 5–1 on aggregate".
- Season eval (`--with-llm`): 4/4 correct, 0 grounding flags.
- Main 20-question baseline: 0.60 / 0.85 / 0.95 / 0.95, MRR 0.733 — unchanged
  (the eval still uses `EVAL_TOP_K=8`).
- Tests: **184 passed, 6 skipped**; all data quality checks pass (224 chunks).

### Files touched

- `.env`, `.env.example` (three values each)
- `config/settings.yaml` (`answer_style`)
- `src/config.py` (`ANSWER_STYLES`, field, validation, redacted)
- `src/generation/answer_chain.py` (rule 7 mapping, `filter_to_season`)
- `tests/test_answer_chain.py` (style and filter tests)
- `tests/test_retrieval.py` (expected `FINAL_TOP_N` now 12)

### Risks and limits

- More chunks and longer answers raise the chance of rambling; the season
  filter protects only questions that name a season.
- A simple factual question may still get a short answer under `normal`; the
  model decides how much detail the question needs.
- The model still expands some surnames ("Bukayo Saka", "Mikel Arteta") even
  though rule 16 forbids it; the grounding heuristic reports it.
- Restart any running chat after editing `.env` or `settings.yaml`.
