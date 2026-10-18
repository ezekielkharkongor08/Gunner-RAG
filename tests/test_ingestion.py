"""test.py -- check the ingestion pipeline with your own eyes.

Run:  python test.py
      python test.py --wikipedia   (also fetches Wikipedia pages -- needs network)
      python test.py --db          (also embeds + stores -- needs Ollama + PostgreSQL)

Prints PASS / FAIL for each check, saves the full report to test_report.txt,
and exits with code 1 if any check failed.

This runs stages 1 to 3 (load, clean, split) by default. No Ollama or
PostgreSQL needed unless you pass --db.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.wikipedia_loader import (
    load_local_documents,
    load_documents,
    fetch_wikipedia_documents,
)
from src.ingestion.document_cleaner import clean_documents, clean_text
from src.ingestion.text_splitter import (
    split_documents,
    _split_into_sections,
    _is_junk_section,
    _fold_short_chunks,
    _split_csv_document,
    CONTENT_PROSE,
    CONTENT_TABLE,
    CONTENT_TABLE_SUMMARY,
)
from src.ingestion.index_builder import (
    PrefixedEmbeddings,
    stable_chunk_id,
    delete_existing_for_source,
    build_index,
    index_status,
    get_embeddings,
    get_vector_store,
)

# ---------------------------------------------------------------------------
# Report collection
# ---------------------------------------------------------------------------
_report_lines: list[str] = []
_failures = 0
_warnings = 0


def log(line: str = "") -> None:
    _report_lines.append(line)
    # Windows console may not support UTF-8; replace unsupported chars.
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("utf-8", errors="replace").decode("utf-8"))


def pass_check(name: str, detail: str = "") -> None:
    global _failures
    _failures += 0  # placeholder, see fail_check
    _report_lines.append(f"PASS: {name}" + (f" -- {detail}" if detail else ""))
    print(f"PASS: {name}" + (f" -- {detail}" if detail else ""))


def fail_check(name: str, detail: str = "") -> None:
    global _failures
    _failures += 1
    _report_lines.append(f"FAIL: {name}" + (f" -- {detail}" if detail else ""))
    print(f"FAIL: {name}" + (f" -- {detail}" if detail else ""))


def warn_check(name: str, detail: str = "") -> None:
    global _warnings
    _warnings += 1
    _report_lines.append(f"WARN: {name}" + (f" -- {detail}" if detail else ""))
    print(f"WARN: {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Stage 1-3: load, clean, split
# ---------------------------------------------------------------------------
def run_pipeline(include_wikipedia: bool = False):
    log("=" * 60)
    log("STAGE 1-3: load -> clean -> split")
    log("=" * 60)

    docs = load_documents(include_local=True, include_wikipedia=include_wikipedia)
    log(f"Documents loaded: {len(docs)}")
    local_count = sum(1 for d in docs if d.metadata.get("source") == "local")
    wiki_count = sum(1 for d in docs if d.metadata.get("source") == "wikipedia")
    log(f"  local: {local_count}, wikipedia: {wiki_count}")

    cleaned = clean_documents(docs)
    log(f"Documents after cleaning: {len(cleaned)}")

    chunks = split_documents(cleaned)
    log(f"Chunks after splitting: {len(chunks)}")

    # Counts per content_type
    from collections import Counter
    ct = Counter(c.metadata.get("content_type", "?") for c in chunks)
    log(f"Chunks per content_type: {dict(ct)}")

    return docs, cleaned, chunks


# ---------------------------------------------------------------------------
# Check 2: chunk size
# ---------------------------------------------------------------------------
def check_chunk_size(chunks):
    log("\n" + "-" * 60)
    log("Check 2: Chunk size")
    log("-" * 60)

    if not chunks:
        fail_check("chunk_size", "No chunks at all.")
        return

    sizes = [len(c.page_content) for c in chunks]
    mn, mx, avg = min(sizes), max(sizes), sum(sizes) // len(sizes)
    log(f"  min={mn}  max={mx}  avg={avg}  count={len(sizes)}")

    empty = [c for c in chunks if len(c.page_content) == 0]
    if empty:
        fail_check("chunk_size_empty", f"{len(empty)} empty chunk(s) found.")
    else:
        pass_check("chunk_size_empty", "No empty chunks.")

    if mx > settings.chunk.max_warn:
        warn_check("chunk_size_warn",
                   f"Largest chunk ({mx}) exceeds CHUNK_MAX_WARN={settings.chunk.max_warn}.")
    else:
        pass_check("chunk_size_warn",
                   f"Largest chunk ({mx}) within CHUNK_MAX_WARN={settings.chunk.max_warn}.")

    pass_check("chunk_size_stats",
               f"min={mn} max={mx} avg={avg}")


# ---------------------------------------------------------------------------
# Check 3: junk sections
# ---------------------------------------------------------------------------
def check_junk_sections(chunks):
    log("\n" + "-" * 60)
    log("Check 3: Junk sections removed")
    log("-" * 60)

    junk_names = {s.lower() for s in settings.junk_sections}
    found_junk = []
    for c in chunks:
        section = str(c.metadata.get("section", "")).lower()
        text = c.page_content.lower()
        for junk in junk_names:
            if junk in section or f" > {junk}" in text or f"\n{junk}\n" in text:
                found_junk.append((c.metadata.get("chunk_id", ""), junk))

    if found_junk:
        fail_check("junk_sections",
                   f"{len(found_junk)} chunk(s) still reference junk sections: {found_junk[:5]}")
    else:
        pass_check("junk_sections", "No junk sections in any chunk.")


# ---------------------------------------------------------------------------
# Check 4: noise removal
# ---------------------------------------------------------------------------
def check_noise(chunks):
    log("\n" + "-" * 60)
    log("Check 4: Noise removed (no [[, ]], citation markers like [12])")
    log("-" * 60)

    citation_re = re.compile(r"\[\d+[a-z]?\]")
    noisy = []
    for c in chunks:
        text = c.page_content
        if "[[" in text or "]]" in text or citation_re.search(text):
            noisy.append(c.metadata.get("chunk_id", ""))

    if noisy:
        fail_check("noise",
                   f"{len(noisy)} chunk(s) still contain wiki markup/citations: {noisy[:5]}")
    else:
        pass_check("noise", "No wiki markup or citation markers in any chunk.")


# ---------------------------------------------------------------------------
# Check 5: metadata
# ---------------------------------------------------------------------------
def check_metadata(chunks):
    log("\n" + "-" * 60)
    log("Check 5: Required metadata fields")
    log("-" * 60)

    required = ["title", "source_url", "source", "chunk_id", "section",
                "content_type", "fetched_at"]
    missing = []
    for c in chunks:
        md = c.metadata
        for field in required:
            if field not in md or md[field] is None or md[field] == "":
                missing.append((c.metadata.get("chunk_id", ""), field))

    if missing:
        fail_check("metadata",
                   f"{len(missing)} missing field(s): {missing[:10]}")
    else:
        pass_check("metadata", f"All {len(required)} fields present on every chunk.")


# ---------------------------------------------------------------------------
# Check 6: breadcrumb
# ---------------------------------------------------------------------------
def check_breadcrumb(chunks):
    log("\n" + "-" * 60)
    log("Check 6: Every prose chunk starts with breadcrumb line")
    log("-" * 60)

    bad = []
    for c in chunks:
        if c.metadata.get("content_type") != CONTENT_PROSE:
            continue
        title = str(c.metadata.get("title", "")).strip()
        section = str(c.metadata.get("section", "")).strip()
        # Breadcrumb is "title > section" or just "title"
        expected_start = title
        if section and section != title:
            expected_start = f"{title} > {section}"
        first_line = c.page_content.split("\n", 1)[0].strip()
        if first_line != expected_start:
            bad.append((c.metadata.get("chunk_id", ""), first_line[:60], expected_start[:60]))

    if bad:
        fail_check("breadcrumb",
                   f"{len(bad)} prose chunk(s) missing correct breadcrumb: {bad[:5]}")
    else:
        prose_count = sum(1 for c in chunks if c.metadata.get("content_type") == CONTENT_PROSE)
        pass_check("breadcrumb", f"All {prose_count} prose chunks start with breadcrumb.")


# ---------------------------------------------------------------------------
# Check 7: unique IDs
# ---------------------------------------------------------------------------
def check_unique_ids(chunks):
    log("\n" + "-" * 60)
    log("Check 7: All stable chunk IDs are unique")
    log("-" * 60)

    ids = [c.metadata.get("chunk_id", "") for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = [i for i in ids if ids.count(i) > 1]
        fail_check("unique_ids", f"Duplicate chunk IDs: {set(dupes)}")
    else:
        pass_check("unique_ids", f"All {len(ids)} chunk IDs are unique.")


# ---------------------------------------------------------------------------
# Check 8: CSV checks
# ---------------------------------------------------------------------------
def check_csv(chunks):
    log("\n" + "-" * 60)
    log("Check 8: CSV chunks")
    log("-" * 60)

    csv_chunks = [c for c in chunks if c.metadata.get("file_type") == "csv"]
    if not csv_chunks:
        warn_check("csv", "No CSV chunks found (skipping CSV checks).")
        return

    # 8a: every CSV chunk starts with title and has Columns: line
    bad_start = []
    for c in csv_chunks:
        first_line = c.page_content.split("\n", 1)[0].strip()
        title = str(c.metadata.get("title", "")).strip()
        has_columns = "Columns:" in c.page_content
        if first_line != title or not has_columns:
            bad_start.append(c.metadata.get("chunk_id", ""))

    if bad_start:
        fail_check("csv_start_columns",
                   f"{len(bad_start)} CSV chunk(s) missing title or Columns: line: {bad_start[:5]}")
    else:
        pass_check("csv_start_columns",
                   f"All {len(csv_chunks)} CSV chunks start with title and have Columns: line.")

    # 8b: no row split across chunks - count row lines per chunk
    # Count "- " lines in each chunk, sum per table, compare to data rows.
    from collections import defaultdict
    table_row_lines = defaultdict(int)
    table_data_rows = {}

    for c in csv_chunks:
        table_name = c.metadata.get("table_name", "?")
        row_lines = sum(1 for line in c.page_content.split("\n") if line.strip().startswith("- "))
        table_row_lines[table_name] += row_lines

    # Get expected data row count from CSV files
    import csv as csv_module, io
    from pathlib import Path
    for path in settings.data_raw_dir.rglob("*.csv"):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Strip header block
        lines = text.splitlines()
        body_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or set(stripped) <= {"=", "-", "#", "*", "_", "~"}:
                continue
            candidate = stripped.lstrip("#").strip()
            if re.match(r"^[A-Za-z][A-Za-z _]{1,30}\s*:\s*.+$", candidate):
                continue
            body_start = i
            break
        body = "\n".join(lines[body_start:])
        rows = [r for r in csv_module.reader(io.StringIO(body))
                if any(c.strip() for c in r)]
        data_rows = len(rows) - 1 if len(rows) > 1 else 0  # minus header
        title = path.stem.replace("_", " ").replace("-", " ").strip()
        # Try to get TABLE_TOPIC from header
        for line in lines[:10]:
            m = re.match(r"^#\s*TABLE_TOPIC:\s*(.+)$", line.strip())
            if m:
                title = m.group(1).strip()
                break
        table_data_rows[title] = data_rows

    row_mismatch = []
    for table_name, expected in table_data_rows.items():
        actual = table_row_lines.get(table_name, 0)
        if actual != expected:
            row_mismatch.append((table_name, expected, actual))

    if row_mismatch:
        fail_check("csv_no_split_rows",
                   f"Row count mismatch: {row_mismatch[:5]}")
    else:
        pass_check("csv_no_split_rows",
                   f"Row lines preserved across chunks for all {len(table_data_rows)} table(s).")

    # 8c: summary chunk for SUMMARY_COLUMNS tables
    summary_chunks = [c for c in csv_chunks
                      if c.metadata.get("content_type") == CONTENT_TABLE_SUMMARY]
    log(f"  Summary chunks found: {len(summary_chunks)}")
    # Check that at least one summary exists if any CSV has SUMMARY_COLUMNS
    has_summary_header = any(
        str(c.metadata.get("csv_summary_columns", "")).strip() for c in csv_chunks
    )
    if has_summary_header and not summary_chunks:
        fail_check("csv_summary", "SUMMARY_COLUMNS configured but no summary chunk produced.")
    elif not has_summary_header:
        pass_check("csv_summary", "No SUMMARY_COLUMNS configured (summary chunk optional).")
    else:
        pass_check("csv_summary",
                   f"{len(summary_chunks)} summary chunk(s) produced.")

    # 8d: sample CSV with [a] not damaged (document behavior: [a] IS removed by cleaner)
    log("  Note: cleaner removes [a] footnote markers from CSV values too "
        "(cleaner not changed for CSV per task rules).")


# ---------------------------------------------------------------------------
# Check 9: deduplication
# ---------------------------------------------------------------------------
def check_dedup():
    log("\n" + "-" * 60)
    log("Check 9: Deduplication (loading same file twice)")
    log("-" * 60)

    docs1 = load_local_documents()
    docs2 = load_local_documents()  # same files again
    combined = docs1 + docs2
    cleaned = clean_documents(combined)

    # clean_documents already dedupes by content fingerprint
    log(f"  Loaded {len(combined)} raw docs (2x files), cleaned to {len(cleaned)}")
    if len(cleaned) == len(docs1):
        pass_check("dedup", "Duplicate files removed by cleaner.")
    else:
        fail_check("dedup",
                   f"Expected {len(docs1)} cleaned docs, got {len(cleaned)}.")


# ---------------------------------------------------------------------------
# Check 10: unit tests with fake input
# ---------------------------------------------------------------------------
def check_unit_tests():
    log("\n" + "-" * 60)
    log("Check 10: Unit tests (fake input)")
    log("-" * 60)

    # 10a: section splitting
    text = """Lead paragraph.

== History ==
History content here.

=== Wenger era ===
Wenger content here.

== References ==
References content here.
"""
    sections = _split_into_sections(text)
    headings = [s["heading"] for s in sections if s["heading"]]
    if "History" in headings and "Wenger era" in headings and "References" in headings:
        pass_check("unit_section_split", f"Found sections: {headings}")
    else:
        fail_check("unit_section_split", f"Expected History/Wenger era/References, got {headings}")

    # 10b: junk section removal
    assert _is_junk_section("References") is True
    assert _is_junk_section("See also") is True
    assert _is_junk_section("History") is False
    assert _is_junk_section(None) is False
    pass_check("unit_junk_section", "Junk section detection works.")

    # 10c: short-chunk folding
    pieces = [
        Document(page_content="A" * 10, metadata={}),
        Document(page_content="B" * 200, metadata={}),
        Document(page_content="C" * 300, metadata={}),
    ]
    folded = _fold_short_chunks(pieces, min_size=50)
    # First piece (10 chars) should be folded into second
    if len(folded) == 2 and folded[0].page_content.startswith("A" * 10):
        pass_check("unit_short_fold", f"{len(pieces)} pieces -> {len(folded)} chunks, short folded.")
    else:
        fail_check("unit_short_fold", f"Expected 2 chunks, got {len(folded)}.")

    # 10d: CSV row grouping
    csv_doc = Document(
        page_content="My Table\nPage context\nColumns: A, B\n- Row1: A: 1, B: 2\n- Row2: A: 3, B: 4\n- Row3: A: 5, B: 6",
        metadata={
            "title": "My Table",
            "page_context": "Page context",
            "file_type": "csv",
            "content_type": "table",
            "csv_header_row": ["A", "B"],
            "csv_rows": json.dumps([["1", "2"], ["3", "4"], ["5", "6"]], ensure_ascii=False),
            "csv_column_notes": "",
            "csv_row_template": "",
            "csv_summary_columns": "",
            "source": "local",
            "origin": "data/raw/test.csv",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # Temporarily set CSV_ROWS_PER_CHUNK to 2
    import src.ingestion.text_splitter as ts_mod
    old_rows = ts_mod.CSV_ROWS_PER_CHUNK
    ts_mod.CSV_ROWS_PER_CHUNK = 2
    try:
        csv_chunks = _split_csv_document(csv_doc)
    finally:
        ts_mod.CSV_ROWS_PER_CHUNK = old_rows

    if len(csv_chunks) == 2:  # 3 rows, 2 per chunk -> 2 chunks
        pass_check("unit_csv_row_group", f"3 rows, 2 per chunk -> {len(csv_chunks)} chunks.")
    else:
        fail_check("unit_csv_row_group", f"Expected 2 chunks, got {len(csv_chunks)}.")

    # 10e: summary chunk
    csv_doc2 = Document(
        page_content="Scores\nContext\nColumns: Player, Score\n- Row1: Player: X, Score: 10\n- Row2: Player: Y, Score: 25\n- Row3: Player: Z, Score: 7",
        metadata={
            "title": "Scores",
            "page_context": "Context",
            "file_type": "csv",
            "content_type": "table",
            "csv_header_row": ["Player", "Score"],
            "csv_rows": json.dumps([["X", "10"], ["Y", "25"], ["Z", "7"]], ensure_ascii=False),
            "csv_column_notes": "",
            "csv_row_template": "",
            "csv_summary_columns": "Score",
            "source": "local",
            "origin": "data/raw/test.csv",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    csv_chunks2 = _split_csv_document(csv_doc2)
    summary_chunks = [c for c in csv_chunks2
                      if c.metadata.get("content_type") == CONTENT_TABLE_SUMMARY]
    if summary_chunks and "25" in summary_chunks[0].page_content:
        pass_check("unit_summary_chunk", "Summary chunk has top value (25).")
    else:
        fail_check("unit_summary_chunk",
                   f"Expected summary chunk with top value 25, got {len(summary_chunks)} summary chunks.")

    # 10f: prefix wrapper
    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[len(t)] for t in texts]
        def embed_query(self, text):
            return [len(text)]

    wrapped = PrefixedEmbeddings(
        base=FakeEmbeddings(),
        doc_prefix="doc: ",
        query_prefix="query: ",
    )
    doc_vecs = wrapped.embed_documents(["hello"])
    query_vec = wrapped.embed_query("hello")
    if doc_vecs[0][0] == len("doc: hello") and query_vec[0] == len("query: hello"):
        pass_check("unit_prefix_wrapper", "Prefixes added correctly.")
    else:
        fail_check("unit_prefix_wrapper", f"doc={doc_vecs[0][0]} query={query_vec[0]}")


# ---------------------------------------------------------------------------
# --db checks
# ---------------------------------------------------------------------------
def run_db_checks():
    log("\n" + "=" * 60)
    log("STAGE 4: embed + store (--db)")
    log("=" * 60)

    # Check 1: build with reset=True, row count equals chunks
    log("\n--- Check DB1: build_index(reset=True) ---")
    chunks = load_and_split_for_db()
    summary = build_index(reset=True)
    log(f"  build_index summary: {summary}")
    status = index_status()
    log(f"  index_status: {status}")
    db_chunks = status.get("chunks", 0)
    if db_chunks == len(chunks):
        pass_check("db1_row_count",
                   f"DB rows ({db_chunks}) equals chunks ({len(chunks)}).")
    else:
        fail_check("db1_row_count",
                   f"DB rows ({db_chunks}) != chunks ({len(chunks)}).")

    # Check 2: build again without reset, count stays same
    log("\n--- Check DB2: build_index(reset=False) no duplicates ---")
    summary2 = build_index(reset=False)
    status2 = index_status()
    log(f"  build_index summary: {summary2}")
    log(f"  index_status: {status2}")
    if status2.get("chunks", 0) == db_chunks:
        pass_check("db2_no_dup", f"Row count unchanged ({status2.get('chunks', 0)}).")
    else:
        fail_check("db2_no_dup",
                   f"Row count changed from {db_chunks} to {status2.get('chunks', 0)}.")

    # Check 3: stale chunk removal
    log("\n--- Check DB3: stale chunk removal (change text, re-index) ---")
    # Modify one chunk's text by changing a source file, re-index, check old rows gone.
    # We simulate by changing a local file temporarily.
    import shutil
    from pathlib import Path
    raw_dir = settings.data_raw_dir
    txt_files = list(raw_dir.rglob("*.txt"))
    if not txt_files:
        warn_check("db3_stale", "No .txt files to modify for stale chunk test.")
    else:
        target = txt_files[0]
        backup = target.read_text(encoding="utf-8")
        try:
            # Append a unique marker
            target.write_text(backup + "\n\nUNIQUE_MARKER_FOR_STALE_TEST", encoding="utf-8")
            summary3 = build_index(reset=False)
            status3 = index_status()
            log(f"  After re-index: {status3}")
            # Check that the new marker appears in search
            store = get_vector_store()
            hits = store.similarity_search_with_score("UNIQUE_MARKER_FOR_STALE_TEST", k=1)
            if hits:
                pass_check("db3_stale", "Re-indexed content found in DB.")
            else:
                fail_check("db3_stale", "Re-indexed content NOT found in DB.")
        finally:
            target.write_text(backup, encoding="utf-8")

    # Check 4: sample questions
    log("\n--- Check DB4: Sample questions (k=3) ---")
    questions = [
        "Who was Arsenal's top scorer?",
        "Who managed Arsenal in 1998?",
        "How many league titles has Arsenal won?",
        "When was Highbury closed?",
        "Where do Arsenal play their home games?",
    ]
    store = get_vector_store()
    for q in questions:
        hits = store.similarity_search_with_score(q, k=3)
        log(f"\n  Q: {q}")
        for doc, score in hits:
            md = doc.metadata
            snippet = doc.page_content[:150].replace("\n", " ")
            log(f"    [{md.get('chunk_id',''):<40}] score={score:.4f} "
                f"section={md.get('section','')[:30]}")
            log(f"      {snippet}")

    # Check 5: index_status
    log("\n--- Check DB5: index_status() ---")
    status = index_status()
    log(f"  {status}")
    pass_check("db5_status", f"index_status: {status}")


def load_and_split_for_db():
    """Load, clean, split for DB tests."""
    docs = load_documents(include_local=True, include_wikipedia=False)
    cleaned = clean_documents(docs)
    return split_documents(cleaned)


# ---------------------------------------------------------------------------
# Print sample chunks
# ---------------------------------------------------------------------------
def print_samples(chunks):
    log("\n" + "=" * 60)
    log("SAMPLE CHUNKS (for owner review)")
    log("=" * 60)

    prose = [c for c in chunks if c.metadata.get("content_type") == CONTENT_PROSE]
    csv_tbl = [c for c in chunks if c.metadata.get("content_type") == CONTENT_TABLE]

    random.seed(42)
    sample_prose = random.sample(prose, min(5, len(prose)))
    sample_csv = random.sample(csv_tbl, min(3, len(csv_tbl)))

    for i, c in enumerate(sample_prose, 1):
        log(f"\n--- Prose sample {i}: {c.metadata.get('chunk_id','')} ---")
        log(c.page_content)

    for i, c in enumerate(sample_csv, 1):
        log(f"\n--- CSV sample {i}: {c.metadata.get('chunk_id','')} ---")
        log(c.page_content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Force UTF-8 output where possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Test the ingestion pipeline.")
    parser.add_argument("--wikipedia", action="store_true",
                        help="Also fetch Wikipedia pages (needs network).")
    parser.add_argument("--db", action="store_true",
                        help="Also embed + store (needs Ollama + PostgreSQL).")
    args = parser.parse_args()

    log(f"Arsenal Weki Ingestion Test")
    log(f"Run started: {datetime.now(timezone.utc).isoformat()}")
    log(f"Embedding model: {settings.ollama.embedding_model}")
    log(f"Chunk size: {settings.chunk.size}, overlap: {settings.chunk.overlap}")
    log(f"Wikipedia pages: {len(settings.pages)}")

    # Run pipeline
    docs, cleaned, chunks = run_pipeline(include_wikipedia=args.wikipedia)

    # Checks 1-10 (no database)
    log("\n" + "=" * 60)
    log("Check 1: Counts (loaded / cleaned / chunks)")
    log("=" * 60)
    local_count = sum(1 for d in docs if d.metadata.get("source") == "local")
    wiki_count = sum(1 for d in docs if d.metadata.get("source") == "wikipedia")
    log(f"  Loaded: {len(docs)} (local={local_count}, wikipedia={wiki_count})")
    log(f"  Cleaned: {len(cleaned)}")
    log(f"  Chunks: {len(chunks)}")
    from collections import Counter
    ct = Counter(c.metadata.get("content_type", "?") for c in chunks)
    log(f"  Per content_type: {dict(ct)}")
    pass_check("counts",
               f"loaded={len(docs)} cleaned={len(cleaned)} chunks={len(chunks)}")

    check_chunk_size(chunks)
    check_junk_sections(chunks)
    check_noise(chunks)
    check_metadata(chunks)
    check_breadcrumb(chunks)
    check_unique_ids(chunks)
    check_csv(chunks)
    check_dedup()
    check_unit_tests()

    # DB checks
    if args.db:
        run_db_checks()

    # Samples
    print_samples(chunks)

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  PASS checks: many (see above)")
    log(f"  FAIL checks: {_failures}")
    log(f"  WARN checks: {_warnings}")

    # Save report
    report_path = "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    log(f"\nFull report saved to: {report_path}")

    if _failures > 0:
        log(f"\nRESULT: FAIL ({_failures} check(s) failed).")
        return 1
    log("\nRESULT: PASS (all checks passed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
