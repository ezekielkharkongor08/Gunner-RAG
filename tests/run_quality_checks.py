"""Quality checks against a real pgvector collection.

Read-only by default. With ``--reindex-check`` it re-embeds the same chunks
into the same collection (no reset) and confirms the row count stays the same.

Usage:
    python tests/run_quality_checks.py --collection arsenal_eval_900_120
    python tests/run_quality_checks.py --collection arsenal_eval_900_120 --reindex-check

Never point ``--reindex-check`` at ``arsenal_documents``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from src.config import settings  # noqa: E402
from src.ingestion.document_cleaner import clean_documents  # noqa: E402
from src.ingestion.index_builder import build_index, get_engine  # noqa: E402
from src.ingestion.section_assembler import assemble_sections  # noqa: E402
from src.ingestion.text_splitter import split_documents  # noqa: E402
from src.ingestion.wikipedia_loader import fetch_wikipedia_documents  # noqa: E402

logger = logging.getLogger(__name__)

REQUIRED_NEW_METADATA = (
    "title", "url", "section", "section_path", "page_type", "seasons",
    "is_lead", "heading_level", "chunk_index", "chunk_count", "chunk_chars",
    "body_chars", "header_chars", "content_hash", "chunk_id",
)


def load_docs(collection: str) -> list[dict]:
    sql = text(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :name
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(sql, {"name": collection}).mappings().all()
    docs = []
    for row in rows:
        metadata = row["cmetadata"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                metadata = {}
        docs.append({"document": row["document"] or "", "metadata": metadata})
    return docs


def body_of(document: str) -> str:
    return document.split("\n\n", 1)[1] if "\n\n" in document else document


def run_checks(docs: list[dict]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    documents = [doc["document"] for doc in docs]
    bodies = [body_of(document) for document in documents]
    metas = [doc["metadata"] for doc in docs]

    checks.append(("collection is not empty", bool(docs), f"{len(docs)} chunks"))
    if not docs:
        return checks

    body_lengths = [int(meta.get("body_chars", len(body)))
                    for meta, body in zip(metas, bodies)]
    too_short = [
        (meta.get("section_path"), length)
        for meta, length in zip(metas, body_lengths)
        if length < settings.chunk.min_size
    ]
    checks.append((
        f"no body < CHUNK_MIN_SIZE ({settings.chunk.min_size})",
        not too_short,
        f"shortest {min(body_lengths)}"
        + (f"; offenders: {too_short[:3]}" if too_short else ""),
    ))

    leading_dot = [
        meta.get("section_path") for meta, body in zip(metas, bodies)
        if body.lstrip().startswith(".")
    ]
    checks.append((
        "no chunk starts with '.'",
        not leading_dot,
        f"{len(leading_dot)} offender(s)",
    ))

    bad_header = []
    for meta, document in zip(metas, documents):
        title = str(meta.get("title", ""))
        path = str(meta.get("section_path", ""))
        expected = f"{title}\nSection: {path}\n\n"
        if not document.startswith(expected):
            bad_header.append(path)
    checks.append((
        "every chunk starts with title + section header",
        not bad_header,
        f"{len(bad_header)} bad header(s)",
    ))

    junk = {s.lower() for s in settings.junk_sections}
    junk_paths = [
        meta.get("section_path") for meta in metas
        if any(part.strip().lower() in junk
               for part in str(meta.get("section_path", "")).split(">"))
    ]
    checks.append((
        "no junk sections",
        not junk_paths,
        f"{len(junk_paths)} junk path(s)",
    ))

    boilerplate = []
    for meta, body in zip(metas, bodies):
        for rule in settings.boilerplate_patterns:
            if (len(body) < int(rule["max_chars"])
                    and re.match(str(rule["pattern"]), body.strip())):
                # Same exception as section_assembler._is_boilerplate: a
                # keep_if_pattern match means the section was kept on purpose.
                keep_if = rule.get("keep_if_pattern")
                if keep_if and re.search(str(keep_if), body):
                    break
                boilerplate.append((meta.get("section_path"), rule["pattern"]))
                break
    checks.append((
        "no boilerplate caption sections",
        not boilerplate,
        f"{len(boilerplate)} offender(s)",
    ))

    paths = [str(meta.get("section_path", "")) for meta in metas]
    kits_leaf = [path for path in paths if path == "Kits"]
    kit_text = [
        body for path, body in zip(paths, bodies)
        if "Kit information" in path
    ]
    checks.append((
        "Kits merged into Kit information",
        not kits_leaf and any("Supplier:" in body for body in kit_text),
        f"kits_leaf={kits_leaf}; kit_chunks={len(kit_text)}",
    ))

    kent_leaf = [path for path in paths if path.endswith("Kent County FA")]
    london = [
        body for path, body in zip(paths, bodies) if path.endswith("London FA")
    ]
    checks.append((
        "Kent County FA not a standalone section",
        not kent_leaf and any("Kent" in body for body in london),
        f"kent_leaf={kent_leaf}; london_chunks={len(london)}",
    ))

    with_breaks = sum(1 for body in bodies if "\n\n" in body)
    checks.append((
        "paragraph breaks survived (\\n\\n)",
        with_breaks > 0,
        f"{with_breaks}/{len(bodies)} chunks contain \\n\\n",
    ))

    missing = []
    for meta in metas:
        absent = [field for field in REQUIRED_NEW_METADATA if field not in meta]
        if absent:
            missing.append((meta.get("section_path"), absent))
    checks.append((
        "all new metadata fields present",
        not missing,
        f"{len(missing)} chunk(s) missing fields"
        + (f"; first: {missing[0]}" if missing else ""),
    ))

    ids = [str(meta.get("chunk_id", "")) for meta in metas]
    checks.append((
        "chunk ids are unique",
        len(ids) == len(set(ids)) and all(ids),
        f"{len(ids) - len(set(ids))} duplicate(s)",
    ))

    headers_ok = all(
        int(meta.get("header_chars", 0)) > 0
        and int(meta.get("chunk_chars", 0)) == len(document)
        for meta, document in zip(metas, documents)
    )
    checks.append((
        "header_chars > 0 and chunk_chars matches text",
        headers_ok,
        "",
    ))

    checks.append((
        "chunk count is reported",
        True,
        f"{len(docs)} chunks; median body {statistics.median(body_lengths):.0f}",
    ))
    return checks


def rebuild_chunks() -> list:
    docs = clean_documents(fetch_wikipedia_documents())
    assembled, _report = assemble_sections(docs)
    return split_documents(assembled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.database.collection)
    parser.add_argument(
        "--reindex-check", action="store_true",
        help="Re-embed the same chunks without reset and compare row counts.",
    )
    args = parser.parse_args()

    if args.reindex_check and args.collection == settings.database.collection:
        parser.error("Refusing to re-index the production collection.")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    docs = load_docs(args.collection)
    checks = run_checks(docs)

    lines = [f"Quality checks: {args.collection}", ""]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))

    if args.reindex_check and docs:
        before = len(docs)
        chunks = rebuild_chunks()
        build_index(chunks, reset=False, collection=args.collection)
        after = len(load_docs(args.collection))
        ok = before == after
        lines.append("")
        lines.append(
            f"[{'PASS' if ok else 'FAIL'}] re-index without reset keeps row "
            f"count ({before} -> {after})"
        )
        checks.append(("re-index without reset", ok, f"{before} -> {after}"))

    report = "\n".join(lines)
    print(report)
    report_path = settings.eval.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {report_path}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
