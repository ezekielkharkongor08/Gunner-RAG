"""Retrieval-only eval for chunking experiments.

No LLM call. For each question we embed the question, fetch the top k chunks
and check whether the chunk(s) that contain the expected fact are in the top k.
This measures chunking + embedding quality only.

Two metrics are reported:

    * primary   -- the chunk must be in the expected section AND contain all
                   expected keywords (or at least one keyword for "any" parts)
    * secondary -- the chunk only has to contain the keywords

A question with several parts (Q1) needs every part to be reachable.
"any" questions (Q14, Q15, Q20) need one of their alternative locations.

Usage:
    python eval/retrieval_eval.py --collection arsenal_documents
    python eval/retrieval_eval.py --collection arsenal_eval_900_120 --show 1,4,14,21

Writes eval/results/<collection>.json and prints a one-row table.
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
from src.ingestion.index_builder import get_engine  # noqa: E402
from src.retrieval.retriever import search_for_question  # noqa: E402
from src.retrieval.section_router import list_sections, match_section  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.json")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_questions(path: Path | None = None) -> list[dict]:
    """Read a questions file (default eval/questions.json)."""
    source = path or QUESTIONS_FILE
    data = json.loads(source.read_text(encoding="utf-8"))
    return data["questions"]


def load_collection_docs(collection: str) -> list[dict]:
    """Return every chunk in a collection: document + metadata."""
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
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append({"document": row["document"] or "", "metadata": metadata})
    return docs


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _keywords(part: dict) -> list[str]:
    return list(part.get("keywords_all", [])) + list(part.get("keywords_any", []))


def matches_part(part: dict, text_value: str, section_path: str,
                 use_section: bool = True) -> bool:
    """True when a chunk satisfies this question part."""
    if use_section:
        expected = str(part.get("section", ""))
        # Case-sensitive: "Final" must not match "Semi-final".
        if expected and expected not in section_path:
            return False

    haystack = text_value.lower()
    required = part.get("keywords_all", [])
    if any(word.lower() not in haystack for word in required):
        return False

    alternatives = part.get("keywords_any", [])
    if alternatives and not any(word.lower() in haystack for word in alternatives):
        return False
    return True


def first_rank(part: dict, results: list[tuple], use_section: bool) -> int | None:
    """1-based rank of the first result matching the part, or None."""
    for rank, (document, _score) in enumerate(results, start=1):
        metadata = document.metadata or {}
        if matches_part(
            part,
            document.page_content,
            str(metadata.get("section_path", "")),
            use_section=use_section,
        ):
            return rank
    return None


def question_rank(question: dict, results: list[tuple],
                  use_section: bool) -> tuple[int | None, list[int | None]]:
    """Return (question rank, per-part ranks)."""
    expected = question.get("expected_section")
    if expected and use_section:
        # Section questions are graded on the stored section_path, not keywords.
        for rank, (document, _score) in enumerate(results, start=1):
            section_path = str((document.metadata or {}).get("section_path", ""))
            if section_path == expected:
                return rank, [rank]
        return None, [None]

    parts = question.get("parts", [])
    part_ranks = [first_rank(part, results, use_section) for part in parts]
    if not part_ranks:
        return None, []

    if question.get("require_any_part"):
        found = [rank for rank in part_ranks if rank is not None]
        return (min(found) if found else None), part_ranks

    # Default: every part is required.
    if any(rank is None for rank in part_ranks):
        return None, part_ranks
    return max(part_ranks), part_ranks


# ---------------------------------------------------------------------------
# Pre-flight: is every expected keyword actually in the collection?
# ---------------------------------------------------------------------------
def keyword_report(questions: list[dict], docs: list[dict]) -> list[str]:
    warnings: list[str] = []
    for question in questions:
        if question.get("negative"):
            continue
        for part in question.get("parts", []):
            expected = str(part.get("section", ""))
            for keyword in _keywords(part):
                hits = [
                    doc for doc in docs
                    if keyword.lower() in doc["document"].lower()
                ]
                if not hits:
                    warnings.append(
                        f"Q{question['id']}: keyword {keyword!r} is NOT in "
                        f"collection."
                    )
                    continue
                in_section = [
                    doc for doc in hits
                    if expected in str(doc["metadata"].get("section_path", ""))
                ]
                if not in_section:
                    warnings.append(
                        f"Q{question['id']}: keyword {keyword!r} is only "
                        f"outside section {expected!r}."
                    )
                elif len(hits) > 5:
                    warnings.append(
                        f"Q{question['id']}: keyword {keyword!r} is weak "
                        f"({len(hits)} chunks contain it)."
                    )
    return warnings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_top(question: dict, results: list[tuple], ranks: list[int | None]) -> None:
    print(f"\nQ{question['id']}: {question['question']}")
    if question.get("negative"):
        print("  (negative question: not scored; top results shown)")
    else:
        print(f"  primary rank: {ranks[0] if len(ranks) == 1 else ranks}")
    display = max(settings.eval.hit_at)
    for rank, (document, score) in enumerate(results[:display], start=1):
        md = document.metadata or {}
        body = int(md.get("body_chars", len(document.page_content)))
        marker = " <-- expected" if (
            not question.get("negative") and rank in ranks
        ) else ""
        print(
            f"  {rank:>2}. dist={score:.4f} chars={len(document.page_content):>4} "
            f"body={body:>4} | {md.get('section_path', '?')}{marker}"
        )
        snippet = document.page_content.replace("\n", " ")[:150]
        print(f"      {snippet}")


# ---------------------------------------------------------------------------
# Optional: answer with the LLM and flag facts that are not in the context
# ---------------------------------------------------------------------------
from src.generation.grounding import (  # noqa: E402
    unsupported_numbers,
    unsupported_terms,
)


def run_llm_checks(questions: list[dict]) -> None:
    """Generate one answer per question and flag unsupported facts.

    Advisory only: the number check is exact (the token must appear in the
    retrieved text), the name check is a heuristic and can false-positive.
    Negative questions (21, 22) must not be answered with a hard fact.
    """
    from src.generation.answer_chain import answer_question

    print("\nLLM answers (current settings):")
    flagged = 0
    for question in questions:
        result = answer_question(question["question"])
        answer = (result.answer or "").strip()
        context = " ".join(chunk.body_text() for chunk in result.sources).lower()

        missing_numbers = unsupported_numbers(answer, context)
        missing_names = unsupported_terms(answer, context)
        negative = bool(question.get("negative"))
        invented = negative and bool(missing_numbers or missing_names)
        if missing_numbers or invented:
            flagged += 1

        print(f"\nQ{question['id']}: {question['question']}")
        print(f"  answer: {answer[:700]}")
        print(f"  chunks: {len(result.sources)}")
        if negative:
            print(
                "  negative: "
                + ("FLAG - possible invented fact" if invented else "OK - no hard fact")
            )
        if missing_numbers:
            print(f"  FLAG numbers not in retrieved text: {missing_numbers}")
        if missing_names:
            print(f"  note names not in retrieved text (heuristic): {missing_names[:6]}")

    print(f"\nLLM check: {flagged} question(s) flagged.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", default=settings.database.collection,
        help="Collection to read. Defaults to PGVECTOR_COLLECTION.",
    )
    parser.add_argument("--k", type=int, default=settings.eval.top_k)
    parser.add_argument(
        "--query-instruction", choices=("on", "off"), default="off",
        help="Apply the Qwen3 query instruction (default: off).",
    )
    parser.add_argument(
        "--query-rewrite", choices=("on", "off"), default="off",
        help="Legacy switch: on == --query-rewrite-mode always (default: off).",
    )
    parser.add_argument(
        "--query-rewrite-mode", choices=("off", "always", "fallback"),
        default=None,
        help="Rewriting mode to measure. Defaults to --query-rewrite.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Write the result JSON here instead of EVAL_RESULTS_DIR.",
    )
    parser.add_argument(
        "--with-llm", action="store_true",
        help="Also generate an answer per question and flag facts that are "
             "not in the retrieved text (slow: one LLM call per question).",
    )
    parser.add_argument(
        "--questions-file", default=None,
        help="Questions JSON to use instead of eval/questions.json.",
    )
    parser.add_argument(
        "--gate-report", action="store_true",
        help="Print best distance, chunks kept by the gate and the section "
             "router result per question.",
    )
    parser.add_argument(
        "--show", default="",
        help="Comma-separated question ids whose top k results to print.",
    )
    parser.add_argument(
        "--max-misses", type=int, default=6,
        help="How many missed questions print their top k results.",
    )
    args = parser.parse_args()
    rewrite_mode = args.query_rewrite_mode or (
        "always" if args.query_rewrite == "on" else "off"
    )

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    hit_at = tuple(sorted(settings.eval.hit_at))
    search_k = max(args.k, max(hit_at))
    show_ids = {
        int(part) for part in args.show.split(",") if part.strip().isdigit()
    }

    questions = load_questions(
        Path(args.questions_file) if args.questions_file else None
    )
    docs = load_collection_docs(args.collection)
    if not docs:
        print(f"Collection {args.collection!r} is empty or does not exist.")
        return 1

    lengths = [len(doc["document"]) for doc in docs]
    body_lengths = [
        int(doc["metadata"]["body_chars"])
        for doc in docs if "body_chars" in doc["metadata"]
    ]
    print(f"Collection: {args.collection}")
    print(
        f"Chunks: {len(docs)} | median chunk chars: "
        f"{statistics.median(lengths):.0f} | median body chars: "
        f"{statistics.median(body_lengths) if body_lengths else 0:.0f}"
    )

    warnings = keyword_report(questions, docs)
    if warnings:
        print("\nPre-flight warnings:")
        for warning in warnings:
            print(f"  ! {warning}")
    else:
        print("Pre-flight: all expected keywords found in the expected sections.")

    per_question: list[dict] = []
    gate_rows: list[dict] = []
    shown = 0

    def _search(question_text: str) -> list[tuple]:
        """One question -> ranked results, honouring the two switches."""
        return search_for_question(
            question_text,
            mode=rewrite_mode,
            k=search_k,
            collection=args.collection,
            instruction_enabled=(args.query_instruction == "on"),
        )

    for question in questions:
        if question.get("negative"):
            per_question.append({"id": question["id"], "negative": True,
                                 "rank": None, "keyword_rank": None,
                                 "part_ranks": []})
            if question["id"] in show_ids:
                results = _search(question["question"])
                print_top(question, results, [])
                shown += 1
            continue

        results = _search(question["question"])
        rank, part_ranks = question_rank(question, results, use_section=True)
        keyword_rank, _ = question_rank(question, results, use_section=False)

        if args.gate_report:
            best = min((score for _, score in results), default=None)
            margin = settings.retrieval.relative_margin
            kept = sum(
                1 for _, score in results
                if best is not None and score <= best + margin
            )
            gate_rows.append({
                "id": question["id"],
                "best": best,
                "kept": min(kept, args.k),
                "weak": (
                    best is not None
                    and best > settings.retrieval.query_rewrite_weak_score
                ),
                "expected_section": question.get("expected_section"),
            })

        per_question.append({
            "id": question["id"],
            "question": question["question"],
            "rank": rank,
            "keyword_rank": keyword_rank,
            "part_ranks": part_ranks,
        })

        if question["id"] in show_ids:
            print_top(question, results, part_ranks)
            shown += 1
        elif rank is None and shown < args.max_misses:
            print_top(question, results, part_ranks)
            shown += 1

    scored = [item for item in per_question if not item.get("negative")]
    metrics: dict[str, float] = {}
    for limit in hit_at:
        hits = sum(
            1 for item in scored
            if item["rank"] is not None and item["rank"] <= limit
        )
        metrics[f"hit@{limit}"] = hits / len(scored) if scored else 0.0
    metrics["mrr"] = (
        sum(1.0 / item["rank"] for item in scored if item["rank"]) / len(scored)
        if scored else 0.0
    )

    keyword_scored = [item for item in scored if item.get("keyword_rank")]
    metrics["keyword_mrr"] = (
        sum(1.0 / item["keyword_rank"] for item in keyword_scored)
        / len(scored) if scored else 0.0
    )

    print("\nPer-question ranks (primary / keyword-only):")
    for item in per_question:
        if item.get("negative"):
            print(f"  Q{item['id']:>2}: negative, not scored")
        else:
            print(
                f"  Q{item['id']:>2}: {str(item['rank']):>4} / "
                f"{str(item['keyword_rank']):>4}"
            )

    header = " | ".join(["hit@" + str(n) for n in hit_at] + ["MRR", "MRR(kw)"])
    row = " | ".join(
        [f"{metrics['hit@' + str(n)]:.2f}" for n in hit_at]
        + [f"{metrics['mrr']:.3f}", f"{metrics['keyword_mrr']:.3f}"]
    )
    print(f"\nTable:\n  collection | chunks | {header}")
    print(f"  {args.collection} | {len(docs)} | {row}")

    if args.gate_report and gate_rows:
        question_text = {q["id"]: q["question"] for q in questions}
        router_paths = [section.path for section in list_sections(args.collection)]
        print(
            "\nGate report: mode=%s, margin=%.2f, weak_score=%.2f"
            % (
                settings.retrieval.retrieval_gate,
                settings.retrieval.relative_margin,
                settings.retrieval.query_rewrite_weak_score,
            )
        )
        print("  id  | best  | kept | rewrite? | section router")
        for row in gate_rows:
            expected = row["expected_section"]
            if expected:
                matched = match_section(
                    question_text[row["id"]],
                    router_paths,
                    settings.section_tokens,
                )
                router_text = "OK" if matched == expected else f"miss ({matched})"
            else:
                router_text = "-"
            print(
                "  Q%-2d | %s | %4d | %-8s | %s"
                % (
                    row["id"],
                    "%.3f" % row["best"] if row["best"] is not None else "  -  ",
                    row["kept"],
                    "yes" if row["weak"] else "no",
                    router_text,
                )
            )

    payload = {
        "collection": args.collection,
        "chunks": len(docs),
        "questions_file": args.questions_file or QUESTIONS_FILE.name,
        "query_instruction": args.query_instruction,
        "query_rewrite": args.query_rewrite,
        "query_rewrite_mode": rewrite_mode,
        "median_chunk_chars": statistics.median(lengths),
        "median_body_chars": (
            statistics.median(body_lengths) if body_lengths else 0
        ),
        "hit_at": list(hit_at),
        "metrics": metrics,
        "questions": per_question,
    }
    out_dir = (
        Path(args.out_dir) if args.out_dir else settings.eval.results_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_rewrite = {"off": "off", "always": "on", "fallback": "fallback"}[
        rewrite_mode
    ]
    # A non-default question file always gets its own name, otherwise it would
    # overwrite the baseline arsenal_documents.json (it did once for the season
    # questions). The grid switches add their suffix only with --out-dir.
    suffix = ""
    if args.out_dir:
        suffix = f"_instr-{args.query_instruction}_rewrite-{suffix_rewrite}"
    if args.questions_file:
        suffix += f"_{Path(args.questions_file).stem}"
    out_name = f"{args.collection}{suffix}.json"
    out_path = out_dir / out_name
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_path}")

    if args.with_llm:
        run_llm_checks(questions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
