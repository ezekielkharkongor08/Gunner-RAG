"""Print one comparison table from eval/results/*.json.

Usage:
    python eval/compare_results.py

The baseline (arsenal_documents) is shown first. Other rows are sorted by the
number of chunks so the chunk-size sweep is easy to read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def main() -> int:
    results_dir = settings.eval.results_dir
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"No result files in {results_dir}. Run eval/retrieval_eval.py first.")
        return 1

    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)

    # Baseline first, then by chunk count (small to large).
    rows.sort(key=lambda row: (
        row["collection"] != settings.database.collection,
        row["chunks"],
    ))

    hit_keys = [f"hit@{n}" for n in settings.eval.hit_at]
    header = ["collection", "chunks", "median"] + hit_keys + ["MRR", "MRR(kw)"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for row in rows:
        metrics = row["metrics"]
        cells = [
            row["collection"],
            str(row["chunks"]),
            f"{row['median_body_chars'] or row['median_chunk_chars']:.0f}",
        ]
        cells += [f"{metrics[key]:.2f}" for key in hit_keys]
        cells += [f"{metrics['mrr']:.3f}", f"{metrics['keyword_mrr']:.3f}"]
        print(" | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
