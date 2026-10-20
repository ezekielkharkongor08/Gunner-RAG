"""Build one experiment collection: arsenal_eval_<tag>.

The app settings are frozen when Python imports src.config, so the cleanest
way to build a different chunk size is a subprocess with env overrides.

Usage:
    python eval/build_eval_collection.py --tag 900_120 --size 900 --overlap 120

This never touches ``arsenal_documents``: the collection name is always
``EVAL_COLLECTION_PREFIX + tag``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Collection tag, e.g. 900_120.")
    parser.add_argument("--size", type=int, default=settings.chunk.size)
    parser.add_argument("--overlap", type=int, default=settings.chunk.overlap)
    parser.add_argument("--batch-size", type=int, default=settings.chunk.batch_size)
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Fetch fresh Wikipedia text instead of using the cache.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.tag):
        parser.error("--tag may only contain letters, numbers and underscores.")
    if args.overlap >= args.size:
        parser.error("--overlap must be smaller than --size.")

    collection = f"{settings.eval.collection_prefix}{args.tag}"
    if collection == settings.database.collection:
        parser.error("Refusing to write to the production collection.")

    env = os.environ.copy()
    env.update({
        "PGVECTOR_COLLECTION": collection,
        "CHUNK_SIZE": str(args.size),
        "CHUNK_OVERLAP": str(args.overlap),
        "INDEX_BATCH_SIZE": str(args.batch_size),
        "USE_FETCH_CACHE": "true",
        "REFRESH_FETCH_CACHE": "true" if args.refresh_cache else "false",
    })

    print(f"Building {collection} (size={args.size}, overlap={args.overlap}) ...")
    subprocess.run(
        [sys.executable, "-m", "src.ingestion.run_ingestion"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(f"Done: {collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
