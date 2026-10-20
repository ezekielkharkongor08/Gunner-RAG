"""run_ingestion.py -- orchestrator for the full ingestion pipeline.

Runs the four stages in order:

    load (wikipedia_loader.py)
        -> clean (document_cleaner.py)
        -> split (text_splitter.py)
        -> embed + store (index_builder.py)

Wikipedia is the only content source.  Articles are fetched through the
MediaWiki Action API, extracted as section-aware prose, cleaned, split,
embedded and stored in PostgreSQL/pgvector.

Usage:
    python src/ingestion/run_ingestion.py                              # default pages from settings.yaml
    python src/ingestion/run_ingestion.py --titles "Arsenal F.C." "History of Arsenal F.C."
    python src/ingestion/run_ingestion.py --titles "Arsenal F.C." --no-reset
    python src/ingestion/run_ingestion.py --titles "Arsenal F.C." --batch-size 16

From Python:
    from src.ingestion.run_ingestion import run_ingestion
    summary = run_ingestion(titles=["Arsenal F.C."], reset=True)
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.ingestion.document_cleaner import clean_documents
from src.ingestion.index_builder import build_index
from src.ingestion.section_assembler import assemble_sections, summarize_report
from src.ingestion.text_splitter import split_documents
from src.ingestion.wikipedia_loader import fetch_wikipedia_documents

logger = logging.getLogger(__name__)


def run_ingestion(
    titles: list[str] | None = None,
    *,
    reset: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Run the full ingestion pipeline: load -> clean -> split -> build_index.

    Args:
        titles: Wikipedia article titles to fetch. Defaults to the ``pages``
            list in ``config/settings.yaml``.
        reset: wipe the pgvector collection before inserting.
        batch_size: chunks per embedding batch.

    Returns:
        The summary dict from build_index().

    Raises:
        ValueError: from build_index() if no valid chunks were produced
            (this is intentional -- a failed run is never reported as success).
    """
    if titles is None:
        titles = list(settings.pages)

    # Stage 1: load / fetch Wikipedia articles (section-aware prose).
    logger.info("Stage 1: fetching %d Wikipedia article(s) via the MediaWiki API", len(titles))
    documents = fetch_wikipedia_documents(titles=titles)
    logger.info("Stage 1 complete: %d document(s) loaded.", len(documents))

    # Stage 2: clean the prose.
    cleaned = clean_documents(documents)
    logger.info("Stage 2 complete: %d document(s) cleaned.", len(cleaned))

    # Stage 2b: assemble sections (drop boilerplate, merge short sections).
    assembled, assembly_report = assemble_sections(cleaned)
    logger.info("Stage 2b complete: %s", summarize_report(assembly_report))

    # Stage 3: split cleaned prose into chunks.
    chunks = split_documents(assembled)
    logger.info("Stage 3 complete: %d chunk(s) produced.", len(chunks))

    # Stage 4: embed + store.  Raises ValueError if there are no valid chunks,
    # so a failed ingestion is never reported as successfully indexed.
    summary = build_index(
        chunks,
        reset=reset,
        batch_size=batch_size or settings.chunk.batch_size,
    )
    summary["assembly"] = assembly_report
    logger.info("Stage 4 complete: %d chunk(s) written.", summary["written"])

    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline.")
    parser.add_argument(
        "--titles", nargs="*", metavar="TITLE",
        help="Wikipedia article title(s) to ingest (defaults to config/settings.yaml pages).",
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Do not wipe the pgvector collection before inserting.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Chunks per embedding batch (default: INDEX_BATCH_SIZE from .env).",
    )
    args = parser.parse_args()

    titles: list[str] | None = None
    if args.titles:
        flat: list[str] = []
        for entry in args.titles:
            flat.extend([part.strip() for part in entry.split(",") if part.strip()])
        titles = flat

    print(f"Wikipedia pages configured: {len(settings.pages)}")
    for title in settings.pages:
        print(f"  - {title}")

    summary = run_ingestion(
        titles=titles,
        reset=not args.no_reset,
        batch_size=args.batch_size,
    )

    print("\n=== Ingestion complete ===")
    print(f"Chunks:       {summary['chunks']}")
    print(f"Written:      {summary['written']}")
    print(f"Collection:   {summary['collection']}")
    print(f"Model:        {summary['embedding_model']}")
    print(f"Reset:        {summary['reset']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
