"""Ingest one annual report PDF into the corpus.

Usage: uv run python -m scripts.ingest data/filings/totalenergies-2025.pdf \
           --slug totalenergies-2025 --company TotalEnergies \
           --title "Universal Registration Document" --year 2025 --language en
"""

import argparse
import logging
from pathlib import Path

from app.ingest.pipeline import TS_CONFIG, DocumentMeta, ingest_document
from app.logs import setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--language", choices=sorted(TS_CONFIG), required=True)
    parser.add_argument("--source-url", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    meta = DocumentMeta(
        slug=args.slug,
        company=args.company,
        title=args.title,
        fiscal_year=args.year,
        language=args.language,
        source_url=args.source_url,
    )
    report = ingest_document(args.pdf, meta)
    log.info(
        "done: document %s (id %d), %d pages, %d chunks",
        meta.slug,
        report.document_id,
        report.pages,
        report.chunks,
    )


if __name__ == "__main__":
    main()
