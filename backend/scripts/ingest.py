"""Ingest one annual report: OCR -> chunk -> embed -> Postgres.

Usage:
  uv run python -m scripts.ingest ../data/filings/totalenergies-2025.pdf \\
      --slug totalenergies-2025 --company TotalEnergies \\
      --title "Document d'enregistrement universel" --year 2025 --language fr \\
      --source-url https://...

Idempotent: re-running for the same slug replaces the document and its chunks.
OCR results are cached on disk, so a re-run only re-embeds (cents, not dollars).
"""

import argparse
from pathlib import Path

from pgvector import Vector

from app.db import get_conn
from app.ingest.chunk import chunk_pages
from app.ingest.ocr import run_ocr
from app.mistral import embed_texts


def embedding_input(company: str, title: str, year: int, section_path: str, content: str) -> str:
    """Deterministic contextual prefix: cheap version of contextual retrieval."""
    prefix = f"{company} - {title} ({year})"
    if section_path:
        prefix += f" > {section_path}"
    return f"{prefix}\n\n{content}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--language", choices=["fr", "en"], required=True)
    parser.add_argument("--source-url", default=None)
    args = parser.parse_args()

    print(f"[1/4] OCR {args.pdf.name} (cached if already done)...")
    pages = run_ocr(args.pdf)
    print(f"      {len(pages)} pages")

    print("[2/4] Chunking...")
    chunks = chunk_pages(pages)
    sizes = [len(c.content) for c in chunks]
    print(f"      {len(chunks)} chunks (avg {sum(sizes) // len(sizes)} chars)")

    print("[3/4] Embedding...")
    inputs = [
        embedding_input(args.company, args.title, args.year, c.section_path, c.content)
        for c in chunks
    ]
    vectors = embed_texts(inputs)

    print("[4/4] Writing to Postgres...")
    ts_config = "french" if args.language == "fr" else "english"
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE slug = %s", (args.slug,))
        doc_id = conn.execute(
            """INSERT INTO documents (slug, company, title, fiscal_year, language, pages, source_url)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (args.slug, args.company, args.title, args.year, args.language, len(pages), args.source_url),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO chunks (document_id, page_start, page_end, section_path, content, embedding, tsv)
                    VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('{ts_config}', %s))""",
                [
                    (doc_id, c.page_start, c.page_end, c.section_path, c.content, Vector(v), c.content)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        conn.commit()
    print(f"done: document {args.slug} (id {doc_id}), {len(chunks)} chunks")


if __name__ == "__main__":
    main()
