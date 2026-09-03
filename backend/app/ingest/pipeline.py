"""Ingest one filing: OCR, chunk, embed, write to Postgres."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pgvector import Vector
from psycopg import Connection

from app.db import get_conn
from app.ingest.chunk import Chunk, chunk_pages
from app.ingest.ocr import run_ocr
from app.mistral import embed_texts

log = logging.getLogger(__name__)

# Postgres full-text configuration used to build each document's tsvector.
TS_CONFIG = {"fr": "french", "en": "english"}


@dataclass(frozen=True)
class DocumentMeta:
    slug: str
    company: str
    title: str
    fiscal_year: int
    language: str
    source_url: str | None = None


@dataclass(frozen=True)
class IngestReport:
    document_id: int
    pages: int
    chunks: int


def embedding_input(meta: DocumentMeta, chunk: Chunk) -> str:
    """Prefix each chunk with its provenance so the embedding carries company, year and section."""
    prefix = f"{meta.company} - {meta.title} ({meta.fiscal_year})"
    if chunk.section_path:
        prefix += f" > {chunk.section_path}"
    return f"{prefix}\n\n{chunk.content}"


def ingest_document(pdf_path: Path, meta: DocumentMeta) -> IngestReport:
    if meta.language not in TS_CONFIG:
        raise ValueError(f"unsupported language {meta.language!r}, expected one of {sorted(TS_CONFIG)}")

    log.info("[1/4] OCR %s (cached if already done)", pdf_path.name)
    pages = run_ocr(pdf_path)
    log.info("      %d pages", len(pages))

    log.info("[2/4] chunking")
    chunks = chunk_pages(pages)
    avg_chars = sum(len(c.content) for c in chunks) // max(len(chunks), 1)
    log.info("      %d chunks (avg %d chars)", len(chunks), avg_chars)

    log.info("[3/4] embedding")
    vectors = embed_texts([embedding_input(meta, c) for c in chunks])

    log.info("[4/4] writing to Postgres")
    with get_conn() as conn:
        doc_id = _replace_document(conn, meta, pages=len(pages))
        _insert_chunks(conn, doc_id, TS_CONFIG[meta.language], chunks, vectors)
    return IngestReport(document_id=doc_id, pages=len(pages), chunks=len(chunks))


def _replace_document(conn: Connection, meta: DocumentMeta, pages: int) -> int:
    """Delete any previous ingestion of this slug (chunks cascade) and insert the new row."""
    conn.execute("DELETE FROM documents WHERE slug = %s", (meta.slug,))
    row = conn.execute(
        """INSERT INTO documents (slug, company, title, fiscal_year, language, pages, source_url)
           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (meta.slug, meta.company, meta.title, meta.fiscal_year, meta.language, pages, meta.source_url),
    ).fetchone()
    if row is None:
        raise RuntimeError("INSERT ... RETURNING id returned no row")
    return row[0]


def _insert_chunks(
    conn: Connection, doc_id: int, ts_config: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (document_id, page_start, page_end, section_path, content, embedding, tsv)
               VALUES (%s, %s, %s, %s, %s, %s, to_tsvector(%s::regconfig, %s))""",
            [
                (doc_id, c.page_start, c.page_end, c.section_path, c.content, Vector(v), ts_config, c.content)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
