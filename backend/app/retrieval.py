"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with RRF.

One readable SQL query does the whole thing. Both retrievers return their
top candidates ranked; Reciprocal Rank Fusion (k=60) merges the two rank
lists without any score normalization. The full-text side runs against both
the French and English configurations because the user's question language
is independent of each document's language.
"""

import logging
import time
from dataclasses import dataclass

from pgvector import Vector

from app.db import get_conn
from app.mistral import embed_texts

log = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 50

HYBRID_SQL = """
WITH vec AS (
    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> %(qvec)s) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE %(company)s::text IS NULL OR d.company = %(company)s
    ORDER BY c.embedding <=> %(qvec)s
    LIMIT %(cand)s
),
fts AS (
    SELECT id, row_number() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT c.id, max(ts_rank_cd(c.tsv, q.query)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN LATERAL (
            VALUES (websearch_to_tsquery('french', %(q)s)),
                   (websearch_to_tsquery('english', %(q)s))
        ) AS q(query)
        WHERE c.tsv @@ q.query
          AND (%(company)s::text IS NULL OR d.company = %(company)s)
        GROUP BY c.id
        ORDER BY score DESC
        LIMIT %(cand)s
    ) ranked
),
fused AS (
    SELECT COALESCE(vec.id, fts.id) AS id,
           COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0)
         + COALESCE(1.0 / (%(rrf_k)s + fts.rank), 0) AS rrf_score
    FROM vec FULL OUTER JOIN fts USING (id)
)
SELECT c.id, d.company, d.title, d.fiscal_year, c.page_start, c.page_end,
       c.section_path, c.content, fused.rrf_score
FROM fused
JOIN chunks c ON c.id = fused.id
JOIN documents d ON d.id = c.document_id
ORDER BY fused.rrf_score DESC
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    chunk_id: int
    company: str
    title: str
    fiscal_year: int
    page_start: int
    page_end: int
    section_path: str
    content: str
    score: float


def pages_label(page_start: int, page_end: int) -> str:
    return f"p. {page_start}" if page_start == page_end else f"p. {page_start}-{page_end}"


def hybrid_search(query: str, company: str | None = None, k: int = 10) -> list[RetrievedChunk]:
    t0 = time.monotonic()
    qvec = embed_texts([query])[0]
    params = {
        "q": query,
        "qvec": Vector(qvec),
        "company": company,
        "cand": CANDIDATES_PER_RETRIEVER,
        "rrf_k": RRF_K,
        "k": k,
    }
    with get_conn() as conn:
        rows = conn.execute(HYBRID_SQL, params).fetchall()
    log.info("search %r company=%s: %d results in %d ms", query, company, len(rows), (time.monotonic() - t0) * 1000)
    return [RetrievedChunk(*row) for row in rows]
