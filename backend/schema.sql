CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    slug        text UNIQUE NOT NULL,        -- e.g. "totalenergies-2025"
    company     text NOT NULL,
    title       text NOT NULL,
    fiscal_year int  NOT NULL,
    language    text NOT NULL CHECK (language IN ('fr', 'en')),
    pages       int  NOT NULL,
    source_url  text
);

CREATE TABLE IF NOT EXISTS chunks (
    id           serial PRIMARY KEY,
    document_id  int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_start   int NOT NULL,
    page_end     int NOT NULL,
    section_path text NOT NULL DEFAULT '',
    content      text NOT NULL,
    embedding    vector(1024) NOT NULL,       -- mistral-embed
    tsv          tsvector NOT NULL            -- french/english config chosen per document at insert time
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
