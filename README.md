# Le Stagiaire

**Live demo: [lestagiaire.genod.ch](https://lestagiaire.genod.ch)**

An analyst agent for annual reports, built end to end on Mistral's platform. Named after the junior who gets handed the 800-page filing: it reads everything, cites the page for every figure, and unlike a real intern it says "I don't know" instead of improvising.

## The problem

Listed companies publish annual reports of 300 to 800 pages. Analysts spend hours locating and cross-referencing facts inside them: net debt, maturity schedules, risk factors, litigation. That is retrieval and synthesis over unstructured text, which models do well. The judgment stays human; the document search should not.

In finance, an answer without a verifiable source is worthless, and a system that invents numbers is worse than no system. Hence the two hard requirements: page-level citations, and refusal when the answer is not in the documents. That is also why this ships with an eval harness rather than a demo alone.

## What it does

Questions in French or English over the FY2025 filings of TotalEnergies, BNP Paribas, Stellantis and ASML (2,471 pages, mixed sectors and languages; all four are publicly reported Mistral customers or partners).

- Point lookups: "What is TotalEnergies' average debt maturity?"
- Synthesis: "Summarize Stellantis' climate-related risk factors."
- Cross-document: "Compare the net debt of TotalEnergies and Stellantis."

## How it works

Every model is served by La Plateforme.

1. **Ingestion**: filings parsed by `mistral-ocr-latest` into page-anchored markdown, tables preserved.
2. **Chunking**: structure-aware, with company, year, page and section kept per chunk. 6,232 chunks.
3. **Indexing**: `mistral-embed` vectors in Postgres with pgvector. Retrieval is hybrid, vector similarity fused with Postgres full-text, because financial questions mix semantics ("climate risk") with exact strings ("CET1").
4. **Agent**: `mistral-medium-latest` in a tool-calling loop over one `search_filings` tool. Simple questions resolve in a single search, comparisons decompose into several. The system prompt enforces grounding, citations and refusal. The run streams over SSE: each search as it fires, then the answer token by token.
5. **Evaluation**: a hand-verified golden set, scored by exact numeric matching for figures and `mistral-small-latest` as judge for prose. Results are in the app's Evals tab.

Python and FastAPI, direct `mistralai` SDK calls, no RAG framework. React front end served by the same process. Deployed as one Docker Compose stack (app plus Postgres/pgvector) behind Caddy. Ingesting the corpus cost about $12 in OCR and embeddings; a question costs about a cent and answers in 3 seconds at the median.

## Evals

13 questions, hand-verified against the source PDFs.

| Category | Result |
|---|---|
| Numeric extraction | 7/7 |
| Synthesis | 1/1 |
| Cross-document comparison | 2/2 |
| Unanswerable (must refuse) | 3/3 |
| Citation accuracy | 10/10 |
| Retrieval recall | 10/10 |
| False refusals | 0/10 |

A 100% pass rate at n=13 says more about the set than about the system: it is small, and the questions are ones the pipeline was expected to handle. The honest reading is that it demonstrates the harness works and catches the failure modes that matter (invented figures, wrong pages, refusing an answerable question), not that the system is solved. Growing the set until it fails is the next step, and the point of building the harness first.

## Deliberate non-choices

Scoping means saying no. Each of these was considered and skipped at this corpus size, with the upgrade path known: Mistral's managed document library (it would black-box the retrieval this project exists to show), rerankers, GraphRAG, semantic and late chunking, multi-vector retrieval, a dedicated vector database, vector quantization, BM25 extensions over native Postgres full-text, eval frameworks (a transparent 200-line runner beats opaque generic metrics at n=13), and observability platforms (traces are JSONL, rendered in the app).

## Running it

The filings are not in the repo; they are public downloads from each issuer.

```bash
cp .env.example .env      # add a Mistral API key
docker compose up -d      # Postgres with pgvector
cd backend
uv run python -m scripts.init_db
uv run python -m scripts.ingest asml-2025.pdf \
  --slug asml-2025 --company ASML --title "Annual Report" --year 2025 --language en
uv run uvicorn app.main:app
```

The public demo is rate-limited and capped daily, since every question spends real tokens.

Checks (from `backend/`): `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`. The tests cover the pure parts (chunking, numeric grading, citation resolution, demo guardrails) and need neither the database nor an API key.
