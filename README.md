# Le Stagiaire

**Status: work in progress (pipeline, agent, evals and UI built; deployment pending).**

An analyst agent for annual reports, built end-to-end on Mistral's platform. Named after the junior who usually gets handed the 800-page filing: it reads everything, cites the page for every figure, and unlike a real intern it says "I don't know" instead of improvising.

## The problem

Listed companies publish annual reports (universal registration documents) of 300 to 800 pages. Credit analysts, equity analysts, M&A juniors and financial journalists spend hours locating and cross-referencing facts inside them: net debt, maturity schedules, risk factors, geographic exposure, litigation. The work is retrieval and synthesis over massive unstructured text, which is exactly what LLMs do well. The judgment stays human; the document search should not.

The catch in finance: an answer without a verifiable source is worthless, and a system that invents numbers is worse than no system. So the two hard requirements are page-level citations and the ability to refuse when the information is not in the documents. That is also why this prototype ships with an evaluation harness, not just a demo.

## What it does

Ask questions in natural language over the latest annual reports of four European companies (TotalEnergies, BNP Paribas, Stellantis, ASML - a deliberate mix of sectors and of French and English filings):

- Point lookups: "What is TotalEnergies' average debt maturity?"
- Synthesis: "Summarize Stellantis' climate-related risk factors."
- Multi-document comparisons: "Compare the net debt of TotalEnergies and Stellantis."

Every claim is cited with company and page. When the answer is not in the corpus, the agent says so instead of guessing.

## How it works

Pipeline (all models served by Mistral's La Plateforme):

1. **Ingestion**: PDF filings parsed with Mistral OCR into page-anchored markdown (tables preserved).
2. **Chunking**: structure-aware splitting, metadata kept per chunk (company, year, page, section).
3. **Indexing**: Mistral embeddings stored in Postgres + pgvector; hybrid retrieval (vector similarity + full-text) because financial questions mix semantics and exact figures.
4. **Agent**: a tool-calling loop around a `search_filings` tool. Simple questions resolve in one search; comparisons decompose into several. System prompt enforces grounding, citations, and refusal. The API streams the whole run over SSE: each search as it fires, then the answer token by token.
5. **Evaluation**: a hand-verified golden set of ~30 questions across four categories (numeric extraction, synthesis, cross-document comparison, unanswerable). Scored with exact numeric matching plus an LLM judge for text answers and citation accuracy. Results are displayed in the app.

Backend: Python / FastAPI, direct `mistralai` SDK calls (no RAG framework). Models: `mistral-medium-latest` (agent), `mistral-small-latest` (eval judge), `mistral-embed`, `mistral-ocr-latest`. Frontend: React. Deployment: Railway.

## Deliberate non-choices

Scoping means saying no. Each of these was considered and skipped for this corpus size (4 documents, ~10k chunks), with the upgrade path known: Mistral's managed document library (would black-box the retrieval this project is meant to demonstrate), rerankers, GraphRAG, semantic/late chunking, ColBERT-style multi-vector retrieval, dedicated vector databases, vector quantization, BM25 extensions over native Postgres full-text, eval frameworks (a transparent 200-line runner beats opaque generic metrics at n=30), and observability platforms (traces are plain JSONL, rendered in the app).

## Why these four companies

They are publicly reported Mistral customers or partners. If the point is to show what an applied AI prototype on Mistral's stack looks like for Mistral's actual market, the corpus should look like that market.

## Roadmap

- [ ] Day 1: ingestion pipeline (OCR, chunking, embeddings), retrieval working from the CLI
- [ ] Day 2: agent loop, hybrid search, web UI with citations
- [ ] Day 3: golden set, eval runner, evals page, deployment

Architecture decisions, eval results and honest limitations will be documented here as they land.
