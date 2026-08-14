"""The analyst agent: a plain tool-calling loop around hybrid retrieval.

One tool, several searches allowed, hard iteration cap, grounding and
citation rules enforced by the system prompt, citations resolved
server-side from the chunk ids the model actually received.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from app.config import settings
from app.db import get_conn
from app.mistral import get_client
from app.retrieval import RetrievedChunk, hybrid_search

MAX_ITERATIONS = 8
TEMPERATURE = 0.1
RESULTS_PER_SEARCH = 8

SYSTEM_PROMPT = """\
You are a financial analyst assistant answering questions about the latest annual \
reports of {companies}. Your only source of truth is the search_filings tool.

Rules:
- Base every statement on search results. Never use outside knowledge for facts or figures.
- Cite the source of every claim by appending the chunk marker, e.g. [c123], right after it. \
Only cite chunk ids that appeared in your search results.
- Copy figures exactly as written in the source (same unit, same scale).
- For comparisons across companies, search each company separately.
- If the information is not in the documents after a few well-chosen searches, say clearly \
that it is not in the reports. Never guess.
- Answer in the language of the question. Be concise and precise.\
"""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_filings",
        "description": (
            "Hybrid search (semantic + keyword) over the annual reports. "
            "Returns the most relevant excerpts with their chunk id, company, pages and section. "
            "Reformulate and search again if results look off-topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, in the language of the target document when known"},
                "company": {
                    "type": "string",
                    "description": "Restrict to one company (exact name), omit to search all",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ToolCallRecord:
    query: str
    company: str | None
    n_results: int
    latency_ms: int


@dataclass
class AgentAnswer:
    answer: str
    citations: list[dict]
    trace: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    capped: bool = False


def _list_companies() -> list[str]:
    with get_conn() as conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT company FROM documents ORDER BY company")]


def _format_results(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No results. Try different terms, another language, or drop the company filter."
    parts = []
    for c in chunks:
        pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"p. {c.page_start}-{c.page_end}"
        header = f"[c{c.chunk_id}] {c.company} {c.fiscal_year}, {pages}"
        if c.section_path:
            header += f" | {c.section_path}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def _normalize_content(content) -> str:
    """The model may answer as plain text with [cN] markers, or as a list of
    TextChunk/ReferenceChunk parts (Mistral's native citation format, where the
    reference ids are the chunk markers we exposed). Normalize both to text
    with inline [cN] markers."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content or []:
        kind = getattr(part, "type", None)
        if kind == "text":
            parts.append(part.text)
        elif kind == "reference":
            refs = "".join(f"[{rid}]" if not str(rid).startswith("[") else str(rid) for rid in part.reference_ids)
            parts.append(f" {refs}")
    return "".join(parts).strip()


def _resolve_citations(answer: str, seen_chunks: dict[int, RetrievedChunk]) -> list[dict]:
    # markers appear standalone [c123] or grouped [c123, c456]
    cited_ids = [
        int(m)
        for group in re.findall(r"\[([^\]]*?c\d+[^\]]*?)\]", answer)
        for m in re.findall(r"c(\d+)", group)
    ]
    citations = []
    for cid in dict.fromkeys(cited_ids):  # unique, in order of appearance
        c = seen_chunks.get(cid)
        if c is None:
            continue  # model cited an id it never received: drop it
        citations.append(
            {
                "chunk_id": cid,
                "company": c.company,
                "title": c.title,
                "fiscal_year": c.fiscal_year,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_path": c.section_path,
                "quote": c.content,
            }
        )
    return citations


def answer_question(
    question: str,
    on_event: Callable[[dict], None] | None = None,
) -> AgentAnswer:
    """Run the agent loop. on_event receives progress dicts (for SSE streaming)."""
    client = get_client()
    companies = _list_companies()
    seen_chunks: dict[int, RetrievedChunk] = {}
    result = AgentAnswer(answer="", citations=[])

    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT.format(companies=", ".join(companies))},
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS + 1):
        result.iterations = iteration + 1
        capped = iteration == MAX_ITERATIONS
        resp = client.chat.complete(
            model=settings.agent_model,
            messages=messages,
            tools=[SEARCH_TOOL],
            tool_choice="none" if capped else "auto",
            temperature=TEMPERATURE,
        )
        if resp.usage:
            result.prompt_tokens += resp.usage.prompt_tokens or 0
            result.completion_tokens += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            result.answer = _normalize_content(msg.content)
            result.capped = capped
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            query, company = args.get("query", ""), args.get("company")
            if on_event:
                on_event({"type": "search", "query": query, "company": company})
            t0 = time.monotonic()
            try:
                chunks = hybrid_search(query, company=company, k=RESULTS_PER_SEARCH)
                content = _format_results(chunks)
                for c in chunks:
                    seen_chunks[c.chunk_id] = c
            except Exception as exc:
                chunks, content = [], json.dumps({"error": str(exc)})
            result.trace.append(
                ToolCallRecord(
                    query=query,
                    company=company,
                    n_results=len(chunks),
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            )
            messages.append(
                {"role": "tool", "name": tc.function.name, "content": content, "tool_call_id": tc.id}
            )

    result.citations = _resolve_citations(result.answer, seen_chunks)
    return result
