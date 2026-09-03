"""The analyst agent: a plain tool-calling loop around hybrid retrieval.

One tool, several searches allowed, hard iteration cap, grounding and
citation rules enforced by the system prompt, citations resolved
server-side from the chunk ids the model actually received.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings
from app.db import get_conn
from app.mistral import get_client
from app.retrieval import RetrievedChunk, hybrid_search, pages_label

log = logging.getLogger(__name__)

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
                "query": {
                    "type": "string",
                    "description": "Search query, in the language of the target document when known",
                },
                "company": {
                    "type": "string",
                    "description": "Restrict to one company (exact name), omit to search all",
                },
            },
            "required": ["query"],
        },
    },
}

# Citation markers appear standalone [c123] or grouped [c123, c456].
_MARKER_GROUP_RE = re.compile(r"\[([^\]]*?c\d+[^\]]*?)\]")
_CHUNK_ID_RE = re.compile(r"c(\d+)")

EventCallback = Callable[[dict], None]


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
    # every chunk the agent saw in search results (not only the cited ones),
    # needed to compute retrieval recall in the evals
    retrieved: list[dict] = field(default_factory=list)


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[dict]
    prompt_tokens: int = 0
    completion_tokens: int = 0


@lru_cache(maxsize=1)
def _list_companies() -> tuple[str, ...]:
    """The corpus only changes when a filing is ingested, which restarts nothing: cached per process."""
    with get_conn() as conn:
        return tuple(r[0] for r in conn.execute("SELECT DISTINCT company FROM documents ORDER BY company"))


def _format_results(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No results. Try different terms, another language, or drop the company filter."
    parts = []
    for c in chunks:
        header = f"[c{c.chunk_id}] {c.company} {c.fiscal_year}, {pages_label(c.page_start, c.page_end)}"
        if c.section_path:
            header += f" | {c.section_path}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def content_to_text(content) -> str:
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
            refs = "".join(rid if str(rid).startswith("[") else f"[{rid}]" for rid in part.reference_ids)
            parts.append(f" {refs}")
    return "".join(parts)


def resolve_citations(answer: str, seen_chunks: dict[int, RetrievedChunk]) -> list[dict]:
    """Map the [cN] markers of the answer to the chunks the model actually received.
    Unique, in order of first appearance; ids the model never saw are dropped."""
    cited_ids = [int(m) for group in _MARKER_GROUP_RE.findall(answer) for m in _CHUNK_ID_RE.findall(group)]
    citations = []
    for cid in dict.fromkeys(cited_ids):
        c = seen_chunks.get(cid)
        if c is None:
            continue
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


class ToolCallAccumulator:
    """Rebuild complete tool calls from streamed deltas (id, name and argument fragments
    arrive across several chunks, keyed by index)."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}

    def add(self, delta_tool_calls) -> None:
        for tc in delta_tool_calls or []:
            idx = getattr(tc, "index", None)
            if idx is None:
                idx = len(self._calls)
            slot = self._calls.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["function"]["name"] = tc.function.name
                args = tc.function.arguments
                if isinstance(args, dict):
                    slot["function"]["arguments"] += json.dumps(args)
                elif args:
                    slot["function"]["arguments"] += args

    def calls(self) -> list[dict]:
        return [self._calls[i] for i in sorted(self._calls)]


def _model_kwargs(messages: list, tool_choice: str) -> dict:
    return {
        "model": settings.agent_model,
        "messages": messages,
        "tools": [SEARCH_TOOL],
        "tool_choice": tool_choice,
        "temperature": TEMPERATURE,
    }


def _stream_turn(messages: list, tool_choice: str, on_token: Callable[[str], None] | None) -> ModelTurn:
    turn = ModelTurn(text="", tool_calls=[])
    text_parts: list[str] = []
    calls = ToolCallAccumulator()
    with get_client().chat.stream(**_model_kwargs(messages, tool_choice)) as stream:
        for event in stream:
            chunk = event.data
            if chunk.usage:
                turn.prompt_tokens = chunk.usage.prompt_tokens or 0
                turn.completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = content_to_text(delta.content) if delta.content else ""
            if piece:
                text_parts.append(piece)
                if on_token:
                    on_token(piece)
            calls.add(delta.tool_calls)
    turn.text = "".join(text_parts).strip()
    turn.tool_calls = calls.calls()
    return turn


def _complete_turn(messages: list, tool_choice: str) -> ModelTurn:
    resp = get_client().chat.complete(**_model_kwargs(messages, tool_choice))
    msg = resp.choices[0].message
    if msg is None:
        raise RuntimeError("chat completion returned no message")
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in (msg.tool_calls or [])
    ]
    turn = ModelTurn(text=content_to_text(msg.content).strip(), tool_calls=tool_calls)
    if resp.usage:
        turn.prompt_tokens = resp.usage.prompt_tokens or 0
        turn.completion_tokens = resp.usage.completion_tokens or 0
    return turn


def _model_turn(messages: list, tool_choice: str, on_token: Callable[[str], None] | None) -> ModelTurn:
    """Streamed call, with a non-streaming fallback if the stream drops mid-way."""
    try:
        return _stream_turn(messages, tool_choice, on_token)
    except Exception:
        log.warning("streamed call failed, falling back to a non-streaming request", exc_info=True)
        return _complete_turn(messages, tool_choice)


def _run_search(tool_call: dict, seen_chunks: dict[int, RetrievedChunk], on_event: EventCallback | None):
    """Execute one search_filings call. Returns (tool message, trace record)."""
    args = json.loads(tool_call["function"]["arguments"])
    query, company = args.get("query", ""), args.get("company")
    if on_event:
        on_event({"type": "search", "query": query, "company": company})
    t0 = time.monotonic()
    try:
        chunks = hybrid_search(query, company=company, k=RESULTS_PER_SEARCH)
        content = _format_results(chunks)
        seen_chunks.update((c.chunk_id, c) for c in chunks)
    except Exception as exc:
        # the model gets the error as a tool result and can rephrase or give up
        log.warning("search failed for %r: %s", query, exc)
        chunks, content = [], json.dumps({"error": str(exc)})
    record = ToolCallRecord(
        query=query, company=company, n_results=len(chunks), latency_ms=int((time.monotonic() - t0) * 1000)
    )
    message = {
        "role": "tool",
        "name": tool_call["function"]["name"],
        "content": content,
        "tool_call_id": tool_call["id"],
    }
    return message, record


def answer_question(question: str, on_event: EventCallback | None = None) -> AgentAnswer:
    """Run the agent loop. on_event receives progress dicts (for SSE streaming):
    'search' when a tool call fires, 'token' for streamed answer text, 'reset'
    when streamed text turns out to be pre-tool-call preamble to discard."""
    seen_chunks: dict[int, RetrievedChunk] = {}
    result = AgentAnswer(answer="", citations=[])
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT.format(companies=", ".join(_list_companies()))},
        {"role": "user", "content": question},
    ]
    on_token = (lambda t: on_event({"type": "token", "text": t})) if on_event else None

    for iteration in range(MAX_ITERATIONS + 1):
        result.iterations = iteration + 1
        capped = iteration == MAX_ITERATIONS
        turn = _model_turn(messages, "none" if capped else "auto", on_token)
        result.prompt_tokens += turn.prompt_tokens
        result.completion_tokens += turn.completion_tokens
        log.info(
            "iteration %d: %d tool calls, %d chars of text",
            iteration + 1,
            len(turn.tool_calls),
            len(turn.text),
        )

        if not turn.tool_calls:
            result.answer = turn.text
            result.capped = capped
            break

        # any text streamed before the tool calls was preamble, not the answer
        if turn.text and on_event:
            on_event({"type": "reset"})
        messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})
        for tc in turn.tool_calls:
            message, record = _run_search(tc, seen_chunks, on_event)
            messages.append(message)
            result.trace.append(record)

    result.citations = resolve_citations(result.answer, seen_chunks)
    result.retrieved = [
        {"company": c.company, "page_start": c.page_start, "page_end": c.page_end} for c in seen_chunks.values()
    ]
    return result
