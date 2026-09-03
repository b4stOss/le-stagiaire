"""The analyst agent: a plain tool-calling loop around hybrid retrieval.

One tool, several searches allowed, hard iteration cap, grounding and
citation rules enforced by the system prompt, citations resolved
server-side from the chunk ids the model actually received.

Read top-down: the loop, then one model turn, then the streaming details,
then the search tool, then the helpers.
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

# Progress events for the UI: {"type": "search" | "token" | "reset", ...}
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
    """What one call to the model produced, whether it was streamed or not."""

    text: str
    tool_calls: list[dict]
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ------------------------------------------------------------------ the loop


def answer_question(question: str, on_event: EventCallback | None = None) -> AgentAnswer:
    """Ask the model; while it asks for searches, run them and ask again; stop at its first
    plain answer or at the iteration cap. on_event streams progress to the UI:
    'search' when a tool call fires, 'token' for answer text, 'reset' when streamed
    text turns out to be preamble before a tool call."""
    seen_chunks: dict[int, RetrievedChunk] = {}
    result = AgentAnswer(answer="", citations=[])
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT.format(companies=", ".join(_list_companies()))},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 2):
        capped = iteration > MAX_ITERATIONS
        turn = _model_turn(messages, tool_choice="none" if capped else "auto", on_event=on_event)
        result.iterations = iteration
        result.prompt_tokens += turn.prompt_tokens
        result.completion_tokens += turn.completion_tokens
        log.info("iteration %d: %d tool calls, %d chars of text", iteration, len(turn.tool_calls), len(turn.text))

        if not turn.tool_calls:
            result.answer = turn.text
            result.capped = capped
            break

        if turn.text and on_event:
            on_event({"type": "reset"})  # text before a tool call is thinking aloud, not the answer
        messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})
        for tool_call in turn.tool_calls:
            tool_message, record = _run_search(tool_call, seen_chunks, on_event)
            messages.append(tool_message)
            result.trace.append(record)

    result.citations = resolve_citations(result.answer, seen_chunks)
    result.retrieved = [
        {"company": c.company, "page_start": c.page_start, "page_end": c.page_end} for c in seen_chunks.values()
    ]
    return result


# ------------------------------------------------------------------ one model turn


def _model_turn(messages: list, tool_choice: str, on_event: EventCallback | None) -> ModelTurn:
    """Streamed call so the UI gets tokens as they come; if the stream drops mid-way,
    the same call is replayed without streaming."""
    try:
        return _stream_turn(messages, tool_choice, on_event)
    except Exception:
        log.warning("streamed call failed, replaying without streaming", exc_info=True)
        return _complete_turn(messages, tool_choice)


def _model_kwargs(messages: list, tool_choice: str) -> dict:
    return {
        "model": settings.agent_model,
        "messages": messages,
        "tools": [SEARCH_TOOL],
        "tool_choice": tool_choice,
        "temperature": TEMPERATURE,
    }


def _complete_turn(messages: list, tool_choice: str) -> ModelTurn:
    resp = get_client().chat.complete(**_model_kwargs(messages, tool_choice))
    msg = resp.choices[0].message
    if msg is None:
        raise RuntimeError("chat completion returned no message")
    tool_calls = [
        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in (msg.tool_calls or [])
    ]
    return ModelTurn(
        text=content_to_text(msg.content).strip(),
        tool_calls=tool_calls,
        prompt_tokens=(resp.usage.prompt_tokens or 0) if resp.usage else 0,
        completion_tokens=(resp.usage.completion_tokens or 0) if resp.usage else 0,
    )


# ------------------------------------------------------------------ streaming
#
# A streamed response is a sequence of chunks, each carrying a delta: a fragment of
# text, and/or fragments of tool calls. Text fragments are forwarded to the UI as they
# arrive. Tool calls arrive in pieces too (id in one chunk, name in another, the JSON
# arguments spread over several), interleaved when there are several calls; each piece
# carries the index of the call it belongs to, which is what the accumulator keys on.


def _stream_turn(messages: list, tool_choice: str, on_event: EventCallback | None) -> ModelTurn:
    text_parts: list[str] = []
    calls = ToolCallAccumulator()
    prompt_tokens = completion_tokens = 0
    with get_client().chat.stream(**_model_kwargs(messages, tool_choice)) as stream:
        for event in stream:
            chunk = event.data
            if chunk.usage:  # sent once, on the last chunk
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = content_to_text(delta.content)
            if piece:
                text_parts.append(piece)
                if on_event:
                    on_event({"type": "token", "text": piece})
            calls.add(delta.tool_calls)
    return ModelTurn(
        text="".join(text_parts).strip(),
        tool_calls=calls.calls(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class ToolCallAccumulator:
    """Rebuild complete tool calls from streamed fragments, keyed by call index."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}

    def add(self, fragments) -> None:
        for fragment in fragments or []:
            index = fragment.index if fragment.index is not None else len(self._calls)
            call = self._calls.setdefault(
                index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if fragment.id:
                call["id"] = fragment.id
            if fragment.function and fragment.function.name:
                call["function"]["name"] = fragment.function.name
            if fragment.function and fragment.function.arguments:
                args = fragment.function.arguments
                call["function"]["arguments"] += json.dumps(args) if isinstance(args, dict) else args

    def calls(self) -> list[dict]:
        return [self._calls[i] for i in sorted(self._calls)]


def content_to_text(content) -> str:
    """Model content is either a string, or a list of parts (text / reference) in
    Mistral's native citation format. Flatten both to text with inline [cN] markers."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.type == "text":
            parts.append(part.text)
        elif part.type == "reference":
            parts.append(" " + "".join(rid if rid.startswith("[") else f"[{rid}]" for rid in part.reference_ids))
    return "".join(parts)


# ------------------------------------------------------------------ the search tool


def _run_search(tool_call: dict, seen_chunks: dict[int, RetrievedChunk], on_event: EventCallback | None):
    """Run one search_filings call. Returns the tool message for the model and a trace record."""
    args = json.loads(tool_call["function"]["arguments"])
    query, company = args.get("query", ""), args.get("company")
    if on_event:
        on_event({"type": "search", "query": query, "company": company})

    t0 = time.monotonic()
    try:
        chunks = hybrid_search(query, company=company, k=RESULTS_PER_SEARCH)
        seen_chunks.update((c.chunk_id, c) for c in chunks)
        content = _format_results(chunks)
    except Exception as exc:
        log.warning("search failed for %r: %s", query, exc)
        chunks, content = [], json.dumps({"error": str(exc)})  # the model can rephrase or give up
    latency_ms = int((time.monotonic() - t0) * 1000)

    tool_message = {"role": "tool", "name": "search_filings", "content": content, "tool_call_id": tool_call["id"]}
    record = ToolCallRecord(query=query, company=company, n_results=len(chunks), latency_ms=latency_ms)
    return tool_message, record


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


# ------------------------------------------------------------------ helpers


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


@lru_cache(maxsize=1)
def _list_companies() -> tuple[str, ...]:
    """The corpus only changes when a filing is ingested, which restarts the app: cached per process."""
    with get_conn() as conn:
        return tuple(r[0] for r in conn.execute("SELECT DISTINCT company FROM documents ORDER BY company"))
