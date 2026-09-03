import json
import logging
import queue
import re
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import answer_question
from app.config import DATA_DIR, FILINGS_DIR, FRONTEND_DIST, settings
from app.db import close_pool, get_conn
from app.guardrails import DemoGuard
from app.logs import setup_logging

setup_logging()
log = logging.getLogger(__name__)

TRACES_FILE = DATA_DIR / "traces.jsonl"
EVALS_FILE = DATA_DIR / "evals" / "results.json"
MAX_QUESTION_CHARS = 400

guard = DemoGuard(settings.demo_daily_budget, settings.demo_hourly_per_ip, settings.demo_max_concurrent)
_traces_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    close_pool()


app = FastAPI(title="Le Stagiaire", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str


def _client_ip(request: Request) -> str:
    """Behind Caddy, the socket address is the proxy: read the forwarded one."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _notice(message: str) -> StreamingResponse:
    """A refused question is not a failure: send it as a 'notice' event so the
    UI shows a message instead of an error."""
    return StreamingResponse(iter([_sse({"type": "notice", "message": message})]), media_type="text/event-stream")


def _append_trace(record: dict) -> None:
    TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _traces_lock, TRACES_FILE.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _answer_in_background(question: str, events: queue.Queue) -> None:
    """Run the agent in a worker thread, pushing progress then the final answer onto the queue.
    The agent and the Mistral SDK are synchronous, so this keeps the event loop free."""
    started = time.time()
    try:
        result = answer_question(question, on_event=events.put)
        record = {
            "answer": result.answer,
            "citations": result.citations,
            "trace": [asdict(t) for t in result.trace],
            "iterations": result.iterations,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "capped": result.capped,
            "duration_s": round(time.time() - started, 2),
        }
        _append_trace({"ts": started, "question": question, **record})
        events.put({"type": "answer", **record})
    except Exception as exc:
        log.exception("agent failed on %r", question)
        events.put({"type": "error", "message": str(exc)})
    finally:
        guard.release()
        events.put(None)


def _drain(events: queue.Queue) -> Iterator[str]:
    while (event := events.get()) is not None:
        yield _sse(event)


@app.get("/api/documents")
def documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT d.company, d.title, d.fiscal_year, d.language, d.pages, count(c.id)
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY d.company"""
        ).fetchall()
    keys = ("company", "title", "fiscal_year", "language", "pages", "chunks")
    return [dict(zip(keys, row, strict=True)) for row in rows]


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> StreamingResponse:
    """SSE stream: 'search' and streamed 'token' events while the agent works
    ('reset' discards tokens that preceded a tool call), one final 'answer' event."""
    question = req.question.strip()
    if not question:
        return _notice("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        return _notice(f"Questions are capped at {MAX_QUESTION_CHARS} characters here.")
    refused = guard.acquire(_client_ip(request))
    if refused:
        log.info("refused question from %s: %s", _client_ip(request), refused.split(".")[0])
        return _notice(refused)

    events: queue.Queue = queue.Queue()
    threading.Thread(target=_answer_in_background, args=(question, events), daemon=True).start()
    return StreamingResponse(_drain(events), media_type="text/event-stream")


@app.get("/api/filings/{slug}")
def filing(slug: str) -> FileResponse:
    """Serve a source PDF so the corpus cards can open the real filing."""
    safe = re.sub(r"[^a-z0-9-]", "", slug.lower())
    path = FILINGS_DIR / f"{safe}-2025.pdf"
    if not safe or not path.exists():
        raise HTTPException(status_code=404, detail="unknown filing")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/health")
def health() -> dict:
    """Cheap liveness probe: the app is only useful if the corpus answers."""
    with get_conn() as conn:
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    return {"status": "ok", "chunks": row[0] if row else 0}


@app.get("/api/evals")
def evals() -> dict:
    if EVALS_FILE.exists():
        return json.loads(EVALS_FILE.read_text())
    return {"status": "not_run"}


# In production the built frontend is served by the same process.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
