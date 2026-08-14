import json
import os
import queue
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import answer_question
from app.config import DATA_DIR
from app.db import get_conn

app = FastAPI(title="Le Stagiaire")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

TRACES_FILE = DATA_DIR / "traces.jsonl"
EVALS_FILE = DATA_DIR / "evals" / "results.json"

# Demo limits. This runs on a public URL against a personal API key, and every
# question costs about a cent, so the app caps what a stranger can spend.
MAX_QUESTION_CHARS = 400
DAILY_BUDGET = int(os.environ.get("DEMO_DAILY_BUDGET", "150"))
HOURLY_PER_IP = int(os.environ.get("DEMO_HOURLY_PER_IP", "12"))
MAX_CONCURRENT = int(os.environ.get("DEMO_MAX_CONCURRENT", "3"))

_guard_lock = threading.Lock()
_slots = threading.Semaphore(MAX_CONCURRENT)
_budget_day = ""
_budget_used = 0
_ip_hits: dict[str, deque] = defaultdict(deque)


class AskRequest(BaseModel):
    question: str


def _client_ip(request: Request) -> str:
    """Behind Caddy, the socket address is the proxy: read the forwarded one."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _claim_budget(ip: str) -> str | None:
    """Reserve one question. Returns None when allowed, else why it was refused."""
    global _budget_day, _budget_used
    now = time.time()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _guard_lock:
        if today != _budget_day:
            _budget_day, _budget_used = today, 0
            _ip_hits.clear()
        if _budget_used >= DAILY_BUDGET:
            return ("This demo has used up its question budget for today. It answers on a "
                    "personal Mistral API key, so daily spending is capped. Try again tomorrow.")
        hits = _ip_hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= HOURLY_PER_IP:
            return (f"That is {HOURLY_PER_IP} questions within an hour from this address, which is "
                    "the demo limit. Try again a little later.")
        hits.append(now)
        _budget_used += 1
        return None


def _notice(message: str) -> StreamingResponse:
    """A refused question is not a failure: send it as a 'notice' event so the
    UI shows a message instead of an error."""
    payload = json.dumps({"type": "notice", "message": message}, ensure_ascii=False)
    return StreamingResponse(iter([f"data: {payload}\n\n"]), media_type="text/event-stream")


@app.get("/api/documents")
def documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT d.company, d.title, d.fiscal_year, d.language, d.pages, count(c.id)
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY d.company"""
        ).fetchall()
    return [
        {"company": r[0], "title": r[1], "fiscal_year": r[2], "language": r[3], "pages": r[4], "chunks": r[5]}
        for r in rows
    ]


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> StreamingResponse:
    """SSE stream: 'search' and streamed 'token' events while the agent works
    ('reset' discards tokens that preceded a tool call), one final 'answer' event."""
    question = req.question.strip()
    if not question:
        return _notice("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        return _notice(f"Questions are capped at {MAX_QUESTION_CHARS} characters here.")
    if not _slots.acquire(blocking=False):
        return _notice("The demo is busy answering other questions. Give it a few seconds.")
    refused = _claim_budget(_client_ip(request))
    if refused:
        _slots.release()
        return _notice(refused)

    events: queue.Queue = queue.Queue()
    done = object()

    def work() -> None:
        started = time.time()
        try:
            result = answer_question(question, on_event=events.put)
            record = {
                "ts": started,
                "question": question,
                "answer": result.answer,
                "citations": result.citations,
                "trace": [asdict(t) for t in result.trace],
                "iterations": result.iterations,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "capped": result.capped,
                "duration_s": round(time.time() - started, 2),
            }
            TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with TRACES_FILE.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            events.put({"type": "answer", **{k: record[k] for k in
                        ("answer", "citations", "trace", "iterations",
                         "prompt_tokens", "completion_tokens", "capped", "duration_s")}})
        except Exception as exc:
            events.put({"type": "error", "message": str(exc)})
        finally:
            _slots.release()
            events.put(done)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is done:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/filings/{slug}")
def filing(slug: str) -> FileResponse:
    """Serve a source PDF so the corpus cards can open the real filing."""
    safe = re.sub(r"[^a-z0-9-]", "", slug.lower())
    path = DATA_DIR / "filings" / f"{safe}-2025.pdf"
    if not safe or not path.exists():
        raise HTTPException(status_code=404, detail="unknown filing")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/health")
def health() -> dict:
    """Cheap liveness probe: the app is only useful if the corpus answers."""
    with get_conn() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": chunks}


@app.get("/api/evals")
def evals() -> dict:
    if EVALS_FILE.exists():
        return json.loads(EVALS_FILE.read_text())
    return {"status": "not_run"}


# In production the built frontend is served by the same process.
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
