import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


class AskRequest(BaseModel):
    question: str


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
def ask(req: AskRequest) -> StreamingResponse:
    """SSE stream: 'search' and streamed 'token' events while the agent works
    ('reset' discards tokens that preceded a tool call), one final 'answer' event."""
    events: queue.Queue = queue.Queue()
    done = object()

    def work() -> None:
        started = time.time()
        try:
            result = answer_question(req.question, on_event=events.put)
            record = {
                "ts": started,
                "question": req.question,
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
            events.put(done)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is done:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/evals")
def evals() -> dict:
    if EVALS_FILE.exists():
        return json.loads(EVALS_FILE.read_text())
    return {"status": "not_run"}


# In production the built frontend is served by the same process.
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
