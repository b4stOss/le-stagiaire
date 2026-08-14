import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AgentEvent,
  AnswerEvent,
  Citation,
  DocumentInfo,
  SearchEvent,
  askStream,
  getDocuments,
  getEvals,
} from "./api";

const EXAMPLES = [
  "What was Stellantis' net revenue in 2025, and how did it evolve vs 2024?",
  "Quelle est la dette nette de TotalEnergies fin 2025 ?",
  "Compare the climate-related risk factors of TotalEnergies and BNP Paribas.",
  "How much did ASML spend on R&D in 2025?",
];

/** Replace [c123] / [c123, c456] markers with numbered footnote links. */
function linkCitations(answer: string, citations: Citation[]): string {
  const indexOf = new Map(citations.map((c, i) => [c.chunk_id, i + 1]));
  return answer.replace(/\[([^\]]*?c\d+[^\]]*?)\]/g, (whole, group: string) => {
    const ids = [...group.matchAll(/c(\d+)/g)].map((m) => Number(m[1]));
    const nums = ids.map((id) => indexOf.get(id)).filter(Boolean);
    if (nums.length === 0) return whole;
    return nums.map((n) => `[[${n}]](#cite-${n})`).join(" ");
  });
}

function AgentActivity({ searches, running }: { searches: SearchEvent[]; running: boolean }) {
  if (searches.length === 0 && !running) return null;
  return (
    <div className="activity">
      {searches.map((s, i) => (
        <div key={i} className="activity-line">
          <span className="activity-icon">⌕</span>
          <span>
            {s.query}
            {s.company ? <span className="activity-company"> · {s.company}</span> : null}
          </span>
        </div>
      ))}
      {running && <div className="activity-line pulse">thinking…</div>}
    </div>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      <h3>Sources</h3>
      {citations.map((c, i) => {
        const pages =
          c.page_start === c.page_end ? `p. ${c.page_start}` : `p. ${c.page_start}-${c.page_end}`;
        return (
          <div key={c.chunk_id} className="citation" id={`cite-${i + 1}`}>
            <button className="citation-header" onClick={() => setOpen(open === i ? null : i)}>
              <span className="citation-num">{i + 1}</span>
              <span className="citation-ref">
                {c.company} {c.fiscal_year}, {pages}
              </span>
              {c.section_path && <span className="citation-section">{c.section_path}</span>}
            </button>
            {open === i && <blockquote className="citation-quote">{c.quote}</blockquote>}
          </div>
        );
      })}
    </div>
  );
}

function AskTab({ documents }: { documents: DocumentInfo[] }) {
  const [question, setQuestion] = useState("");
  const [searches, setSearches] = useState<SearchEvent[]>([]);
  const [result, setResult] = useState<AnswerEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const answerRef = useRef<HTMLDivElement>(null);

  const ask = async (q: string) => {
    if (!q.trim() || running) return;
    setQuestion(q);
    setSearches([]);
    setResult(null);
    setError(null);
    setRunning(true);
    try {
      await askStream(q, (event: AgentEvent) => {
        if (event.type === "search") setSearches((prev) => [...prev, event]);
        else if (event.type === "answer") setResult(event);
        else if (event.type === "error") setError(event.message);
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <>
      <div className="corpus">
        {documents.map((d) => (
          <span key={d.company} className="chip" title={`${d.title} · ${d.chunks} chunks`}>
            {d.company} <span className="chip-meta">{d.fiscal_year} · {d.pages} p.</span>
          </span>
        ))}
      </div>

      <form
        className="ask-form"
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything about these annual reports…"
          disabled={running}
        />
        <button type="submit" disabled={running || !question.trim()}>
          {running ? "…" : "Ask"}
        </button>
      </form>

      {!result && !running && (
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example" onClick={() => ask(ex)}>
              {ex}
            </button>
          ))}
        </div>
      )}

      <AgentActivity searches={searches} running={running} />

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="answer" ref={answerRef}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {linkCitations(result.answer, result.citations)}
          </ReactMarkdown>
          <CitationList citations={result.citations} />
          <div className="stats">
            {result.trace.length} searches · {result.iterations} steps ·{" "}
            {result.prompt_tokens + result.completion_tokens} tokens · {result.duration_s}s
            {result.capped ? " · hit iteration cap" : ""}
          </div>
        </div>
      )}
    </>
  );
}

function EvalsTab() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    getEvals().then(setData);
  }, []);
  if (!data) return <p className="muted">Loading…</p>;
  if (data.status === "not_run") {
    return <p className="muted">Evals have not been run yet. Coming on day 3.</p>;
  }
  return <pre className="evals-raw">{JSON.stringify(data, null, 2)}</pre>;
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [tab, setTab] = useState<"ask" | "evals">("ask");

  useEffect(() => {
    getDocuments().then(setDocuments);
  }, []);

  return (
    <div className="page">
      <header>
        <h1>Rapport</h1>
        <p className="tagline">An analyst agent over annual reports, built on Mistral.</p>
        <nav>
          <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>
            Ask
          </button>
          <button className={tab === "evals" ? "active" : ""} onClick={() => setTab("evals")}>
            Evals
          </button>
        </nav>
      </header>
      <main>{tab === "ask" ? <AskTab documents={documents} /> : <EvalsTab />}</main>
      <footer>
        Answers are grounded in the 2025 annual reports listed above, with page-level citations.
      </footer>
    </div>
  );
}
