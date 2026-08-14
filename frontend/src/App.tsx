import { useEffect, useState } from "react";
import Answer from "./Answer";
import Evals from "./Evals";
import {
  AgentEvent,
  AnswerEvent,
  DocumentInfo,
  SearchEvent,
  askStream,
  getDocuments,
} from "./api";

const EXAMPLES = [
  "What was Stellantis' net revenue in 2025, and how did it evolve vs 2024?",
  "Quelle est la dette nette hors location de TotalEnergies fin 2025 ?",
  "Compare the climate-related risk factors of TotalEnergies and BNP Paribas.",
  "Between Stellantis and ASML, which company was more profitable in 2025?",
];

function AgentLog({ searches, running }: { searches: SearchEvent[]; running: boolean }) {
  if (searches.length === 0 && !running) return null;
  return (
    <div className="agent-log" aria-live="polite">
      {searches.map((s, i) => (
        <div key={i} className="log-line">
          <span className="log-verb">search</span>
          <span className="log-query">{s.query}</span>
          {s.company && <span className="log-scope">{s.company}</span>}
        </div>
      ))}
      {running && (
        <div className="log-line pulse">
          <span className="log-verb">agent</span>
          <span className="log-query">reading results…</span>
        </div>
      )}
    </div>
  );
}

function AskTab({ documents }: { documents: DocumentInfo[] }) {
  const [question, setQuestion] = useState("");
  const [searches, setSearches] = useState<SearchEvent[]>([]);
  const [result, setResult] = useState<AnswerEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

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

  const totalPages = documents.reduce((sum, d) => sum + d.pages, 0);

  return (
    <>
      <p className="corpus-line">
        {documents.map((d, i) => (
          <span key={d.company}>
            {i > 0 && <span className="corpus-sep"> · </span>}
            <span className="corpus-company">{d.company}</span>
          </span>
        ))}
        {documents.length > 0 && (
          <span className="corpus-meta">
            {" "}
            — FY2025 filings, {totalPages.toLocaleString()} pages indexed
          </span>
        )}
      </p>

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
          placeholder="Ask about these filings, in English or French…"
          disabled={running}
          aria-label="Question"
        />
        <button type="submit" disabled={running || !question.trim()}>
          {running ? "Working…" : "Ask"}
        </button>
      </form>

      {!result && !running && !error && (
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example" onClick={() => ask(ex)}>
              {ex}
            </button>
          ))}
        </div>
      )}

      <AgentLog searches={searches} running={running} />

      {error && (
        <div className="error">
          The request failed: {error}. Check that the server is running, then try again.
        </div>
      )}

      {result && <Answer result={result} />}
    </>
  );
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [tab, setTab] = useState<"ask" | "evals">("ask");

  useEffect(() => {
    getDocuments().then(setDocuments);
  }, []);

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-row">
          <h1 className="wordmark">Rapport</h1>
          <nav aria-label="Sections">
            <button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>
              Ask
            </button>
            <button className={tab === "evals" ? "active" : ""} onClick={() => setTab("evals")}>
              Evals
            </button>
          </nav>
        </div>
        <p className="tagline">
          An analyst agent over annual reports. Every figure cites its page.
        </p>
      </header>
      <main>{tab === "ask" ? <AskTab documents={documents} /> : <Evals />}</main>
      <footer>
        Built on Mistral: OCR, embeddings and agent all run on La Plateforme.
        Answers come only from the filings; when the information is not there, the agent says so.
      </footer>
    </div>
  );
}
