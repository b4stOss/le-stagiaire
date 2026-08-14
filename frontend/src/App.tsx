import { useEffect, useRef, useState } from "react";
import Answer, { StreamingAnswer } from "./Answer";
import Evals from "./Evals";
import FlameMark from "./FlameMark";
import Trace from "./Trace";
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

function AskTab({ documents }: { documents: DocumentInfo[] }) {
  const [question, setQuestion] = useState("");
  const [searches, setSearches] = useState<SearchEvent[]>([]);
  const [streamText, setStreamText] = useState("");
  const [result, setResult] = useState<AnswerEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(0);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setElapsed((Date.now() - startedAt.current) / 1000), 100);
    return () => clearInterval(timer);
  }, [running]);

  const ask = async (q: string) => {
    if (!q.trim() || running) return;
    setQuestion(q);
    setSearches([]);
    setStreamText("");
    setResult(null);
    setError(null);
    setElapsed(0);
    startedAt.current = Date.now();
    setRunning(true);
    try {
      await askStream(q, (event: AgentEvent) => {
        if (event.type === "search") setSearches((prev) => [...prev, event]);
        else if (event.type === "token") setStreamText((prev) => prev + event.text);
        else if (event.type === "reset") setStreamText("");
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
            - FY2025 filings, {totalPages.toLocaleString()} pages indexed
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

      <Trace
        searches={searches}
        result={result}
        running={running}
        streaming={streamText.length > 0}
        elapsed={elapsed}
      />

      {error && (
        <div className="error">
          The request failed: {error}. Check that the server is running, then try again.
        </div>
      )}

      {result ? <Answer result={result} /> : streamText && <StreamingAnswer text={streamText} />}
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
          <h1 className="wordmark">
            <FlameMark px={7} />
            Le Stagiaire
          </h1>
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
          The intern who actually reads the filings. Every figure cites its page.
        </p>
      </header>
      <main>{tab === "ask" ? <AskTab documents={documents} /> : <Evals />}</main>
      <footer>
        Built on Mistral: OCR, embeddings and agent all run on La Plateforme. When the answer
        is not in the filings, Le Stagiaire says so instead of guessing.
      </footer>
    </div>
  );
}
