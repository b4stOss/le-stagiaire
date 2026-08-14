import { useState } from "react";
import FlameMark from "./FlameMark";
import { AnswerEvent, SearchEvent } from "./api";

interface TraceEntry {
  query: string;
  company: string | null;
  n_results?: number;
  latency_ms?: number;
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="trace-chevron"
      style={{ transform: open ? "rotate(180deg)" : "rotate(0)" }}
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

function Timeline({ entries }: { entries: TraceEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="trace-rail">
      {entries.map((e, i) => (
        <div key={i} className="trace-line" style={{ animationDelay: `${Math.min(i, 3) * 40}ms` }}>
          <span className="trace-verb">search</span>
          <span className="trace-query">{e.query}</span>
          {e.company && <span className="trace-scope">{e.company}</span>}
          {e.n_results !== undefined && (
            <span className="trace-meta">
              {e.n_results} results · {((e.latency_ms ?? 0) / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

interface TraceProps {
  searches: SearchEvent[];
  result: AnswerEvent | null;
  running: boolean;
  streaming: boolean;
  elapsed: number;
}

/** The agent's work log. Live while it runs (animated mark, shimmer status,
    elapsed time); collapses into a one-line receipt once the answer lands. */
export default function Trace({ searches, result, running, streaming, elapsed }: TraceProps) {
  const [open, setOpen] = useState(false);

  if (!running && !result) return null;

  if (result) {
    const tokens = result.prompt_tokens + result.completion_tokens;
    return (
      <div className="trace done">
        <button className="trace-head" aria-expanded={open} onClick={() => setOpen(!open)}>
          <FlameMark px={4} />
          <span className="trace-label">
            {result.trace.length} {result.trace.length === 1 ? "search" : "searches"} ·{" "}
            {result.duration_s}s · {tokens.toLocaleString()} tokens
            {result.capped ? " · stopped at iteration cap" : ""}
          </span>
          <Chevron open={open} />
        </button>
        <div className={`trace-body${open ? "" : " closed"}`}>
          <div>
            <Timeline entries={result.trace} />
          </div>
        </div>
      </div>
    );
  }

  const label = streaming
    ? "Writing the answer…"
    : searches.length === 0
      ? "Reading the question…"
      : "Going through the filings…";

  return (
    <div className="trace" aria-live="polite">
      <div className="trace-head">
        <FlameMark px={4} animate />
        <span className="trace-label shimmer">{label}</span>
        <span className="trace-elapsed">{elapsed.toFixed(1)}s</span>
      </div>
      <div className="trace-body">
        <div>
          <Timeline entries={searches} />
        </div>
      </div>
    </div>
  );
}
