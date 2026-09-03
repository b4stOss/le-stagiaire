import { useEffect, useState } from "react";
import { EvalQuestion, EvalResults, Rate, getEvals } from "./api";

const CATEGORY_INFO: Record<string, { label: string; how: string; desc: string }> = {
  numeric: {
    label: "Numeric extraction",
    how: "deterministic",
    desc: "Exact figures, no LLM in the grading",
  },
  synthesis: {
    label: "Synthesis",
    how: "LLM judge",
    desc: "Graded against hand-verified key facts",
  },
  comparison: {
    label: "Cross-document comparison",
    how: "LLM judge",
    desc: "Several reports combined in one answer",
  },
  unanswerable: {
    label: "Unanswerable",
    how: "abstention check",
    desc: "Traps: the agent must decline, not guess",
  },
};

/** One dot per golden-set question: filled green = passed, hollow red = failed.
    Pass/fail is encoded by fill and color, so it survives colorblindness. */
function Dots({ rate }: { rate: Rate }) {
  return (
    <div className="dots" role="img" aria-label={`${rate.passed} of ${rate.total} passed`}>
      {Array.from({ length: rate.total }, (_, i) => (
        <span key={i} className={`dot${i < rate.passed ? " pass" : " fail"}`} />
      ))}
    </div>
  );
}

function QuestionRow({ q }: { q: EvalQuestion }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="eval-q">
      <button className="eval-q-head" onClick={() => setOpen(!open)}>
        <span className={`verdict ${q.passed ? "pass" : "fail"}`}>{q.passed ? "PASS" : "FAIL"}</span>
        <span className="eval-q-text">{q.question}</span>
        <span className="eval-q-id">{q.id}</span>
      </button>
      {open && (
        <div className="eval-q-body">
          <p className="eval-q-answer">{q.answer}</p>
          <div className="eval-q-meta">
            {q.citation_ok !== undefined && <span>citation {q.citation_ok ? "ok" : "missed"}</span>}
            {q.retrieval_recall !== undefined && <span>recall {q.retrieval_recall ? "ok" : "missed"}</span>}
            <span>
              {q.n_searches} {q.n_searches === 1 ? "search" : "searches"} · {q.duration_s}s
            </span>
          </div>
          {q.judge_reasoning && <p className="eval-q-judge">Judge: {q.judge_reasoning}</p>}
        </div>
      )}
    </div>
  );
}

export default function Evals() {
  const [data, setData] = useState<EvalResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getEvals().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="error">Could not load the evals: {error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  if (data.status === "not_run" || !data.summary) {
    return <p className="muted">Evals have not been run yet.</p>;
  }
  const s = data.summary;
  const questions = data.questions ?? [];

  return (
    <div className="evals">
      <div className="evals-intro">
        <div className="evals-overall">
          <span className="evals-score">
            {s.overall.passed}/{s.overall.total}
          </span>
          {s.overall.pct !== null && (
            <span className={`evals-pct${s.overall.passed === s.overall.total ? " full" : ""}`}>
              {Math.round(s.overall.pct)}%
            </span>
          )}
          <span className="evals-score-label">
            golden-set questions passed · run {data.run_at}
          </span>
        </div>
        <p className="evals-note">
          Gold answers hand-verified against the source PDFs; graded deterministically where
          possible, by an LLM judge otherwise.
        </p>
      </div>

      <div className="evals-cats">
        {Object.entries(CATEGORY_INFO).map(([key, info]) => {
          const rate = s.by_category[key];
          if (!rate) return null;
          return (
            <div key={key} className="evals-cat">
              <div className="evals-cat-head">
                <span className="evals-cat-label">{info.label}</span>
                <span className="evals-cat-how">{info.how}</span>
                <span className={`evals-cat-score${rate.passed === rate.total ? " full" : ""}`}>
                  {rate.passed}/{rate.total}
                </span>
              </div>
              <Dots rate={rate} />
              <p className="evals-cat-desc">{info.desc}</p>
            </div>
          );
        })}
      </div>

      <div className="evals-metrics">
        <div className="metric">
          <span className="metric-value">
            {s.citation_accuracy.passed}/{s.citation_accuracy.total}
          </span>
          <span className="metric-label">citations point at the verified page</span>
        </div>
        <div className="metric">
          <span className="metric-value">
            {s.retrieval_recall.passed}/{s.retrieval_recall.total}
          </span>
          <span className="metric-label">gold page found by retrieval</span>
        </div>
        <div className="metric">
          <span className={`metric-value${s.false_refusal.count === 0 ? " good" : ""}`}>
            {s.false_refusal.count}
          </span>
          <span className="metric-label">
            wrong refusals across {s.false_refusal.total} answerable questions
          </span>
        </div>
      </div>

      <div className="evals-questions">
        <h3>Questions</h3>
        {questions.map((q) => (
          <QuestionRow key={q.id} q={q} />
        ))}
      </div>
    </div>
  );
}
