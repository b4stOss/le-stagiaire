import { useEffect, useState } from "react";
import { getEvals } from "./api";

interface Rate {
  passed: number;
  total: number;
  pct: number | null;
}

interface EvalQuestion {
  id: string;
  category: string;
  question: string;
  verified: boolean;
  answer: string;
  passed: boolean;
  judge_reasoning?: string;
  citation_ok?: boolean;
  retrieval_recall?: boolean;
  abstained?: boolean;
  n_searches: number;
  duration_s: number;
}

interface EvalResults {
  status?: string;
  run_at?: string;
  summary?: {
    overall: Rate;
    by_category: Record<string, Rate>;
    citation_accuracy: Rate;
    retrieval_recall: Rate;
    correct_abstention: Rate;
    false_refusal: { count: number; total: number };
    verified_share: Rate;
  };
  questions?: EvalQuestion[];
}

const CATEGORY_INFO: Record<string, { label: string; how: string; desc: string }> = {
  numeric: {
    label: "Numeric extraction",
    how: "deterministic",
    desc: "Exact figures, graded by parsing and comparing numbers, no LLM involved",
  },
  synthesis: {
    label: "Synthesis",
    how: "LLM judge",
    desc: "Open questions graded pass/fail against hand-verified key facts",
  },
  comparison: {
    label: "Cross-document comparison",
    how: "LLM judge",
    desc: "Questions that require searching several reports and combining results",
  },
  unanswerable: {
    label: "Unanswerable",
    how: "abstention check",
    desc: "Traps whose answer is not in the corpus; the agent must decline, not guess",
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
  useEffect(() => {
    getEvals().then((d) => setData(d as EvalResults));
  }, []);

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
          Every gold answer was hand-verified against the source PDFs. Figures and refusals are
          graded deterministically; only synthesis and comparison use an LLM judge. This is a
          regression harness, not a proof: it makes quality measurable across changes.
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
