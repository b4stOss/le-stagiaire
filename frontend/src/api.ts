export interface DocumentInfo {
  company: string;
  title: string;
  fiscal_year: number;
  language: string;
  pages: number;
  chunks: number;
}

export interface Citation {
  chunk_id: number;
  company: string;
  title: string;
  fiscal_year: number;
  page_start: number;
  page_end: number;
  section_path: string;
  quote: string;
}

export interface SearchEvent {
  type: "search";
  query: string;
  company: string | null;
}

export interface AnswerEvent {
  type: "answer";
  answer: string;
  citations: Citation[];
  trace: { query: string; company: string | null; n_results: number; latency_ms: number }[];
  iterations: number;
  prompt_tokens: number;
  completion_tokens: number;
  capped: boolean;
  duration_s: number;
}

export interface TokenEvent {
  type: "token";
  text: string;
}

/** Streamed text so far was pre-tool-call preamble: discard it. */
export interface ResetEvent {
  type: "reset";
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

/** The demo declined to run the question (budget, rate limit, length). Not a failure. */
export interface NoticeEvent {
  type: "notice";
  message: string;
}

export type AgentEvent =
  | SearchEvent
  | TokenEvent
  | ResetEvent
  | AnswerEvent
  | ErrorEvent
  | NoticeEvent;

export interface Rate {
  passed: number;
  total: number;
  pct: number | null;
}

export interface EvalQuestion {
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

export interface EvalResults {
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

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

export function getDocuments(): Promise<DocumentInfo[]> {
  return getJson("/api/documents");
}

export function getEvals(): Promise<EvalResults> {
  return getJson("/api/evals");
}

export async function askStream(
  question: string,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`/api/ask: ${res.status}`);
  if (!res.body) throw new Error("no response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith("data: ")) {
        onEvent(JSON.parse(line.slice(6)));
      }
    }
  }
}

