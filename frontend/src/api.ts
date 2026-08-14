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

export async function getDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch("/api/documents");
  return res.json();
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

export async function getEvals(): Promise<Record<string, unknown>> {
  const res = await fetch("/api/evals");
  return res.json();
}
