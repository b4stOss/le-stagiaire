import { useEffect, useLayoutEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnswerEvent, Citation } from "./api";

/** Replace [c123] / [c123, c456] markers with numbered links to sources. */
function linkCitations(answer: string, citations: Citation[]): string {
  const indexOf = new Map(citations.map((c, i) => [c.chunk_id, i + 1]));
  return answer.replace(/\[([^\]]*?c\d+[^\]]*?)\]/g, (whole, group: string) => {
    const ids = [...group.matchAll(/c(\d+)/g)].map((m) => Number(m[1]));
    const nums = ids.map((id) => indexOf.get(id)).filter(Boolean);
    if (nums.length === 0) return whole;
    return nums.map((n) => `[${n}](#cite-${n})`).join("");
  });
}

function pagesLabel(c: Citation): string {
  return c.page_start === c.page_end ? `p. ${c.page_start}` : `p. ${c.page_start}–${c.page_end}`;
}

interface SidenoteProps {
  citation: Citation;
  n: number;
  top?: number;
  active: boolean;
  onHover: (n: number | null) => void;
  measureRef?: (el: HTMLDivElement | null) => void;
}

function Sidenote({ citation, n, top, active, onHover, measureRef }: SidenoteProps) {
  const [open, setOpen] = useState(false);
  return (
    <div
      ref={measureRef}
      id={`cite-${n}`}
      className={`sidenote${active ? " active" : ""}`}
      style={top !== undefined ? { top } : undefined}
      onMouseEnter={() => onHover(n)}
      onMouseLeave={() => onHover(null)}
    >
      <div className="sidenote-head">
        <span className="sidenote-n">{n}</span>
        <span className="sidenote-src">
          {citation.company} <span className="sidenote-pages">{pagesLabel(citation)}</span>
        </span>
      </div>
      {citation.section_path && <div className="sidenote-section">{citation.section_path}</div>}
      <button className="sidenote-toggle" onClick={() => setOpen(!open)}>
        {open ? "Hide excerpt" : "Read excerpt"}
      </button>
      {open && <blockquote className="sidenote-quote">{citation.quote}</blockquote>}
    </div>
  );
}

export default function Answer({ result }: { result: AnswerEvent }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const noteRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [tops, setTops] = useState<number[] | null>(null);
  const [wide, setWide] = useState(() => window.matchMedia("(min-width: 1080px)").matches);
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1080px)");
    const onChange = (e: MediaQueryListEvent) => setWide(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Align each sidenote with its first marker in the text, then resolve
  // collisions by stacking downward.
  useLayoutEffect(() => {
    if (!wide || !bodyRef.current) {
      setTops(null);
      return;
    }
    const body = bodyRef.current;
    const compute = () => {
      const bodyTop = body.getBoundingClientRect().top;
      const desired: number[] = result.citations.map((_, i) => {
        const marker = body.querySelector<HTMLElement>(`a[href="#cite-${i + 1}"]`);
        return marker ? marker.getBoundingClientRect().top - bodyTop : i * 90;
      });
      const heights = noteRefs.current.map((el) => (el ? el.offsetHeight : 80));
      const adjusted: number[] = [];
      let floor = 0;
      desired.forEach((d, i) => {
        const top = Math.max(d, floor);
        adjusted.push(top);
        floor = top + heights[i] + 12;
      });
      setTops((prev) => (JSON.stringify(prev) === JSON.stringify(adjusted) ? prev : adjusted));
    };
    compute();
    const observer = new ResizeObserver(compute);
    observer.observe(body);
    return () => observer.disconnect();
  }, [wide, result]);

  const markerRenderer = {
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.slice(6));
        return (
          <a
            href={href}
            className={`cite-marker${active === n ? " active" : ""}`}
            onMouseEnter={() => setActive(n)}
            onMouseLeave={() => setActive(null)}
            onClick={(e) => {
              if (wide) e.preventDefault();
              document.getElementById(`cite-${n}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
            }}
          >
            {children}
          </a>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <div className={`answer-layout${wide ? " wide" : ""}`}>
      <div className="answer-sheet" ref={bodyRef}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markerRenderer}>
          {linkCitations(result.answer, result.citations)}
        </ReactMarkdown>
        <div className="answer-stats">
          {result.trace.length} {result.trace.length === 1 ? "search" : "searches"} ·{" "}
          {result.iterations} steps · {(result.prompt_tokens + result.completion_tokens).toLocaleString()} tokens ·{" "}
          {result.duration_s}s{result.capped ? " · stopped at iteration cap" : ""}
        </div>
      </div>
      <div className="answer-margin">
        {result.citations.map((c, i) => (
          <Sidenote
            key={c.chunk_id}
            citation={c}
            n={i + 1}
            top={wide && tops ? tops[i] : undefined}
            active={active === i + 1}
            onHover={setActive}
            measureRef={(el) => (noteRefs.current[i] = el)}
          />
        ))}
      </div>
    </div>
  );
}
