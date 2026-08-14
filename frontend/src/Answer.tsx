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

/** While streaming, number markers live by order of appearance (the same
    order the server resolves them in) and hide a trailing half-written marker. */
function liveCitations(text: string): string {
  const settled = text.replace(/\[[^\]]*$/, "");
  const order = new Map<number, number>();
  return settled.replace(/\[([^\]]*?c\d+[^\]]*?)\]/g, (_whole, group: string) => {
    const ids = [...group.matchAll(/c(\d+)/g)].map((m) => Number(m[1]));
    return ids
      .map((id) => {
        if (!order.has(id)) order.set(id, order.size + 1);
        return `[${order.get(id)}](#cite-live)`;
      })
      .join("");
  });
}

const liveMarkerRenderer = {
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
    if (href === "#cite-live") return <span className="cite-marker">{children}</span>;
    return (
      <a href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  },
};

/** The answer sheet while tokens are still arriving: live markers, blinking caret. */
export function StreamingAnswer({ text }: { text: string }) {
  return (
    <div className="answer-layout">
      <div className="answer-sheet streaming">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={liveMarkerRenderer}>
          {liveCitations(text)}
        </ReactMarkdown>
      </div>
    </div>
  );
}

function pagesLabel(c: Citation): string {
  return c.page_start === c.page_end ? `p. ${c.page_start}` : `p. ${c.page_start}–${c.page_end}`;
}

interface SidenoteProps {
  citation: Citation;
  n: number;
  top?: number;
  active: boolean;
  flash: boolean;
  open: boolean;
  onToggle: (n: number) => void;
  onHover: (n: number | null) => void;
  measureRef?: (el: HTMLDivElement | null) => void;
}

function Sidenote({ citation, n, top, active, flash, open, onToggle, onHover, measureRef }: SidenoteProps) {
  return (
    <div
      ref={measureRef}
      id={`cite-${n}`}
      className={`sidenote${active ? " active" : ""}${flash ? " flash" : ""}`}
      style={{
        ...(top !== undefined ? { top } : undefined),
        animationDelay: `${Math.min(n - 1, 5) * 70}ms`,
      }}
      onMouseEnter={() => onHover(n)}
      onMouseLeave={() => onHover(null)}
    >
      <button className="sidenote-head" aria-expanded={open} onClick={() => onToggle(n)}>
        <span className="sidenote-n">{n}</span>
        <span className="sidenote-src">
          {citation.company} <span className="sidenote-pages">{pagesLabel(citation)}</span>
        </span>
        <svg
          className="sidenote-chevron"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0)" }}
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {citation.section_path && <div className="sidenote-section">{citation.section_path}</div>}
      <div className={`sidenote-drop${open ? " open" : ""}`}>
        <div>
          <blockquote className="sidenote-quote">{citation.quote}</blockquote>
        </div>
      </div>
    </div>
  );
}

export default function Answer({ result }: { result: AnswerEvent }) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const noteRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [tops, setTops] = useState<number[] | null>(null);
  const [wide, setWide] = useState(() => window.matchMedia("(min-width: 1080px)").matches);
  const [active, setActive] = useState<number | null>(null);
  const [openNotes, setOpenNotes] = useState<Set<number>>(new Set());
  const [flash, setFlash] = useState<number | null>(null);
  const flashTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1080px)");
    const onChange = (e: MediaQueryListEvent) => setWide(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    setOpenNotes(new Set());
    setFlash(null);
    return () => window.clearTimeout(flashTimer.current);
  }, [result]);

  // Align each sidenote with its first marker in the text, then resolve
  // collisions by stacking downward. Notes are observed too, so opening an
  // excerpt pushes everything below it instead of overlapping.
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
    noteRefs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [wide, result]);

  const toggleNote = (n: number) =>
    setOpenNotes((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });

  /** Marker clicked: open the note's excerpt, bring it into view, flash it. */
  const revealNote = (n: number) => {
    setOpenNotes((prev) => new Set(prev).add(n));
    window.clearTimeout(flashTimer.current);
    setFlash(null);
    window.setTimeout(() => setFlash(n), 30);
    flashTimer.current = window.setTimeout(() => setFlash(null), 1500);
    requestAnimationFrame(() => {
      document.getElementById(`cite-${n}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  };

  const markerRenderer = {
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.slice(6));
        const c = result.citations[n - 1];
        return (
          <a
            href={href}
            className={`cite-marker${active === n ? " active" : ""}`}
            title={c ? `${c.company} · ${pagesLabel(c)}` : undefined}
            onMouseEnter={() => setActive(n)}
            onMouseLeave={() => setActive(null)}
            onClick={(e) => {
              e.preventDefault();
              revealNote(n);
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
      </div>
      <div className="answer-margin">
        {result.citations.map((c, i) => (
          <Sidenote
            key={c.chunk_id}
            citation={c}
            n={i + 1}
            top={wide && tops ? tops[i] : undefined}
            active={active === i + 1}
            flash={flash === i + 1}
            open={openNotes.has(i + 1)}
            onToggle={toggleNote}
            onHover={setActive}
            measureRef={(el) => (noteRefs.current[i] = el)}
          />
        ))}
      </div>
    </div>
  );
}
