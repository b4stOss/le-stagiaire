"""Grading logic for the golden set.

Deterministic wherever possible (numbers, citations, retrieval recall);
an LLM judge (mistral-small, temperature 0, binary verdict, reference-guided)
only where text quality genuinely needs judgment (synthesis, comparison)
and to classify answer-vs-abstention on unanswerable questions.
"""

import json
import re

from app.agent import AgentAnswer
from app.config import settings
from app.mistral import chat_complete

# ---------------------------------------------------------------- numeric

_SCALES = {
    "billion": 1000.0, "billions": 1000.0, "bn": 1000.0,
    "milliard": 1000.0, "milliards": 1000.0, "md": 1000.0,
    "million": 1.0, "millions": 1.0, "mn": 1.0,
    "thousand": 0.001, "milliers": 0.001, "k": 0.001,
}

_NUM_RE = re.compile(r"\(?-?\d[\d\s  ,.]*\d|\(?-?\d\)?")


def _parse_number(raw: str) -> float | None:
    """Handle '153,508' (EN thousands), '20 215' (FR thousands), '14,7' (FR decimal),
    '4.7', and accounting negatives '(22,332)'."""
    negative = raw.strip().startswith("(") or raw.strip().startswith("-")
    s = raw.strip().strip("()-").replace(" ", " ").replace(" ", " ")
    s = s.replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rindex(".") > s.rindex(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        # comma + exactly 3 digits and no other commas in tail -> thousands separator
        s = s.replace(",", "") if len(tail) == 3 and head.replace(",", "").isdigit() else s.replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def extract_candidates_in_millions(text: str) -> list[float]:
    """All numbers in the text, normalized to millions using a nearby scale word."""
    candidates: list[float] = []
    for match in _NUM_RE.finditer(text):
        value = _parse_number(match.group())
        if value is None:
            continue
        window = text[match.end() : match.end() + 25].lower()
        factor = 1.0
        for word, f in _SCALES.items():
            if re.match(rf"\s*(?:de\s+)?{re.escape(word)}\b", window) or re.match(rf"\s*{re.escape(word)}[€$]", window):
                factor = f
                break
        candidates.append(value * factor)
        if factor == 1.0:
            candidates.append(value)  # raw value too, in case it was already in millions
    return candidates


def grade_numeric(answer: str, gold: dict) -> bool:
    """gold: {"value": 153508, "unit": "EUR", "scale": "million"|"percent"|"raw", "tolerance_pct": 0.5}

    "million": candidates normalized to millions via nearby scale words.
    "percent"/"raw": plain number comparison (ratios, kboe/d, counts...)."""
    gold_value = float(gold["value"])
    tolerance = abs(gold_value) * float(gold.get("tolerance_pct", 0.5)) / 100
    if gold.get("scale") in ("percent", "raw"):
        candidates = [v for m in _NUM_RE.finditer(answer) if (v := _parse_number(m.group())) is not None]
    else:
        candidates = extract_candidates_in_millions(answer)
    # compare magnitudes: "a net loss of EUR 22,332 million" is a correct answer
    # for a gold value of -22332 (the sign is carried by the wording)
    return any(abs(abs(c) - abs(gold_value)) <= tolerance for c in candidates)


# ---------------------------------------------------------------- citations & retrieval

def _pages_overlap(page_start: int, page_end: int, gold_pages: list[int], slack: int = 1) -> bool:
    return any(page_start - slack <= p <= page_end + slack for p in gold_pages)


def grade_citation(result: AgentAnswer, gold_pages: list[dict]) -> bool:
    """At least one citation points at a gold page (±1) of the right company."""
    for spec in gold_pages:
        for c in result.citations:
            if c["company"] == spec["company"] and _pages_overlap(c["page_start"], c["page_end"], spec["pages"]):
                return True
    return False


def grade_retrieval_recall(result: AgentAnswer, gold_pages: list[dict]) -> bool:
    """The gold page was in the search results the agent saw (cited or not)."""
    for spec in gold_pages:
        for r in result.retrieved:
            if r["company"] == spec["company"] and _pages_overlap(r["page_start"], r["page_end"], spec["pages"]):
                return True
    return False


# ---------------------------------------------------------------- LLM judge

def _judge_call(prompt: str) -> dict:
    resp = chat_complete(
        model=settings.judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp.choices[0].message.content
    return json.loads(content if isinstance(content, str) else content[0].text)


JUDGE_PROMPT = """\
You are grading an answer produced by a document-grounded assistant, against reference \
key facts that were verified by a human in the source documents.

Question: {question}

Reference key facts (ground truth): {gold_facts}

Candidate answer:
{answer}

Grade PASS if the candidate answer is consistent with the reference key facts and \
actually answers the question. Grade FAIL ONLY if it contradicts the reference or \
misses its core facts. IMPORTANT: the reference is a MINIMAL set of key facts, not an \
exhaustive one; additional details beyond the reference must NOT cause a FAIL. \
Do NOT reward length or style.

Respond in JSON: {{"reasoning": "<2-3 sentences>", "verdict": "pass" | "fail"}}"""


ABSTENTION_PROMPT = """\
A document-grounded assistant was asked a question whose answer is NOT in its documents. \
Classify its response.

Question: {question}

Response:
{answer}

Did the assistant correctly abstain (clearly state the information is not in the documents, \
without providing a substantive made-up answer)? Respond in JSON: \
{{"reasoning": "<1-2 sentences>", "verdict": "abstained" | "answered"}}"""


def grade_with_judge(question: str, gold_facts: str, answer: str) -> tuple[bool, str]:
    out = _judge_call(JUDGE_PROMPT.format(question=question, gold_facts=gold_facts, answer=answer))
    return out.get("verdict") == "pass", out.get("reasoning", "")


def grade_abstention(question: str, answer: str) -> tuple[bool, str]:
    out = _judge_call(ABSTENTION_PROMPT.format(question=question, answer=answer))
    return out.get("verdict") == "abstained", out.get("reasoning", "")
