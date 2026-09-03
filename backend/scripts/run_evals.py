"""Run the golden set through the agent and grade it.

Usage: uv run python -m scripts.run_evals [--only num-01,una-02]

Writes data/evals/results.json (served by the API, displayed in the UI)
and prints a per-category summary.
"""

import argparse
import json
import logging
import time

from app.agent import answer_question
from app.config import DATA_DIR, REPO_ROOT
from app.evals import (
    grade_abstention,
    grade_citation,
    grade_numeric,
    grade_retrieval_recall,
    grade_with_judge,
)
from app.logs import setup_logging

GOLDEN_FILE = REPO_ROOT / "backend" / "evals" / "golden.json"
RESULTS_FILE = DATA_DIR / "evals" / "results.json"

log = logging.getLogger(__name__)


def run_one(q: dict) -> dict:
    t0 = time.time()
    result = answer_question(q["question"])
    record = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "verified": q.get("verified", False),
        "answer": result.answer,
        "n_searches": len(result.trace),
        "tokens": result.prompt_tokens + result.completion_tokens,
        "duration_s": round(time.time() - t0, 2),
    }

    if q["category"] == "unanswerable":
        abstained, reasoning = grade_abstention(q["question"], result.answer)
        record.update(passed=abstained, abstained=abstained, judge_reasoning=reasoning)
        return record

    # answerable categories
    if q["category"] == "numeric":
        record["passed"] = grade_numeric(result.answer, q["gold"])
    else:  # synthesis / comparison
        passed, reasoning = grade_with_judge(q["question"], q["gold_facts"], result.answer)
        record.update(passed=passed, judge_reasoning=reasoning)

    if q.get("gold_pages"):
        record["citation_ok"] = grade_citation(result, q["gold_pages"])
        record["retrieval_recall"] = grade_retrieval_recall(result, q["gold_pages"])

    # false-refusal check: did the agent wrongly abstain on an answerable question?
    abstained, _ = grade_abstention(q["question"], result.answer)
    record["abstained"] = abstained
    return record


def aggregate(records: list[dict]) -> dict:
    def rate(items: list[dict], key: str) -> dict:
        graded = [r for r in items if key in r]
        passed = sum(1 for r in graded if r[key])
        return {
            "passed": passed,
            "total": len(graded),
            "pct": round(100 * passed / len(graded)) if graded else None,
        }

    by_category = {}
    for cat in ("numeric", "synthesis", "comparison", "unanswerable"):
        items = [r for r in records if r["category"] == cat]
        if items:
            by_category[cat] = rate(items, "passed")

    answerable = [r for r in records if r["category"] != "unanswerable"]
    return {
        "overall": rate(records, "passed"),
        "by_category": by_category,
        "citation_accuracy": rate(records, "citation_ok"),
        "retrieval_recall": rate(records, "retrieval_recall"),
        "correct_abstention": rate([r for r in records if r["category"] == "unanswerable"], "abstained"),
        "false_refusal": {
            "count": sum(1 for r in answerable if r.get("abstained")),
            "total": len(answerable),
        },
        "verified_share": rate(records, "verified"),
    }


def _status_line(record: dict) -> str:
    status = "PASS" if record["passed"] else "FAIL"
    extras = []
    if "citation_ok" in record:
        extras.append(f"cite {'ok' if record['citation_ok'] else 'KO'}")
    if "retrieval_recall" in record:
        extras.append(f"recall {'ok' if record['retrieval_recall'] else 'KO'}")
    return status + (f" ({', '.join(extras)})" if extras else "")


def print_summary(summary: dict) -> None:
    print("\n=== Summary ===")
    for name, s in [("overall", summary["overall"]), *summary["by_category"].items()]:
        print(f"{name:14} {s['passed']}/{s['total']}" + (f" ({s['pct']}%)" if s["pct"] is not None else ""))
    ca, rr = summary["citation_accuracy"], summary["retrieval_recall"]
    print(f"{'citations':14} {ca['passed']}/{ca['total']}")
    print(f"{'recall':14} {rr['passed']}/{rr['total']}")
    print(f"{'false refusal':14} {summary['false_refusal']['count']}/{summary['false_refusal']['total']}")


def main() -> None:
    setup_logging(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated question ids")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_FILE.read_text())["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        golden = [q for q in golden if q["id"] in wanted]

    records = []
    for q in golden:
        print(f"[{q['id']}] {q['question'][:70]}...", flush=True)
        record = run_one(q)
        print(f"    {_status_line(record)}", flush=True)
        records.append(record)

    summary = aggregate(records)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {"run_at": time.strftime("%Y-%m-%d %H:%M"), "summary": summary, "questions": records},
            ensure_ascii=False,
            indent=2,
        )
    )
    print_summary(summary)
    print(f"\nresults -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
