"""Ask the agent a question from the CLI.

Usage: uv run python -m scripts.ask "Compare the net debt of TotalEnergies and Stellantis"
"""

import argparse

from app.agent import answer_question
from app.logs import setup_logging
from app.retrieval import pages_label


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    result = answer_question(args.question)

    print("\n" + "=" * 70)
    print(result.answer)
    print("=" * 70)
    print(
        f"\n{result.iterations} iterations, {len(result.trace)} searches, "
        f"{result.prompt_tokens}+{result.completion_tokens} tokens" + (", CAPPED" if result.capped else "")
    )
    for c in result.citations:
        pages = pages_label(c["page_start"], c["page_end"])
        print(f"  [c{c['chunk_id']}] {c['company']} {c['fiscal_year']}, {pages} | {c['section_path'][:70]}")


if __name__ == "__main__":
    main()
