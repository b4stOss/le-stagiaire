"""Quick CLI check of hybrid retrieval, before any agent exists.

Usage: uv run python -m scripts.search "average debt maturity" [--company TotalEnergies]
"""

import argparse

from app.retrieval import hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--company", default=None)
    parser.add_argument("-k", type=int, default=8)
    args = parser.parse_args()

    for i, c in enumerate(hybrid_search(args.query, company=args.company, k=args.k), 1):
        pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"p. {c.page_start}-{c.page_end}"
        print(f"\n--- [{i}] {c.company} {c.fiscal_year}, {pages} (rrf {c.score:.4f})")
        if c.section_path:
            print(f"    {c.section_path}")
        print(c.content[:400].replace("\n", " ") + ("..." if len(c.content) > 400 else ""))


if __name__ == "__main__":
    main()
