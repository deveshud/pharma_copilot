from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.retriever import LocalRetriever


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_PATH = OUTPUTS_DIR / "retrieval_embeddings.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve the top matching chunks for a user query.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural-language user query to retrieve against the chunk embeddings.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the retrieval embeddings JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print score, section title, chunk length, and preview for ranked chunks.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Embeddings JSON not found: {input_path.resolve()}")

    query = args.query.strip() if args.query else input("Enter your query: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    retriever = LocalRetriever()
    if args.debug:
        results = retriever.retrieve_debug_from_file(query, input_path, top_k=args.top_k)
    else:
        results = retriever.retrieve_from_file(query, input_path, top_k=args.top_k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
