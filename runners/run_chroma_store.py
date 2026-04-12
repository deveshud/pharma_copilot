from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.chroma_store import DEFAULT_CHROMA_COLLECTION, ChromaEmbeddingStore


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_PATH = OUTPUTS_DIR / "retrieval_embeddings.json"
DEFAULT_PERSIST_PATH = OUTPUTS_DIR / "chroma_db"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Store generated retrieval embeddings in a persistent ChromaDB collection.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the generated retrieval embeddings JSON file.",
    )
    parser.add_argument(
        "--persist-path",
        default=str(DEFAULT_PERSIST_PATH),
        help="Directory where ChromaDB should persist the vector database.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of records to upsert per ChromaDB call.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before inserting records.",
    )
    return parser


def print_store_summary(summary: dict[str, object]) -> None:
    print("ChromaDB store summary:")
    print(f"- Persist path: {summary['persist_path']}")
    print(f"- Collection: {summary['collection_name']}")
    print(f"- Records seen: {summary['records_seen']}")
    print(f"- Records stored: {summary['records_stored']}")
    print(f"- Reset collection: {summary['reset_collection']}")


def main() -> int:
    args = build_argument_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Embeddings JSON not found: {input_path.resolve()}")

    store = ChromaEmbeddingStore(
        persist_path=args.persist_path,
        collection_name=args.collection,
    )
    summary = store.store_file(
        input_path,
        reset_collection=args.reset,
        batch_size=args.batch_size,
    )

    print_store_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
