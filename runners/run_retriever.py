from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.chroma_store import DEFAULT_CHROMA_COLLECTION
from models.retrieval_encoder import DEFAULT_RETRIEVAL_MODEL
from models.retriever import LocalRetriever


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PERSIST_PATH = OUTPUTS_DIR / "chroma_db"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top chunks from the persistent ChromaDB vector store.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural-language user query to retrieve against ChromaDB embeddings.",
    )
    parser.add_argument(
        "--persist-path",
        default=str(DEFAULT_PERSIST_PATH),
        help="Directory containing the persistent ChromaDB vector database.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_CHROMA_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence Transformers model used for query encoding. Defaults to the model stored in Chroma metadata.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final reranked chunks to return.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=25,
        help="Number of Chroma vector candidates to fetch before metadata-aware reranking.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw vector score, rerank delta, and reranking reasons.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only cleaned chunk text instead of chunk records with metadata.",
    )
    return parser


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": result["score"],
        "chunk_id": result["chunk_id"],
        "file_name": result["file_name"],
        "section_title": result["section_title"],
        "page_number": result["page_number"],
        "chunk_length": result["chunk_length"],
        "preview": result["preview"],
        "metadata": result["metadata"],
        "text": result["text"],
    }


def main() -> int:
    args = build_argument_parser().parse_args()

    query = args.query.strip() if args.query else input("Enter your query: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    persist_path = Path(args.persist_path)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_path.resolve()}. "
            "Run runners/run_chroma_store.py first."
        )

    retriever = LocalRetriever()
    collection = retriever.open_chroma_collection(
        persist_path=persist_path,
        collection_name=args.collection,
    )
    model_name = args.model or retriever.infer_chroma_model_name(collection, DEFAULT_RETRIEVAL_MODEL)

    results = retriever.retrieve_debug_from_chroma(
        query,
        collection,
        model_name=model_name,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    if args.text_only:
        output: object = [result["text"] for result in results]
    elif args.debug:
        output = results
    else:
        output = [compact_result(result) for result in results]

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
