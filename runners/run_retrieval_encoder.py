from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.retrieval_encoder import DEFAULT_RETRIEVAL_MODEL, RetrievalEncoder


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_PATH = OUTPUTS_DIR / "structural_chunks.json"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "retrieval_embeddings.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode structural chunks into retrieval-oriented embeddings.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the chunked JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where encoded embeddings JSON should be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RETRIEVAL_MODEL,
        help="Sentence Transformers model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use during embedding generation.",
    )
    return parser


def print_embedding_summary(encoded_output: dict[str, object]) -> None:
    stats = encoded_output["stats"]
    model = encoded_output["model"]
    print("Embedding summary:")
    print(f"- Source documents: {stats['source_document_count']}")
    print(f"- Chunks encoded: {stats['chunk_count']}")
    print(f"- Model: {model['name']}")
    print(f"- Embedding dimension: {model['embedding_dimension']}")


def main() -> int:
    args = build_argument_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path.resolve()}")

    encoder = RetrievalEncoder(model_name=args.model, batch_size=args.batch_size)
    encoded_output = encoder.encode_file(input_path)
    saved_path = encoder.save_embeddings_output(encoded_output, output_path)

    print_embedding_summary(encoded_output)
    print(f"\nSaved retrieval embeddings to: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
