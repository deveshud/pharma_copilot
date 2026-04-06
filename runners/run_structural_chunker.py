from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chunkers.structural_chunker import StructuralChunker


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_PATH = OUTPUTS_DIR / "normalized_ingestion_blocks.json"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "structural_chunks.json"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create structure-aware chunks from normalized ingestion blocks.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the consolidated parsed JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where chunked JSON should be written.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum character budget per non-table chunk.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=380,
        help="Maximum token budget per non-table chunk, leaving room for retrieval metadata.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=40,
        help="Token overlap carried from one chunk to the next within the same section.",
    )
    return parser


def print_chunking_summary(chunked_output: dict[str, list[dict[str, object]]]) -> None:
    print("Chunking summary:")
    for file_name, chunks in chunked_output.items():
        print(f"- {file_name}: {len(chunks)} chunks")


def main() -> int:
    args = build_argument_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path.resolve()}")

    chunker = StructuralChunker(
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    consolidated_blocks = chunker.load_consolidated_blocks(input_path)
    chunked_output = chunker.chunk_consolidated_blocks(consolidated_blocks)
    saved_path = chunker.save_chunked_output(chunked_output, output_path)

    print_chunking_summary(chunked_output)
    print(f"\nSaved chunk output to: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
