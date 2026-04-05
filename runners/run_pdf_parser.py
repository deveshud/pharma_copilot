from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.pdf_parser import PdfParser, parse_pdf_folder, save_results_to_json


if __name__ == "__main__":
    folder = PROJECT_ROOT / "data" / "sample_docs"
    print(f"Looking for PDF files in: {folder.resolve()}")

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
    else:
        parsed_results = parse_pdf_folder(
            str(folder),
            PdfParser(include_document_metadata=True, include_empty_pages=True),
        )
        for file_name, blocks in parsed_results.items():
            print(f"{file_name}: {len(blocks)} blocks")

        save_results_to_json(parsed_results, "parsed_outputs\\normalized_pdf_blocks.json")
        print("Saved: normalized_pdf_blocks.json")
