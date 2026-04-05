from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.xlsx_parser import XlsxParser, parse_xlsx_folder, save_results_to_json


if __name__ == "__main__":
    folder = PROJECT_ROOT / "data" / "sample_docs"
    print(f"Looking for XLSX files in: {folder.resolve()}")

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
    else:
        parsed_results = parse_xlsx_folder(
            str(folder),
            XlsxParser(
                include_workbook_metadata=True,
                include_sheet_metadata=True,
                include_empty_rows=False,
                infer_header_row=True,
            ),
        )
        for file_name, blocks in parsed_results.items():
            print(f"{file_name}: {len(blocks)} blocks")

        save_results_to_json(parsed_results, "parsed_outputs\\normalized_xlsx_blocks.json")
        print("Saved: normalized_xlsx_blocks.json")
