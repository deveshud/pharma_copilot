from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.docx_parser import DocxParser


def parse_docx_folder(folder_path: str, parser: DocxParser | None = None):
    parser = parser or DocxParser(heading_threshold=3)
    folder = Path(folder_path)
    parsed_results = {}

    for file_path in sorted(folder.glob("*.docx")):
        if file_path.name.startswith("~$"):
            continue
        parsed_results[file_path.name] = parser.parse(str(file_path))

    return parsed_results


def save_docx_output(parsed_results, output_file: str = "parsed_outputs\\normalized_docx_blocks.json") -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_ready = {
        file_name: [asdict(block) for block in blocks]
        for file_name, blocks in parsed_results.items()
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    folder = PROJECT_ROOT / "data" / "sample_docs"
    print(f"Looking for DOCX files in: {folder.resolve()}")

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
    else:
        parsed_results = parse_docx_folder(str(folder))
        for file_name, blocks in parsed_results.items():
            print(f"{file_name}: {len(blocks)} blocks")

        save_docx_output(parsed_results)
        print("Saved: normalized_docx_blocks.json")
